"""Tests for `datasets stage` — the release pipeline's download step.

This runs in CI with an AWS role, so it must never touch the dataset API or the
interactive token flow, and it must fail the build rather than produce an image
with missing or oversized data.
"""

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from compartment.datasets.manifest import ManifestError
from compartment.datasets.stage import (
    DEFAULT_BUCKET,
    object_key,
    resolve_bucket,
    resolve_max_total_bytes,
    stage_model_dir,
)

MANIFEST = """\
datasets:
  - name: kenya-contact-matrix
    version: "1"
    file: data/kenya-contacts.csv
  - name: kenya-admin-zones
    version: "2.1"
    file: data/zones.json
"""

BUCKET = "collaboratory-datasets"
LIMIT = 2 * 1024 * 1024 * 1024


def _model_dir(tmp_path, manifest=MANIFEST):
    model_dir = tmp_path / "my_model"
    model_dir.mkdir()
    if manifest is not None:
        (model_dir / "datasets.yaml").write_text(manifest)
    return model_dir


@pytest.fixture
def s3(monkeypatch):
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 2048}

    def fake_download(bucket, key, dest):
        # Mirror boto3: the file exists afterwards.
        from pathlib import Path

        Path(dest).write_bytes(b"x" * 2048)

    client.download_file.side_effect = fake_download

    fake_boto3 = MagicMock()
    fake_boto3.client.return_value = client
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake_boto3)
    return client


def test_no_manifest_is_not_an_error(tmp_path):
    """Most models ship no datasets; the CI step must not fail for them."""
    model_dir = _model_dir(tmp_path, manifest=None)

    lines = stage_model_dir(model_dir, BUCKET, LIMIT)

    assert "nothing to stage" in lines[0]


def test_downloads_each_entry_to_its_manifest_path(tmp_path, s3):
    model_dir = _model_dir(tmp_path)

    stage_model_dir(model_dir, BUCKET, LIMIT)

    downloaded = {call.args[1]: call.args[2] for call in s3.download_file.call_args_list}
    assert downloaded == {
        "datasets/kenya-contact-matrix/1/kenya-contacts.csv":
            str(model_dir / "data" / "kenya-contacts.csv"),
        "datasets/kenya-admin-zones/2.1/zones.json":
            str(model_dir / "data" / "zones.json"),
    }
    assert (model_dir / "data" / "kenya-contacts.csv").is_file()


def test_object_key_matches_the_bucket_layout(tmp_path):
    from compartment.datasets.manifest import load_manifest

    model_dir = _model_dir(tmp_path)
    entries = load_manifest(model_dir / "datasets.yaml")

    assert object_key(entries[0]) == "datasets/kenya-contact-matrix/1/kenya-contacts.csv"


def test_missing_key_fails_the_build(tmp_path, s3):
    model_dir = _model_dir(tmp_path)
    s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "404"}}, "HeadObject"
    )

    with pytest.raises(ManifestError) as exc:
        stage_model_dir(model_dir, BUCKET, LIMIT)

    message = str(exc.value)
    assert "datasets/kenya-contact-matrix/1/kenya-contacts.csv" in message
    # The fix is to push it, so say so.
    assert "push" in message
    s3.download_file.assert_not_called()


def test_existing_file_of_matching_size_is_not_redownloaded(tmp_path, s3):
    model_dir = _model_dir(tmp_path)
    target = model_dir / "data" / "kenya-contacts.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"y" * 2048)

    lines = stage_model_dir(model_dir, BUCKET, LIMIT)

    keys = [call.args[1] for call in s3.download_file.call_args_list]
    assert "datasets/kenya-contact-matrix/1/kenya-contacts.csv" not in keys
    assert any("already present" in line for line in lines)
    # The other entry still downloads.
    assert "datasets/kenya-admin-zones/2.1/zones.json" in keys


def test_size_mismatch_forces_a_redownload(tmp_path, s3):
    """A truncated or stale local file must not be trusted."""
    model_dir = _model_dir(tmp_path)
    target = model_dir / "data" / "kenya-contacts.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"short")

    stage_model_dir(model_dir, BUCKET, LIMIT)

    keys = [call.args[1] for call in s3.download_file.call_args_list]
    assert "datasets/kenya-contact-matrix/1/kenya-contacts.csv" in keys


def test_aggregate_ceiling_fails_the_build(tmp_path, s3):
    """500 MB is per dataset; ten of them still bloats the image."""
    model_dir = _model_dir(tmp_path)
    s3.head_object.return_value = {"ContentLength": 400 * 1024 * 1024}

    with pytest.raises(ManifestError) as exc:
        stage_model_dir(model_dir, BUCKET, max_total_bytes=500 * 1024 * 1024)

    message = str(exc.value)
    assert "PANSIM_MAX_STAGED_BYTES" in message


def test_total_is_reported_for_the_build_log(tmp_path, s3):
    model_dir = _model_dir(tmp_path)

    lines = stage_model_dir(model_dir, BUCKET, LIMIT)

    assert "2 dataset(s) staged" in lines[-1]


# ---------------------------------------------------------------------------
# Environment resolution
# ---------------------------------------------------------------------------


def test_bucket_defaults_to_the_shared_services_bucket(monkeypatch):
    monkeypatch.delenv("PANSIM_DATASETS_BUCKET", raising=False)
    assert resolve_bucket() == DEFAULT_BUCKET


def test_bucket_can_be_overridden(monkeypatch):
    monkeypatch.setenv("PANSIM_DATASETS_BUCKET", "some-rebuild-bucket")
    assert resolve_bucket() == "some-rebuild-bucket"


def test_max_staged_bytes_rejects_nonsense(monkeypatch):
    monkeypatch.setenv("PANSIM_MAX_STAGED_BYTES", "loads")
    with pytest.raises(ManifestError):
        resolve_max_total_bytes()

    monkeypatch.setenv("PANSIM_MAX_STAGED_BYTES", "0")
    with pytest.raises(ManifestError):
        resolve_max_total_bytes()


def test_max_staged_bytes_override(monkeypatch):
    monkeypatch.setenv("PANSIM_MAX_STAGED_BYTES", "1234")
    assert resolve_max_total_bytes() == 1234
