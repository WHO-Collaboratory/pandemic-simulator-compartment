"""Tests for the client-side per-dataset size cap on `datasets push`.

The dataset API rejects an oversized upload too, but only after the bytes have
crossed the wire — and the scan-complete Lambda is the only layer that sees the
true size. This is the layer that saves a modeler the upload.
"""

from unittest.mock import MagicMock

import pytest

from compartment.datasets.cli import _cmd_push
from compartment.datasets.manifest import MAX_DATASET_BYTES, ManifestError


def _manifest(tmp_path, entries: list[tuple[str, str]]) -> str:
    lines = ["datasets:"]
    for name, filename in entries:
        lines.append(f"  - name: {name}")
        lines.append('    version: "1"')
        lines.append(f"    file: {filename}")
    manifest = tmp_path / "datasets.yaml"
    manifest.write_text("\n".join(lines) + "\n")
    return str(manifest)


def _sparse_file(path, size: int):
    """A file of a given apparent size without writing that many bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.truncate(size)


class _Args:
    def __init__(self, manifest):
        self.manifest = manifest
        self.names = []


def test_oversized_file_is_rejected_before_any_upload(tmp_path):
    _sparse_file(tmp_path / "big.csv", MAX_DATASET_BYTES + 1)
    manifest = _manifest(tmp_path, [("too-big", "big.csv")])
    api = MagicMock()

    with pytest.raises(ManifestError) as exc:
        _cmd_push(_Args(manifest), api, "token")

    api.push.assert_not_called()
    api.upload.assert_not_called()

    message = str(exc.value)
    assert "500 MB" in message
    # The cap is about image size, and the modeler needs to know why.
    assert "container image" in message


def test_an_oversized_entry_blocks_its_siblings(tmp_path):
    """Validation is up front, so a bad last entry doesn't leave a half-push."""
    _sparse_file(tmp_path / "small.csv", 1024)
    _sparse_file(tmp_path / "big.csv", MAX_DATASET_BYTES + 1)
    manifest = _manifest(tmp_path, [("fine", "small.csv"), ("too-big", "big.csv")])
    api = MagicMock()

    with pytest.raises(ManifestError):
        _cmd_push(_Args(manifest), api, "token")

    api.push.assert_not_called()


def test_file_exactly_at_the_limit_is_accepted(tmp_path):
    """The limit is inclusive; an exact-size file must not be rejected."""
    _sparse_file(tmp_path / "exact.csv", MAX_DATASET_BYTES)
    manifest = _manifest(tmp_path, [("exact", "exact.csv")])
    api = MagicMock()
    api.push.return_value = {"upload_id": "abc", "upload_url": "https://example/put"}

    assert _cmd_push(_Args(manifest), api, "token") == 0

    api.push.assert_called_once()
    assert api.push.call_args.args[4] == MAX_DATASET_BYTES


def test_client_limit_matches_the_server(tmp_path):
    """A client cap looser than the server's produces a late, confusing
    rejection; a tighter one blocks uploads the platform would accept.

    The server value lives in infra/tofu/shared-services/variables.tf as
    dataset_max_upload_size_bytes. Update both together.
    """
    assert MAX_DATASET_BYTES == 524288000
