"""Unit tests for the Bring-Your-Own-Dataset SDK (offline, no network).

Covers the supported format allowlist (csv/tsv/json/ndjson/parquet), version
precedence (arg > config pin > yaml > latest), the missing-dataset pull hint,
the load-once cache, and the parser-exploit guard (pickle and other
code-deserializing readers are never used).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from compartment import datasets
from compartment.datasets import loader


@pytest.fixture(autouse=True)
def _reset_sdk():
    """Reset SDK module state (resolver + frame cache) around every test."""
    loader._reset()
    yield
    loader._reset()


def _model_dir_with_manifest(tmp_path: Path, name: str, version: str) -> Path:
    md = tmp_path / "model"
    md.mkdir(exist_ok=True)
    (md / "datasets.yaml").write_text(
        textwrap.dedent(
            f"""
            datasets:
              - name: {name}
                version: "{version}"
                format: csv
            """
        )
    )
    return md


def _seed_csv(cache: Path, name: str, version: str, marker: str, filename="data.csv"):
    d = cache / name / version
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"which": [marker]}).to_csv(d / filename, index=False)


# ---------------------------------------------------------------------------
# Format coverage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename,writer",
    [
        ("data.csv", lambda df, p: df.to_csv(p, index=False)),
        ("data.tsv", lambda df, p: df.to_csv(p, sep="\t", index=False)),
        ("data.json", lambda df, p: df.to_json(p, orient="records")),
        ("data.ndjson", lambda df, p: df.to_json(p, orient="records", lines=True)),
        ("data.jsonl", lambda df, p: df.to_json(p, orient="records", lines=True)),
    ],
)
def test_load_supported_formats(tmp_path, filename, writer):
    expected = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    d = tmp_path / "ns/slug" / "1.0.0"
    d.mkdir(parents=True)
    writer(expected, d / filename)

    datasets.configure(mode="local", cache_root=tmp_path)
    out = datasets.load("ns/slug", version="1.0.0")

    assert list(out.columns) == ["a", "b"]
    assert out["a"].tolist() == [1, 2]
    assert out["b"].tolist() == [3, 4]


# ---------------------------------------------------------------------------
# Version precedence
# ---------------------------------------------------------------------------
def test_version_precedence_arg_pin_yaml(tmp_path):
    cache = tmp_path / "cache"
    for marker in ("v-yaml", "v-pin", "v-arg"):
        _seed_csv(cache, "mob/kenya", marker, marker)
    md = _model_dir_with_manifest(tmp_path, "mob/kenya", "v-yaml")

    # explicit arg > config pin > yaml
    datasets.configure(
        mode="local",
        cache_root=cache,
        model_dir=md,
        pins=[{"slug": "mob/kenya", "version": "v-pin"}],
    )
    assert datasets.load("mob/kenya", version="v-arg")["which"].iloc[0] == "v-arg"
    assert datasets.load("mob/kenya")["which"].iloc[0] == "v-pin"

    # pin absent -> yaml version
    datasets.configure(mode="local", cache_root=cache, model_dir=md)
    assert datasets.load("mob/kenya")["which"].iloc[0] == "v-yaml"


def test_latest_local_picks_newest_version(tmp_path):
    cache = tmp_path / "cache"
    for v in ("2026-01-01", "2026-05-01", "2026-03-01"):
        _seed_csv(cache, "s/d", v, v)
    md = _model_dir_with_manifest(tmp_path, "s/d", "latest")

    datasets.configure(mode="local", cache_root=cache, model_dir=md)
    assert datasets.load("s/d")["which"].iloc[0] == "2026-05-01"


# ---------------------------------------------------------------------------
# Missing dataset
# ---------------------------------------------------------------------------
def test_missing_dataset_raises_with_pull_hint(tmp_path):
    datasets.configure(mode="local", cache_root=tmp_path)
    with pytest.raises(FileNotFoundError) as ei:
        datasets.load("nope/missing", version="9.9.9")
    msg = str(ei.value)
    assert "datasets pull nope/missing" in msg
    assert "9.9.9" in msg


def test_load_requires_configure():
    loader._reset()
    with pytest.raises(RuntimeError):
        datasets.load("x/y")


# ---------------------------------------------------------------------------
# Optional datasets — required: false degrades instead of raising
# ---------------------------------------------------------------------------
def test_optional_dataset_missing_returns_none(tmp_path):
    """An optional dataset that isn't in the cache yields None, not an error."""
    datasets.configure(mode="local", cache_root=tmp_path)
    assert datasets.load("nope/missing", version="9.9.9", required=False) is None


def test_optional_dataset_survives_unconfigured_sdk():
    """required=False tolerates a never-configured SDK.

    This is the path that model unit tests hit: they construct a model directly,
    so ``Model.__init__`` -> ``load_datasets()`` runs with no resolver at all.
    A demo model must not explode there.
    """
    loader._reset()
    assert datasets.load("x/y", required=False) is None


def test_required_from_manifest_defaults_to_raising(tmp_path):
    """``required:`` in datasets.yaml drives the default when not passed."""
    md = _model_dir_with_manifest(tmp_path, "s/d", "1.0.0")
    datasets.configure(mode="local", cache_root=tmp_path, model_dir=md)
    # The fixture manifest omits `required`, so it defaults to True -> raises.
    with pytest.raises(FileNotFoundError):
        datasets.load("s/d")


def test_explicit_required_overrides_manifest(tmp_path):
    """An explicit required= argument wins over the manifest declaration."""
    md = _model_dir_with_manifest(tmp_path, "s/d", "1.0.0")
    datasets.configure(mode="local", cache_root=tmp_path, model_dir=md)
    assert datasets.load("s/d", required=False) is None


# ---------------------------------------------------------------------------
# 'latest' from the registry is intentionally unsupported
# ---------------------------------------------------------------------------
def test_registry_latest_lookup_raises_with_both_remedies():
    """Resolving 'latest' at load time must stay unimplemented.

    It is only reachable from a cloud run loading an undeclared dataset, where
    guessing a version would silently break reproducibility. Guard it with a test
    so nobody helpfully implements it -- the raise is the intended behaviour.
    """
    from compartment.datasets import registry_client

    with pytest.raises(NotImplementedError) as ei:
        registry_client.get_latest_published_version("mobility/kenya")

    message = str(ei.value)
    assert "mobility/kenya" in message
    # Must tell the caller both ways out, or it's just a dead end.
    assert "datasets.yaml" in message
    assert "dataset_pins" in message


# ---------------------------------------------------------------------------
# Parser-exploit guard — pickle and friends are never read
# ---------------------------------------------------------------------------
def test_pickle_file_is_refused_not_read(tmp_path):
    d = tmp_path / "evil/data" / "1.0.0"
    d.mkdir(parents=True)
    pd.DataFrame({"a": [1]}).to_pickle(d / "data.pkl")

    datasets.configure(mode="local", cache_root=tmp_path)
    with pytest.raises(ValueError) as ei:
        datasets.load("evil/data", version="1.0.0")
    assert "pkl" in str(ei.value).lower() or "pickle" in str(ei.value).lower()


def test_no_code_deserializing_readers_exist():
    # The allowlist must never grow a pickle / numpy-pickle / hdf reader.
    assert "pickle" not in loader._READERS
    for banned in (".pkl", ".pickle", ".npy", ".npz", ".h5", ".hdf5", ".xlsx"):
        assert banned not in loader.ALLOWED_SUFFIXES


# ---------------------------------------------------------------------------
# Cache + manifest introspection
# ---------------------------------------------------------------------------
def test_load_is_cached_once(tmp_path):
    _seed_csv(tmp_path, "s/d", "1.0.0", "x")
    datasets.configure(mode="local", cache_root=tmp_path)
    first = datasets.load("s/d", version="1.0.0")
    second = datasets.load("s/d", version="1.0.0")
    assert first is second  # same cached frame, not re-read


def test_list_declared(tmp_path):
    md = _model_dir_with_manifest(tmp_path, "a/b", "1.0.0")
    datasets.configure(mode="local", cache_root=tmp_path, model_dir=md)
    declared = datasets.list_declared()
    assert [d.name for d in declared] == ["a/b"]
    assert declared[0].version == "1.0.0"


def test_ambiguous_version_dir_errors(tmp_path):
    d = tmp_path / "s/d" / "1.0.0"
    d.mkdir(parents=True)
    pd.DataFrame({"a": [1]}).to_csv(d / "one.csv", index=False)
    pd.DataFrame({"a": [1]}).to_csv(d / "two.csv", index=False)
    datasets.configure(mode="local", cache_root=tmp_path)
    with pytest.raises(ValueError) as ei:
        datasets.load("s/d", version="1.0.0")
    assert "multiple data files" in str(ei.value)


def test_sidecar_file_is_ignored(tmp_path):
    d = tmp_path / "s/d" / "1.0.0"
    d.mkdir(parents=True)
    pd.DataFrame({"a": [7]}).to_csv(d / "data.csv", index=False)
    (d / "data.csv.sha256").write_text("deadbeef  data.csv\n")
    datasets.configure(mode="local", cache_root=tmp_path)
    out = datasets.load("s/d", version="1.0.0")
    assert out["a"].iloc[0] == 7
