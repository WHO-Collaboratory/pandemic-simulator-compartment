"""Offline unit tests for the datasets CLI pure helpers (no network/S3).

Network + S3 paths (push upload, pull download, list query) require a
GraphQL endpoint and are exercised in the staging CLI round-trip
(plan Verification step 4), not here.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from compartment import datasets_cli as cli


def test_sha256_file_matches_known(tmp_path):
    p = tmp_path / "f.csv"
    p.write_bytes(b"a,b\n1,2\n")
    import hashlib

    assert cli.sha256_file(p) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()


def test_deterministic_ids():
    assert cli.dataset_id_for("mobility/kenya") == "dataset-mobility-kenya"
    assert (
        cli.version_id_for("mobility/kenya", "2026-06-01")
        == "dataset-mobility-kenya-2026-06-01"
    )


def test_object_key_layout():
    assert (
        cli.object_key("mobility/kenya", "2026-06-01", "/tmp/x/mob.csv")
        == "mobility/kenya/2026-06-01/mob.csv"
    )


def test_bucket_names():
    assert cli.datasets_bucket("dev") == "collaboratory-datasets-dev"
    assert cli.quarantine_bucket("dev") == "collaboratory-datasets-quarantine-dev"


@pytest.mark.parametrize("ext", [".csv", ".tsv", ".json", ".ndjson", ".jsonl"])
def test_validate_extension_allows_supported(tmp_path, ext):
    p = tmp_path / f"f{ext}"
    p.write_text("x")
    assert cli.validate_extension(p) == ext


@pytest.mark.parametrize("ext", [".pkl", ".pickle", ".npy", ".xlsx", ".h5"])
def test_validate_extension_rejects_unsupported(tmp_path, ext):
    p = tmp_path / f"f{ext}"
    p.write_bytes(b"\x00")
    with pytest.raises(ValueError):
        cli.validate_extension(p)


def test_compute_metadata_csv(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(p, index=False)
    meta = cli.compute_metadata(p)
    assert meta["row_count"] == 3
    assert meta["file_name"] == "d.csv"
    assert meta["file_size"] > 0
    schema = json.loads(meta["column_schema"])
    assert [c["name"] for c in schema["columns"]] == ["a", "b"]
    # column_schema carries only names + dtypes, never data rows
    assert "1" not in meta["column_schema"] and "x" not in meta["column_schema"]


def test_compute_metadata_refuses_pickle(tmp_path):
    p = tmp_path / "evil.pkl"
    pd.DataFrame({"a": [1]}).to_pickle(p)
    with pytest.raises(ValueError):
        cli.compute_metadata(p)


def test_formula_injection_detection():
    assert cli.is_formula_injection("=cmd()")
    assert cli.is_formula_injection("+1+1")
    assert cli.is_formula_injection("-2")
    assert cli.is_formula_injection("@SUM(A1)")
    assert not cli.is_formula_injection("hello")
    assert not cli.is_formula_injection(42)


def test_formula_injection_cells(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"name": ["ok", "=HYPERLINK(x)"], "val": [1, 2]}).to_csv(
        p, index=False
    )
    hits = cli.formula_injection_cells(p)
    assert hits == ["name[1]"]


def test_push_dry_run_makes_no_calls(tmp_path, capsys, monkeypatch):
    # Guard: dry-run must not construct a GraphQL client or touch S3.
    monkeypatch.setattr(
        cli, "_client", lambda *a, **k: pytest.fail("client built in dry-run")
    )
    monkeypatch.setattr(
        cli, "_s3", lambda *a, **k: pytest.fail("s3 used in dry-run")
    )
    p = tmp_path / "mob.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)

    parser = cli.build_parser()
    args = parser.parse_args(
        ["push", "--file", str(p), "--slug", "mobility/kenya",
         "--version", "2026-06-01", "--dry-run"]
    )
    assert args.func(args) == 0
    assert "DRY RUN" in capsys.readouterr().out
