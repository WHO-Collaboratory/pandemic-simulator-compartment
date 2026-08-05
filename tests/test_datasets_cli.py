"""Offline unit tests for the datasets CLI pure helpers (no network/S3).

Network + S3 paths (push upload, pull download, list query) require a
GraphQL endpoint and are exercised in the staging CLI round-trip
(plan Verification step 4), not here.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def test_cli_needs_no_aws_credentials_or_api_key():
    """The CLI must stay usable by a modeler with no AWS access at all.

    It reaches S3 only through presigned URLs minted by the platform, so it must
    not import boto3, hold a GraphQL api key, or know bucket names. This is the
    whole point of the presign layer -- if someone reintroduces a direct S3 call,
    external modelers silently lose the ability to publish.
    """
    source = Path(cli.__file__).read_text()
    for forbidden in ("boto3", "GQL_API_KEY", "x-api-key", "collaboratory-datasets"):
        assert forbidden not in source, f"datasets_cli.py must not reference {forbidden}"

    assert not hasattr(cli, "datasets_bucket")
    assert not hasattr(cli, "quarantine_bucket")


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
    assert cli.is_formula_injection("@SUM(A1)")
    assert cli.is_formula_injection("-2+3")
    assert not cli.is_formula_injection("hello")
    assert not cli.is_formula_injection(42)


def test_formula_injection_exempts_plain_numbers():
    """Signed numbers are data, not formulas.

    This mirrors the numeric carve-out in the authoritative gate
    (validate.ts::isFormulaInjectionCell). Without it, the client-side pre-flight
    warns about ordinary numeric values that happen to sit in an object-dtype
    column (e.g. one containing NAs) — contradicting the gate it is supposed to
    predict, and training modelers to ignore the warning.
    """
    for value in ("-2", "+3.2", "-0", "+1e6"):
        assert not cli.is_formula_injection(value), value


def test_formula_injection_cells(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"name": ["ok", "=HYPERLINK(x)"], "val": [1, 2]}).to_csv(
        p, index=False
    )
    hits = cli.formula_injection_cells(p)
    assert hits == ["name[1]"]


def test_push_dry_run_makes_no_calls(tmp_path, capsys, monkeypatch):
    # Guard: dry-run must neither authenticate nor call the platform.
    monkeypatch.setattr(
        cli, "ApiClient", lambda *a, **k: pytest.fail("API client built in dry-run")
    )
    monkeypatch.setattr(
        cli.auth, "id_token", lambda *a, **k: pytest.fail("token used in dry-run")
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


# ---------------------------------------------------------------------------
# Scan-gate reporting: exit codes + verdict rendering
# ---------------------------------------------------------------------------
def test_exit_codes_distinguish_scan_outcomes():
    """`push --wait` / `status` are meant to gate CI, so the codes must differ.

    In particular REJECTED (2) must not collapse into either success or a generic
    error: a pipeline needs to tell "the scan rejected my data" apart from "the
    command broke".
    """
    assert cli.exit_code_for("PUBLISHED") == cli.EXIT_OK == 0
    assert cli.exit_code_for("REJECTED") == cli.EXIT_REJECTED == 2
    assert cli.exit_code_for("PENDING_SCAN") == cli.EXIT_PENDING == 3
    assert cli.exit_code_for("QUARANTINED") == cli.EXIT_PENDING
    assert cli.exit_code_for(None) == cli.EXIT_ERROR == 1
    assert cli.exit_code_for("ARCHIVED") == cli.EXIT_ERROR


def test_terminal_statuses_stop_the_wait_loop():
    """Anything not terminal must keep polling, or --wait exits early."""
    assert set(cli.TERMINAL_STATUSES) == {"PUBLISHED", "REJECTED", "ARCHIVED"}
    for pending in ("PENDING_SCAN", "QUARANTINED"):
        assert pending not in cli.TERMINAL_STATUSES


def test_format_verdict_splits_category_from_detail():
    # handler.ts writes "malware:<status>", "validation:<reasons>", or "clean".
    assert cli.format_verdict("validation:formula injection in 2 cell(s)") == (
        "validation: formula injection in 2 cell(s)"
    )
    assert cli.format_verdict("malware:THREATS_FOUND EICAR") == (
        "malware: THREATS_FOUND EICAR"
    )
    assert cli.format_verdict("clean") == "clean"
    assert cli.format_verdict(None) == ""


# ---------------------------------------------------------------------------
# Key contract — three implementations must agree
# ---------------------------------------------------------------------------
def test_key_contract_matches_route_and_scan_gate():
    """object_key / dataset_id_for / version_id_for are duplicated in TypeScript.

    They appear in app/frontend/.../api/_helpers/datasetKeys.ts (which derives the
    authoritative key) and app/backend/dataset-scan/src/handler.ts (which parses it
    back). If these drift, a push uploads to one key while the registry row and the
    scanner look at another, and the upload silently never gets promoted.

    These literals are the contract. Update all three together.
    """
    assert cli.object_key("amr/antibiotic-use", "2026-01-01", "use.csv") == (
        "amr/antibiotic-use/2026-01-01/use.csv"
    )
    assert cli.dataset_id_for("amr/antibiotic-use") == "dataset-amr-antibiotic-use"
    assert cli.version_id_for("amr/antibiotic-use", "2026-01-01") == (
        "dataset-amr-antibiotic-use-2026-01-01"
    )

    # parseKey() in handler.ts pops filename then version and joins the rest, so a
    # round-trip must recover the original slug.
    key = cli.object_key("amr/antibiotic-use", "2026-01-01", "use.csv")
    parts = key.split("/")
    file_name, version, slug = parts[-1], parts[-2], "/".join(parts[:-2])
    assert (slug, version, file_name) == ("amr/antibiotic-use", "2026-01-01", "use.csv")


def test_object_key_strips_path_components_from_filename():
    """A filename must never be able to inject extra path segments.

    The server re-derives the key and rejects anything that isn't a simple name,
    but the client shouldn't construct a traversal either.
    """
    assert cli.object_key("a/b", "1", "../../etc/passwd") == "a/b/1/passwd"
    assert cli.object_key("a/b", "1", "/abs/path.csv") == "a/b/1/path.csv"
