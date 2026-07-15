"""Slice 3 (reproducibility plumbing) offline tests.

The cloud round-trip itself (job-create freezes pins -> re-run reproduces
identical bytes) is plan Verification step 5 and needs a deployed env. Here we
lock down the offline-checkable invariants that make it work:
  - dataset_pins is in GRAPHQL_QUERY (else cloud runs silently lose pins),
  - generate_artifact embeds dataset_dependencies from datasets.yaml,
  - a pinned content_hash mismatch fails loudly at load time.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from compartment import datasets
from compartment.datasets import loader


@pytest.fixture(autouse=True)
def _reset_sdk():
    loader._reset()
    yield
    loader._reset()


# ---------------------------------------------------------------------------
# Load-bearing: dataset_pins must be selected by GRAPHQL_QUERY
# ---------------------------------------------------------------------------
def test_graphql_query_selects_dataset_pins():
    from compartment.cloud_helpers.graphql_queries import GRAPHQL_QUERY

    assert "dataset_pins" in GRAPHQL_QUERY
    # the pin subfields the resolver relies on
    for field in ("slug", "version", "content_hash", "bucket", "key"):
        assert field in GRAPHQL_QUERY


# ---------------------------------------------------------------------------
# generate_artifact embeds dataset_dependencies from datasets.yaml
# ---------------------------------------------------------------------------
def test_generate_artifact_emits_dataset_dependencies():
    from compartment.generate_artifact import (
        _augment_with_dependencies,
        _dataset_dependencies,
    )
    from compartment.models.test_klebsiella_amr_model.model import KlebsiellaAmrModel

    deps = _dataset_dependencies(KlebsiellaAmrModel)
    assert any(d["name"] == "amr/antibiotic-use" for d in deps)
    dep = next(d for d in deps if d["name"] == "amr/antibiotic-use")
    assert dep["version"] == "2026-01-01"
    assert dep["format"] == "csv"

    artifact = _augment_with_dependencies({}, KlebsiellaAmrModel)
    assert "dataset_dependencies" in artifact


def test_augment_omits_key_when_no_manifest(tmp_path):
    # A model dir with no datasets.yaml adds no dataset_dependencies key.
    from compartment.generate_artifact import _augment_with_dependencies
    from compartment.models.test_klebsiella_amr_model.model import KlebsiellaAmrModel

    artifact = _augment_with_dependencies({}, KlebsiellaAmrModel, model_dir=str(tmp_path))
    assert "dataset_dependencies" not in artifact


# ---------------------------------------------------------------------------
# Reproducibility integrity: a pinned content_hash mismatch fails loudly
# ---------------------------------------------------------------------------
def _seed(cache, name, version, body="region,cases\nA,10\n"):
    d = cache / name / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.csv").write_text(body)
    return hashlib.sha256(body.encode()).hexdigest()


def test_matching_pin_hash_loads(tmp_path):
    body = "region,cases\nA,10\nB,20\n"
    digest = _seed(tmp_path, "epi/cases", "2026-01-01", body)
    datasets.configure(
        mode="local",
        cache_root=tmp_path,
        pins=[{"slug": "epi/cases", "version": "2026-01-01", "content_hash": digest}],
    )
    df = datasets.load("epi/cases")
    assert len(df) == 2


def test_mismatched_pin_hash_raises(tmp_path):
    _seed(tmp_path, "epi/cases", "2026-01-01")
    datasets.configure(
        mode="local",
        cache_root=tmp_path,
        pins=[
            {"slug": "epi/cases", "version": "2026-01-01", "content_hash": "sha256:deadbeef"}
        ],
    )
    with pytest.raises(ValueError) as ei:
        datasets.load("epi/cases")
    assert "hash mismatch" in str(ei.value).lower()


def test_pin_version_drives_resolution(tmp_path):
    # The config pin's version is what resolves (the frozen reproducibility path).
    _seed(tmp_path, "epi/cases", "2026-01-01", "region,cases\nA,1\n")
    _seed(tmp_path, "epi/cases", "2026-09-01", "region,cases\nA,999\n")
    datasets.configure(
        mode="local",
        cache_root=tmp_path,
        pins=[{"slug": "epi/cases", "version": "2026-01-01"}],
    )
    df = datasets.load("epi/cases")
    assert df["cases"].iloc[0] == 1  # the pinned version, not the newer one
