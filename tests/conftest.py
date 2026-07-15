"""Shared pytest fixtures.

Points the Bring-Your-Own-Dataset SDK at the committed fixture cache under
``tests/data`` for every test, so models that declare a ``datasets.yaml`` (the
Klebsiella AMR demo) resolve their datasets offline during smoke tests.

Tests that need a different cache pass ``cache_root=...`` to
``datasets.configure()`` directly, which takes precedence over this env var.
"""

import os
import pathlib

import pytest

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


@pytest.fixture(autouse=True)
def _byod_dataset_cache(monkeypatch):
    monkeypatch.setenv("WHO_DATASET_CACHE", str(DATA_DIR))
    # Never leak a cloud signal into local test runs.
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
