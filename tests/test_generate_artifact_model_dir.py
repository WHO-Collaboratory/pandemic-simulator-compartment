"""Tests for --model-dir dynamic discovery in generate_artifact CLI."""

import json
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Unit tests for _discover_model_from_dir
# ---------------------------------------------------------------------------


def test_discover_model_from_dir_mpox():
    from compartment.generate_artifact import _discover_model_from_dir
    from compartment.models.mpox_jax_model.model import MpoxJaxModel

    result = _discover_model_from_dir("compartment/models/mpox_jax_model")
    assert result is MpoxJaxModel


def test_discover_model_from_dir_trailing_slash():
    from compartment.generate_artifact import _discover_model_from_dir
    from compartment.models.mpox_jax_model.model import MpoxJaxModel

    result = _discover_model_from_dir("compartment/models/mpox_jax_model/")
    assert result is MpoxJaxModel


def test_discover_model_from_dir_resolves_disease_type():
    from compartment.generate_artifact import _discover_model_from_dir

    model_class = _discover_model_from_dir("compartment/models/mpox_jax_model")
    schema = model_class._build_parameter_schema()
    assert schema.disease_type == "MONKEYPOX"


def test_discover_model_from_dir_invalid_path(capsys):
    from compartment.generate_artifact import _discover_model_from_dir

    with pytest.raises(SystemExit):
        _discover_model_from_dir("compartment/models/nonexistent_model")


def test_discover_model_from_dir_covid():
    from compartment.generate_artifact import _discover_model_from_dir
    from compartment.models.covid_jax_model.model import CovidJaxModel

    result = _discover_model_from_dir("compartment/models/covid_jax_model")
    assert result is CovidJaxModel


def test_discover_model_from_dir_dengue():
    from compartment.generate_artifact import _discover_model_from_dir
    from compartment.models.dengue_jax_model.model import DengueJaxModel

    result = _discover_model_from_dir("compartment/models/dengue_jax_model")
    assert result is DengueJaxModel


# ---------------------------------------------------------------------------
# Integration: CLI --model-dir produces same artifact as positional arg
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cli_model_dir_matches_disease_type_output():
    """Artifact from --model-dir should match artifact from positional disease_type."""
    result_dir = subprocess.run(
        [sys.executable, "-m", "compartment.generate_artifact", "--model-dir",
         "compartment/models/mpox_jax_model"],
        capture_output=True, text=True,
    )
    result_type = subprocess.run(
        [sys.executable, "-m", "compartment.generate_artifact", "MONKEYPOX"],
        capture_output=True, text=True,
    )
    assert result_dir.returncode == 0, result_dir.stderr
    assert result_type.returncode == 0, result_type.stderr
    assert json.loads(result_dir.stdout) == json.loads(result_type.stdout)


@pytest.mark.integration
def test_cli_model_dir_invalid_exits_nonzero():
    result = subprocess.run(
        [sys.executable, "-m", "compartment.generate_artifact", "--model-dir",
         "compartment/models/nonexistent"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
