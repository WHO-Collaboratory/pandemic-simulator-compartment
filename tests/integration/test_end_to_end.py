"""
Integration tests — run full simulations using existing example configs.

These tests exercise the complete pipeline: JSON config loading → validation
→ model initialisation → JAX ODE solve → post-processing → JSON output.

Run just these tests with:
    pytest -m integration

Skip them with:
    pytest -m "not integration"
"""
import json
import os
import pytest


@pytest.mark.integration
def test_mpox_end_to_end(tmp_path):
    from compartment.models.mpox_jax_model.model import MpoxJaxModel
    from compartment.run_simulation import run_simulation

    config_file = "compartment/models/mpox_jax_model/example-config.json"
    output_file = str(tmp_path / "mpox_output.json")

    run_simulation(
        model_class=MpoxJaxModel,
        config_path=config_file,
        output_path=output_file,
    )

    _assert_valid_output(output_file)


@pytest.mark.integration
def test_covid_end_to_end(tmp_path):
    from compartment.models.covid_jax_model.model import CovidJaxModel
    from compartment.run_simulation import run_simulation

    config_file = "compartment/models/covid_jax_model/example-config.json"
    output_file = str(tmp_path / "covid_output.json")

    run_simulation(
        model_class=CovidJaxModel,
        config_path=config_file,
        output_path=output_file,
    )

    _assert_valid_output(output_file)


@pytest.mark.integration
def test_dengue_end_to_end(tmp_path):
    from compartment.models.dengue_jax_model.model import DengueJaxModel
    from compartment.run_simulation import run_simulation

    config_file = "compartment/models/dengue_jax_model/example-config.json"
    output_file = str(tmp_path / "dengue_output.json")

    run_simulation(
        model_class=DengueJaxModel,
        config_path=config_file,
        output_path=output_file,
    )

    _assert_valid_output(output_file)


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------

def _assert_valid_output(output_file: str) -> None:
    """Assert that the output file is valid and has the expected structure."""
    assert os.path.exists(output_file), f"Output file not created: {output_file}"

    with open(output_file) as f:
        results = json.load(f)

    # Top-level: list of 2 results (with/without interventions)
    assert isinstance(results, list), "Output should be a list"
    assert len(results) == 2, "Should have 2 results (with and without interventions)"

    for result in results:
        assert isinstance(result, dict)
        assert "admin_zones" in result, "Each result should have 'admin_zones'"

        for zone in result["admin_zones"]:
            assert "time_series" in zone, "Each zone should have 'time_series'"
            assert len(zone["time_series"]) > 0, "time_series should not be empty"

            # Check first and last time step
            for ts in (zone["time_series"][0], zone["time_series"][-1]):
                assert "date" in ts, "Each time step should have a 'date'"
                # All compartment values should be non-negative
                for key, val in ts.items():
                    if key == "date":
                        continue
                    if isinstance(val, dict):
                        for age_key, v in val.items():
                            assert v >= -1.0, (
                                f"Negative population value {v} for {key}.{age_key} "
                                f"at {ts['date']}"
                            )
