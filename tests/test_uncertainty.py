"""Uncertainty (LHS) run tests.

Verify that uncertainty mode produces valid confidence intervals across models.

Run:
    python3 -m pytest tests/test_uncertainty.py -v -m integration
"""

import json
import math
import pathlib
import tempfile
import pytest
from helpers import _run_model


class TestUncertaintyRuns:
    """Test uncertainty (LHS) runs produce valid confidence intervals."""

    @pytest.mark.integration
    def test_covid_uncertainty_varying_params(self):
        """COVID uncertainty run with variance on beta should produce CI bands."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = {
            "Disease": {
                "disease_type": "COVID_SEIHDR",
                "immunity_period": 14,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
            "run_mode": "UNCERTAINTY",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0, "population": 500000, "infected_population": 1.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"},
                        "value": 0.3,
                        "FieldConfigs": {"items": [{
                            "field_key": "value",
                            "has_variance": True,
                            "distribution_type": "UNIFORM",
                            "disease_param": "BETA",
                            "min": 0.15,
                            "max": 0.45,
                        }]},
                    },
                    {
                        "transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"},
                        "value": 0.1,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "GAMMA", "min": 0, "max": 0}]},
                    },
                ]
            },
            "Interventions": {"items": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))

        assert len(results) == 2
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            assert len(ts) > 0

            mid_point = ts[len(ts) // 2]
            for comp in ["S", "I", "R"]:
                assert comp in mid_point
                val = mid_point[comp]
                assert "mean" in val, f"Expected uncertainty format (mean/lower/upper) for {comp}"
                assert "lower" in val
                assert "upper" in val

            for entry in ts:
                for comp in ["S", "I", "R"]:
                    val = entry[comp]
                    for stat in ["mean", "lower", "upper"]:
                        assert not math.isnan(val[stat]), (
                            f"NaN in {comp}.{stat} on {entry['date']}"
                        )

    @pytest.mark.integration
    def test_covid_uncertainty_no_nan(self):
        """COVID uncertainty run should produce no NaN values."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = {
            "Disease": {
                "disease_type": "COVID_SEIHDR",
                "immunity_period": 14,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-02-01",
            "run_mode": "UNCERTAINTY",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0, "population": 100000, "infected_population": 1.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"},
                        "value": 0.25,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": True, "distribution_type": "UNIFORM", "disease_param": "BETA", "min": 0.1, "max": 0.4}]},
                    },
                    {
                        "transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"},
                        "value": 0.1,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "GAMMA", "min": 0, "max": 0}]},
                    },
                ]
            },
            "Interventions": {"items": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))

        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            for entry in ts:
                for key, val in entry.items():
                    if key == "date":
                        continue
                    if isinstance(val, dict):
                        assert "mean" in val and "lower" in val and "upper" in val, (
                            f"Expected uncertainty format for {key}, got keys: {val.keys()}"
                        )
                        for stat in ["mean", "lower", "upper"]:
                            assert not math.isnan(val[stat]), (
                                f"NaN in {key}.{stat} on {entry['date']}"
                            )

    @pytest.mark.integration
    def test_dengue_uncertainty_varying_interventions(self):
        """Dengue uncertainty run with variance on intervention adherence."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        config = {
            "Disease": {
                "disease_type": "VECTOR_BORNE",
                "immunity_period": 240,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-07-01",
            "run_mode": "UNCERTAINTY",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 2.0, "center_lon": 45.0, "population": 500000, "infected_population": 5, "seroprevalence": 30, "temp_min": 20, "temp_max": 35, "temp_mean": 28},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "Interventions": {
                "items": [
                    {
                        "Intervention": {"name": "PHYSICAL", "display_name": "Bite Reduction"},
                        "adherence_min": 50.0,
                        "transmission_percentage": 50.0,
                        "start_date": "2025-01-01",
                        "end_date": "2025-07-01",
                        "FieldConfigs": {
                            "items": [{
                                "field_key": "adherence_min",
                                "has_variance": True,
                                "distribution_type": "UNIFORM",
                                "min": 20.0,
                                "max": 80.0,
                            }]
                        },
                    }
                ]
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(DengueJaxModel, pathlib.Path(config_path))

        assert len(results) == 2

        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert len(ts) > 0

        mid_point = ts[len(ts) // 2]
        sample_comp = [k for k in mid_point if k != "date"][0]
        val = mid_point[sample_comp]
        assert "mean" in val, f"Expected uncertainty format for {sample_comp}, got {val}"

        ctrl_run = next(r for r in results if r["control_run"])
        ctrl_ts = ctrl_run["parent_admin_total"]["time_series"]
        assert len(ctrl_ts) > 0

    @pytest.mark.integration
    def test_uncertainty_lower_le_mean_le_upper(self):
        """In uncertainty output, lower <= mean <= upper should hold."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = {
            "Disease": {
                "disease_type": "COVID_SEIHDR",
                "immunity_period": 14,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-02-15",
            "run_mode": "UNCERTAINTY",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0, "population": 200000, "infected_population": 1.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"},
                        "value": 0.3,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": True, "distribution_type": "UNIFORM", "disease_param": "BETA", "min": 0.2, "max": 0.5}]},
                    },
                    {
                        "transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"},
                        "value": 0.1,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "GAMMA", "min": 0, "max": 0}]},
                    },
                ]
            },
            "Interventions": {"items": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))

        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            for entry in ts:
                for key, val in entry.items():
                    if key == "date":
                        continue
                    if isinstance(val, dict) and "mean" in val:
                        assert val["lower"] <= val["mean"] + 0.01, (
                            f"lower ({val['lower']}) > mean ({val['mean']}) "
                            f"for {key} on {entry['date']}"
                        )
                        assert val["mean"] <= val["upper"] + 0.01, (
                            f"mean ({val['mean']}) > upper ({val['upper']}) "
                            f"for {key} on {entry['date']}"
                        )
