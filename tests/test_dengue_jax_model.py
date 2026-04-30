"""Dengue model tests.

Covers vector dynamics, multi-strain infection, seroprevalence effects,
immunity period behaviour, and intervention impact.

Run:
    python3 -m pytest tests/test_dengue.py -v -m integration
"""

import json
import pathlib
import tempfile
import pytest
from helpers import _run_model, MODELS_DIR


def _get_dengue_val(entry, comp):
    """Extract the numeric value from a time series entry (handles both formats)."""
    val = entry[comp]
    if isinstance(val, dict):
        return val.get("age_all", val.get("mean", 0))
    return val


class TestDengueDisease:
    """Tests specific to Dengue model dynamics."""

    @pytest.mark.integration
    def test_vector_population_emerges(self):
        """Mosquito population should grow from 0 via temperature dynamics."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        results = _run_model(
            DengueJaxModel,
            MODELS_DIR / "dengue_jax_model" / "example-config.json",
        )
        ts = results[0]["parent_admin_total"]["time_series"]

        mid = len(ts) // 2
        iv_mid = _get_dengue_val(ts[mid], "IV")
        assert iv_mid > 0, (
            f"Infectious vector population is 0 at day {mid}. "
            "Temperature-driven vector dynamics may be broken."
        )

    @pytest.mark.integration
    def test_secondary_infections_occur(self):
        """I2 (secondary infections) should be non-zero by end of a multi-year run."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        results = _run_model(
            DengueJaxModel,
            MODELS_DIR / "dengue_jax_model" / "example-config.json",
        )
        ts = results[0]["parent_admin_total"]["time_series"]

        i2_end = _get_dengue_val(ts[-1], "I2")
        assert i2_end > 0, (
            "No secondary infections (I2) by end of simulation. "
            "Cross-immunity waning or secondary infection pathway may be broken."
        )

    def test_seroprevalence_affects_initial_s0(self):
        """Higher seroprevalence should reduce initial S0 population."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        compartment_list = DengueJaxModel.COMPARTMENT_LIST
        pop = 1_000_000
        zone_base = {
            "name": "Test", "center_lat": 2.0, "center_lon": 45.0,
            "population": pop, "infected_population": 5.0,
            "seroprevalence": 10.0,
        }
        zone_high = {**zone_base, "seroprevalence": 60.0}

        pop_low = DengueJaxModel.get_initial_population(
            [zone_base], compartment_list
        )
        pop_high = DengueJaxModel.get_initial_population(
            [zone_high], compartment_list
        )

        s0_idx = compartment_list.index("S0")
        s0_low = pop_low[0, s0_idx]
        s0_high = pop_high[0, s0_idx]
        assert s0_high < s0_low, (
            f"Higher seroprevalence should reduce S0: "
            f"sero=10% -> S0={s0_low:.0f}, sero=60% -> S0={s0_high:.0f}"
        )

    @pytest.mark.integration
    def test_immunity_period_matters(self):
        """immunity_period=0 vs 240 should produce different trajectories."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        base_config = {
            "Disease": {"disease_type": "VECTOR_BORNE", "immunity_period": 240},
            "start_date": "2026-01-01",
            "end_date": "2028-01-01",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 2.0, "center_lon": 45.0, "population": 500000, "infected_population": 5.0, "seroprevalence": 30, "temp_min": 20, "temp_max": 35, "temp_mean": 28},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "Interventions": {"items": []},
        }

        config_no_immunity = json.loads(json.dumps(base_config))
        config_no_immunity["Disease"]["immunity_period"] = 0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(base_config, f)
            path_with = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_no_immunity, f)
            path_without = f.name

        results_with = _run_model(DengueJaxModel, pathlib.Path(path_with))
        results_without = _run_model(DengueJaxModel, pathlib.Path(path_without))

        ts_with = results_with[0]["parent_admin_total"]["time_series"]
        ts_without = results_without[0]["parent_admin_total"]["time_series"]

        snot_with = _get_dengue_val(ts_with[-1], "Snot")
        snot_without = _get_dengue_val(ts_without[-1], "Snot")
        assert snot_with != snot_without, (
            f"immunity_period=240 and immunity_period=0 produced identical Snot "
            f"({snot_with}). Parameter may be ignored."
        )

    @pytest.mark.integration
    def test_intervention_reduces_infections(self):
        """Bite reduction intervention should reduce cumulative infections vs control."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        results = _run_model(
            DengueJaxModel,
            MODELS_DIR / "dengue_jax_model" / "example-config.json",
        )
        with_run = next(r for r in results if not r["control_run"])
        ctrl_run = next(r for r in results if r["control_run"])

        with_ts = with_run["parent_admin_total"]["time_series"]
        ctrl_ts = ctrl_run["parent_admin_total"]["time_series"]

        r_with = _get_dengue_val(with_ts[-1], "R")
        r_ctrl = _get_dengue_val(ctrl_ts[-1], "R")
        assert r_with <= r_ctrl, (
            f"Intervention run has MORE recoveries ({r_with:.0f}) than "
            f"control ({r_ctrl:.0f}). Intervention may be broken."
        )

    @pytest.mark.integration
    def test_uses_fixed_compartments(self):
        """Dengue should always use its fixed strain-specific compartments."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        results = _run_model(
            DengueJaxModel,
            MODELS_DIR / "dengue_jax_model" / "example-config.json",
        )
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = {k for k in ts[0].keys() if k != "date"}
        assert "S0" in compartments or "SV" in compartments, (
            f"Expected dengue-specific compartments, got {compartments}"
        )


# ---------------------------------------------------------------------------
# Uncertainty (LHS) runs
# ---------------------------------------------------------------------------


class TestDengueUncertainty:
    """Test Dengue uncertainty (LHS) runs produce valid confidence intervals."""

    @pytest.mark.integration
    def test_uncertainty_varying_interventions(self):
        """Uncertainty run with variance on intervention adherence."""
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
