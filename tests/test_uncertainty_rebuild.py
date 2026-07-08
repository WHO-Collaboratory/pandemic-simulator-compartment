"""Behavioral tests for rebuild-from-config uncertainty sampling.

The runner reconstructs the model from an overridden config per LHS sample
(Model.build_overridden_config + BatchSimulationManager._single_run), so every
value declared via add_disease_parameter / add_intervention /
add_transmission_edge actually varies — even when the model bakes it into a
derived constant in __init__ (e.g. 1/latent_period) or a frozen Intervention
object.

These tests assert real spread in the output CI bands, not just that the
uncertainty output format is present.

Run:
    python -m pytest tests/test_uncertainty_rebuild.py -v -m integration
"""

import json
import pathlib
import tempfile

import pytest
from helpers import run_model, MODELS_DIR


def _ci_dict_values(entry):
    return [v for k, v in entry.items() if k != "date" and isinstance(v, dict)]


def _has_spread(time_series):
    """True if any CI band has lower != upper at the midpoint or end."""
    if not time_series:
        return False
    for idx in (len(time_series) // 2, -1):
        for v in _ci_dict_values(time_series[idx]):
            if v.get("lower") != v.get("upper"):
                return True
    return False


def _is_uncertainty(time_series):
    vals = _ci_dict_values(time_series[0]) if time_series else []
    return bool(vals) and all("mean" in v for v in vals)


def _run(config):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        path = f.name
    from compartment.models.dengue_jax_model.model import DengueJaxModel
    return run_model(DengueJaxModel, pathlib.Path(path))


def _dengue_config(**overrides):
    cfg = {
        "Disease": {"disease_type": "VECTOR_BORNE", "immunity_period": 240},
        "start_date": "2025-01-01",
        "end_date": "2027-01-01",  # 2-yr window so dynamics diverge
        "n_simulations": 6,
        "admin_zones": [
            {
                "name": "Zone A", "center_lat": 2.0, "center_lon": 45.0,
                "population": 200000, "infected_population": 5,
                "seroprevalence": 30, "temp_min": 20, "temp_max": 35, "temp_mean": 28,
            }
        ],
        "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
        "travel_volume": {"leaving": 20},
        "Interventions": {"items": []},
    }
    cfg.update(overrides)
    return cfg


class TestBakedDiseaseParamVaries:
    """Disease params that __init__ bakes into derived constants
    (self.delta_H = 1/latent_period, self.zeta_cross = 1/immunity_period)
    only vary because the model is rebuilt from an overridden config."""

    @pytest.mark.integration
    def test_latent_period_produces_spread(self):
        cfg = _dengue_config()
        cfg["Disease"]["variance_params"] = [
            {"param": "latent_period", "dist": "uniform", "min": 3.0, "max": 12.0}
        ]
        results = _run(cfg)
        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert _is_uncertainty(ts)
        assert _has_spread(ts), (
            "latent_period variance produced no spread — the baked "
            "delta_H = 1/latent_period is not being re-derived per sample."
        )

    @pytest.mark.integration
    def test_immunity_period_produces_spread(self):
        cfg = _dengue_config()
        cfg["Disease"]["variance_params"] = [
            {"param": "immunity_period", "dist": "uniform", "min": 30, "max": 720}
        ]
        results = _run(cfg)
        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert _is_uncertainty(ts)
        assert _has_spread(ts), (
            "immunity_period variance produced no spread — the baked "
            "zeta_cross = 1/immunity_period is not being re-derived per sample."
        )


class TestInterventionVaries:
    """Intervention objects are frozen in __init__; only a rebuild propagates
    a sampled adherence/transmission_percentage into the run."""

    @pytest.mark.integration
    def test_intervention_adherence_produces_spread(self):
        cfg = _dengue_config()
        cfg["Interventions"] = {
            "items": [
                {
                    "Intervention": {"name": "PHYSICAL", "display_name": "Bite Reduction"},
                    "adherence_min": 50.0,
                    "transmission_percentage": 50.0,
                    "start_date": "2025-01-01",
                    "end_date": "2027-01-01",
                    "FieldConfigs": {
                        "items": [
                            {
                                "field_key": "adherence_min",
                                "has_variance": True,
                                "distribution_type": "UNIFORM",
                                "min": 5.0,
                                "max": 95.0,
                            }
                        ]
                    },
                }
            ]
        }
        results = _run(cfg)
        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert _is_uncertainty(ts)
        assert _has_spread(ts), (
            "Intervention adherence variance produced no spread — the frozen "
            "Intervention object is not being rebuilt per sample."
        )


class TestControlRunStaysInterventionless:
    """Regression: rebuilding the control batch from its source config must not
    re-introduce interventions."""

    @pytest.mark.integration
    def test_control_has_more_infections_than_intervention_run(self):
        cfg = _dengue_config()
        # Strong intervention; variance on a disease param triggers UNCERTAINTY.
        cfg["Disease"]["variance_params"] = [
            {"param": "latent_period", "dist": "uniform", "min": 4.0, "max": 8.0}
        ]
        cfg["Interventions"] = {
            "items": [
                {
                    "Intervention": {"name": "PHYSICAL", "display_name": "Bite Reduction"},
                    "adherence_min": 90.0,
                    "transmission_percentage": 90.0,
                    "start_date": "2025-01-01",
                    "end_date": "2027-01-01",
                    "FieldConfigs": {"items": []},
                }
            ]
        }
        results = _run(cfg)
        with_run = next(r for r in results if not r["control_run"])
        ctrl_run = next(r for r in results if r["control_run"])

        def _mean_R(run):
            v = run["parent_admin_total"]["time_series"][-1]["R"]
            return v["mean"] if isinstance(v, dict) else v

        r_with = _mean_R(with_run)
        r_ctrl = _mean_R(ctrl_run)
        assert r_ctrl >= r_with, (
            f"Control run ({r_ctrl:.0f} recovered) should have >= infections than "
            f"the intervention run ({r_with:.0f}). If they're ~equal, the control "
            f"batch is wrongly re-introducing interventions on rebuild."
        )


class TestTransmissionEdgeVaries:
    """Covid edge rates route through transmission_dict + _to_rate on rebuild."""

    @pytest.mark.integration
    def test_covid_beta_produces_spread(self):
        from compartment.models.covid_jax_model.model import CovidJaxModel

        cfg = json.loads(
            (MODELS_DIR / "covid_jax_model" / "example-config.json").read_text()
        )
        cfg["run_mode"] = "UNCERTAINTY"
        cfg["n_simulations"] = 6
        edges = cfg["TransmissionEdges"]["items"]
        # First edge is susceptible->exposed (beta). Attach variance.
        edges[0]["FieldConfigs"] = {
            "items": [
                {
                    "disease_param": "beta",
                    "has_variance": True,
                    "distribution_type": "UNIFORM",
                    "min": 0.1,
                    "max": 0.6,
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name
        results = run_model(CovidJaxModel, pathlib.Path(path))
        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert _is_uncertainty(ts)
        assert _has_spread(ts), "Covid beta edge variance produced no spread."
