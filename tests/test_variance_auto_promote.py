"""Tests for auto-promotion of run_mode to UNCERTAINTY via Disease.variance_params.

Covers two new behaviours in run_simulation:
  1. Local-mode configs can supply variance_params inside the Disease section.
  2. Any non-empty uncertainty_params collection auto-promotes run_mode to
     UNCERTAINTY even when run_mode is absent or explicitly DETERMINISTIC.

Run:
    python -m pytest tests/test_variance_auto_promote.py -v -m integration
"""

import json
import pathlib
import tempfile

import pytest
from helpers import run_model

DENGUE_BASE_CONFIG = {
    "Disease": {
        "disease_type": "VECTOR_BORNE",
        "immunity_period": 240,
    },
    "start_date": "2025-01-01",
    "end_date": "2025-04-01",
    "n_simulations": 4,
    "admin_zones": [
        {
            "name": "Zone A",
            "center_lat": 2.0,
            "center_lon": 45.0,
            "population": 200000,
            "infected_population": 5,
            "seroprevalence": 30,
            "temp_min": 20,
            "temp_max": 35,
            "temp_mean": 28,
        }
    ],
    "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
    "travel_volume": {"leaving": 20},
    "Interventions": {"items": []},
}


def _has_uncertainty_format(time_series: list) -> bool:
    """Return True if every non-date value in the first timestep is a CI dict."""
    if not time_series:
        return False
    first = time_series[0]
    values = [v for k, v in first.items() if k != "date"]
    return values and all(isinstance(v, dict) and "mean" in v for v in values)


def _run_config(config: dict) -> list:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        path = f.name
    from compartment.models.dengue_jax_model.model import DengueJaxModel
    return run_model(DengueJaxModel, pathlib.Path(path))


class TestVarianceAutoPromote:

    @pytest.mark.integration
    def test_auto_promotes_when_no_run_mode_set(self):
        """No run_mode in config + variance_params → UNCERTAINTY output format."""
        config = json.loads(json.dumps(DENGUE_BASE_CONFIG))
        config["Disease"]["variance_params"] = [
            {"param": "immunity_period", "dist": "uniform", "min": 90, "max": 365}
        ]
        # Deliberately omit run_mode — should auto-promote

        results = _run_config(config)
        assert len(results) == 2

        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert _has_uncertainty_format(ts), (
            "Expected uncertainty (CI) format when variance_params present but "
            "run_mode not set. Got scalar values — auto-promotion may be broken."
        )

    @pytest.mark.integration
    def test_auto_promotes_when_run_mode_is_deterministic(self):
        """Explicit run_mode=DETERMINISTIC + variance_params → still promoted to UNCERTAINTY."""
        config = json.loads(json.dumps(DENGUE_BASE_CONFIG))
        config["run_mode"] = "DETERMINISTIC"
        config["Disease"]["variance_params"] = [
            {"param": "immunity_period", "dist": "uniform", "min": 90, "max": 365}
        ]

        results = _run_config(config)
        assert len(results) == 2

        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert _has_uncertainty_format(ts), (
            "Expected UNCERTAINTY output even with explicit run_mode=DETERMINISTIC "
            "when variance_params are present."
        )

    @pytest.mark.integration
    def test_no_variance_params_stays_deterministic(self):
        """Config with no variance_params and no run_mode stays DETERMINISTIC (scalar output)."""
        config = json.loads(json.dumps(DENGUE_BASE_CONFIG))
        # No variance_params, no run_mode

        results = _run_config(config)
        assert len(results) == 2

        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert not _has_uncertainty_format(ts), (
            "Expected scalar (DETERMINISTIC) output when no variance_params present."
        )

    # NOTE: assertions that variance actually moves the output (CI spread)
    # live in test_uncertainty_rebuild.py, since real spread depends on the
    # rebuild-from-config mechanism. This file only covers run_mode promotion
    # and the uncertainty output *format*.

    @pytest.mark.integration
    def test_variance_params_default_dist_is_uniform(self):
        """Omitting 'dist' from a variance_param entry should default to uniform without error."""
        config = json.loads(json.dumps(DENGUE_BASE_CONFIG))
        config["Disease"]["variance_params"] = [
            {"param": "immunity_period", "min": 90, "max": 365}
            # no 'dist' key
        ]

        results = _run_config(config)
        assert len(results) == 2
        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert _has_uncertainty_format(ts)
