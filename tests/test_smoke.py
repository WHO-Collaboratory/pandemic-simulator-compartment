"""Smoke tests: run every model end-to-end with its example config.

Each test loads the model's example-config.json, runs the full simulation
pipeline (validation → model init → ODE solve → post-processing), and
checks basic correctness of the output.

These are integration tests — they exercise the real code path, not mocks.
They take a few seconds each due to JAX compilation.

Run all smoke tests:
    python3 -m pytest tests/test_smoke.py -v -m integration

Run just one model:
    python3 -m pytest tests/test_smoke.py -v -m integration -k "covid_jax_model"
"""

import math
import pytest
from helpers import MODEL_CONFIGS, _import_class, _run_model


@pytest.mark.integration
class TestModelSmoke:
    """Run each model end-to-end and verify output structure."""

    @pytest.fixture(params=list(MODEL_CONFIGS.keys()))
    def model_run(self, request):
        """Run a model and return (model_id, results)."""
        model_id, class_path, config_path = MODEL_CONFIGS[request.param]
        model_class = _import_class(class_path)
        results = _run_model(model_class, config_path)
        return model_id, results

    def test_returns_two_runs(self, model_run):
        """Output should contain exactly two runs (with/without interventions)."""
        _, results = model_run
        assert isinstance(results, list)
        assert len(results) == 2

    def test_control_run_flags(self, model_run):
        """One run should be control, one should not."""
        _, results = model_run
        control_flags = {r["control_run"] for r in results}
        assert control_flags == {True, False}

    def test_has_parent_admin_total(self, model_run):
        """Both runs should have a parent_admin_total section."""
        _, results = model_run
        for run in results:
            assert "parent_admin_total" in run
            assert "time_series" in run["parent_admin_total"]

    def test_time_series_not_empty(self, model_run):
        """Time series should have at least one entry."""
        _, results = model_run
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            assert len(ts) > 0

    def test_time_series_has_date(self, model_run):
        """Every time series entry should have a date field."""
        _, results = model_run
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            for entry in ts:
                assert "date" in entry

    def test_compartments_present(self, model_run):
        """Time series entries should have at least 2 compartments."""
        _, results = model_run
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = [k for k in ts[0].keys() if k != "date"]
        assert len(compartments) >= 2, f"Expected at least 2 compartments, got {compartments}"

    def test_compartment_deltas_match_time_series(self, model_run):
        """compartment_deltas keys should match the compartments in time_series."""
        _, results = model_run
        for run in results:
            assert "compartment_deltas" in run, "Output missing compartment_deltas"
            deltas = run["compartment_deltas"]
            ts = run["parent_admin_total"]["time_series"]
            ts_compartments = {k for k in ts[0].keys() if k != "date"}
            delta_compartments = set(deltas.keys())
            assert delta_compartments == ts_compartments, (
                f"compartment_deltas keys {sorted(delta_compartments)} "
                f"don't match time_series compartments {sorted(ts_compartments)}"
            )

    def test_no_nan_values(self, model_run):
        """No compartment should contain NaN values."""
        _, results = model_run
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            for entry in ts:
                for key, val in entry.items():
                    if key == "date":
                        continue
                    if isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            if isinstance(sub_val, (int, float)):
                                assert not math.isnan(sub_val), (
                                    f"NaN in {key}.{sub_key} on {entry['date']}"
                                )
                            elif isinstance(sub_val, dict):
                                for stat, v in sub_val.items():
                                    if isinstance(v, (int, float)):
                                        assert not math.isnan(v), (
                                            f"NaN in {key}.{sub_key}.{stat} on {entry['date']}"
                                        )

    def test_no_negative_compartments(self, model_run):
        """No compartment population should be significantly negative."""
        _, results = model_run
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            for entry in ts:
                for key, val in entry.items():
                    if key == "date":
                        continue
                    if isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            if isinstance(sub_val, (int, float)):
                                assert sub_val >= -1.0, (
                                    f"Negative value {sub_val} in {key}.{sub_key} "
                                    f"on {entry['date']}"
                                )

    def test_has_admin_zones(self, model_run):
        """Output should contain admin_zones with time series."""
        _, results = model_run
        for run in results:
            assert "admin_zones" in run
            zones = run["admin_zones"]
            assert len(zones) > 0
            for zone in zones:
                assert "admin_zone_id" in zone
                assert "time_series" in zone

    def test_has_interventions(self, model_run):
        """Output should have an interventions list."""
        _, results = model_run
        for run in results:
            assert "interventions" in run

    def test_dates_are_sequential(self, model_run):
        """Dates in time series should be monotonically increasing."""
        from datetime import datetime
        _, results = model_run
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            if len(ts) < 2:
                continue
            dates = [datetime.strptime(e["date"], "%Y-%m-%d") for e in ts]
            for i in range(1, len(dates)):
                assert dates[i] > dates[i - 1], (
                    f"Dates not sequential: {dates[i-1]} >= {dates[i]}"
                )
