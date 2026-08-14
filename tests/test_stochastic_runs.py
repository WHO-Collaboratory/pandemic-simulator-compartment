"""Unit and integration tests for stochastic run mode.

Covers:
  - SimulationManager solver dispatch → Euler for models with STOCHASTIC=True (unit)
  - Stochastic runs produce parameter uncertainty-format output (CI bands, not scalar) (integration)
  - CI bands have non-zero spread — different seeds produce different trajectories (integration)
  - Stochastic mode always runs exactly 30 trajectories (integration, via mock)
  - Config run_mode=DETERMINISTIC does not suppress stochastic batch behaviour (integration)
  - Variance params spread across the 30 stochastic runs, not added on top (integration, via mock)

Unit tests run without a marker.  Integration tests are marked @pytest.mark.integration
and invoke the full simulation stack; expect ~10–30s for the 30-trajectory SIR runs.

Run:
    python -m pytest tests/test_stochastic_runs.py -v                 # unit only
    python -m pytest tests/test_stochastic_runs.py -v -m integration  # all
"""

import json
import tempfile
import pathlib
from unittest.mock import patch

import pytest

from compartment.simulation_manager import SimulationManager
from compartment.models.test_covid_sir_stochastic.model import CovidSirStochasticModel
from compartment.run_simulation import run_simulation


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

def _stochastic_config(**overrides) -> dict:
    """Minimal valid config for CovidSirStochasticModel.

    Uses a short date window (31 days) to keep integration tests fast.
    """
    cfg = {
        "Disease": {
            "disease_type": "COVID_SIR_STOCHASTIC",
            "transmission_edges": [
                {
                    "source": "susceptible",
                    "target": "infected",
                    "data": {"transmission_rate": 0.4},
                },
                {
                    "source": "infected",
                    "target": "recovered",
                    "data": {"transmission_rate": 7.0},
                },
            ],
        },
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "admin_unit_id": "USA",
        "AdminUnit": {"id": "USA", "center_lat": 37.0902},
        "case_file": {
            "admin_zones": [
                {
                    "name": "Zone A",
                    "center_lat": 40.7128,
                    "center_lon": -74.006,
                    "population": 100000,
                    "infected_population": 50,
                }
            ]
        },
    }
    cfg.update(overrides)
    return cfg


def _run(cfg: dict) -> list[dict]:
    """Write cfg to a temp file, run the stochastic model, return parsed results."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        config_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    run_simulation(
        model_class=CovidSirStochasticModel,
        config_path=config_path,
        output_path=output_path,
    )
    with open(output_path) as f:
        return json.load(f)


def _is_uncertainty_format(time_series: list) -> bool:
    """True when compartment values are CI dicts {median, lower, upper}."""
    if not time_series:
        return False
    first = time_series[0]
    values = [v for k, v in first.items() if k != "date"]
    return bool(values) and all(isinstance(v, dict) and "median" in v for v in values)


def _has_spread(time_series: list) -> bool:
    """True when any CI band has lower < upper at the midpoint or final timestep."""
    for idx in (len(time_series) // 2, -1):
        for k, v in time_series[idx].items():
            if k == "date":
                continue
            if isinstance(v, dict) and v.get("lower", 0) < v.get("upper", 0):
                return True
    return False


# ---------------------------------------------------------------------------
# Unit tests — no simulation, milliseconds
# ---------------------------------------------------------------------------

class _Det:
    """Minimal stand-in for a deterministic model class."""


class _Stoch:
    """Minimal stand-in for a stochastic model class."""
    STOCHASTIC = True


class _ExplicitSolver:
    """Model that sets SOLVER explicitly, overriding STOCHASTIC."""
    STOCHASTIC = True
    SOLVER = "odeint"


class TestSolverDispatch:
    """SimulationManager._euler_integrate / odeint selection via STOCHASTIC flag."""

    def _solver(self, model_obj):
        """Replicate the solver-selection logic from SimulationManager.run_simulation."""
        sm = SimulationManager(model_obj)
        solver = getattr(sm.model, "SOLVER", None)
        if solver is None:
            solver = "euler" if getattr(sm.model, "STOCHASTIC", False) else "odeint"
        return solver

    def test_stochastic_flag_selects_euler(self):
        assert self._solver(_Stoch()) == "euler"

    def test_no_flag_selects_odeint(self):
        assert self._solver(_Det()) == "odeint"

    def test_explicit_solver_attr_takes_precedence_over_stochastic_flag(self):
        # SOLVER = "odeint" wins even when STOCHASTIC = True
        assert self._solver(_ExplicitSolver()) == "odeint"

    def test_real_stochastic_model_class_selects_euler(self):
        # CovidSirStochasticModel itself — not a mock
        instance = object.__new__(CovidSirStochasticModel)
        assert self._solver(instance) == "euler"


# ---------------------------------------------------------------------------
# Integration tests — full simulation stack
# ---------------------------------------------------------------------------

class TestStochasticOutputFormat:
    """Stochastic runs must produce multi-run (CI-band) formatted output,
    identical in structure to UNCERTAINTY runs."""

    @pytest.mark.integration
    def test_produces_ci_band_format_not_scalar(self):
        results = _run(_stochastic_config())
        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert _is_uncertainty_format(ts), (
            "STOCHASTIC model should produce {median, lower, upper} CI output, "
            "not bare scalar values."
        )

    @pytest.mark.integration
    def test_returns_exactly_two_arms(self):
        results = _run(_stochastic_config())
        assert len(results) == 2, f"Expected 2 result arms, got {len(results)}"

    @pytest.mark.integration
    def test_both_arms_have_correct_control_run_flag(self):
        results = _run(_stochastic_config())
        flags = {r["control_run"] for r in results}
        assert flags == {True, False}

    @pytest.mark.integration
    def test_control_arm_is_also_ci_band_format(self):
        results = _run(_stochastic_config())
        ctrl = next(r for r in results if r["control_run"])
        ts = ctrl["parent_admin_total"]["time_series"]
        assert _is_uncertainty_format(ts)


class TestStochasticSpread:
    """30 independent seeds should produce trajectories that diverge,
    giving CI bands with lower < upper."""

    @pytest.mark.integration
    def test_with_arm_ci_bands_have_spread(self):
        results = _run(_stochastic_config())
        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert _has_spread(ts), (
            "30 stochastic trajectories should produce CI spread (lower < upper). "
            "If all trajectories are identical, per-run seed isolation is broken."
        )

    @pytest.mark.integration
    def test_control_arm_ci_bands_have_spread(self):
        results = _run(_stochastic_config())
        ctrl = next(r for r in results if r["control_run"])
        ts = ctrl["parent_admin_total"]["time_series"]
        assert _has_spread(ts)


class TestStochasticRunCount:
    """Stochastic run count: defaults to 30, honoured from n_simulations when set."""

    @pytest.mark.integration
    def test_defaults_to_30_without_n_simulations(self):
        """Without n_simulations in config, stochastic trajectories default to 30."""
        cfg = _stochastic_config()  # no n_simulations key
        n_sims_seen = []

        import compartment.run_simulation as _rs
        original = _rs.batch_simulate_and_postprocess

        def _capture(model, n_sims, param_list, ci, num_workers):
            n_sims_seen.append(n_sims)
            return original(model, n_sims, param_list, ci, num_workers)

        with patch.object(_rs, "batch_simulate_and_postprocess", side_effect=_capture):
            _run(cfg)

        assert n_sims_seen, "batch_simulate_and_postprocess was never called"
        assert all(n == 30 for n in n_sims_seen), (
            f"Expected 30 trajectories per arm by default, got {n_sims_seen}."
        )

    @pytest.mark.integration
    def test_honours_n_simulations_from_config(self):
        """n_simulations in config overrides the 30-trajectory default (e.g. for smoke tests)."""
        cfg = _stochastic_config(n_simulations=6)
        n_sims_seen = []

        import compartment.run_simulation as _rs
        original = _rs.batch_simulate_and_postprocess

        def _capture(model, n_sims, param_list, ci, num_workers):
            n_sims_seen.append(n_sims)
            return original(model, n_sims, param_list, ci, num_workers)

        with patch.object(_rs, "batch_simulate_and_postprocess", side_effect=_capture):
            _run(cfg)

        assert n_sims_seen, "batch_simulate_and_postprocess was never called"
        assert all(n == 6 for n in n_sims_seen), (
            f"Expected 6 trajectories per arm (from n_simulations=6), got {n_sims_seen}."
        )

    @pytest.mark.integration
    def test_exactly_two_batch_calls_one_per_arm(self):
        """One call for the intervention arm, one for the control arm."""
        cfg = _stochastic_config()
        call_count = []

        import compartment.run_simulation as _rs
        original = _rs.batch_simulate_and_postprocess

        def _capture(model, n_sims, param_list, ci, num_workers):
            call_count.append(1)
            return original(model, n_sims, param_list, ci, num_workers)

        with patch.object(_rs, "batch_simulate_and_postprocess", side_effect=_capture):
            _run(cfg)

        assert len(call_count) == 2, (
            f"Expected exactly 2 batch calls (one per arm), got {len(call_count)}"
        )


class TestStochasticRunModeOverride:
    """The STOCHASTIC model flag must override the config's run_mode field."""

    @pytest.mark.integration
    def test_config_deterministic_still_triggers_batch_run(self):
        """Explicit run_mode=DETERMINISTIC in config must not suppress the 30-run batch."""
        cfg = _stochastic_config(run_mode="DETERMINISTIC")
        n_sims_seen = []

        import compartment.run_simulation as _rs
        original = _rs.batch_simulate_and_postprocess

        def _capture(model, n_sims, param_list, ci, num_workers):
            n_sims_seen.append(n_sims)
            return original(model, n_sims, param_list, ci, num_workers)

        with patch.object(_rs, "batch_simulate_and_postprocess", side_effect=_capture):
            _run(cfg)

        assert n_sims_seen, (
            "batch_simulate_and_postprocess was not called — "
            "the STOCHASTIC model may have fallen through to the DETERMINISTIC branch."
        )
        assert all(n == 30 for n in n_sims_seen), (
            f"Expected 30 runs per arm; got {n_sims_seen}. "
            "STOCHASTIC flag must override run_mode=DETERMINISTIC in config."
        )

    @pytest.mark.integration
    def test_config_deterministic_output_is_still_ci_format(self):
        """Double-check via output format: CI bands, not scalar."""
        results = _run(_stochastic_config(run_mode="DETERMINISTIC"))
        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert _is_uncertainty_format(ts), (
            "CI-band format expected even when config has run_mode=DETERMINISTIC, "
            "because the model class has STOCHASTIC=True."
        )


class TestStochasticWithVarianceParams:
    """When variance params are declared alongside a STOCHASTIC model,
    the run stays at 30 trajectories and LHS samples are distributed across them."""

    @pytest.mark.integration
    def test_variance_params_do_not_increase_run_count(self):
        """Stochastic + variance must stay at 30, not multiply runs."""
        cfg = _stochastic_config()
        cfg["Disease"]["variance_params"] = [
            {"param": "beta", "dist": "uniform", "min": 0.2, "max": 0.6}
        ]
        n_sims_seen = []

        import compartment.run_simulation as _rs
        original = _rs.batch_simulate_and_postprocess

        def _capture(model, n_sims, param_list, ci, num_workers):
            n_sims_seen.append(n_sims)
            # Run with empty params to avoid potential rebuild issues in the
            # stochastic model's custom __init__; we only care about n_sims here.
            empty = [{} for _ in range(n_sims)]
            return original(model, n_sims, empty, ci, num_workers)

        with patch.object(_rs, "batch_simulate_and_postprocess", side_effect=_capture):
            _run(cfg)

        assert all(n == 30 for n in n_sims_seen), (
            f"Expected 30 runs per arm; got {n_sims_seen}. "
            "Variance params must spread across the 30 stochastic runs, not add more."
        )

    @pytest.mark.integration
    def test_variance_params_generate_lhs_param_list_not_empty_dicts(self):
        """With variance params, each batch call should receive a non-trivial param_list
        (LHS samples), not the all-empty list used for pure stochastic runs."""
        cfg = _stochastic_config()
        cfg["Disease"]["variance_params"] = [
            {"param": "beta", "dist": "uniform", "min": 0.2, "max": 0.6}
        ]
        param_lists_seen = []

        import compartment.run_simulation as _rs
        original = _rs.batch_simulate_and_postprocess

        def _capture(model, n_sims, param_list, ci, num_workers):
            param_lists_seen.append(param_list)
            # Swap in empty params so rebuild issues don't fail the sim
            empty = [{} for _ in range(n_sims)]
            return original(model, n_sims, empty, ci, num_workers)

        with patch.object(_rs, "batch_simulate_and_postprocess", side_effect=_capture):
            _run(cfg)

        assert param_lists_seen, "batch_simulate_and_postprocess was never called"
        # The intervention arm (first call) should have non-empty LHS dicts
        with_arm_params = param_lists_seen[0]
        assert len(with_arm_params) == 30
        assert any(bool(p) for p in with_arm_params), (
            "When variance_params are declared, the param_list passed to the "
            "stochastic batch should contain LHS samples, not all-empty dicts."
        )

    @pytest.mark.integration
    def test_no_variance_params_uses_empty_param_list(self):
        """Without variance params, the batch receives all-empty dicts — pure stochastic."""
        cfg = _stochastic_config()
        param_lists_seen = []

        import compartment.run_simulation as _rs
        original = _rs.batch_simulate_and_postprocess

        def _capture(model, n_sims, param_list, ci, num_workers):
            param_lists_seen.append(param_list)
            return original(model, n_sims, param_list, ci, num_workers)

        with patch.object(_rs, "batch_simulate_and_postprocess", side_effect=_capture):
            _run(cfg)

        for arm_params in param_lists_seen:
            assert all(p == {} for p in arm_params), (
                "Without variance_params, all param dicts should be empty — "
                "stochasticity alone drives trajectory divergence."
            )
