"""Tests for compartment/interventions.py."""
import jax.numpy as jnp
import numpy as np
import pytest

from compartment.interventions import jax_prop_intervention, jax_timestep_intervention


# ---------------------------------------------------------------------------
# Helpers / shared data
# ---------------------------------------------------------------------------

def _make_covid_rates(beta=0.5):
    return {"beta": beta}


def _make_social_isolation_cfg(start_ord, end_ord=None, adherence=0.8, reduc=0.5):
    return {
        "start_date_ordinal": start_ord,
        "end_date_ordinal": end_ord,
        "adherence_min": adherence,
        "transmission_percentage": reduc,
    }


def _make_threshold_cfg(start_th=0.01, end_th=0.005, adherence=0.8, reduc=0.5):
    return {
        "start_threshold": start_th,
        "end_threshold": end_th,
        "adherence_min": adherence,
        "transmission_percentage": reduc,
    }


# ---------------------------------------------------------------------------
# jax_timestep_intervention
# ---------------------------------------------------------------------------

class TestTimestepIntervention:
    def test_rate_reduced_when_in_window(self):
        rates = _make_covid_rates(beta=0.5)
        cfg = _make_social_isolation_cfg(start_ord=100, end_ord=200)
        idict = {"social_isolation": cfg}
        istat = {"social_isolation": False}
        tm = jnp.eye(1)

        new_rates, _, _ = jax_timestep_intervention(idict, 150, rates, istat, tm)
        # expected: 0.5 * (1 - 0.8 * 0.5) = 0.5 * 0.6 = 0.3
        assert float(new_rates["beta"]) == pytest.approx(0.3)

    def test_rate_unchanged_outside_window(self):
        rates = _make_covid_rates(beta=0.5)
        cfg = _make_social_isolation_cfg(start_ord=100, end_ord=200)
        idict = {"social_isolation": cfg}
        istat = {"social_isolation": False}
        tm = jnp.eye(1)

        new_rates, _, _ = jax_timestep_intervention(idict, 300, rates, istat, tm)
        assert float(new_rates["beta"]) == pytest.approx(0.5)

    def test_rate_unchanged_before_window(self):
        rates = _make_covid_rates(beta=0.5)
        cfg = _make_social_isolation_cfg(start_ord=100, end_ord=200)
        idict = {"social_isolation": cfg}
        istat = {"social_isolation": False}
        tm = jnp.eye(1)

        new_rates, _, _ = jax_timestep_intervention(idict, 50, rates, istat, tm)
        assert float(new_rates["beta"]) == pytest.approx(0.5)

    def test_no_start_date_never_activates(self):
        rates = _make_covid_rates(beta=0.5)
        cfg = {
            "start_date_ordinal": None,
            "end_date_ordinal": None,
            "adherence_min": 0.8,
            "transmission_percentage": 0.5,
        }
        idict = {"social_isolation": cfg}
        istat = {"social_isolation": False}
        tm = jnp.eye(1)

        # Any ordinal day should leave rate unchanged
        new_rates, new_stat, _ = jax_timestep_intervention(idict, 9999, rates, istat, tm)
        assert float(new_rates["beta"]) == pytest.approx(0.5)

    def test_no_end_date_stays_active_after_start(self):
        rates = _make_covid_rates(beta=0.5)
        cfg = _make_social_isolation_cfg(start_ord=100, end_ord=None)
        idict = {"social_isolation": cfg}
        istat = {"social_isolation": False}
        tm = jnp.eye(1)

        # Well past start date, no end date — should be active
        new_rates, _, _ = jax_timestep_intervention(idict, 10_000, rates, istat, tm)
        assert float(new_rates["beta"]) == pytest.approx(0.3)

    def test_lockdown_travel_matrix_becomes_identity(self):
        rates = _make_covid_rates(beta=0.5)
        non_identity = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        cfg = {
            "start_date_ordinal": 100,
            "end_date_ordinal": 200,
            "adherence_min": 1.0,
            "transmission_percentage": 1.0,
        }
        idict = {"lock_down": cfg}
        istat = {"lock_down": False}

        _, _, new_tm = jax_timestep_intervention(idict, 150, rates, istat, non_identity)
        expected = jnp.eye(2)
        assert jnp.allclose(new_tm, expected)

    def test_lockdown_travel_matrix_unchanged_outside_window(self):
        rates = _make_covid_rates(beta=0.5)
        non_identity = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        cfg = {
            "start_date_ordinal": 100,
            "end_date_ordinal": 200,
            "adherence_min": 1.0,
            "transmission_percentage": 1.0,
        }
        idict = {"lock_down": cfg}
        istat = {"lock_down": False}

        _, _, new_tm = jax_timestep_intervention(idict, 300, rates, istat, non_identity)
        assert jnp.allclose(new_tm, non_identity)

    def test_empty_intervention_dict_returns_unchanged(self):
        rates = _make_covid_rates(beta=0.5)
        tm = jnp.eye(2)
        new_rates, new_stat, new_tm = jax_timestep_intervention({}, 150, rates, {}, tm)
        assert float(new_rates["beta"]) == pytest.approx(0.5)
        assert jnp.allclose(new_tm, tm)


# ---------------------------------------------------------------------------
# jax_prop_intervention
# ---------------------------------------------------------------------------

class TestPropIntervention:
    def test_activates_when_threshold_crossed(self):
        rates = _make_covid_rates(beta=0.5)
        cfg = _make_threshold_cfg(start_th=0.01, end_th=0.005, adherence=0.8, reduc=0.5)
        idict = {"social_isolation": cfg}
        istat = {"social_isolation": False}
        tm = jnp.eye(1)

        # 2% infected > 1% threshold → activates
        new_rates, new_stat, _ = jax_prop_intervention(idict, 0.02, rates, istat, tm)
        assert float(new_rates["beta"]) == pytest.approx(0.3)  # 0.5*(1-0.8*0.5)
        assert bool(new_stat["social_isolation"]) is True

    def test_does_not_activate_below_threshold(self):
        rates = _make_covid_rates(beta=0.5)
        cfg = _make_threshold_cfg(start_th=0.01)
        idict = {"social_isolation": cfg}
        istat = {"social_isolation": False}
        tm = jnp.eye(1)

        # 0.5% infected < 1% threshold → no activation
        new_rates, new_stat, _ = jax_prop_intervention(idict, 0.005, rates, istat, tm)
        assert float(new_rates["beta"]) == pytest.approx(0.5)
        assert bool(new_stat["social_isolation"]) is False

    def test_deactivates_below_end_threshold(self):
        rates = _make_covid_rates(beta=0.5)
        cfg = _make_threshold_cfg(start_th=0.01, end_th=0.005, adherence=0.8, reduc=0.5)
        idict = {"social_isolation": cfg}
        # Start as already active
        istat = {"social_isolation": True}
        tm = jnp.eye(1)

        # 0.4% infected < 0.5% end_threshold → deactivates
        new_rates, new_stat, _ = jax_prop_intervention(idict, 0.004, rates, istat, tm)
        assert float(new_rates["beta"]) == pytest.approx(0.5)
        assert bool(new_stat["social_isolation"]) is False

    def test_rate_reduction_formula(self):
        rates = _make_covid_rates(beta=1.0)
        cfg = _make_threshold_cfg(start_th=0.01, end_th=None, adherence=0.6, reduc=0.4)
        idict = {"social_isolation": cfg}
        istat = {"social_isolation": False}
        tm = jnp.eye(1)

        new_rates, _, _ = jax_prop_intervention(idict, 0.02, rates, istat, tm)
        # 1.0 * (1 - 0.6 * 0.4) = 1.0 * 0.76 = 0.76
        assert float(new_rates["beta"]) == pytest.approx(0.76)

    def test_no_start_threshold_never_activates(self):
        rates = _make_covid_rates(beta=0.5)
        cfg = {
            "start_threshold": None,
            "end_threshold": None,
            "adherence_min": 0.8,
            "transmission_percentage": 0.5,
        }
        idict = {"social_isolation": cfg}
        istat = {"social_isolation": False}
        tm = jnp.eye(1)

        new_rates, new_stat, _ = jax_prop_intervention(idict, 0.5, rates, istat, tm)
        assert float(new_rates["beta"]) == pytest.approx(0.5)
        assert bool(new_stat["social_isolation"]) is False

    def test_lockdown_travel_matrix_becomes_identity_on_threshold(self):
        rates = _make_covid_rates(beta=0.5)
        non_identity = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        cfg = _make_threshold_cfg(start_th=0.01, end_th=0.005)
        idict = {"lock_down": cfg}
        istat = {"lock_down": False}

        _, _, new_tm = jax_prop_intervention(idict, 0.05, rates, istat, non_identity)
        assert jnp.allclose(new_tm, jnp.eye(2))

    def test_empty_intervention_dict_returns_unchanged(self):
        rates = _make_covid_rates(beta=0.5)
        tm = jnp.eye(2)
        new_rates, _, new_tm = jax_prop_intervention({}, 0.1, rates, {}, tm)
        assert float(new_rates["beta"]) == pytest.approx(0.5)
        assert jnp.allclose(new_tm, tm)
