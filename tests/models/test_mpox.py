"""Tests for MpoxJaxModel (simplest SIR reference implementation)."""
import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# disease_type property
# ---------------------------------------------------------------------------

def test_disease_type(mpox_model):
    assert mpox_model.disease_type == "MONKEYPOX"


# ---------------------------------------------------------------------------
# get_params
# ---------------------------------------------------------------------------

def test_get_params_returns_beta_gamma(mpox_model):
    params = mpox_model.get_params()
    assert len(params) == 2
    beta, gamma = params
    assert float(beta) == pytest.approx(0.3)
    assert float(gamma) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# prepare_initial_state
# ---------------------------------------------------------------------------

def test_prepare_initial_state_shape(mpox_model):
    y, compartments = mpox_model.prepare_initial_state()
    # shape should be (n_compartments, n_regions) = (3, 1)
    assert y.shape == (3, 1)


def test_prepare_initial_state_compartment_list(mpox_model):
    _, compartments = mpox_model.prepare_initial_state()
    assert compartments == ["S", "I", "R"]


def test_prepare_initial_state_population_values(mpox_model):
    y, compartments = mpox_model.prepare_initial_state()
    # 100k population, 1% infected → 1000 infected, 99000 susceptible
    s_idx = compartments.index("S")
    i_idx = compartments.index("I")
    r_idx = compartments.index("R")
    assert float(y[s_idx, 0]) == pytest.approx(99_000.0, rel=1e-3)
    assert float(y[i_idx, 0]) == pytest.approx(1_000.0, rel=1e-3)
    assert float(y[r_idx, 0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# derivative
# ---------------------------------------------------------------------------

def test_derivative_zero_infected_means_zero_ds_dt(mpox_model):
    # When I=0, dS/dt should be 0 (no transmission)
    y = jnp.array([[100_000.0], [0.0], [0.0]])
    p = mpox_model.get_params()
    derivs = mpox_model.derivative(y, 0.0, p)
    # derivs shape: (3, 1)
    assert float(derivs[0, 0]) == pytest.approx(0.0, abs=1e-6)   # dS/dt
    assert float(derivs[1, 0]) == pytest.approx(0.0, abs=1e-6)   # dI/dt


def test_derivative_population_conservation(mpox_model):
    # Sum of all derivatives should be ≈ 0 (closed population)
    y = jnp.array([[99_000.0], [1_000.0], [0.0]])
    p = mpox_model.get_params()
    derivs = mpox_model.derivative(y, 0.0, p)
    assert float(derivs.sum()) == pytest.approx(0.0, abs=1e-4)


def test_derivative_infection_flows_correctly(mpox_model):
    # With I > 0: dS/dt < 0, dR/dt > 0
    y = jnp.array([[99_000.0], [1_000.0], [0.0]])
    p = mpox_model.get_params()
    derivs = mpox_model.derivative(y, 0.0, p)
    assert float(derivs[0, 0]) < 0    # dS/dt < 0
    assert float(derivs[2, 0]) > 0    # dR/dt > 0


def test_derivative_shape(mpox_model):
    y = jnp.array([[99_000.0], [1_000.0], [0.0]])
    p = mpox_model.get_params()
    derivs = mpox_model.derivative(y, 0.0, p)
    assert derivs.shape == (3, 1)


# ---------------------------------------------------------------------------
# get_initial_population (classmethod)
# ---------------------------------------------------------------------------

def test_get_initial_population_si_totals(mpox_model):
    from compartment.models.mpox_jax_model.model import MpoxJaxModel

    admin_zones = [{"population": 100_000, "infected_population": 5.0}]
    compartment_list = ["S", "I", "R"]
    pop = MpoxJaxModel.get_initial_population(admin_zones, compartment_list)
    # 5% of 100k = 5000 infected
    assert pop[0, compartment_list.index("I")] == pytest.approx(5_000.0)
    assert pop[0, compartment_list.index("S")] + pop[0, compartment_list.index("I")] == pytest.approx(100_000.0)
    assert pop[0, compartment_list.index("R")] == pytest.approx(0.0)


def test_get_initial_population_multiple_zones():
    from compartment.models.mpox_jax_model.model import MpoxJaxModel

    admin_zones = [
        {"population": 100_000, "infected_population": 1.0},
        {"population": 50_000, "infected_population": 2.0},
    ]
    compartment_list = ["S", "I", "R"]
    pop = MpoxJaxModel.get_initial_population(admin_zones, compartment_list)
    assert pop.shape == (2, 3)
    assert pop[0, 0] + pop[0, 1] == pytest.approx(100_000.0)
    assert pop[1, 0] + pop[1, 1] == pytest.approx(50_000.0)
