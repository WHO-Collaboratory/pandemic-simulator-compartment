"""Tests for CovidJaxModel."""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# disease_type
# ---------------------------------------------------------------------------

def test_disease_type(covid_model):
    assert covid_model.disease_type == "RESPIRATORY"


# ---------------------------------------------------------------------------
# prepare_initial_state
# ---------------------------------------------------------------------------

def test_prepare_initial_state_shape_with_age_demographics(covid_model):
    y, compartments = covid_model.prepare_initial_state()
    # SIR + I_total = 4 compartments, 3 age groups, 1 region → (4, 3, 1)
    assert y.ndim == 3
    n_comp, n_age, n_regions = y.shape
    assert n_age == 3          # age_0_17, age_18_55, age_56_plus
    assert n_regions == 1


def test_prepare_initial_state_compartment_list_includes_totals(covid_model):
    _, compartments = covid_model.prepare_initial_state()
    # SIR: only I should get a _total because _add_cumulative_compartments adds for ["E","I","H"]
    assert "I_total" in compartments
    # E and H are not in ["S", "I", "R"], so they should NOT appear
    assert "E_total" not in compartments
    assert "H_total" not in compartments


def test_prepare_initial_state_population_values_non_negative(covid_model):
    y, _ = covid_model.prepare_initial_state()
    assert float(y.min()) >= 0.0


def test_prepare_initial_state_total_population_preserved(covid_model):
    y, compartments = covid_model.prepare_initial_state()
    # Sum over S, I, R (not I_total) across all age groups should equal original population
    sir_indices = [compartments.index(c) for c in ["S", "I", "R"]]
    total = float(y[sir_indices, :, :].sum())
    assert total == pytest.approx(100_000.0, rel=1e-3)


# ---------------------------------------------------------------------------
# get_params
# ---------------------------------------------------------------------------

def test_get_params_returns_7_elements(covid_model):
    # (interaction_matrix, theta, gamma, zeta, eta, epsilon, delta)
    params = covid_model.get_params()
    assert len(params) == 7


def test_get_params_interaction_matrix_shape(covid_model):
    params = covid_model.get_params()
    interaction_matrix = params[0]
    assert interaction_matrix.shape == (3, 3)


def test_get_params_rates_match_config(covid_model):
    params = covid_model.get_params()
    _, theta, gamma, zeta, eta, epsilon, delta = params
    assert float(theta) == pytest.approx(0.2)
    assert float(gamma) == pytest.approx(0.14)
    assert float(zeta) == pytest.approx(0.05)
    assert float(eta) == pytest.approx(0.1)
    assert float(epsilon) == pytest.approx(0.02)
    assert float(delta) == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Cumulative compartments
# ---------------------------------------------------------------------------

def test_cumulative_compartments_non_negative_after_prepare(covid_model):
    y, compartments = covid_model.prepare_initial_state()
    if "I_total" in compartments:
        idx = compartments.index("I_total")
        assert float(y[idx].min()) >= 0.0
