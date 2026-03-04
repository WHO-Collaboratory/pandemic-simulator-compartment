"""Tests for DengueJaxModel (4-serotype vector-borne model)."""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# disease_type
# ---------------------------------------------------------------------------

def test_disease_type(dengue_model):
    assert dengue_model.disease_type == "VECTOR_BORNE"


# ---------------------------------------------------------------------------
# prepare_initial_state
# ---------------------------------------------------------------------------

def test_prepare_initial_state_shape(dengue_model):
    from compartment.models.dengue_jax_model.model import DengueJaxModel

    y, compartments = dengue_model.prepare_initial_state()
    n_base = len(DengueJaxModel.COMPARTMENT_LIST)   # 58
    n_cumulative = 8                                  # added by _add_cumulative_compartments
    expected_comps = n_base + n_cumulative
    assert y.shape == (expected_comps, 1)             # 1 region
    assert len(compartments) == expected_comps


def test_prepare_initial_state_compartments_include_base_and_cumulative(dengue_model):
    from compartment.models.dengue_jax_model.model import DengueJaxModel

    _, compartments = dengue_model.prepare_initial_state()
    # Base compartments should be present
    for comp in ["SV", "S0", "I1", "R1"]:
        assert comp in compartments
    # Cumulative compartments should be appended
    for comp in ["E_total", "I_total", "C_total", "Snot_total", "E2_total", "I2_total", "H_total", "R_total"]:
        assert comp in compartments


# ---------------------------------------------------------------------------
# COMPARTMENT_DELTA_GROUPING
# ---------------------------------------------------------------------------

def test_compartment_delta_grouping_count():
    from compartment.models.dengue_jax_model.model import DengueJaxModel

    grouping = DengueJaxModel.COMPARTMENT_DELTA_GROUPING
    # 12 groups: SV, EV, IV, S, E, I, C, Snot, E2, I2, H, R
    assert len(grouping) == 12


def test_compartment_delta_grouping_no_total_keys_as_group_names():
    from compartment.models.dengue_jax_model.model import DengueJaxModel

    grouping = DengueJaxModel.COMPARTMENT_DELTA_GROUPING
    for group_name in grouping:
        assert not group_name.endswith("_total"), (
            f"Group name '{group_name}' should not have _total suffix"
        )


def test_compartment_delta_grouping_covers_all_base_compartments():
    from compartment.models.dengue_jax_model.model import DengueJaxModel

    grouping = DengueJaxModel.COMPARTMENT_DELTA_GROUPING
    all_members = {comp for members in grouping.values() for comp in members}
    for comp in DengueJaxModel.COMPARTMENT_LIST:
        assert comp in all_members, f"{comp} not covered in COMPARTMENT_DELTA_GROUPING"


# ---------------------------------------------------------------------------
# get_initial_population (classmethod)
# ---------------------------------------------------------------------------

def test_get_initial_population_seroprevalence_sets_snot():
    from compartment.models.dengue_jax_model.model import DengueJaxModel

    admin_zones = [
        {
            "population": 100_000,
            "infected_population": 3.0,
            "seroprevalence": 30.0,  # 30% previously exposed
        }
    ]
    compartment_list = DengueJaxModel.COMPARTMENT_LIST
    pop = DengueJaxModel.get_initial_population(admin_zones, compartment_list)

    # Each of Snot1..4 should hold 25% × 30% × 100k = 7500
    for i in range(1, 5):
        snot_idx = compartment_list.index(f"Snot{i}")
        assert pop[0, snot_idx] == pytest.approx(7_500.0, rel=1e-3)


def test_get_initial_population_infected_population_set():
    from compartment.models.dengue_jax_model.model import DengueJaxModel

    admin_zones = [
        {
            "population": 100_000,
            "infected_population": 4.0,  # 4% infected
            "seroprevalence": 0.0,
        }
    ]
    compartment_list = DengueJaxModel.COMPARTMENT_LIST
    pop = DengueJaxModel.get_initial_population(admin_zones, compartment_list)

    # Total infected across I1-I4 should be 4% × 100k = 4000
    i_indices = [compartment_list.index(f"I{i}") for i in range(1, 5)]
    total_infected = sum(pop[0, idx] for idx in i_indices)
    assert total_infected == pytest.approx(4_000.0, rel=1e-3)


def test_get_initial_population_s0_is_remainder():
    from compartment.models.dengue_jax_model.model import DengueJaxModel

    admin_zones = [
        {
            "population": 100_000,
            "infected_population": 3.0,
            "seroprevalence": 10.0,
        }
    ]
    compartment_list = DengueJaxModel.COMPARTMENT_LIST
    pop = DengueJaxModel.get_initial_population(admin_zones, compartment_list)

    s0_idx = compartment_list.index("S0")
    # Snot total = 10% × 100k = 10000; infected total = 3% × 100k = 3000
    # S0 = 100000 - 10000 - 3000 = 87000
    assert pop[0, s0_idx] == pytest.approx(87_000.0, rel=1e-3)


def test_get_initial_population_no_seroprevalence():
    from compartment.models.dengue_jax_model.model import DengueJaxModel

    admin_zones = [{"population": 100_000, "infected_population": 0.0, "seroprevalence": 0.0}]
    compartment_list = DengueJaxModel.COMPARTMENT_LIST
    pop = DengueJaxModel.get_initial_population(admin_zones, compartment_list)

    # All Snot compartments should be 0
    for i in range(1, 5):
        snot_idx = compartment_list.index(f"Snot{i}")
        assert pop[0, snot_idx] == pytest.approx(0.0)

    # S0 should equal full population
    s0_idx = compartment_list.index("S0")
    assert pop[0, s0_idx] == pytest.approx(100_000.0)
