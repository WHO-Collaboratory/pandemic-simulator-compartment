"""Tests for pure helper functions in compartment/helpers.py."""
import numpy as np
import pytest
from datetime import date, datetime

from compartment.helpers import (
    compute_jax_compartment_deltas,
    compute_multi_run_compartment_deltas,
    convert_dates,
    create_compartment_list,
    create_initial_population_matrix,
    create_intervention_dict,
    create_transmission_dict,
    extract_admin_units,
    get_compartment_delta_grouping,
    get_simulation_step_size,
    transform_interventions,
)


# ---------------------------------------------------------------------------
# get_compartment_delta_grouping
# ---------------------------------------------------------------------------

class _ModelWithGrouping:
    COMPARTMENT_DELTA_GROUPING = {"S": ["S"], "I": ["I1", "I2"], "R": ["R"]}


class _ModelWithoutGrouping:
    pass


def test_get_compartment_delta_grouping_uses_model_attribute():
    compartment_list = ["S", "I1", "I2", "R"]
    result = get_compartment_delta_grouping(_ModelWithGrouping, compartment_list)
    assert result == _ModelWithGrouping.COMPARTMENT_DELTA_GROUPING


def test_get_compartment_delta_grouping_default_1to1():
    compartment_list = ["S", "I", "R"]
    result = get_compartment_delta_grouping(_ModelWithoutGrouping, compartment_list)
    assert result == {"S": ["S"], "I": ["I"], "R": ["R"]}


def test_get_compartment_delta_grouping_default_excludes_total():
    compartment_list = ["S", "I", "R", "I_total"]
    result = get_compartment_delta_grouping(None, compartment_list)
    assert "I_total" not in result
    assert set(result.keys()) == {"S", "I", "R"}


def test_get_compartment_delta_grouping_none_model():
    compartment_list = ["S", "I", "R"]
    result = get_compartment_delta_grouping(None, compartment_list)
    assert result == {"S": ["S"], "I": ["I"], "R": ["R"]}


# ---------------------------------------------------------------------------
# create_compartment_list
# ---------------------------------------------------------------------------

def test_create_compartment_list_sir():
    nodes = [{"id": "susceptible"}, {"id": "infected"}, {"id": "recovered"}]
    result = create_compartment_list(nodes)
    assert result == ["S", "I", "R"]


def test_create_compartment_list_seihdr_order():
    # Order should follow master_order: S, E, I, H, D, R
    nodes = [
        {"id": "recovered"},
        {"id": "susceptible"},
        {"id": "hospitalized"},
        {"id": "exposed"},
        {"id": "deceased"},
        {"id": "infected"},
    ]
    result = create_compartment_list(nodes)
    assert result == ["S", "E", "I", "H", "D", "R"]


def test_create_compartment_list_excludes_unknown_nodes():
    nodes = [
        {"id": "susceptible"},
        {"id": "infected"},
        {"id": "vector"},  # unknown node
    ]
    result = create_compartment_list(nodes)
    assert result == ["S", "I"]
    assert "vector" not in result


# ---------------------------------------------------------------------------
# create_transmission_dict
# ---------------------------------------------------------------------------

def test_create_transmission_dict_susceptible_to_infected():
    edges = [{"id": "susceptible->infected", "data": {"transmission_rate": 0.3}}]
    result = create_transmission_dict(edges)
    assert result == {"beta": 0.3}


def test_create_transmission_dict_multiple_edges():
    edges = [
        {"id": "susceptible->infected", "data": {"transmission_rate": 0.3}},
        {"id": "infected->recovered", "data": {"transmission_rate": 0.1}},
    ]
    result = create_transmission_dict(edges)
    assert result["beta"] == 0.3
    assert result["gamma"] == 0.1


def test_create_transmission_dict_skips_unrecognized_edges():
    edges = [
        {"id": "susceptible->infected", "data": {"transmission_rate": 0.3}},
        {"id": "vector->host", "data": {"transmission_rate": 0.5}},  # unrecognized
    ]
    result = create_transmission_dict(edges)
    assert "beta" in result
    assert len(result) == 1


def test_create_transmission_dict_defaults_missing_rate():
    edges = [{"id": "susceptible->infected", "data": {}}]
    result = create_transmission_dict(edges)
    assert result["beta"] == 0


# ---------------------------------------------------------------------------
# create_initial_population_matrix
# ---------------------------------------------------------------------------

def test_create_initial_population_matrix_basic():
    case_file = [{"population": 100_000, "infected_population": 1.0}]
    compartment_list = ["S", "I", "R"]
    result = create_initial_population_matrix(case_file, compartment_list)
    assert result.shape == (1, 3)
    # 1% infected → 1000 infected
    assert result[0, compartment_list.index("I")] == pytest.approx(1000.0)
    assert result[0, compartment_list.index("S")] == pytest.approx(99_000.0)
    assert result[0, compartment_list.index("R")] == pytest.approx(0.0)


def test_create_initial_population_matrix_multiple_zones():
    case_file = [
        {"population": 100_000, "infected_population": 1.0},
        {"population": 50_000, "infected_population": 2.0},
    ]
    compartment_list = ["S", "I", "R"]
    result = create_initial_population_matrix(case_file, compartment_list)
    assert result.shape == (2, 3)
    # Zone 1: 1% of 100k = 1000 infected
    assert result[0, 1] == pytest.approx(1000.0)
    # Zone 2: 2% of 50k = 1000 infected
    assert result[1, 1] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# compute_jax_compartment_deltas
# ---------------------------------------------------------------------------

def test_compute_jax_compartment_deltas_3d_uses_final_step():
    # shape (T, C, R) = (5, 3, 1)
    pm = np.zeros((5, 3, 1))
    pm[0] = [[100_000.0], [1000.0], [0.0]]
    pm[-1] = [[95_000.0], [500.0], [4500.0]]
    compartment_list = ["S", "I", "R"]
    deltas = compute_jax_compartment_deltas(pm, "MONKEYPOX", 1, compartment_list)
    assert deltas["S"] == pytest.approx(95_000.0)
    assert deltas["I"] == pytest.approx(500.0)
    assert deltas["R"] == pytest.approx(4500.0)


def test_compute_jax_compartment_deltas_4d_sums_age_axis():
    # shape (T, C, A, R) = (5, 3, 2, 1): 2 age groups
    pm = np.zeros((5, 3, 2, 1))
    pm[-1, 0, 0, 0] = 40_000.0   # S, age 0
    pm[-1, 0, 1, 0] = 55_000.0   # S, age 1
    pm[-1, 1, 0, 0] = 200.0      # I, age 0
    pm[-1, 1, 1, 0] = 300.0      # I, age 1
    pm[-1, 2, 0, 0] = 2000.0     # R, age 0
    pm[-1, 2, 1, 0] = 2500.0     # R, age 1
    compartment_list = ["S", "I", "R"]
    deltas = compute_jax_compartment_deltas(pm, "RESPIRATORY", 1, compartment_list)
    assert deltas["S"] == pytest.approx(95_000.0)
    assert deltas["I"] == pytest.approx(500.0)
    assert deltas["R"] == pytest.approx(4500.0)


def test_compute_jax_compartment_deltas_prefers_total_column():
    # shape (T, C, R) = (5, 4, 1) — includes I_total
    pm = np.zeros((5, 4, 1))
    pm[-1, 0, 0] = 95_000.0   # S
    pm[-1, 1, 0] = 500.0      # I (current)
    pm[-1, 2, 0] = 4500.0     # R
    pm[-1, 3, 0] = 5500.0     # I_total (cumulative > current I)
    compartment_list = ["S", "I", "R", "I_total"]
    deltas = compute_jax_compartment_deltas(pm, "MONKEYPOX", 1, compartment_list)
    # "I" group should use I_total column, not the raw I column
    assert deltas["I"] == pytest.approx(5500.0)


def test_compute_jax_compartment_deltas_multiple_regions():
    # shape (T, C, R) = (3, 2, 2) — 2 regions
    pm = np.zeros((3, 2, 2))
    pm[-1, 0, 0] = 90_000.0   # S, region 0
    pm[-1, 0, 1] = 45_000.0   # S, region 1
    pm[-1, 1, 0] = 10_000.0   # I, region 0
    pm[-1, 1, 1] = 5_000.0    # I, region 1
    compartment_list = ["S", "I"]
    deltas = compute_jax_compartment_deltas(pm, "MONKEYPOX", 2, compartment_list)
    assert deltas["S"] == pytest.approx(135_000.0)
    assert deltas["I"] == pytest.approx(15_000.0)


# ---------------------------------------------------------------------------
# compute_multi_run_compartment_deltas
# ---------------------------------------------------------------------------

def test_compute_multi_run_compartment_deltas_averages_simulations():
    # 2 simulation runs, each (T=3, C=2, R=1)
    run1 = np.zeros((3, 2, 1))
    run1[-1] = [[90_000.0], [10_000.0]]
    run2 = np.zeros((3, 2, 1))
    run2[-1] = [[80_000.0], [20_000.0]]
    population_matrix = [run1, run2]
    compartment_list = ["S", "I"]
    avg = compute_multi_run_compartment_deltas(population_matrix, "MONKEYPOX", 1, compartment_list)
    assert avg["S"] == pytest.approx(85_000.0)
    assert avg["I"] == pytest.approx(15_000.0)


# ---------------------------------------------------------------------------
# get_simulation_step_size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_timesteps,expected_step", [
    (365, 1),
    (730, 2),
    (100, 1),
    (1, 1),
    (366, 2),
])
def test_get_simulation_step_size(n_timesteps, expected_step):
    assert get_simulation_step_size(n_timesteps) == expected_step


# ---------------------------------------------------------------------------
# convert_dates
# ---------------------------------------------------------------------------

def test_convert_dates_date_to_iso():
    d = date(2025, 3, 15)
    assert convert_dates(d) == "2025-03-15"


def test_convert_dates_datetime_to_iso():
    dt = datetime(2025, 3, 15, 12, 0, 0)
    result = convert_dates(dt)
    assert result.startswith("2025-03-15")


def test_convert_dates_ndarray_to_list():
    arr = np.array([1.0, 2.0, 3.0])
    result = convert_dates(arr)
    assert result == [1.0, 2.0, 3.0]
    assert isinstance(result, list)


def test_convert_dates_nested_dict_recurses():
    data = {"date_field": date(2025, 1, 1), "value": 42}
    result = convert_dates(data)
    assert result["date_field"] == "2025-01-01"
    assert result["value"] == 42


def test_convert_dates_list_recurses():
    data = [date(2025, 1, 1), date(2025, 6, 1)]
    result = convert_dates(data)
    assert result == ["2025-01-01", "2025-06-01"]


def test_convert_dates_passthrough_scalar():
    assert convert_dates(42) == 42
    assert convert_dates("hello") == "hello"


# ---------------------------------------------------------------------------
# transform_interventions
# ---------------------------------------------------------------------------

def test_transform_interventions_strips_ordinals():
    data = {
        "social_isolation": {
            "start_date": "2025-03-01",
            "end_date": "2025-04-01",
            "start_date_ordinal": 738580,
            "end_date_ordinal": 738611,
            "adherence_min": 0.8,
        }
    }
    result = transform_interventions(data)
    assert len(result) == 1
    item = result[0]
    assert "start_date_ordinal" not in item
    assert "end_date_ordinal" not in item


def test_transform_interventions_preserves_other_fields():
    data = {
        "mask_wearing": {
            "start_date": "2025-03-01",
            "start_date_ordinal": 738580,
            "adherence_min": 0.5,
            "transmission_percentage": 0.3,
        }
    }
    result = transform_interventions(data)
    item = result[0]
    assert item["id"] == "mask_wearing"
    assert item["start_date"] == "2025-03-01"
    assert item["adherence_min"] == 0.5
    assert item["transmission_percentage"] == 0.3


def test_transform_interventions_multiple():
    data = {
        "int_a": {"start_date_ordinal": 100, "val": 1},
        "int_b": {"end_date_ordinal": 200, "val": 2},
    }
    result = transform_interventions(data)
    ids = {item["id"] for item in result}
    assert ids == {"int_a", "int_b"}


# ---------------------------------------------------------------------------
# extract_admin_units
# ---------------------------------------------------------------------------

def test_extract_admin_units():
    case_file = [{"name": "Zone A"}, {"name": "Zone B"}, {"name": "Zone C"}]
    result = extract_admin_units(case_file)
    assert result == ["Zone A", "Zone B", "Zone C"]


def test_extract_admin_units_single():
    case_file = [{"name": "New York"}]
    result = extract_admin_units(case_file)
    assert result == ["New York"]


# ---------------------------------------------------------------------------
# create_intervention_dict
# ---------------------------------------------------------------------------

def test_create_intervention_dict_date_strings_become_ordinals():
    interventions = [
        {
            "id": "social_isolation",
            "start_date": "2025-03-01",
            "end_date": "2025-04-01",
            "adherence_min": 80.0,
            "transmission_percentage": 50.0,
            "start_threshold": None,
            "end_threshold": None,
        }
    ]
    result = create_intervention_dict(interventions, "2025-01-01")
    entry = result["social_isolation"]
    expected_start = date(2025, 3, 1).toordinal()
    expected_end = date(2025, 4, 1).toordinal()
    assert entry["start_date_ordinal"] == expected_start
    assert entry["end_date_ordinal"] == expected_end


def test_create_intervention_dict_missing_dates_remain_none():
    # start_threshold is set, so dates should not be auto-filled
    interventions = [
        {
            "id": "social_isolation",
            "start_threshold": 5.0,
            "end_threshold": 1.0,
            "adherence_min": 80.0,
            "transmission_percentage": 50.0,
        }
    ]
    result = create_intervention_dict(interventions, "2025-01-01")
    entry = result["social_isolation"]
    assert entry["start_date_ordinal"] is None
    assert entry["end_date_ordinal"] is None


def test_create_intervention_dict_normalizes_percentages():
    interventions = [
        {
            "id": "mask_wearing",
            "start_date": "2025-06-01",
            "adherence_min": 80.0,          # should be divided by 100
            "transmission_percentage": 50.0, # should be divided by 100
            "start_threshold": None,
            "end_threshold": None,
        }
    ]
    result = create_intervention_dict(interventions, "2025-01-01")
    entry = result["mask_wearing"]
    assert entry["adherence_min"] == pytest.approx(0.80)
    assert entry["transmission_percentage"] == pytest.approx(0.50)
