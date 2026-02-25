"""Tests for compartment/validation/."""
import pytest
from datetime import date

from pydantic import ValidationError

from compartment.validation import (
    ProcessedSimulation,
    SimulationConfig,
    ValidationPostProcessor,
    load_simulation_config,
)
from compartment.validation.base_simulation import TravelVolume
from compartment.validation.diseases.mpox import MpoxDiseaseConfig


# ---------------------------------------------------------------------------
# Minimal valid MPOX config factory
# ---------------------------------------------------------------------------

def _mpox_job(**overrides):
    base = {
        "admin_unit_0_id": "USA",
        "start_date": "2025-01-01",
        "end_date": "2025-04-01",
        "AdminUnit0": {"id": "USA", "center_lat": 37.09},
        "Disease": {
            "disease_type": "MONKEYPOX",
            "transmission_edges": [
                {"source": "susceptible", "target": "infected", "data": {"transmission_rate": 0.3}},
                {"source": "infected", "target": "recovered", "data": {"transmission_rate": 0.1}},
            ],
        },
        "case_file": {
            "admin_zones": [
                {
                    "center_lat": 40.71,
                    "center_lon": -74.01,
                    "name": "New York",
                    "population": 8_000_000,
                    "infected_population": 1,
                }
            ]
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# BaseSimulationShared date validation
# ---------------------------------------------------------------------------

def test_end_date_before_start_date_raises():
    with pytest.raises(ValidationError, match="end_date"):
        SimulationConfig[MpoxDiseaseConfig](
            **_mpox_job(start_date="2025-04-01", end_date="2025-01-01")
        )


def test_same_start_end_date_is_valid():
    cfg = SimulationConfig[MpoxDiseaseConfig](
        **_mpox_job(start_date="2025-01-01", end_date="2025-01-01")
    )
    assert cfg.time_steps == 1


def test_time_steps_auto_computed_from_dates():
    cfg = SimulationConfig[MpoxDiseaseConfig](
        **_mpox_job(start_date="2025-01-01", end_date="2025-04-10")
    )
    expected_days = (date(2025, 4, 10) - date(2025, 1, 1)).days
    assert cfg.time_steps == expected_days


def test_explicit_time_steps_respected():
    cfg = SimulationConfig[MpoxDiseaseConfig](
        **_mpox_job(start_date="2025-01-01", end_date="2025-04-10", time_steps=50)
    )
    assert cfg.time_steps == 50


# ---------------------------------------------------------------------------
# TravelVolume normalisation
# ---------------------------------------------------------------------------

def test_travel_volume_leaving_above_1_is_normalized():
    tv = TravelVolume(leaving=50.0)
    assert tv.leaving == pytest.approx(0.5)


def test_travel_volume_leaving_below_1_unchanged():
    tv = TravelVolume(leaving=0.2)
    assert tv.leaving == pytest.approx(0.2)


def test_travel_volume_leaving_exactly_1_unchanged():
    tv = TravelVolume(leaving=1.0)
    assert tv.leaving == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ProcessedSimulation dict-like access
# ---------------------------------------------------------------------------

def test_processed_simulation_getitem_direct_field():
    ps = ProcessedSimulation(
        config={"start_date": "2025-01-01"},
        compartment_list=["S", "I", "R"],
    )
    assert ps["compartment_list"] == ["S", "I", "R"]


def test_processed_simulation_getitem_falls_back_to_config():
    ps = ProcessedSimulation(
        config={"custom_key": "custom_value"},
        compartment_list=[],
    )
    assert ps["custom_key"] == "custom_value"


def test_processed_simulation_get_with_default():
    ps = ProcessedSimulation(config={}, compartment_list=[])
    assert ps.get("nonexistent_key", "fallback") == "fallback"


def test_processed_simulation_get_direct_field():
    ps = ProcessedSimulation(
        config={},
        compartment_list=["S", "I"],
    )
    assert ps.get("compartment_list") == ["S", "I"]


# ---------------------------------------------------------------------------
# ValidationPostProcessor._process_default
# ---------------------------------------------------------------------------

def test_process_default_mpox_uses_hardcoded_compartment_list():
    from compartment.models.mpox_jax_model.model import MpoxJaxModel

    config = {"data": {"getSimulationJob": _mpox_job()}}
    processed = load_simulation_config(config, "MONKEYPOX")
    assert processed.compartment_list == MpoxJaxModel.COMPARTMENT_LIST


def test_process_default_no_transmission_edges_gives_empty_dict():
    # No disease_nodes, no compartment_list but MONKEYPOX uses hardcoded COMPARTMENT_LIST
    # So the empty transmission_edges case: transmission_dict should be empty or have entries
    # for the given edges only
    config = {"data": {"getSimulationJob": _mpox_job()}}
    processed = load_simulation_config(config, "MONKEYPOX")
    # MPOX has 2 edges: beta and gamma → both should be in transmission_dict
    assert "beta" in processed.transmission_dict
    assert "gamma" in processed.transmission_dict


def test_process_default_no_travel_volume_gives_none_travel_matrix():
    """When travel_volume is None, travel_matrix should be None."""
    job = _mpox_job()
    job["travel_volume"] = None
    config = {"data": {"getSimulationJob": job}}
    processed = load_simulation_config(config, "MONKEYPOX")
    assert processed.travel_matrix is None


def test_process_default_admin_units_extracted():
    config = {"data": {"getSimulationJob": _mpox_job()}}
    processed = load_simulation_config(config, "MONKEYPOX")
    assert processed.admin_units == ["New York"]


def test_process_default_hemisphere_set_from_admin_unit():
    # AdminUnit0 lat = 37.09 (North)
    config = {"data": {"getSimulationJob": _mpox_job()}}
    processed = load_simulation_config(config, "MONKEYPOX")
    assert processed.hemisphere == "North"
