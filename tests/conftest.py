"""Shared fixtures for the test suite."""
import numpy as np
import pytest
from datetime import date


@pytest.fixture
def minimal_admin_zone():
    return {
        "id": "zone-1",
        "name": "Zone 1",
        "center_lat": 40.0,
        "center_lon": -74.0,
        "population": 100_000,
        "infected_population": 1.0,
        "seroprevalence": 10.0,
        "temp_min": 15.0,
        "temp_max": 30.0,
        "temp_mean": 25.0,
    }


@pytest.fixture
def two_admin_zones():
    return [
        {
            "id": "zone-1",
            "name": "Zone 1",
            "center_lat": 40.0,
            "center_lon": -74.0,
            "population": 100_000,
            "infected_population": 1.0,
            "seroprevalence": 10.0,
            "temp_min": 15.0,
            "temp_max": 30.0,
            "temp_mean": 25.0,
        },
        {
            "id": "zone-2",
            "name": "Zone 2",
            "center_lat": 34.0,
            "center_lon": -118.0,
            "population": 50_000,
            "infected_population": 2.0,
            "seroprevalence": 5.0,
            "temp_min": 10.0,
            "temp_max": 35.0,
            "temp_mean": 22.0,
        },
    ]


@pytest.fixture
def mpox_config_dict():
    from compartment.models.mpox_jax_model.model import MpoxJaxModel

    admin_zones = [
        {
            "id": "zone-1",
            "name": "Zone 1",
            "center_lat": 40.0,
            "center_lon": -74.0,
            "population": 100_000,
            "infected_population": 1.0,
        }
    ]
    compartment_list = MpoxJaxModel.COMPARTMENT_LIST
    initial_population = MpoxJaxModel.get_initial_population(admin_zones, compartment_list)
    return {
        "initial_population": initial_population,
        "transmission_dict": {"beta": 0.3, "gamma": 0.1},
        "start_date": date(2025, 1, 1),
        "time_steps": 90,
        "admin_units": ["Zone 1"],
        "compartment_list": compartment_list,
        "case_file": {"admin_zones": admin_zones, "demographics": {"age_all": 100.0}},
        "intervention_dict": {},
    }


@pytest.fixture
def mpox_model(mpox_config_dict):
    from compartment.models.mpox_jax_model.model import MpoxJaxModel

    return MpoxJaxModel(mpox_config_dict)


@pytest.fixture
def covid_config_dict():
    from compartment.models.covid_jax_model.model import CovidJaxModel

    admin_zones = [
        {
            "id": "zone-1",
            "name": "Zone 1",
            "center_lat": 40.0,
            "center_lon": -74.0,
            "population": 100_000,
            "infected_population": 1.0,
        }
    ]
    compartment_list = ["S", "I", "R"]
    initial_population = CovidJaxModel.get_initial_population(admin_zones, compartment_list)
    # Demographics must have exactly 3 groups to match the hardcoded 3x3 interaction matrix
    demographics = {"age_0_17": 25.0, "age_18_55": 50.0, "age_56_plus": 25.0}
    return {
        "initial_population": initial_population,
        "travel_matrix": np.eye(1),
        "compartment_list": compartment_list,
        "travel_volume": {"leaving": 0.1},
        "transmission_dict": {
            "beta": 0.25,
            "gamma": 0.14,
            "theta": 0.2,
            "zeta": 0.05,
            "delta": 0.01,
            "eta": 0.1,
            "epsilon": 0.02,
        },
        "start_date": date(2025, 1, 1),
        "time_steps": 60,
        "admin_units": ["Zone 1"],
        "case_file": {"admin_zones": admin_zones, "demographics": demographics},
        "intervention_dict": {},
    }


@pytest.fixture
def covid_model(covid_config_dict):
    from compartment.models.covid_jax_model.model import CovidJaxModel

    return CovidJaxModel(covid_config_dict)


@pytest.fixture
def dengue_config_dict():
    from compartment.models.dengue_jax_model.model import DengueJaxModel

    admin_zones = [
        {
            "id": "zone-1",
            "name": "Zone 1",
            "center_lat": -18.0,
            "center_lon": 46.0,
            "population": 100_000,
            "infected_population": 3.0,
            "seroprevalence": 30.0,
        }
    ]
    compartment_list = DengueJaxModel.COMPARTMENT_LIST
    initial_population = DengueJaxModel.get_initial_population(admin_zones, compartment_list)
    return {
        "initial_population": initial_population,
        "start_date": date(2025, 1, 1),
        "time_steps": 365,
        "travel_matrix": np.eye(1),
        "admin_units": ["Zone 1"],
        "case_file": {"admin_zones": admin_zones, "demographics": {"age_all": 100.0}},
        "travel_volume": {"leaving": 0.1},
        "hemisphere": "South",
        "temperature": {"temp_min": 15.0, "temp_max": 30.0, "temp_mean": 25.0},
        "intervention_dict": {},
        "Disease": {"immunity_period": 240},
    }


@pytest.fixture
def dengue_model(dengue_config_dict):
    from compartment.models.dengue_jax_model.model import DengueJaxModel

    return DengueJaxModel(dengue_config_dict)
