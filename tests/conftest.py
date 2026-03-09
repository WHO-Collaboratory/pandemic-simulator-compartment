"""Pytest configuration and shared fixtures for all tests"""
import json
import pytest
from pathlib import Path
from datetime import datetime
import numpy as np


@pytest.fixture
def base_config():
    """Base configuration dictionary for testing"""
    return {
        "start_date": datetime(2024, 1, 1),
        "time_steps": 100,
        "admin_units": ["Zone1", "Zone2"],
        "case_file": {
            "admin_zones": [
                {
                    "admin_id": "Zone1",
                    "population": 10000,
                    "infected_population": 1.0
                },
                {
                    "admin_id": "Zone2",
                    "population": 20000,
                    "infected_population": 0.5
                }
            ],
            "demographics": {
                "0-17": 0.25,
                "18-64": 0.65,
                "65+": 0.10
            }
        },
        "travel_volume": {
            "leaving": 0.01
        },
        "travel_matrix": [[1.0, 0.1], [0.1, 1.0]],
        "interventions": [],
        "intervention_dict": {
            "lock_down": [],
            "mask_wearing": [],
            "social_isolation": []
        }
    }


@pytest.fixture
def abc_config(base_config):
    """Configuration for ABC model"""
    config = base_config.copy()
    config.update({
        "compartment_list": ["A", "B", "C"],
        "initial_population": [[9900, 100, 0], [19900, 100, 0]],
        "transmission_dict": {
            "alpha": 0.3,
            "beta": 0.1
        }
    })
    return config


@pytest.fixture
def covid_config(base_config):
    """Configuration for COVID model"""
    config = base_config.copy()
    config.update({
        "compartment_list": ["S", "E", "I", "H", "R", "D"],
        "initial_population": [[9900, 0, 100, 0, 0, 0], [19900, 0, 100, 0, 0, 0]],
        "transmission_dict": {
            "beta": 0.5,
            "gamma": 0.1,
            "theta": 0.2,
            "zeta": 0.05,
            "delta": 0.01,
            "eta": 0.15,
            "epsilon": 0.02
        }
    })
    return config


@pytest.fixture
def mpox_config(base_config):
    """Configuration for Mpox model"""
    config = base_config.copy()
    config.update({
        "compartment_list": ["S", "E", "I", "R"],
        "initial_population": [[9900, 0, 100, 0], [19900, 0, 100, 0]],
        "transmission_dict": {
            "beta": 0.3,
            "sigma": 0.1,
            "gamma": 0.05
        }
    })
    return config


@pytest.fixture
def dengue_config(base_config):
    """Configuration for Dengue model"""
    config = base_config.copy()
    config.update({
        "compartment_list": ["S_h", "I_h", "R_h", "S_v", "I_v"],
        "initial_population": [[9900, 100, 0, 50000, 1000], [19900, 100, 0, 100000, 2000]],
        "transmission_dict": {
            "beta_h": 0.3,
            "beta_v": 0.4,
            "gamma_h": 0.1,
            "mu_v": 0.05
        }
    })
    return config


@pytest.fixture
def example_config_path():
    """Get path to example config files"""
    def _get_path(model_name):
        return Path(__file__).parent.parent / "compartment" / "models" / model_name / "example-config.json"
    return _get_path


@pytest.fixture
def load_example_config(example_config_path):
    """Load an example config file"""
    def _load(model_name):
        config_path = example_config_path(model_name)
        if not config_path.exists():
            pytest.skip(f"Example config not found: {config_path}")
        with open(config_path) as f:
            return json.load(f)
    return _load
