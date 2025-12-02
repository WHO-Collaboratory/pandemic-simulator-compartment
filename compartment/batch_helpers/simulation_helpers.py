"""Helpers which reduce duplication between Batch programs."""
from os import environ

def get_simulation_params()->dict:

    simulation_params = {
        "SIMULATION_JOB_ID": environ[("SIMULATION_JOB_ID")],
        "GRAPHQL_APIKEY": environ[("GRAPHQL_APIKEY")],
        "GRAPHQL_ENDPOINT": environ[("GRAPHQL_ENDPOINT")],
        "ENVIRONMENT": environ.get("ENVIRONMENT", None)
    }
    return simulation_params
