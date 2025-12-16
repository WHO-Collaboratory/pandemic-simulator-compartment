"""Helpers which reduce duplication between Batch programs."""
from os import environ

def get_simulation_params(simulation_job_id:str)->dict:

    simulation_params = {
        "SIMULATION_JOB_ID": simulation_job_id,
        "GRAPHQL_APIKEY": environ[("GRAPHQL_APIKEY")],
        "GRAPHQL_ENDPOINT": environ[("GRAPHQL_ENDPOINT")],
        "ENVIRONMENT": environ.get("ENVIRONMENT", None)
    }
    return simulation_params
