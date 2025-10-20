"""Helpers which reduce duplication between Batch programs."""
import logging
from os import environ

def setup_logging():
    """Sets up logging for AWS Batch (Console Only)"""
    logging.basicConfig(
        format="[%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()]
    )
def get_simulation_params()->dict:

    simulation_params = {
        "SIMULATION_JOB_ID": environ[("SIMULATION_JOB_ID")],
        "GRAPHQL_APIKEY": environ[("GRAPHQL_APIKEY")],
        "GRAPHQL_ENDPOINT": environ[("GRAPHQL_ENDPOINT")],
        "ENVIRONMENT": environ.get("ENVIRONMENT", None)
    }
    return simulation_params
