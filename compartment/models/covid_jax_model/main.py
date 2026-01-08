import logging
import argparse
from compartment.driver import drive_simulation
from compartment.models.covid_jax_model.model import CovidJaxModel
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logger = logging.getLogger()

def lambda_handler(event, context):
    drive_simulation(
        model_class=CovidJaxModel,
        args={
            "mode": "cloud",
            "simulation_job_id": event["simulation_job_id"]
        }
    )
    return None

def run_local():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run pandemic simulation with specified config file')
    parser.add_argument('--mode', choices=['local', 'cloud'], default='local')
    parser.add_argument('--config_file', help='Path to the JSON config file for the simulation')
    parser.add_argument('--output_file', nargs='?', default=None, help='Pass a relative path from your execution directory. Use * to generate a default output filename with timestamp.')
    parser.add_argument('--simulation_job_id', nargs='?', default=None, help='Existing simulation job id in a graphql backend.')
    args = parser.parse_args()

    return drive_simulation(model_class=CovidJaxModel, args=vars(args))
