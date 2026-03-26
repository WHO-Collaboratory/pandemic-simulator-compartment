import logging
import argparse
from compartment.driver import drive_simulation
from compartment.models.test_covid_sir_stochastic.model import CovidSirStochasticModel

logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def lambda_handler(event, context):
    drive_simulation(
        model_class=CovidSirStochasticModel,
        args={"mode": "cloud",
              "simulation_job_id": event["simulation_job_id"]},
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run COVID stochastic SIR simulation"
    )
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--config_file", help="Path to the JSON config file")
    parser.add_argument("--output_file", nargs="?", default=None)
    parser.add_argument("--simulation_job_id", nargs="?", default=None)
    args = parser.parse_args()

    drive_simulation(model_class=CovidSirStochasticModel, args=vars(args))
