import logging
import argparse
from compartment.driver import drive_simulation
from compartment.models.example_parameter_uncertainty_custom_model.model import ExampleParameterUncertaintyCustomModel

logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def lambda_handler(event, context):
    drive_simulation(
        model_class=ExampleParameterUncertaintyCustomModel,
        args={"mode": "cloud", "simulation_job_id": event["simulation_job_id"]},
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Example Disease with Parameter Uncertainty and Custom Equation simulation with specified config file"
    )
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--config_file", help="Path to the JSON config file for the simulation")
    parser.add_argument(
        "--output_file",
        nargs="?",
        default=None,
        help="Output file path. Use * to generate a default filename with timestamp.",
    )
    parser.add_argument(
        "--simulation_job_id",
        nargs="?",
        default=None,
        help="Existing simulation job id in a graphql backend.",
    )
    args = parser.parse_args()
    drive_simulation(model_class=ExampleParameterUncertaintyCustomModel, args=vars(args))
