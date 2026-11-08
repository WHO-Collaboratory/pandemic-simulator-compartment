import logging
import argparse
from compartment.driver import drive_simulation
from compartment.models.cholera_svirb_ocv_maity_model.model import CholeraSvirbOcvMaityModel

logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def lambda_handler(event, context):
    """Run a simulation job on the platform (AWS Lambda entry point).

    Args:
        event (dict): Lambda event; must contain ``simulation_job_id``.
        context (object): Lambda context (unused).

    Returns:
        None: Results are written by the driver.
    """
    drive_simulation(
        model_class=CholeraSvirbOcvMaityModel,
        args={"mode": "cloud", "simulation_job_id": event["simulation_job_id"]},
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Cholera – SVIRB, reactive OCV (extends Maity et al. 2023) simulation with specified config file"
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
    drive_simulation(model_class=CholeraSvirbOcvMaityModel, args=vars(args))
