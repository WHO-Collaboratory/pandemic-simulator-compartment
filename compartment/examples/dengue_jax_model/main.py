import logging
import argparse
import json
import os
import time
import tracemalloc
from datetime import datetime
from compartment.batch_helpers.simulation_helpers import get_simulation_params
from compartment.run_simulation import run_simulation

logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run pandemic simulation with specified config file')
    parser.add_argument('--config_file', help='Path to the JSON config file for the simulation')
    parser.add_argument('--output_file', nargs='?', help='Path to the output JSON file (optional, defaults to input file with results-timestamp suffix)')
    args = parser.parse_args()

    # Start tracking memory usage
    tracemalloc.start()
    start_time = time.time()
    logger.info("Memory tracking started...")

    if args.config_file:
        if args.output_file:
            output_file = args.output_file
        else:
            # Generate default output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            config_basename = os.path.splitext(os.path.basename(args.config_file))[0]
            output_file = f"{config_basename}-results-{timestamp}.json"
        run_metadata = run_simulation(config_path=args.config_file, output_path=output_file)
    else:
        #simulation_params = get_simulation_params()
        simulation_params = {
            "SIMULATION_JOB_ID": "4c8efee7-7310-452e-ba76-1788e69f46c2",#"fb1eab7d-73cb-4c1e-a0c6-16b53600fdb7",#covid:"9814c00a-98f7-4726-a8a9-fa30cb9572de",#"9c9d1052-cfcb-4edc-8b54-3177bf678c97",#"dc60bf28-b4b5-40c1-81bd-09f304b67190",
            "GRAPHQL_ENDPOINT": "https://ftvz6ss74ncwxnn2a4a6dia664.appsync-api.us-east-1.amazonaws.com/graphql",
            "GRAPHQL_APIKEY": "da2-4pf5tbnxgfddlel57zztt6id4y",
            "ENVIRONMENT": "dev"
        }
        run_metadata = run_simulation(simulation_params=simulation_params)

    logger.info(f"run_metadata: {run_metadata}")
    # Capture and log memory usage
    current, peak = tracemalloc.get_traced_memory() 
    tracemalloc.stop()
    end_time = time.time()

    elapsed_time = round(end_time - start_time, 2)

    logger.info(f"Memory tracking stopped. Peak memory usage: {peak / (1024 * 1024):.2f} MB")
    logger.info(f"Elapsed time: {elapsed_time} seconds")