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
    parser.add_argument('--mode', choices=['local', 'cloud'], default='local')
    parser.add_argument('--config_file', help='Path to the JSON config file for the simulation')
    parser.add_argument('--output_file', nargs='?', default=None, help='Pass a relative path from your execution directory. Use * to generate a default output filename with timestamp.')
    parser.add_argument('--simulation_job_id', nargs='?', default=None, help='Existing simulation job id in a graphql backend.')
    args = parser.parse_args()

    # Start tracking memory usage
    tracemalloc.start()
    start_time = time.time()
    logger.info("Memory tracking started...")

    if args.mode == 'local':
        if args.config_file is None:
            raise ValueError("config_file is required for local mode")
        if args.output_file:
            output_file = args.output_file
        elif args.output_file == "*":
            # Generate default output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            config_basename = os.path.splitext(os.path.basename(args.config_file))[0]
            output_file = f"{config_basename}-results-{timestamp}.json"
        elif args.output_file is None:
            output_file = None
        run_metadata = run_simulation(config_path=args.config_file, output_path=output_file)
    elif args.mode == 'cloud':
        simulation_job_id = args.simulation_job_id
        if simulation_job_id is None:
            raise ValueError("simulation_job_id is required for cloud mode")
        simulation_params = get_simulation_params(simulation_job_id=simulation_job_id)
        run_metadata = run_simulation(simulation_params=simulation_params)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    logger.info(f"run_metadata: {run_metadata}")
    # Capture and log memory usage
    current, peak = tracemalloc.get_traced_memory() 
    tracemalloc.stop()
    end_time = time.time()

    elapsed_time = round(end_time - start_time, 2)

    logger.info(f"Memory tracking stopped. Peak memory usage: {peak / (1024 * 1024):.2f} MB")
    logger.info(f"Elapsed time: {elapsed_time} seconds")