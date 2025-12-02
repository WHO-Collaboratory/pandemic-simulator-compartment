import logging
import argparse
import json
import os
import time
import tracemalloc
from datetime import datetime

from compartment.run_simulation import run_simulation
from compartment.examples.covid_jax_model.model import CovidJaxModel

logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run pandemic simulation with specified config file')
    parser.add_argument('config_file', help='Path to the JSON config file for the simulation')
    parser.add_argument('output_file', nargs='?', help='Path to the output JSON file (optional, defaults to input file with results-timestamp suffix)')
    args = parser.parse_args()

    # Start tracking memory usage
    tracemalloc.start()
    start_time = time.time()
    logger.info("Memory tracking started...")


    # Determine output file path
    if args.output_file:
        output_file = args.output_file
    else:
        # Generate default output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        config_basename = os.path.splitext(os.path.basename(args.config_file))[0]
        output_file = f"{config_basename}-results-{timestamp}.json"

    # Run model
    output_data = run_simulation(CovidJaxModel, args.config_file)

    # Write results to JSON file
    logger.info(f"Writing results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Capture and log memory usage
    current, peak = tracemalloc.get_traced_memory() 
    tracemalloc.stop()
    end_time = time.time()

    elapsed_time = round(end_time - start_time, 2)

    logger.info(f"Memory tracking stopped. Peak memory usage: {peak / (1024 * 1024):.2f} MB")
    logger.info(f"Elapsed time: {elapsed_time} seconds")
    logger.info(f"Results saved to: {output_file}")