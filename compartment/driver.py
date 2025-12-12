"""Reusable driver code for managing and running simulations"""
import logging
import os
import time
import tracemalloc
from datetime import datetime
from compartment.batch_helpers.simulation_helpers import get_simulation_params
from compartment.helpers import setup_logging
from compartment.model import Model
from compartment.run_simulation import run_simulation

logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

def drive_simulation(model_class:Model, args:dict):
    # Setup logging first to ensure CloudWatch logs work properly
    setup_logging()
    logger = logging.getLogger()
    tracemalloc.start()
    start_time = time.time()
    logger.info("Memory tracking started...")
    if args["mode"] == 'local':
        if args["config_file"] is None:
            raise ValueError("config_file is required for local mode")
        if args["output_file"]:
            output_file = args["output_file"]
        elif args["output_file"] == "*":
            # Generate default output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            config_basename = os.path.splitext(os.path.basename(args["config_file"]))[0]
            output_file = f"{config_basename}-results-{timestamp}.json"
        elif args["output_file"] is None:
            output_file = None
        run_metadata = run_simulation(model_class=model_class, config_path=args["config_file"], output_path=output_file)
    elif args["mode"] == 'cloud':
        simulation_job_id = args["simulation_job_id"]
        if simulation_job_id is None:
            raise ValueError("simulation_job_id is required for cloud mode")
        simulation_params = get_simulation_params(simulation_job_id=simulation_job_id)
        run_metadata = run_simulation(model_class=model_class, mode='cloud', simulation_params=simulation_params)
    else:
        raise ValueError(f"Invalid mode: {args["mode"]}")
    
    logger.info(f"run_metadata: {run_metadata}")
    # Capture and log memory usage
    current, peak = tracemalloc.get_traced_memory() 
    tracemalloc.stop()
    end_time = time.time()

    elapsed_time = round(end_time - start_time, 2)

    logger.info(f"Memory tracking stopped. Peak memory usage: {peak / (1024 * 1024):.2f} MB")
    logger.info(f"Elapsed time: {elapsed_time} seconds")
    
    return None
