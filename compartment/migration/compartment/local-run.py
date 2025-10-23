import logging
# Remove unnecessary jax INFO logging
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
from batch_simulation_manager import BatchSimulationManager
from concurrent.futures import ProcessPoolExecutor
from covid_jax_model import CovidJaxModel
from dengue_jax_model import DengueJaxModel
from helpers import clean_payload, generate_LHS_samples, build_uncertainty_params
from math import ceil
from simulation_manager import SimulationManager
from simulation_postprocessor import SimulationPostProcessor
import argparse
import boto3
import copy
import json
import multiprocessing
import numpy as np
import os
import time
import tracemalloc


#from temp_uncertainty_graphs import plot_simulation_results, plot_compartments_all_regions, plot_all_regions_and_compartments
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})

# Initialize logging
logger = logging.getLogger(__name__)

def simulate_and_postprocess(model): 
  all_output = SimulationManager(model).run_simulation()
  processor = SimulationPostProcessor(model, all_output)
  return processor.process()

def batch_simulate_and_postprocess(model, n_sims, param_list, ci, num_workers):
  batcher = BatchSimulationManager(max_workers=num_workers)
  all_output = batcher.run_batch(model, n_sims, param_list)
  # Using the first model as a template for metadata
  model.prepare_initial_state()
  processor = SimulationPostProcessor(model, all_output)
  return processor.process(ci=ci)

def main(simulation_params, config_file_path):

    logger.info("Starting the simulation...")
    simulation_job_id = simulation_params.get('SIMULATION_JOB_ID')
    logger.info(f"simulation_job_id: {simulation_job_id}")
    logger.info(f"graphql_endpoint: {simulation_params.get('GRAPHQL_ENDPOINT')}")

    # Grab simulation job and clean payload for model
    #config = get_simulation_job(simulation_params, GRAPHQL_QUERY)
    #print(f"\n\n{config}\n\n")
    #time.sleep(100000000)
    config = json.load(open(config_file_path))
    #print(f"\n\n{config}\n\n")
    #time.sleep(100000000)
    cleaned_config = clean_payload(config)
    #print(f"\n\n{cleaned_config}\n\n")
    #time.sleep(100000000)

    disease_type = config['data']['getSimulationJob']['Disease']['disease_type']

    run_metadata = {
        "simulation_job_id": simulation_job_id,
        "admin_zone_count": len(cleaned_config.get('admin_units')),
        "simulation_time_steps": config['data']['getSimulationJob']['time_steps'],
        "owner": config['data']['getSimulationJob']['owner'],
        "disease_type": disease_type
    }

    logger.info(f"owner: {config['data']['getSimulationJob']['owner']}")
    logger.info(f"number of admin units: {len(cleaned_config.get('admin_units'))}")
    logger.info(f"compartment_list: {cleaned_config.get('compartment_list')}")
    logger.info(f"disease_type: {disease_type}")
    logger.info(f"transmission_dict: {cleaned_config.get('transmission_dict')}")
    logger.info(f"intervention_dict: {cleaned_config.get('intervention_dict')}")

    # Map of disease_type to model_class
    DISEASE_MODEL_MAP = {
      "VECTOR_BORNE": DengueJaxModel,
      "RESPIRATORY": CovidJaxModel    
    }

    try:
        model_class = DISEASE_MODEL_MAP[disease_type]
    except KeyError:
        logger.error(f"Invalid disease type: {disease_type}")
        raise ValueError(f"Invalid disease type: {disease_type}")
    
    run_mode = config['data']['getSimulationJob']['run_mode']
    logger.info(f"run_mode: {run_mode}")

    # Create business as usual model
    model_with = model_class(config, cleaned_config)
    #print(f"cleaned_config: {cleaned_config}")
    #time.sleep(100000000)
    # Create a deepcopy for the interventionless model
    model_without = copy.deepcopy(model_with)
    model_without.intervention_dict = {}

    # Split cpu in half for top level workers
    # Then split the remaining cpu in half for low level workers
    top_level_workers = 2
    low_level_workers = ceil(os.cpu_count() / top_level_workers) - 1
    logger.info(f"top_level_workers: {top_level_workers}")
    logger.info(f"low_level_workers: {low_level_workers}")
    
    if run_mode == "DETERMINISTIC":
        with ProcessPoolExecutor(max_workers=top_level_workers) as executor:
            future_with = executor.submit(simulate_and_postprocess, model_with)
            future_without = executor.submit(simulate_and_postprocess, model_without)
            results_with = future_with.result()
            results_without = future_without.result()
    else:
        n_sims = 30
        ci = 0.95
        uncertainty_params = build_uncertainty_params(
            config['data']['getSimulationJob']['Disease']['transmission_edges'],
            config['data']['getSimulationJob']['interventions']
          )
        
        logger.info(f"Number of simulations: {n_sims}")
        logger.info(f"Confidence interval: {ci}")
        logger.info(f"Uncertainty parameters: {uncertainty_params}")

        param_list = generate_LHS_samples(n_sims, uncertainty_params)
        # remove intervention params from param_list
        interventionless_param_list = [
            {k: v for k, v in d.items() if not k.startswith("intervention.")}
            for d in param_list
        ]

        # Run both batches in parallel
        with ProcessPoolExecutor(max_workers=top_level_workers) as executor:
            future_with = executor.submit(batch_simulate_and_postprocess, model_with, n_sims, param_list, ci, low_level_workers)
            future_without = executor.submit(batch_simulate_and_postprocess, model_without, n_sims, interventionless_param_list, ci, low_level_workers)
            results_with = future_with.result()
            results_without = future_without.result()

    results_with["control_run"] = False
    results_without["control_run"] = True
    results = [results_with, results_without]
    s3_results = copy.deepcopy(results)

    return run_metadata

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run pandemic simulation with specified config file')
    parser.add_argument('config_file', help='Path to the JSON config file for the simulation')
    args = parser.parse_args()

    # Start tracking memory usage
    tracemalloc.start()
    start_time = time.time()
    logger.info("Memory tracking started...")


    # Run model
    run_metadata = main(simulation_params, args.config_file)
    #result = main(simulation_params)
    #logger.info(f"response from gql: {result}")

    run_metadata_query = """
    mutation CreateSimulationRunMetadata($input: CreateSimulationRunMetadataInput!) {
            createSimulationRunMetadata(input: $input) {
                id
                simulation_job_id
            }
        }
    """

    # Capture and log memory usage
    current, peak = tracemalloc.get_traced_memory() 
    tracemalloc.stop()
    end_time = time.time()

    elapsed_time = round(end_time - start_time, 2)
    #run_metadata["run_time"] = elapsed_time

    # DONT RUN THIS LOCALLY
    # Write run metadata to gql 
    #run_metadata_response = gql_write_helper(simulation_params, run_metadata, run_metadata_query)
    #logger.info(f"run_metadata_response: {run_metadata_response}")

    logger.info(f"Memory tracking stopped. Peak memory usage: {peak / (1024 * 1024):.2f} MB")
    logger.info(f"Elapsed time: {elapsed_time} seconds")