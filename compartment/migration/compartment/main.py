import logging
# Remove unnecessary jax INFO logging
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

from batch_helpers.gql import get_simulation_job, write_to_gql
from batch_helpers.graphql_queries import GRAPHQL_QUERY
from batch_helpers.s3 import write_to_s3
from batch_helpers.simulation_helpers import get_simulation_params, setup_logging
from batch_simulation_manager import BatchSimulationManager
from concurrent.futures import ProcessPoolExecutor
from covid_jax_model import CovidJaxModel
from dengue_jax_model import DengueJaxModel
from helpers import clean_payload, load_config, generate_LHS_samples, get_simulation_step_size, format_uncertainty_output, build_uncertainty_params
from math import ceil
from simulation_manager import SimulationManager
from simulation_postprocessor import SimulationPostProcessor
import boto3
import copy
import json
import multiprocessing
import numpy as np
import os
import time
import tracemalloc
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})

# Initialize logging
setup_logging()
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

def main(simulation_params):
    logger.info("Starting the simulation...")
    simulation_job_id = simulation_params.get('SIMULATION_JOB_ID')
    logger.info(f"simulation_job_id: {simulation_job_id}")
    logger.info(f"graphql_endpoint: {simulation_params.get('GRAPHQL_ENDPOINT')}")

    # Grab simulation job and clean payload for model
    config = get_simulation_job(simulation_params, GRAPHQL_QUERY)
    cleaned_config = clean_payload(config)

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
    
    # Write to GQL and S3
    s3_client = boto3.client('s3', region_name='us-east-1')
    bucket_name = f'compartmental-results-{simulation_params.get("ENVIRONMENT")}'
    for i in range(len(results)):
        logger.info(f"id: {results[i].get('id')}")
        gql_statuses = write_to_gql(simulation_params, results[i])
        logger.info(f"gql_statuses_{i}: {gql_statuses}")
        s3_statuses = write_to_s3(s3_client, bucket_name, s3_results[i], simulation_job_id)
        logger.info(f"s3_statuses_{i}: {s3_statuses}")
    
    # Invoke get-ai-summary lambda
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    payload_data = {"simulation_job_id": simulation_job_id}
    payload = json.dumps(payload_data).encode('utf-8')
    lambda_client.invoke(
        FunctionName=f'get-ai-summary-{simulation_params.get("ENVIRONMENT")}',
        InvocationType='Event',
        Payload=payload
    )

    return run_metadata

if __name__ == "__main__":
    # Start tracking memory usage
    tracemalloc.start()
    start_time = time.time()
    logger.info("Memory tracking started...")

    # Spawn fresh python interpreter and pull down sim params
    multiprocessing.set_start_method("spawn", force=True)
    simulation_params = get_simulation_params()

    run_metadata = main(simulation_params)

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
    run_metadata["run_time"] = elapsed_time

    # Write run metadata to gql
    #run_metadata_response = gql_write_helper(simulation_params, run_metadata, run_metadata_query)

    #logger.info(f"run_metadata_response: {run_metadata_response}")
    logger.info(f"Memory tracking stopped. Peak memory usage: {peak / (1024 * 1024):.2f} MB")
    logger.info(f"Elapsed time: {elapsed_time} seconds")