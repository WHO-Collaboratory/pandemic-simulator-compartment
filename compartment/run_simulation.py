import logging
import json
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np # should we use jax.numpy?
from copy import deepcopy
from math import ceil
import boto3
from compartment.model import Model
from compartment.simulation_manager import SimulationManager
from compartment.simulation_postprocessor import SimulationPostProcessor
from compartment.batch_simulation_manager import BatchSimulationManager
from compartment.helpers import (
  generate_LHS_samples, 
  build_uncertainty_params,
  load_config_from_json,
  write_results_to_local
)
from compartment.validation import CovidSimulationConfig, DengueSimulationConfig
from compartment.batch_helpers.graphql_queries import GRAPHQL_QUERY
from compartment.batch_helpers.gql import get_simulation_job
from compartment.batch_helpers.s3 import write_to_s3
from compartment.batch_helpers.simulation_helpers import get_simulation_params
from compartment.batch_helpers.gql import write_to_gql
#from compartment.model import DengueJaxModel, CovidJaxModel
#from compartment.examples.dengue_jax_model.model import DengueJaxModel
from compartment.examples.covid_jax_model.model import CovidJaxModel
# Makes sure unix implementations don't deadlock
multiprocessing.set_start_method('spawn', force=True)

# Remove unnecessary jax INFO logging
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})


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

def run_simulation(simulation_params=None, config_path: str = None, output_path: str = None):
    logger.info("Starting the simulation...")
    
    # Determine if running in local or cloud mode
    is_local_mode = config_path is not None
    
    if is_local_mode:
        logger.info("Running in LOCAL mode")
        # Load config from local JSON file
        config = load_config_from_json(config_path)
        simulation_job_id = config['data']['getSimulationJob'].get('id', 'local-simulation')
        owner = "local-user"
    else:
        logger.info("Running in CLOUD mode")
        if simulation_params is None:
            raise ValueError("simulation_params is required for cloud mode")
        simulation_job_id = simulation_params.get('SIMULATION_JOB_ID')
        logger.info(f"simulation_job_id: {simulation_job_id}")
        logger.info(f"graphql_endpoint: {simulation_params.get('GRAPHQL_ENDPOINT')}")
        # Grab simulation job and clean validated config for model
        config = get_simulation_job(simulation_params, GRAPHQL_QUERY)
        owner = config['data']['getSimulationJob'].get('owner')
    
    # Validate and create config object
    disease_type = config['data']['getSimulationJob']['Disease']['disease_type']
    if disease_type == "VECTOR_BORNE":
        cleaned_config = DengueSimulationConfig(**config['data']['getSimulationJob'])
    else:
        cleaned_config = CovidSimulationConfig(**config['data']['getSimulationJob'])

    disease_type = cleaned_config.Disease.disease_type

    run_metadata = {
        "simulation_job_id": simulation_job_id,
        "admin_zone_count": len(cleaned_config.admin_units),
        "simulation_time_steps": cleaned_config.time_steps,
        "owner": owner,
        "disease_type": disease_type
    }

    logger.info(f"owner: {owner}")
    logger.info(f"number of admin units: {len(cleaned_config.admin_units)}")
    logger.info(f"compartment_list: {cleaned_config.compartment_list}")
    logger.info(f"disease_type: {disease_type}")
    if disease_type == "RESPIRATORY":
        logger.info(f"transmission_dict: {cleaned_config.transmission_dict}")
    logger.info(f"intervention_dict: {cleaned_config.intervention_dict}")

    # Map of disease_type to model_class
    DISEASE_MODEL_MAP = {
        #"VECTOR_BORNE": DengueJaxModel,
        "RESPIRATORY": CovidJaxModel    
    }

    try:
        model_class = DISEASE_MODEL_MAP[disease_type]
    except KeyError:
        logger.error(f"Invalid disease type: {disease_type}")
        raise ValueError(f"Invalid disease type: {disease_type}")
    
    run_mode = cleaned_config.run_mode
    logger.info(f"run_mode: {run_mode}")

    # Create business as usual model
    model_with = model_class(cleaned_config.model_dump())
    # Create a deepcopy for the interventionless model
    model_without = deepcopy(model_with)
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

        if disease_type == "VECTOR_BORNE":
          transmission_edges_dicts = None
        else:
          transmission_edges_dicts = [
              edge.model_dump() for edge in cleaned_config.Disease.transmission_edges
          ]

        interventions_dicts = [intervention.model_dump() for intervention in cleaned_config.interventions]

        uncertainty_params = build_uncertainty_params(
            transmission_edges_dicts,
            interventions_dicts
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
    
    if is_local_mode:
        # Local mode: write results to local file
        if output_path is None:
            output_path = f"simulation_results_{simulation_job_id}.json"
        write_results_to_local(results, output_path)
        logger.info(f"Results saved to: {output_path}")
    else:
        # Cloud mode: write to GQL and S3, invoke lambda
        s3_results = deepcopy(results)
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