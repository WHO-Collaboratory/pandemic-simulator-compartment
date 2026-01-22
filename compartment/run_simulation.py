import logging
import json
import os
import multiprocessing
import numpy as np # should we use jax.numpy?
from copy import deepcopy
import boto3
from compartment.helpers import get_executor_class
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
from compartment.cloud_helpers.graphql_queries import GRAPHQL_QUERY
from compartment.cloud_helpers.gql import get_simulation_job
from compartment.cloud_helpers.s3 import write_to_s3, record_and_upload_validation
from compartment.cloud_helpers.gql import write_to_gql
from compartment.cloud_helpers.simulation_helpers import transform_normalized_interventions
from compartment.model import Model

# Makes sure unix implementations don't deadlock
multiprocessing.set_start_method('spawn', force=True)
ExecutorClass = get_executor_class()

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

def run_simulation(model_class, simulation_params=None, mode:str='local', config_path: str = None, output_path: str = None):
    logger.info("Starting the simulation...")

    if mode == 'local':
        logger.info("Running in LOCAL mode")
        # Load config from local JSON file
        config = load_config_from_json(config_path)
        simulation_job_id = config['data']['getSimulationJob'].get('id', 'local-simulation')
        owner = "local-user"
    elif mode == 'cloud':
        logger.info("Running in CLOUD mode")
        if simulation_params is None:
            raise ValueError("simulation_params is required for cloud mode")
        simulation_job_id = simulation_params.get('SIMULATION_JOB_ID')
        logger.info(f"simulation_job_id: {simulation_job_id}")
        logger.info(f"graphql_endpoint: {simulation_params.get('GRAPHQL_ENDPOINT')}")
        # Grab simulation job and clean validated config for model
        config = get_simulation_job(simulation_params, GRAPHQL_QUERY)

        ########### START NORMALIZATION OF INTERVENTIONS ###########
        # Transform normalized interventions from join tables to legacy format
        # Priority: Use new Interventions join table if available, fall back to embedded interventions
        normalized_interventions = config['data']['getSimulationJob'].get('Interventions', {})
        normalized_items = normalized_interventions.get('items', []) if normalized_interventions else []

        logger.info(f"normalized_items: {normalized_items}")
        
        if normalized_items:
            logger.info(f"Using {len(normalized_items)} interventions from normalized join tables")
            config['data']['getSimulationJob']['interventions'] = transform_normalized_interventions(normalized_items)
        else:
            logger.info("No normalized interventions found, using embedded interventions field")
        ########### END NORMALIZATION OF INTERVENTIONS ###########

        owner = config['data']['getSimulationJob'].get('owner')
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Validate and create config object
    disease_type = config['data']['getSimulationJob']['Disease']['disease_type']
    validation_success, cleaned_config = record_and_upload_validation(
        simulation_job_id,
        config,
        disease_type,
        environment=simulation_params.get("ENVIRONMENT", "dev") if simulation_params else None,
        mode=mode
    )
    if not validation_success:
        logger.error("Halting due to validation failure. See S3 logs for details.")
        return None

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
    if getattr(cleaned_config, "transmission_dict", None) is not None:
        logger.info(f"transmission_dict: {cleaned_config.transmission_dict}")
    logger.info(f"intervention_dict: {cleaned_config.intervention_dict}")

    # Validate that model_class is a subclass of Model
    if not issubclass(model_class, Model):
        logger.error(f"model_class must be a subclass of Model, got {model_class}")
        raise ValueError(f"model_class must be a subclass of Model, got {model_class}")
    
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
    low_level_workers = 2#ceil(os.cpu_count() / top_level_workers) - 1
    logger.info(f"top_level_workers: {top_level_workers}")
    logger.info(f"low_level_workers: {low_level_workers}")
    
    if run_mode == "DETERMINISTIC":
        with ExecutorClass(max_workers=top_level_workers) as executor:
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
        with ExecutorClass(max_workers=top_level_workers) as executor:
            future_with = executor.submit(batch_simulate_and_postprocess, model_with, n_sims, param_list, ci, low_level_workers)
            future_without = executor.submit(batch_simulate_and_postprocess, model_without, n_sims, interventionless_param_list, ci, low_level_workers)
            results_with = future_with.result()
            results_without = future_without.result()

    results_with["control_run"] = False
    results_without["control_run"] = True
    results = [results_with, results_without]
    
    if mode == 'local':
        if output_path is None:
            print(results)
        elif output_path is not None:
            write_results_to_local(results, output_path)
            logger.info(f"Results saved to: {output_path}")
    else:
        # Cloud mode: write to GQL and S3, invoke lambda
        s3_results = deepcopy(results)
        s3_client = boto3.client('s3', region_name='us-east-1')
        bucket_name = f'compartmental-results-{simulation_params.get("ENVIRONMENT")}'

        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = min(32, max(4, len(results) * 2))  # tune as needed
        gql_statuses_by_i = [None] * len(results)
        s3_statuses_by_i = [None] * len(results)

        future_to_meta = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, result in enumerate(results):
                future_to_meta[executor.submit(write_to_gql, simulation_params, result)] = (i, "gql")
                future_to_meta[
                    executor.submit(write_to_s3, s3_client, bucket_name, s3_results[i], simulation_job_id)
                ] = (i, "s3")

            for fut in as_completed(future_to_meta):
                i, kind = future_to_meta[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    logger.exception("Write failed", extra={"i": i, "kind": kind})
                    out = {"status": "error", "error": str(e)}

                if kind == "gql":
                    gql_statuses_by_i[i] = out
                else:
                    s3_statuses_by_i[i] = out

        for i in range(len(results)):
            logger.info(f"gql_statuses_{i}: {gql_statuses_by_i[i]}")
            logger.info(f"s3_statuses_{i}: {s3_statuses_by_i[i]}")
        
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
