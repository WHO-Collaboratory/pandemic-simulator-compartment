import logging
import json
import os
import multiprocessing
import numpy as np  # should we use jax.numpy?
from copy import deepcopy
import boto3
from compartment.model import Model
from compartment.simulation_manager import SimulationManager
from compartment.simulation_postprocessor import SimulationPostProcessor
from compartment.batch_simulation_manager import BatchSimulationManager
from compartment.helpers import (
    get_executor_class,
    generate_LHS_samples,
    build_uncertainty_params,
    load_config_from_json,
    write_results_to_local,
    as_dict_list,
)
from compartment.cloud_helpers.graphql_queries import GRAPHQL_QUERY
from compartment.cloud_helpers.gql import (
    get_simulation_job,
    write_to_gql,
)
from compartment.cloud_helpers.s3 import write_to_s3, record_and_upload_validation
from compartment.model import Model

# Makes sure unix implementations don't deadlock
multiprocessing.set_start_method("spawn", force=True)
ExecutorClass = get_executor_class()

# Remove unnecessary jax INFO logging
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})


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


def run_simulation(
    model_class,
    simulation_params=None,
    mode: str = "local",
    config_path: str = None,
    output_path: str = None,
):
    logger.info("Starting the simulation...")

    disease_param_field_configs = []

    if mode == "local":
        logger.info("Running in LOCAL mode")
        # Load config from local JSON file
        config = load_config_from_json(config_path)
        simulation_job_id = config["data"]["getSimulationJob"].get(
            "id", "local-simulation"
        )
        owner = "local-user"
    elif mode == "cloud":
        logger.info("Running in CLOUD mode")
        if simulation_params is None:
            raise ValueError("simulation_params is required for cloud mode")
        simulation_job_id = simulation_params.get("SIMULATION_JOB_ID")
        logger.info(f"simulation_job_id: {simulation_job_id}")
        logger.info(f"graphql_endpoint: {simulation_params.get('GRAPHQL_ENDPOINT')}")
        # Grab simulation job config from GraphQL
        config = get_simulation_job(simulation_params, GRAPHQL_QUERY)

        owner = config["data"]["getSimulationJob"].get("owner")

        # Extract disease parameter variance configs (if any) before
        # validation strips them.  These are appended to uncertainty_params
        # for UNCERTAINTY runs so LHS sampling covers disease custom fields.
        disease_param_field_configs = config.pop("_disease_param_field_configs", [])
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Resolve model_class from the artifact's disease_type (e.g. COVID_SIHR)
    # so variant selection routes to the correct subclass. ModelArtifact is
    # preferred over Disease because it carries the specific variant type;
    # Disease.disease_type is the base type (e.g. COVID_SEIHDR).
    # The caller-supplied model_class is kept as a final fallback.
    from compartment.registry import resolve as _resolve

    job = config["data"]["getSimulationJob"]
    config_disease_type = (job.get("ModelArtifact") or {}).get("disease_type") or (
        job.get("Disease") or {}
    ).get("disease_type")
    if config_disease_type:
        resolved = _resolve(config_disease_type)
        if resolved is not None:
            model_class = resolved

    disease_type = model_class.DISEASE_TYPE
    config["data"]["getSimulationJob"]["Disease"]["disease_type"] = disease_type

    validation_success, cleaned_config = record_and_upload_validation(
        simulation_job_id,
        config,
        disease_type,
        environment=simulation_params.get("ENVIRONMENT", "dev")
        if simulation_params
        else None,
        mode=mode,
    )
    if not validation_success:
        logger.error("Halting due to validation failure. See S3 logs for details.")
        return None

    run_metadata = {
        "simulation_job_id": simulation_job_id,
        "admin_zone_count": len(cleaned_config.admin_units),
        "simulation_time_steps": cleaned_config.time_steps,
        "owner": owner,
        "disease_type": disease_type,
    }

    logger.info(f"owner: {owner}")
    logger.info(f"number of admin units: {len(cleaned_config.admin_units)}")
    logger.info(f"compartment_list: {cleaned_config.compartment_list}")
    logger.info(f"disease_type: {disease_type}")
    if getattr(cleaned_config, "transmission_dict", None) is not None:
        logger.info(f"transmission_dict: {cleaned_config.transmission_dict}")
    if getattr(cleaned_config, "intervention_dict", None) is not None:
        logger.info(f"intervention_dict: {cleaned_config.intervention_dict}")

    # Validate that model_class is a subclass of Model
    if not issubclass(model_class, Model):
        logger.error(f"model_class must be a subclass of Model, got {model_class}")
        raise ValueError(f"model_class must be a subclass of Model, got {model_class}")

    run_mode = cleaned_config.run_mode
    logger.info(f"run_mode: {run_mode}")

    # Create business as usual model
    model_with = model_class(cleaned_config)
    # Create a deepcopy for the interventionless model
    model_without = deepcopy(model_with)
    model_without.intervention_dict = {}

    # Split cpu in half for top level workers
    # Then split the remaining cpu in half for low level workers
    top_level_workers = 2
    low_level_workers = 2  # ceil(os.cpu_count() / top_level_workers) - 1
    logger.info(f"top_level_workers: {top_level_workers}")
    logger.info(f"low_level_workers: {low_level_workers}")

    if run_mode == "DETERMINISTIC":
        with ExecutorClass(max_workers=top_level_workers) as executor:
            future_with = executor.submit(simulate_and_postprocess, model_with)
            future_without = executor.submit(simulate_and_postprocess, model_without)
            results_with = future_with.result()
            results_without = future_without.result()
    else:
        n_sims = getattr(cleaned_config, "n_simulations", None) or 30
        ci = 0.95

        # Extract normalized TransmissionEdges and Interventions for uncertainty
        transmission_edge_items = []
        interventions_items = []

        transmission_edges_section = getattr(cleaned_config, "TransmissionEdges", None)
        if transmission_edges_section:
            if isinstance(transmission_edges_section, dict):
                transmission_edge_items = transmission_edges_section.get("items", [])
            else:
                transmission_edge_items = getattr(
                    transmission_edges_section, "items", []
                )

        interventions_section = getattr(cleaned_config, "Interventions", None)
        if interventions_section:
            if isinstance(interventions_section, dict):
                interventions_items = interventions_section.get("items", [])
            else:
                interventions_items = getattr(interventions_section, "items", [])

        # Convert to dicts if they're Pydantic models
        transmission_edge_dicts = as_dict_list(transmission_edge_items)
        intervention_dicts = as_dict_list(interventions_items)

        uncertainty_params = build_uncertainty_params(
            transmission_edge_dicts, intervention_dicts
        )

        # Append disease parameter custom field variance configs (if any).
        # These were collected by get_simulation_job() from FieldConfig
        # records on SimulationJobCustomField entries.
        if disease_param_field_configs:
            uncertainty_params.extend(disease_param_field_configs)
            logger.info(
                f"Added {len(disease_param_field_configs)} disease parameter variance config(s) to uncertainty params"
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
            future_with = executor.submit(
                batch_simulate_and_postprocess,
                model_with,
                n_sims,
                param_list,
                ci,
                low_level_workers,
            )
            future_without = executor.submit(
                batch_simulate_and_postprocess,
                model_without,
                n_sims,
                interventionless_param_list,
                ci,
                low_level_workers,
            )
            results_with = future_with.result()
            results_without = future_without.result()

    results_with["control_run"] = False
    results_without["control_run"] = True
    results = [results_with, results_without]

    if mode == "local":
        if output_path is None:
            print(results)
        elif output_path is not None:
            write_results_to_local(results, output_path)
            logger.info(f"Results saved to: {output_path}")
    else:
        # Cloud mode: write results to S3 (full data) and GQL (metadata only)
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket_name = f"compartmental-results-{simulation_params.get('ENVIRONMENT')}"

        from concurrent.futures import ThreadPoolExecutor, as_completed

        gql_statuses = [None] * len(results)
        s3_statuses = [None] * len(results)
        future_to_meta = {}

        with ThreadPoolExecutor(max_workers=len(results) * 2) as executor:
            for i, result in enumerate(results):
                # GQL: write SimulationJobResult metadata only (no time series).
                # Extract metadata before submitting to avoid races with S3
                # writer which pops admin_zones/parent_admin_total.
                gql_metadata = {
                    k: v
                    for k, v in result.items()
                    if k not in ("admin_zones", "parent_admin_total")
                }
                future_to_meta[
                    executor.submit(write_to_gql, simulation_params, gql_metadata)
                ] = (i, "gql")

                # S3: write full data (time series + metadata)
                future_to_meta[
                    executor.submit(
                        write_to_s3,
                        s3_client,
                        bucket_name,
                        result,
                        simulation_job_id,
                    )
                ] = (i, "s3")

            for fut in as_completed(future_to_meta):
                i, kind = future_to_meta[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    logger.exception("Write failed", extra={"i": i, "kind": kind})
                    out = {"status": "error", "error": str(e)}

                if kind == "gql":
                    gql_statuses[i] = out
                else:
                    s3_statuses[i] = out

        for i in range(len(results)):
            logger.info(f"gql_statuses_{i}: {gql_statuses[i]}")
            logger.info(f"s3_statuses_{i}: {s3_statuses[i]}")

        # Invoke get-ai-summary lambda
        lambda_client = boto3.client("lambda", region_name="us-east-1")
        payload_data = {"simulation_job_id": simulation_job_id}
        payload = json.dumps(payload_data).encode("utf-8")
        lambda_client.invoke(
            FunctionName=f"get-ai-summary-{simulation_params.get('ENVIRONMENT')}",
            InvocationType="Event",
            Payload=payload,
        )

    return run_metadata
