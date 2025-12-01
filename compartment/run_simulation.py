import logging
import json
import numpy as np # should we use jax.numpy?
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from math import ceil
import os
from compartment.model import Model
from compartment.simulation_manager import SimulationManager
from compartment.simulation_postprocessor import SimulationPostProcessor
from compartment.batch_simulation_manager import BatchSimulationManager
from compartment.helpers import (
  clean_payload,
  generate_LHS_samples, 
  build_uncertainty_params
)
# Remove unnecessary jax INFO logging
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

#from temp_uncertainty_graphs import plot_simulation_results, plot_compartments_all_regions, plot_all_regions_and_compartments
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})

# Initialize logging

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

def run_simulation(model:Model, config_file_path:str):
  logger.info("Starting the simulation...")
  config = json.load(open(config_file_path))
  cleaned_config = clean_payload(config)

  #TODO determine if run_mode is shared between models or only for dengue
  run_mode = config['data']['getSimulationJob']['run_mode']
  logger.info(f"run_mode: {run_mode}")

  #TODO is an intervention_dict a property of all models?
  model_with = model(cleaned_config)

  model_without = deepcopy(model_with)
  model_without.intervention_dict = {}

  run_metadata = {
    "admin_zone_count": len(cleaned_config.get('admin_units')),
    "simulation_time_steps": config['data']['getSimulationJob']['time_steps'],
    "owner": config['data']['getSimulationJob']['owner'],
    "disease_type": model_with.disease_type
  }

  logger.info(f"owner: {config['data']['getSimulationJob']['owner']}")
  logger.info(f"number of admin units: {len(cleaned_config.get('admin_units'))}")
  logger.info(f"compartment_list: {cleaned_config.get('compartment_list')}")
  logger.info(f"disease_type: {Model.disease_type}")
  logger.info(f"transmission_dict: {cleaned_config.get('transmission_dict')}")
  logger.info(f"intervention_dict: {cleaned_config.get('intervention_dict')}")
  
  # Split cpu in half for top level workers
  # Then split the remaining cpu in half for low level workers
  top_level_workers = 2
  low_level_workers = ceil(os.cpu_count() / top_level_workers) - 1
  logger.info(f"top_level_workers: {top_level_workers}")
  logger.info(f"low_level_workers: {low_level_workers}")

  #TODO determine if run_mode is shared between models or only for dengue
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

    with ProcessPoolExecutor(max_workers=top_level_workers) as executor:
      future_with = executor.submit(batch_simulate_and_postprocess, model_with, n_sims, param_list, ci, low_level_workers)
      future_without = executor.submit(batch_simulate_and_postprocess, model_without, n_sims, interventionless_param_list, ci, low_level_workers)
      results_with = future_with.result()
      results_without = future_without.result()

  results_with["control_run"] = False
  results_without["control_run"] = True
  results = [results_with, results_without]

  output_data = {
    "run_metadata": run_metadata,
    "results": results
  }

  return output_data