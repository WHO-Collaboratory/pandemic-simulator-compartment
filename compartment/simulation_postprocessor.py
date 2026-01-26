import logging
import numpy as np
import scipy.stats as st
import pandas as pd

from compartment.helpers import (
    setup_logging,
    format_jax_output,
    format_uncertainty_output,
    get_simulation_step_size,
    compute_multi_run_compartment_deltas,
    dengue_compartment_grouping,
    covid_compartment_grouping

)

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class SimulationPostProcessor:
    def __init__(self, model, population_matrix):
        self.model = model
        self.population_matrix = population_matrix if isinstance(population_matrix, list) else [population_matrix]
        self.compartment_list = model.compartment_list
        self.admin_units = model.admin_units
        self.start_date = model.start_date
        self.n_timesteps = model.n_timesteps
        self.demographics = getattr(model, "demographics", None) or {'age_0_17': 25.0, 'age_18_55': 50.0, 'age_56_plus': 25.0}
        self.disease_type = model.disease_type
        self.step = get_simulation_step_size(model.n_timesteps)
        self.intervention_dict = getattr(model, "intervention_dict", {})
        self.payload = model.payload
        self.n_runs = len(self.population_matrix)

    def _aggregate_age_groups(self, arr: np.ndarray):
        """
        Ensure age groups are aggregated
        If arr has shape (T, C, A, R) sum over the age axis (T, C, R)
        If arr has shape (T, C, R) return as is
        """
        if arr.ndim == 4:
            return arr.sum(axis=2)
        elif arr.ndim == 3:
            return arr
        else:
            raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")  

    def _calculate_simulation_based_interval(self, all_sims: np.ndarray, ci: float = 0.95):
    
        #Generate simulation-based interval metrics
        median_values = np.median(all_sims, axis=0) 
        lower_bounds = np.percentile(all_sims, ((1-ci)/2)*100, axis=0) # defaults to 2.5th percentile along columns
        upper_bounds = np.percentile(all_sims, (1-((1-ci)/2))*100, axis=0) # defaults to  97.5th percentile along columns

        return median_values, lower_bounds, upper_bounds

    
    def _group_compartments(self, arr: np.ndarray, grouping: dict) -> np.ndarray:
        """
        Given an age-aggregated array of shape (T, C, R), group compartments
        according to the provided grouping and return an array of shape (T, G, R).
        """
        T, C, R = arr.shape
        # Build mapping from original compartment to group
        col2grp = {
            comp: grp
            for grp, comps in grouping.items()
            for comp in comps
        }

        if self.disease_type == "RESPIRATORY":
            group_names = []
            for comp in self.compartment_list:
                if comp in grouping.keys():
                    group_names.append(comp)
            # Filter col2grp to only include compartments in the compartment list
            col2grp = {
                comp: col2grp[comp]
                for comp in self.compartment_list
                if comp in col2grp
            }
            
        else:
            group_names = list(grouping.keys())

        # Prepare output array
        new_arr = np.zeros((T, len(group_names), R))
        # For each region, group via DataFrame
        for r in range(R):
            df = pd.DataFrame(arr[:, :, r], columns=self.compartment_list)
            df_grp = df.T.groupby(col2grp).sum().T
            # Ensure consistent column order
            df_grp = df_grp[group_names]
            new_arr[:, :, r] = df_grp.values
        return new_arr


    def process(self, ci: float = 0.95):
        # Single run case - hand it to the formatter
        if self.n_runs == 1:
            logger.info(f"Handing payload to single run formatter")
            return format_jax_output(
                self.intervention_dict,
                self.payload,
                self.population_matrix[0],
                self.compartment_list,
                len(self.admin_units),
                self.start_date,
                self.n_timesteps,
                self.demographics,
                self.disease_type,
                self.step
            )
        
        # Choose grouping and group compartments before confidence intervals
        if self.disease_type == "VECTOR_BORNE":
            grouping = dengue_compartment_grouping
            grouped_compartment_list = list(grouping.keys())
        else:
            grouping = covid_compartment_grouping
            grouped_compartment_list = [comp for comp in self.compartment_list if comp in grouping.keys()]

        # Use helper to group each simulation's compartments
        logger.info("Grouping compartments before confidence intervals via _group_compartments")
        grouped_sims = [
            self._group_compartments(self._aggregate_age_groups(sim), grouping)
            for sim in self.population_matrix
        ]
        all_sims = np.stack(grouped_sims, axis=0)
        means_child, lower_child, upper_child = self._calculate_simulation_based_interval(all_sims, ci)

        # Total population: sum across regions (last axis is region)
        logger.info(f"Calculating confidence intervals for parent")
        all_sims_total = all_sims.sum(axis=-1)  # shape: (n_runs, T, C)
        means_parent, lower_parent, upper_parent = self._calculate_simulation_based_interval(all_sims_total, ci)

        # Calculate multi run compartment deltas
        logger.info(f"Averaging compartment deltas")
        avg_compartment_deltas = compute_multi_run_compartment_deltas(
            self.population_matrix, 
            self.disease_type, 
            len(self.admin_units), 
            self.compartment_list
        )
        
        # Format output
        logger.info(f"Handing payload to multi run formatter")
        return format_uncertainty_output(
            means_child, lower_child, upper_child,
            means_parent, lower_parent, upper_parent,
            self.payload,
            grouped_compartment_list,
            self.admin_units,
            str(self.start_date),
            self.n_timesteps,
            self.step,
            avg_compartment_deltas
        )
        