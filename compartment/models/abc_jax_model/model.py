import jax.numpy as np
import logging
import numpy as onp
from datetime import datetime
from compartment.helpers import setup_logging
from compartment.interventions import jax_timestep_intervention, jax_prop_intervention
from compartment.model import Model

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


class ABCJaxModel(Model):
    """
    A simple ABC compartmental model where:
    - A represents the first compartment (analogous to Susceptible)
    - B represents the second compartment (analogous to Infected)
    - C represents the third compartment (analogous to Recovered)
    
    The model supports configurable transmission rates between compartments.
    """
    
    def __init__(self, config):
        """Initialize the ABC Model with a configuration dictionary"""
        # Load population and travel data
        self.population_matrix = np.array(config["initial_population"])
        self.travel_matrix = np.fill_diagonal(np.array(config["travel_matrix"]), 1.0, inplace=False)
        self.compartment_list = config["compartment_list"]
        self.sigma = config['travel_volume']['leaving']

        # Load transmission parameters from transmission_edges
        transmission_dict = config.get("transmission_dict", {})
        self.alpha = transmission_dict.get("alpha", 0.3)  # A → B transmission rate
        self.beta = transmission_dict.get("beta", 0.1)    # B → C transmission rate
        
        # Store original rates for interventions
        self.original_rates = {
            "alpha": self.alpha,
            "beta": self.beta
        }

        # Simulation parameters
        self.start_date = config["start_date"]
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = config["time_steps"]

        # Administrative units & demographics
        self.admin_units = config["admin_units"]
        self.demographics = config["case_file"]["demographics"]
        self.age_stratification = list(self.demographics.values())
        self.age_groups = list(self.demographics.keys())
        
        # Simple interaction matrix for ABC model
        self.interaction_matrix = np.array([
            [5.46, 5.18, 0.93], 
            [1.70, 9.18, 1.68], 
            [0.83, 5.90, 3.80]
        ])

        # Interventions
        self.intervention_dict = config["intervention_dict"]
        self.intervention_statuses = {
            "lock_down": False,
            "mask_wearing": False,
            "social_isolation": False,
        }

        self.payload = config
    
    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """
        Initial population setup for ABC model.
        Uses A-B split (analogous to S-I in traditional SIR models).
        
        Returns:
            numpy array of shape (n_compartments, n_zones)
        """
        import numpy as np
        
        column_mapping = {value: index for index, value in enumerate(compartment_list)}
        initial_population = np.zeros((len(admin_zones), len(compartment_list)))

        for i, zone in enumerate(admin_zones):
            # infected_population represents the initial proportion in compartment B
            b_population = round(zone['infected_population'] / 100 * zone['population'], 2)
            a_population = zone['population'] - b_population
            
            initial_population[i, column_mapping['A']] = a_population
            initial_population[i, column_mapping['B']] = b_population
            # C starts at 0
            initial_population[i, column_mapping['C']] = 0

        # Transpose to (n_compartments, n_zones) format expected by Model
        return initial_population.T

    @property
    def disease_type(self):
        return "ABC"

    def get_params(self):
        """Get params tuple for ODE solver"""
        return (
            self.interaction_matrix,
            self.alpha,
            self.beta
        )
    
    def _add_cumulative_compartments(self):
        """Add cumulative tracking compartments"""
        A_idx = self.compartment_list.index('A')
        cumulative_shape = onp.zeros((1, *self.population_matrix[A_idx].shape))
        
        # Add cumulative compartments for B
        if 'B' in self.compartment_list:
            self.population_matrix = onp.vstack((self.population_matrix, cumulative_shape))
            self.compartment_list = self.compartment_list + ['B_total']
    
    def prepare_initial_state(self):
        """Prepare initial state with proper age stratification"""
        # Ensure population matrix has proper shape with age groups
        num_age_groups = len(self.age_groups)
        num_admin_units = len(self.admin_units)
        
        new_population = []
        for comp_idx, comp in enumerate(self.compartment_list):
            if self.population_matrix[comp_idx].ndim == 1:
                # Expand to include age groups
                comp_pop = self.population_matrix[comp_idx]
                age_stratified = onp.zeros((num_age_groups, num_admin_units))
                for age_idx, age_strat in enumerate(self.age_stratification):
                    age_stratified[age_idx, :] = comp_pop * age_strat
                new_population.append(age_stratified)
            else:
                new_population.append(self.population_matrix[comp_idx])
        
        self.population_matrix = onp.array(new_population)
        
        # Normalize interaction matrix
        if self.interaction_matrix.shape[0] != num_age_groups:
            # Use a simple default if sizes don't match
            self.interaction_matrix = np.eye(num_age_groups) * 5.0
        
        self._add_cumulative_compartments()
        return self.population_matrix, self.compartment_list

    def derivative(self, y, t, p):
        """
        Calculates how each compartment changes over time, used by ODE solver.
        
        Dynamics:
        - A → B: Individuals move from A to B at rate alpha (influenced by contact with B individuals)
        - B → C: Individuals move from B to C at rate beta
        """
        # Unpack rates
        age_trans, alpha, beta = p

        # Extract compartments
        states = {comp: y[i] for i, comp in enumerate(self.compartment_list)}
        A = states['A']
        B = states['B']
        C = states['C']

        current_ordinal_day = self.start_date_ordinal + t

        # Compute total population (excluding cumulative compartments)
        N_total = (A + B + C).sum(axis=0)
        B_frac = B / N_total[None, :]

        # Apply interventions (modifying transmission rates)
        rates = {'alpha': alpha}
        prop_B_scalar = B.sum() / N_total.sum()
        
        rates, self.intervention_statuses, contact_matrix = jax_timestep_intervention(
            self.intervention_dict,
            current_ordinal_day,
            rates,
            self.intervention_statuses,
            self.travel_matrix
        )
        
        rates, self.intervention_statuses, contact_matrix = jax_prop_intervention(
            self.intervention_dict,
            prop_B_scalar,
            rates,
            self.intervention_statuses,
            self.travel_matrix
        )

        # Compute force of infection (A→B transition influenced by B prevalence)
        ALPHA = ((rates['alpha'] * contact_matrix) @ B_frac.T).T
        omega = age_trans @ ALPHA

        # Define derivatives
        derivs = {}
        derivs['A'] = -A * omega
        derivs['B'] = A * omega - beta * B
        derivs['C'] = beta * B

        # Cumulative compartments
        if 'B_total' in self.compartment_list:
            derivs['B_total'] = A * omega

        return np.stack([derivs[comp] for comp in self.compartment_list])
