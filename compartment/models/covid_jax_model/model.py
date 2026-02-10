import jax.numpy as np
import logging
import numpy as onp
from datetime import datetime
from compartment.helpers import setup_logging, prepare_covid_initial_state
from compartment.interventions import jax_timestep_intervention, jax_prop_intervention
from compartment.model import Model
# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class CovidJaxModel(Model):
    """ A class representing a compartmental model with dynamic travel and intervention mechanisms """
    
    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """
        Standard SEIHDR initial population for respiratory diseases.
        Uses base class implementation (S-I split).
        """
        return super().get_initial_population(admin_zones, compartment_list, **kwargs)
    
    def __init__(self, config):
        """ Initialize the CompartmentalModel with a configuration dictionary """
        # Load population and travel data
        self.population_matrix = np.array(config["initial_population"])
        self.travel_matrix = np.fill_diagonal(np.array(config["travel_matrix"]), 1.0, inplace=False)
        self.compartment_list = config["compartment_list"]
        self.sigma = config['travel_volume']['leaving'] 

        # Load disease transmission parameters
        transmission_dict = config.get("transmission_dict", {})
        self.beta = transmission_dict.get("beta", None)         # susceptible → infected
        self.gamma = transmission_dict.get("gamma", None)       # infected → recovered
        self.theta = transmission_dict.get("theta", None)       # exposed → infected
        self.zeta = transmission_dict.get("zeta", None)         # infected → hospitalized
        self.delta = transmission_dict.get("delta", None)       # infected → deceased
        self.eta = transmission_dict.get("eta", None)           # hospitalized → recovered
        self.epsilon = transmission_dict.get("epsilon", None)   # hospitalized → deceased
        self.original_rates = {"beta": self.beta}

        # Simulation parameters
        self.start_date = config["start_date"]
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = config["time_steps"]

        # Administrative units & demographics
        self.admin_units = config["admin_units"]
        self.demographics = config["case_file"]["demographics"]
        self.age_stratification = list(self.demographics.values())
        self.age_groups = list(self.demographics.keys())
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
            "vaccination": False
        }

        self.payload = config

    @property
    def disease_type(self):
        return "RESPIRATORY"

    def get_params(self):
        """ Get params tuple for ODE solver """
        return (
            self.interaction_matrix,
            self.theta, 
            self.gamma, 
            self.zeta,
            self.eta, 
            self.epsilon, 
            self.delta
        )
    
    def _add_cumulative_compartments(self):
        # stack compartment totals
        S_idx = self.compartment_list.index('S')
        cumulative_shape = onp.zeros((1, *self.population_matrix[S_idx].shape))
        for comp in ["E", "I", "H"]:
            if comp in self.compartment_list:
                self.population_matrix = onp.vstack((self.population_matrix, cumulative_shape))
                self.compartment_list = self.compartment_list + [f"{comp}_total"]
    
    def prepare_initial_state(self):
        self.population_matrix, self.interaction_matrix = prepare_covid_initial_state(
            self.population_matrix,
            self.interaction_matrix,
            self.demographics
        )
        self._add_cumulative_compartments()

        return self.population_matrix, self.compartment_list

    def derivative(self, y, t, p):
        """
        Calculates how each compartment changes over time, used by ODE solver
        Dynamically handles optional compartments in self.compartment_list
        """
        # Unpack rates
        age_trans, theta, gamma, zeta, eta, epsilon, delta = p

        states = {comp: y[i] for i, comp in enumerate(self.compartment_list)}
        S = states['S']
        I = states['I']
        R = states['R']
        # Optional compartments
        E = states.get('E')
        H = states.get('H')
        D = states.get('D')

        current_ordinal_day = self.start_date_ordinal + t

        # Compute total population excluding deaths and cumulative compartments
        pop_terms = [comp for comp in (S, E, I, H, R) if comp is not None]
        N_total = sum(pop_terms).sum(axis=0)
        I_frac = I / N_total[None, :]

        # Interventions
        rates = {'beta': self.beta}
        prop_infective_scalar = I.sum() / N_total.sum()
        rates, self.intervention_statuses, contact_matrix = jax_timestep_intervention(
            self.intervention_dict,
            current_ordinal_day,
            rates,
            self.intervention_statuses,
            self.travel_matrix
        )
        rates, self.intervention_statuses, contact_matrix = jax_prop_intervention(
            self.intervention_dict,
            prop_infective_scalar,
            rates,
            self.intervention_statuses,
            self.travel_matrix
        )

        # Compute force of infection
        BETA = ((rates['beta'] * contact_matrix) @ I_frac.T).T
        omega = age_trans @ BETA

        # Effective rate parameters based on compartment presence
        theta = theta if E is not None else 0
        zeta = zeta if H is not None else 0
        delta = delta if D is not None else 0
        eta = eta if H is not None else 0
        epsilon = epsilon if H is not None and D is not None else 0

        # Add derivatives to dictionary then stack results
        derivs = {}

        derivs['S'] = -S * omega
        if E is not None:
            derivs['E'] = S * omega - theta * E
        derivs['I'] = (theta * E if E is not None else S * omega) - (gamma + zeta + delta) * I
        if H is not None:
            derivs['H'] = zeta * I - (eta + epsilon) * H
        if D is not None:
            derivs['D'] = (delta * I) + (epsilon * H if H is not None else 0)
        derivs['R'] = gamma * I + (eta * (H if H is not None else 0))

        # Cumulative compartments
        if 'E_total' in self.compartment_list:
            derivs['E_total'] = S * omega
        derivs['I_total'] = (theta * E if E is not None else S * omega)
        if 'H_total' in self.compartment_list:
            derivs['H_total'] = zeta * I

        return np.stack([derivs[comp] for comp in self.compartment_list])