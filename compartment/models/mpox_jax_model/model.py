import jax.numpy as np
import logging
import numpy as onp
from compartment.helpers import setup_logging
from compartment.model import Model

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class MpoxJaxModel(Model):
    """ A simple SIR compartmental model for MPOX """
    
    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """
        Simple SIR initial population for MPOX.
        Uses base class implementation (S-I split).
        """
        return super().get_initial_population(admin_zones, compartment_list, **kwargs)
    
    def __init__(self, config):
        """ Initialize the MPOX SIR model with a configuration dictionary """
        # Load population data
        self.population_matrix = np.array(config["initial_population"]).T
        self.compartment_list = config["compartment_list"]
        
        # Load disease transmission parameters
        transmission_dict = config.get("transmission_dict", {})
        self.beta = transmission_dict.get("beta", None)   # susceptible → infected
        self.gamma = transmission_dict.get("gamma", None) # infected → recovered
        
        # Simulation parameters
        self.start_date = config["start_date"]
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = config["time_steps"]
        
        # Administrative units
        self.admin_units = config["admin_units"]
        
        self.payload = config

    @property
    def disease_type(self):
        return "MONKEYPOX"

    def get_params(self):
        """ Get params tuple for ODE solver """
        return (self.beta, self.gamma)
    
    def prepare_initial_state(self):
        #self._add_cumulative_compartments()
        return self.population_matrix, self.compartment_list

    def derivative(self, y, t, p):
        """
        Calculates the SIR derivative for MPOX
        S -> I -> R
        """
        # Unpack parameters
        beta, gamma = p
        
        # Extract compartments
        states = {comp: y[i] for i, comp in enumerate(self.compartment_list)}
        S = states['S']
        I = states['I']
        R = states['R']
        
        # Compute total population
        N_total = S + I + R
        
        # Force of infection (simple frequency-dependent transmission)
        lambda_force = beta * I / (N_total + 1e-10)  # Add small epsilon to avoid division by zero
        
        # SIR derivatives
        derivs = {}
        derivs['S'] = -S * lambda_force
        derivs['I'] = S * lambda_force - gamma * I
        derivs['R'] = gamma * I
        
        return np.stack([derivs['S'], derivs['I'], derivs['R']])

