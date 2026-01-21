import jax.numpy as np
import logging
import numpy as onp
from datetime import datetime
from compartment.helpers import setup_logging
from compartment.model import Model

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class MonkeypoxJaxModel(Model):
    """A simple SIR compartmental model for Monkeypox with travel between regions"""
    def __init__(self, config):
        """Initialize the Monkeypox SIR model with a configuration dictionary"""
        super().__init__(config)
        
        # Load population and travel data
        self.compartment_list = config["compartment_list"]
        # initial_population from helper is (n_regions, n_compartments), transpose to (n_compartments, n_regions)
        initial_pop = np.array(config["initial_population"])
        if initial_pop.shape[1] == len(self.compartment_list):
            # If shape is (n_regions, n_compartments), transpose to (n_compartments, n_regions)
            initial_pop = initial_pop.T
        self.population_matrix = initial_pop
        # travel_matrix from helpers already has correct structure:
        # diagonal = 1-sigma (staying), off-diagonal = sigma * gravity_rate (moving)
        # Do NOT modify it!
        self.travel_matrix = np.array(config["travel_matrix"])
        self.sigma = config.get('travel_volume', {}).get('leaving', 0.05)

        # Load disease transmission parameters
        transmission_dict = config.get("transmission_dict", {})
        self.beta = transmission_dict.get("beta", 0.2)  # susceptible → infected
        self.gamma = transmission_dict.get("gamma", 0.1)  # infected → recovered

        # Simulation parameters
        self.start_date = config["start_date"]
        if isinstance(self.start_date, str):
            from datetime import datetime
            self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = config["time_steps"]

        # Demographics
        self.demographics = config["case_file"]["demographics"]
        # Administrative units
        self.admin_units = config["admin_units"]
        
        # No interventions for simple SIR model
        self.intervention_dict = {}
        
        # Store payload for postprocessing
        self.payload = config

    @property
    def disease_type(self):
        return "MONKEYPOX"

    def get_params(self):
        """Get params tuple for ODE solver"""
        return (self.beta, self.gamma)

    def prepare_initial_state(self):
        """Prepare initial state - no age stratification for simple SIR"""
        return self.population_matrix, self.compartment_list

    def derivative(self, y, t, p):
        """
        Calculates how each compartment changes over time for SIR model with travel
        """
        beta, gamma = p
        
        S, I, R = y  # shape: (n_regions,)
        
        # Compute total population per region
        N = S + I + R
        
        # Force of infection per region (local transmission)
        I_frac = I / N
        omega = beta * I_frac
        
        # SIR transitions within each region
        dSdt = -S * omega
        dIdt = S * omega - gamma * I
        dRdt = gamma * I
        
        # Travel between regions
        # travel_matrix structure: rows = destination, columns = origin
        # travel_matrix[i,j] = probability of person in region j ending up in region i
        # Diagonal = 1-sigma (staying), off-diagonal = sigma * normalized_gravity_rate
        # Each column sums to 1, so it's a proper stochastic matrix
        
        # For ODE: travel_matrix @ compartment gives the distribution after travel
        # The rate of change is: (travel_matrix @ compartment) - compartment
        # This conserves total population since columns sum to 1
        
        # Apply travel flow (this should conserve population)
        dSdt = dSdt + (self.travel_matrix @ S - S)
        dIdt = dIdt + (self.travel_matrix @ I - I)
        dRdt = dRdt + (self.travel_matrix @ R - R)
        
        return np.stack([dSdt, dIdt, dRdt])
