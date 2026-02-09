import jax.numpy as np
import logging
import numpy as onp
from compartment.helpers import setup_logging
from compartment.model import Model

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class Dengue2StrainModel(Model):
    def __init__(self, config):
        # same as covid jax model
        self.population_matrix = np.array(config["initial_population"])
        self.compartment_list = config["compartment_list"]
        self.start_date = config["start_date"]
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = config["time_steps"]
        # for Jax replace travel matrix diag with 1.0
        self.travel_matrix = np.fill_diagonal(np.array(config["travel_matrix"]), 1.0, inplace=False)
        self.admin_units = config["admin_units"]
        self.sigma = config['travel_volume']['leaving'] 

        # Extra
        self.payload = config

        # Dengue 2-strain parameters
        self.beta_0 = .03
        self.eta = .2
        self.omega=0.5*np.pi/365
        self.rho= 1./pow(10,5)
        self.epsilon = 1.5
        self.phi= 0.
        self.gamma= 0.02
        self.mu= 1/(65*365)
        self.b = self.mu
        self.alpha = 1/(2*365)

    @property
    def disease_type(self):
        return "VECTOR_BORNE_2STRAIN"

    def _add_cumulative_compartments(self):
        # stack compartment totals
        I_total = R1_total = S2_total = I2_total = R2_total = onp.zeros(len(self.admin_units))
        self.population_matrix = onp.vstack((self.population_matrix, I_total, R1_total, S2_total, I2_total, R2_total))
        self.compartment_list = self.compartment_list + ['I_total', 'R1_total', 'S2_total', 'I2_total', 'R2_total']

    def prepare_initial_state(self):
        # For jax dengue we are switching from rows being regions to columns being regions
        self.population_matrix = self.population_matrix.T
        self._add_cumulative_compartments()
        
        return self.population_matrix , self.compartment_list

    def derivative(self, y, t, p):

        # Unpack parameters
        # Clip to avoid infs
        y = np.clip(y, 0.0, 1e9) # clip to avoid infs

        # Unpack state variables
        S, I1, I2, R1, R2, S1, S2, I12, I21, R = y
        
        # Unpack parameters
        beta_0,eta,omega,rho,phi,epsilon,gamma,mu,b,alpha,N = p
        
        # Seasonality
        beta = beta_0 * (1 + eta * np.cos(omega * (t + phi)))

        # Calculate derivatives
        dS_dt = -beta/N * S * (I1 + I2 + rho*N + epsilon*I21 + eta*I12) + b*N - mu*S
        
        dI1_dt = beta/N * S * (I1 + rho*N + epsilon*I21) - (gamma + mu) * I1
        
        dI2_dt = beta/N * S * (I2 + rho*N + epsilon*I12) - (gamma + mu) * I2
        
        I_total = beta/N * S * (I1 + I2 + 2*(rho*N) + epsilon*(I21 + I12))

        dR1_dt = gamma * I1 - (alpha + mu) * R1
        
        dR2_dt = gamma * I2 - (alpha + mu) * R2
        
        R1_total = gamma * (I1 + I2)

        dS1_dt = -beta/N * S1 * (I2 + rho*N + epsilon*I12) + alpha*R1 - mu*S1
        
        dS2_dt = -beta/N * S2 * (I1 + rho*N + epsilon*I21) + alpha*R2 - mu*S2
        
        S2_total = -beta/N * S1 * (I2 + rho*N + epsilon*I12) + alpha*R1 + -beta/N * S2 * (I1 + rho*N + epsilon*I21) + alpha*R2
        
        dI12_dt = beta/N * S1 * (I2 + rho*N + epsilon*I12) - (gamma + mu) * I12
        
        dI21_dt = beta/N * S2 * (I1 + rho*N + epsilon*I21) - (gamma + mu) * I21
        
        I2_total = beta/N * (S1 * (I2 + rho*N + epsilon*I12) + S2 * (I1 + rho*N + epsilon*I21))

        dR_dt = gamma * (I12 + I21) - mu * R

        R2_total = gamma * (I12 + I21)
        
        return np.stack([dS_dt, dI1_dt, dI2_dt, I_total, dR1_dt, dR2_dt, R1_total, dS1_dt, dS2_dt, S2_total, dI12_dt, dI21_dt, I2_total, dR_dt, R2_total])