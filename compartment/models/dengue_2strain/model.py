import jax.numpy as np
import logging
import numpy as onp
from compartment.helpers import setup_logging
from compartment.model import Model

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class Dengue2StrainModel(Model):
    
    # Fixed compartment structure for 2-strain dengue model
    COMPARTMENT_LIST = ['S', 'I1', 'I2', 'R1', 'R2', 'S1', 'S2', 'I12', 'I21', 'R']
    
    def __init__(self, config):
        self.population_matrix = np.array(config["initial_population"])
        self.compartment_list = self.COMPARTMENT_LIST  # Use class attribute
        self.start_date = config["start_date"]
        self.n_timesteps = config["time_steps"]
        self.admin_units = config["admin_units"]
        self.payload = config

        # Dengue 2-strain parameters
        self.beta_0 = 0.03
        self.eta = 0.2
        self.omega = 0.5 * np.pi / 365
        self.rho = 1.0 / pow(10, 5)
        self.epsilon = 1.5
        self.phi = 0.0
        self.gamma = 0.02
        self.mu = 1 / (65 * 365)
        self.b = self.mu
        self.alpha = 1/(2*365)
    
    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        import numpy as onp
        column_mapping = {value: index for index, value in enumerate(compartment_list)}
        initial_population = onp.zeros((len(admin_zones), len(compartment_list)))

        for i, zone in enumerate(admin_zones):
            population = zone['population']
            seroprevalence = zone.get('seroprevalence', 0) or 0
            infected_population = zone.get('infected_population', 0) or 0

            # Split infected_population 50/50 between I1 and I2
            I1 = round(infected_population / 200 * population, 2)
            I2 = round(infected_population / 200 * population, 2)
            
            # Split seroprevalence 50/50 between S1 and S2
            S1 = round(seroprevalence / 200 * population, 2)
            S2 = round(seroprevalence / 200 * population, 2)
            
            # Rest goes to S (susceptible)
            S = population - I1 - I2 - S1 - S2
            
            # Assign to compartments
            initial_population[i, column_mapping['S']] = S
            initial_population[i, column_mapping['I1']] = I1
            initial_population[i, column_mapping['I2']] = I2
            initial_population[i, column_mapping['S1']] = S1
            initial_population[i, column_mapping['S2']] = S2
            # All other compartments (R1, R2, I12, I21, R) start at 0

        return initial_population

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
    
    def get_params(self):
        return np.array([self.beta_0, self.eta, self.omega, self.rho, self.phi, self.epsilon, self.gamma, self.mu, self.b, self.alpha])

    def derivative(self, y, t, p):
        y = np.clip(y, 0.0, 1e9) # clip to avoid infs

        # Unpack state variables
        error_val = 1e-6
        S, I1, I2, R1, R2, S1, S2, I12, I21, R, I_total, R1_total, S2_total, I2_total, R2_total = y
        N = error_val + sum([S, I1, I2, R1, R2, S1, S2, I12, I21, R])
        
        # Unpack parameters
        beta_0,eta,omega,rho,phi,epsilon,gamma,mu,b,alpha = p
        
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