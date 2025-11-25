import jax.numpy as np
import logging
import numpy as onp
from helpers import setup_logging
from datetime import datetime
from interventions import jax_prop_intervention, jax_timestep_intervention
from temperature import temperature_seasonality_jax, calculate_thermal_responses, calculate_surviving_offspring, get_carrying_capacity

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class DengueJaxModel:
    def __init__(self, config, cleaned_config):
        # same as covid jax model
        self.population_matrix = np.array(cleaned_config["initial_population"])
        self.compartment_list = cleaned_config["compartment_list"]
        self.start_date = datetime.strptime(config["data"]["getSimulationJob"]["start_date"], "%Y-%m-%d").date()
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = config["data"]["getSimulationJob"]["time_steps"]
        # for Jax replace travel matrix diag with 1.0
        self.travel_matrix = np.fill_diagonal(np.array(cleaned_config["travel_matrix"]), 1.0, inplace=False)
        self.admin_units = cleaned_config["admin_units"]
        self.demographics = config["data"]["getSimulationJob"]["case_file"]["demographics"]
        self.age_stratification = list(self.demographics.values())
        self.age_groups = list(self.demographics.keys())
        self.disease_type = config['data']['getSimulationJob']['Disease']['disease_type']
        self.sigma = cleaned_config['travel_volume']['leaving'] 

        # Temperature
        self.hemisphere = cleaned_config['hemisphere']
        self.temp_min = cleaned_config['temperature']['temp_min']
        self.temp_max = cleaned_config['temperature']['temp_max']
        self.temp_average = cleaned_config['temperature']['temp_mean']

        # Interventions      
        self.intervention_dict = cleaned_config["intervention_dict"]
        self.intervention_statuses = {
            "physical": False,
            "chemical": False
        }

        # Extra
        self.payload = config
        
        # Rate of loss of cross-protection
        # zeta = 1/immunity_period, default = 1/240, if immunity_period is 0 then zeta = 0 
        self.zeta = (
            1 / config['data']['getSimulationJob']['Disease']['immunity_period']
            if config['data']['getSimulationJob']['Disease'].get('immunity_period') not in [None, 0]
            else 0 if config['data']['getSimulationJob']['Disease'].get('immunity_period') == 0
            else 1 / 240
        )

        # Dengue default parameters
        self.delta_H = 1/5.9    # Host exposed --> infected
        self.eta = 1/5          # Host infected --> recovered
        self.E_a = 0.05         # Activation energy 
        self.k = 8.617e-5       # Boltzmann-Arrhenius constant 
        self.N_v_m = 1.5        # Maximum carrying capacity
        self.T_0 = 29           # Reference temperature where carrying capacity is greatest
        self.kappa = 1e-5      # Helper population for mosquitos 
        self.theta = 0.01       # hospitalized --> infected. This value is from pg. 4 of Paz-Bailey et al.
        self.omega = 1/4.9      # hospitalized --> recovered

    def get_params(self):
        """ Get params tuple for ODE solver """
        numeric_day = int(self.start_date.strftime("%j"))
        hemisphere_flag = 0 if self.hemisphere == "North" else 1
        return (
            self.temp_max, 
            self.temp_min,
            self.temp_average, 
            numeric_day,
            hemisphere_flag
        )
    
    def _add_cumulative_compartments(self):
        # stack compartment totals
        E_total = I_total = C_total = Snot_total = E2_total = I2_total = H_total = R_total = onp.zeros(len(self.admin_units))
        self.population_matrix = onp.vstack((self.population_matrix, E_total, I_total, C_total, Snot_total, E2_total, I2_total, H_total, R_total))
        self.compartment_list = self.compartment_list + ['E_total', 'I_total', 'C_total', 'Snot_total', 'E2_total', 'I2_total', 'H_total', 'R_total']

    def prepare_initial_state(self):
        # For jax dengue we are switching from rows being regions to columns being regions
        self.population_matrix = self.population_matrix.T
        self._add_cumulative_compartments()
        
        return self.population_matrix , self.compartment_list

    def derivative(self, y, t, p):
        y = np.clip(y, 0.0, 1e9) # clip to avoid infs
        Tmax, Tmin, Tmean, numeric_day, hemisphere = p

        current_day = numeric_day + t
        current_ordinal_day = self.start_date_ordinal + t
        n_regions = self.travel_matrix.shape[0]

        #Compartments
        error_val = 1e-6
        SV, EV1, EV2, EV3, EV4, IV1, IV2, IV3, IV4, S0, E1, E2, E3, E4, I1, I2, I3, I4, C1, C2, C3, C4, Snot1, Snot2, Snot3, Snot4, E12, E13, E14, E21, E23, E24, E31, E32, E34, E41, E42, E43, I12, I13, I14, I21, I23, I24, I31, I32, I34, I41, I42, I43, H1, H2, H3, H4, R1, R2, R3, R4, E_total, I_total, C_total, Snot_total, E2_total, I2_total, H_total, R_total = y 
        #Sum of vectors and sum of people
        NV = error_val + sum([SV, EV1, EV2, EV3, EV4, IV1, IV2, IV3, IV4])
        N = error_val + sum([S0, E1, E2, E3, E4, I1, I2, I3, I4, C1, C2, C3, C4, Snot1, Snot2, Snot3, Snot4, E12, E13, E14, E21, E23, E24, E31, E32, E34, E41, E42, E43, I12, I13, I14, I21, I23, I24, I31, I32, I34, I41, I42, I43, H1, H2, H3, H4, R1, R2, R3, R4]) 

        mu = 1 / (73 * 365) # global death rate for 73-year lifespan
        births = mu * N # births per timestep
        
        #Temperature-related functions - these need to be run every timestep
        temperature = temperature_seasonality_jax(Tmax, Tmin, Tmean, current_day, hemisphere)
        l_V_T, s_V_T, d_V_T, epsilon_V_T, delta_V_T, gamma_T, b_V_T, mu_V_T = calculate_thermal_responses(temperature)
        vector_surviving_offspring, mu_V_T0 = calculate_surviving_offspring()
        carrying_capacity = get_carrying_capacity(temperature, vector_surviving_offspring, mu_V_T0, self.N_v_m, self.E_a, N, T0=29, k=self.k)
        I_H_total = sum([I1, I2, I3, I4, I12, I13, I14, I21, I23, I24, I31, I32, I34, I41, I42, I43]) #This is the count of all infective mosquitos, this is used for the FOI for all susceptible humans (S0)
        I_H_prop_infected = np.sum(I_H_total) / np.sum(N) # convert to scalar, proportion of total infected humans

        rates = {"b_V_T": b_V_T, "s_V_T": s_V_T}
        #INTERVENTIONS
        rates, self.intervention_statuses, contact_matrix = jax_timestep_intervention(self.intervention_dict, current_ordinal_day, rates, self.intervention_statuses, self.travel_matrix)
        rates, self.intervention_statuses, contact_matrix = jax_prop_intervention(self.intervention_dict, I_H_prop_infected, rates, self.intervention_statuses, self.travel_matrix)
        #Unpack rates
        b_V_T = rates['b_V_T']
        s_V_T = rates['s_V_T']

        #Converting scalars to arrays
        l_V_T = np.repeat(l_V_T, n_regions)
        d_V_T = np.repeat(d_V_T, n_regions)
        epsilon_V_T = np.repeat(epsilon_V_T, n_regions)
        mu_V_T = np.repeat(mu_V_T, n_regions)

        #FOI for hosts
        transmission_rate_H = b_V_T * gamma_T #In this case, this value is a scalar since temperature is the same across regions
        BETA_H = transmission_rate_H * contact_matrix
        I_V_total = sum([IV1, IV2, IV3, IV4]) #This is the count of all infective mosquitos, this is used for the FOI for all susceptible humans (S0)
        LAMBDA_H_all = BETA_H @ (I_V_total/N) 
        LAMBDA_H_1 = BETA_H @ (IV1/N)    
        LAMBDA_H_2 = BETA_H @ (IV2/N)
        LAMBDA_H_3 = BETA_H @ (IV3/N)
        LAMBDA_H_4 = BETA_H @ (IV4/N)
        LAMBDA_H_Snot1 = BETA_H @ (sum([IV2, IV3, IV4])/N)
        LAMBDA_H_Snot2 = BETA_H @ (sum([IV1, IV3, IV4])/N)
        LAMBDA_H_Snot3 = BETA_H @ (sum([IV1, IV2, IV4])/N)
        LAMBDA_H_Snot4 = BETA_H @ (sum([IV1, IV2, IV3])/N)


        #FOI for vectors
        transmission_rate_V = b_V_T * delta_V_T #In this case, this value is a scalar since temperature is the same across regions
        BETA_V = transmission_rate_V * contact_matrix 
        LAMBDA_V_all = BETA_V @ (I_H_total/N)
        LAMBDA_V_1 = BETA_V @ (sum([I1, I21, I31, I41])/N)
        LAMBDA_V_2 = BETA_V @ (sum([I2, I12, I32, I42])/N)
        LAMBDA_V_3 = BETA_V @ (sum([I3, I13, I23, I43])/N)
        LAMBDA_V_4 = BETA_V @ (sum([I4, I14, I24, I34])/N)
        
        ## vector ode's:
        SV_dot = ((l_V_T * s_V_T * d_V_T) / mu_V_T) * NV * (1 - (NV / carrying_capacity)) - (SV * LAMBDA_V_all) - mu_V_T * SV
        
        #Includes humans who are infected with each serotype on the first infection and on the 2nd infection
        EV1_dot = (SV * LAMBDA_V_1) - (epsilon_V_T + mu_V_T) * EV1
        EV2_dot = (SV * LAMBDA_V_2) - (epsilon_V_T + mu_V_T) * EV2
        EV3_dot = (SV * LAMBDA_V_3) - (epsilon_V_T + mu_V_T) * EV3
        EV4_dot = (SV * LAMBDA_V_4) - (epsilon_V_T + mu_V_T) * EV4
        
        IV1_dot = epsilon_V_T * EV1 - mu_V_T * IV1 + self.kappa
        IV2_dot = epsilon_V_T * EV2 - mu_V_T * IV2 + self.kappa
        IV3_dot = epsilon_V_T * EV3 - mu_V_T * IV3 + self.kappa
        IV4_dot = epsilon_V_T * EV4 - mu_V_T * IV4 + self.kappa

        #host ode'set
        S0_dot = -S0 * LAMBDA_H_all + births - mu * S0

        E1_dot = S0 * LAMBDA_H_1 - self.delta_H * E1 - mu * E1
        E2_dot = S0 * LAMBDA_H_2 - self.delta_H * E2 - mu * E2
        E3_dot = S0 * LAMBDA_H_3 - self.delta_H * E3 - mu * E3
        E4_dot = S0 * LAMBDA_H_4 - self.delta_H * E4 - mu * E4
        E_total_dot = S0 * (LAMBDA_H_1 + LAMBDA_H_2 + LAMBDA_H_3 + LAMBDA_H_4)

        I1_dot = self.delta_H * E1 - self.eta * I1 - mu * I1
        I2_dot = self.delta_H * E2 - self.eta * I2 - mu * I2
        I3_dot = self.delta_H * E3 - self.eta * I3 - mu * I3
        I4_dot = self.delta_H * E4 - self.eta * I4 - mu * I4
        I_total_dot = self.delta_H * (E1 + E2 + E3 + E4)

        C1_dot = self.eta * I1 - self.zeta * C1 - mu * C1
        C2_dot = self.eta * I2 - self.zeta * C2 - mu * C2
        C3_dot = self.eta * I3 - self.zeta * C3 - mu * C3
        C4_dot = self.eta * I4 - self.zeta * C4 - mu * C4
        C_total_dot = self.eta * (I1 + I2 + I3 + I4)

        #People who already got bitten with DENV1, now only susceptible to other serotypes
        Snot1_dot = self.zeta * C1 - Snot1 * LAMBDA_H_Snot1 - mu * Snot1
        Snot2_dot = self.zeta * C2 - Snot2 * LAMBDA_H_Snot2 - mu * Snot2
        Snot3_dot = self.zeta * C3 - Snot3 * LAMBDA_H_Snot3 - mu * Snot3
        Snot4_dot = self.zeta * C4 - Snot4 * LAMBDA_H_Snot4 - mu * Snot4
        Snot_total_dot = self.zeta * (C1 + C2 + C3 + C4)
        
        E12_dot = Snot1 * LAMBDA_H_2 - self.delta_H * E12 - mu * E12
        E13_dot = Snot1 * LAMBDA_H_3 - self.delta_H * E13 - mu * E13
        E14_dot = Snot1 * LAMBDA_H_4 - self.delta_H * E14 - mu * E14
        
        E21_dot = Snot2 * LAMBDA_H_1 - self.delta_H * E21 - mu * E21
        E23_dot = Snot2 * LAMBDA_H_3 - self.delta_H * E23 - mu * E23
        E24_dot = Snot2 * LAMBDA_H_4 - self.delta_H * E24 - mu * E24

        E31_dot = Snot3 * LAMBDA_H_1 - self.delta_H * E31 - mu * E31
        E32_dot = Snot3 * LAMBDA_H_2 - self.delta_H * E32 - mu * E32
        E34_dot = Snot3 * LAMBDA_H_4 - self.delta_H * E34 - mu * E34

        E41_dot = Snot4 * LAMBDA_H_1 - self.delta_H * E41 - mu * E41
        E42_dot = Snot4 * LAMBDA_H_2 - self.delta_H * E42 - mu * E42
        E43_dot = Snot4 * LAMBDA_H_3 - self.delta_H * E43 - mu * E43
        E2_total_dot = (Snot1 * (LAMBDA_H_2 + LAMBDA_H_3 + LAMBDA_H_4)) + \
            (Snot2 * (LAMBDA_H_1 + LAMBDA_H_3 + LAMBDA_H_4)) + \
            (Snot3 * (LAMBDA_H_1 + LAMBDA_H_2 + LAMBDA_H_4)) + \
            (Snot4 * (LAMBDA_H_1 + LAMBDA_H_2 + LAMBDA_H_3))

        I12_dot = self.delta_H * E12 - (self.theta + self.eta) * I12 - mu * I12
        I13_dot = self.delta_H * E13 - (self.theta + self.eta) * I13 - mu * I13
        I14_dot = self.delta_H * E14 - (self.theta + self.eta) * I14 - mu * I14

        I21_dot = self.delta_H * E21 - (self.theta + self.eta) * I21 - mu * I21
        I23_dot = self.delta_H * E23 - (self.theta + self.eta) * I23 - mu * I23
        I24_dot = self.delta_H * E24 - (self.theta + self.eta) * I24 - mu * I24

        I31_dot = self.delta_H * E31 - (self.theta + self.eta) * I31 - mu * I31
        I32_dot = self.delta_H * E32 - (self.theta + self.eta) * I32 - mu * I32
        I34_dot = self.delta_H * E34 - (self.theta + self.eta) * I34 - mu * I34

        I41_dot = self.delta_H * E41 - (self.theta + self.eta) * I41 - mu * I41
        I42_dot = self.delta_H * E42 - (self.theta + self.eta) * I42 - mu * I42
        I43_dot = self.delta_H * E43 - (self.theta + self.eta) * I43 - mu * I43
        I2_total_dot = self.delta_H * (E12 + E13 + E14 + E21 + E23 + E24 + E31 + E32 + E34 + E41 + E42 + E43)

        H1_dot = self.theta * (I21 + I31 + I41) - self.omega * (H1) - mu * H1
        H2_dot = self.theta * (I12 + I32 + I42) - self.omega * (H2) - mu * H2
        H3_dot = self.theta * (I13 + I23 + I43) - self.omega * (H3) - mu * H3
        H4_dot = self.theta * (I14 + I24 + I34) - self.omega * (H4) - mu * H4
        H_total_dot = self.theta * (I21 + I31 + I41 + I12 + I32 + I42 + I13 + I23 + I43 + I14 + I24 + I34)

        #Recovered from 2nd infection since 2nd infection is what causes illness ore recovered from hospitalization
        R1_dot = self.eta * (I21 + I31 + I41) + self.omega * (H1) - mu * R1
        R2_dot = self.eta * (I12 + I32 + I42) + self.omega * (H2) - mu * R2
        R3_dot = self.eta * (I13 + I23 + I43) + self.omega * (H3) - mu * R3
        R4_dot = self.eta * (I14 + I24 + I34) + self.omega * (H4) - mu * R4
        R_total_dot = (self.eta * (I21 + I31 + I41 + I12 + I32 + I42 + I13 + I23 + I43 + I14 + I24 + I34)) + (self.omega * (H1 + H2 + H3 + H4))

        return np.stack([SV_dot, EV1_dot, EV2_dot, EV3_dot, EV4_dot, IV1_dot, IV2_dot, IV3_dot, IV4_dot, S0_dot, E1_dot, E2_dot, E3_dot, E4_dot, I1_dot, I2_dot, 
                        I3_dot, I4_dot, C1_dot, C2_dot, C3_dot, C4_dot, Snot1_dot, Snot2_dot, Snot3_dot, Snot4_dot, E12_dot, E13_dot, E14_dot, E21_dot, E23_dot, 
                        E24_dot, E31_dot, E32_dot, E34_dot, E41_dot, E42_dot, E43_dot, I12_dot, I13_dot, I14_dot, I21_dot, I23_dot, I24_dot, I31_dot, I32_dot, 
                        I34_dot, I41_dot, I42_dot, I43_dot, H1_dot, H2_dot, H3_dot, H4_dot, R1_dot, R2_dot, R3_dot, R4_dot,
                        E_total_dot, I_total_dot, C_total_dot, Snot_total_dot, E2_total_dot, I2_total_dot, H_total_dot, R_total_dot])