import jax.numpy as np
import logging
import numpy as onp
from compartment.helpers import setup_logging
from compartment.model import Model
from compartment.parameters import ValueType
from compartment.interventions import jax_prop_intervention, jax_timestep_intervention
from compartment.temperature import (
    temperature_seasonality_jax,
    calculate_thermal_responses,
    calculate_surviving_offspring,
    get_carrying_capacity,
)

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


class DengueJaxModel(Model):
    """4-serotype dengue vector-borne model with temperature-driven dynamics.

    Compartments are hardcoded (56 core + 8 cumulative) because the
    4-serotype × 2-infection structure is fixed.  Users configure
    transmission edges, interventions, and the immunity period.
    """

    DISEASE_TYPE = "VECTOR_BORNE"

    @classmethod
    def _add_total_compartments(cls, schema):
        # Suppress framework auto-generation of per-edge _total
        # compartments.  Dengue declares its own aggregate totals
        # (E_total, I_total, etc.) covering all serotypes.
        pass

    # Output grouping: collapse serotype detail for display
    COMPARTMENT_DELTA_GROUPING = {
        "SV": ["SV"],
        "EV": ["EV1", "EV2", "EV3", "EV4"],
        "IV": ["IV1", "IV2", "IV3", "IV4"],
        "S": ["S0"],
        "E": ["E1", "E2", "E3", "E4"],
        "I": ["I1", "I2", "I3", "I4"],
        "C": ["C1", "C2", "C3", "C4"],
        "Snot": ["Snot1", "Snot2", "Snot3", "Snot4"],
        "E2": [
            "E12", "E13", "E14", "E21", "E23", "E24",
            "E31", "E32", "E34", "E41", "E42", "E43",
        ],
        "I2": [
            "I12", "I13", "I14", "I21", "I23", "I24",
            "I31", "I32", "I34", "I41", "I42", "I43",
        ],
        "H": ["H1", "H2", "H3", "H4"],
        "R": ["R1", "R2", "R3", "R4"],
    }

    # ------------------------------------------------------------------
    # Declarative parameter schema
    # ------------------------------------------------------------------

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="VECTOR_BORNE",
            label="Dengue (4-serotype)",
            description="A 4-serotype vector-borne dengue model with temperature-driven mosquito dynamics",
        )

        # ---- Vector compartments ----
        schema.add_compartment("SV", "Susceptible Vectors", "Susceptible mosquito population")
        for s in range(1, 5):
            schema.add_compartment(
                f"EV{s}", f"Exposed Vectors (serotype {s})",
                f"Mosquitoes exposed to serotype {s}",
            )
        for s in range(1, 5):
            schema.add_compartment(
                f"IV{s}", f"Infectious Vectors (serotype {s})",
                f"Mosquitoes infectious with serotype {s}",
            )

        # ---- Human primary infection compartments ----
        schema.add_compartment("S0", "Susceptible", "Fully susceptible human population")
        for s in range(1, 5):
            schema.add_compartment(
                f"E{s}", f"Exposed (serotype {s})",
                f"Humans exposed to serotype {s} (primary infection)",
            )
        for s in range(1, 5):
            schema.add_compartment(
                f"I{s}", f"Infected (serotype {s})",
                f"Humans infected with serotype {s} (primary infection)",
                infective=True,
            )

        # ---- Cross-protection compartments ----
        for s in range(1, 5):
            schema.add_compartment(
                f"C{s}", f"Cross-protected (serotype {s})",
                f"Recovered from serotype {s}, temporarily cross-protected",
            )
        for s in range(1, 5):
            schema.add_compartment(
                f"Snot{s}", f"Susceptible (not {s})",
                f"Immune to serotype {s}, susceptible to other serotypes",
            )

        # ---- Human secondary infection compartments ----
        serotype_pairs = [
            (i, j) for i in range(1, 5) for j in range(1, 5) if i != j
        ]
        for i, j in serotype_pairs:
            schema.add_compartment(
                f"E{i}{j}", f"Exposed 2nd ({i}→{j})",
                f"Previously infected with serotype {i}, now exposed to serotype {j}",
            )
        for i, j in serotype_pairs:
            schema.add_compartment(
                f"I{i}{j}", f"Infected 2nd ({i}→{j})",
                f"Secondary infection: previously serotype {i}, now serotype {j}",
                infective=True,
            )

        # ---- Hospitalized & Recovered ----
        for s in range(1, 5):
            schema.add_compartment(
                f"H{s}", f"Hospitalized (serotype {s})",
                f"Hospitalized from secondary infection leading to serotype {s}",
            )
        for s in range(1, 5):
            schema.add_compartment(
                f"R{s}", f"Recovered (serotype {s})",
                f"Recovered from secondary infection with serotype {s}",
            )

        # ---- Cumulative tracking (aggregated across serotypes) ----
        schema.add_compartment("E_total", "Exposed Total", "Cumulative primary exposures")
        schema.add_compartment("I_total", "Infected Total", "Cumulative primary infections")
        schema.add_compartment("C_total", "Cross-protected Total", "Cumulative cross-protected")
        schema.add_compartment("Snot_total", "Partially Susceptible Total", "Cumulative partially susceptible")
        schema.add_compartment("E2_total", "Exposed 2nd Total", "Cumulative secondary exposures")
        schema.add_compartment("I2_total", "Infected 2nd Total", "Cumulative secondary infections")
        schema.add_compartment("H_total", "Hospitalized Total", "Cumulative hospitalizations")
        schema.add_compartment("R_total", "Recovered Total", "Cumulative recoveries")

        # ---- Disease parameters ----
        schema.add_disease_parameter(
            # FOR FUTURE IMPLEMENTATION
            name="num_serotypes",
            label="Number of Serotypes",
            description="Number of dengue serotypes modelled.",
            value_type=ValueType.INTEGER,
            default=4,
            default_min=4,
            default_max=4,
            min_value=4,
            max_value=4,
        )
        schema.add_disease_parameter(
            name="immunity_period",
            label="Cross-Immunity Period",
            description="Duration of cross-serotype immunity after primary infection (days). 0 = no cross-immunity.",
            value_type=ValueType.INTEGER,
            default=240,
            default_min=90,
            default_max=365,
            min_value=0,
            max_value=730,
            unit="days",
        )
        schema.add_disease_parameter(
            name="latent_period",
            label="Host Latent Period",
            description="Mean duration of the exposed (E) period before becoming infectious (days).",
            value_type=ValueType.FLOAT,
            default=5.9,
            default_min=3.0,
            default_max=14.0,
            min_value=1.0,
            max_value=30.0,
            unit="days",
        )
        schema.add_disease_parameter(
            name="infectious_period",
            label="Host Infectious Period",
            description="Mean duration of the infectious (I) period before recovery (days).",
            value_type=ValueType.FLOAT,
            default=5.0,
            default_min=2.0,
            default_max=10.0,
            min_value=1.0,
            max_value=30.0,
            unit="days",
        )
        schema.add_disease_parameter(
            name="hospitalization_rate",
            label="Hospitalization Rate",
            description="Proportion of secondary infections that progress to hospitalization.",
            value_type=ValueType.FLOAT,
            default=0.01,
            default_min=0.001,
            default_max=0.1,
            min_value=0.0,
            max_value=1.0,
        )
        schema.add_disease_parameter(
            name="hospital_stay",
            label="Hospital Stay Duration",
            description="Mean duration of hospitalization before recovery (days).",
            value_type=ValueType.FLOAT,
            default=4.9,
            default_min=2.0,
            default_max=14.0,
            min_value=1.0,
            max_value=30.0,
            unit="days",
        )
        schema.add_disease_parameter(
            name="vector_activation_energy",
            label="Vector Activation Energy",
            description="Arrhenius thermal activation energy for vector mortality rate.",
            value_type=ValueType.FLOAT,
            default=0.05,
            default_min=0.01,
            default_max=0.2,
            min_value=0.001,
            max_value=1.0,
        )
        schema.add_disease_parameter(
            name="boltzmann_constant",
            label="Boltzmann-Arrhenius Constant",
            description="Boltzmann constant used in the thermal response model (eV/K).",
            value_type=ValueType.FLOAT,
            default=8.617e-5,
            default_min=8.617e-5,
            default_max=8.617e-5,
            min_value=1e-6,
            max_value=1e-3,
            unit="eV/K",
        )
        schema.add_disease_parameter(
            name="max_vector_capacity",
            label="Maximum Vector Carrying Capacity",
            description="Maximum mosquito carrying capacity as a multiple of the human population.",
            value_type=ValueType.FLOAT,
            default=1.5,
            default_min=0.5,
            default_max=5.0,
            min_value=0.1,
            max_value=20.0,
        )
        schema.add_disease_parameter(
            name="reference_temperature",
            label="Reference Temperature",
            description="Baseline temperature for the Arrhenius thermal response model (°C).",
            value_type=ValueType.FLOAT,
            default=29.0,
            default_min=25.0,
            default_max=32.0,
            min_value=15.0,
            max_value=40.0,
            unit="°C",
        )
        schema.add_disease_parameter(
            name="vector_seed",
            label="Vector Seed (kappa)",
            description="Small constant added to each infectious vector compartment to prevent numerical extinction.",
            value_type=ValueType.FLOAT,
            default=1e-5,
            default_min=1e-7,
            default_max=1e-3,
            min_value=0.0,
            max_value=0.01,
        )

        # ---- Admin zone fields (dengue-specific) ----
        schema.add_admin_zone_field(
            name="seroprevalence",
            label="Seroprevalence",
            description="Percentage of population with prior dengue exposure",
            value_type=ValueType.PERCENTAGE,
            default=30.0,
            min_value=0.0,
            max_value=100.0,
            unit="%",
        )
        schema.add_admin_zone_field(
            name="temp_min",
            label="Min Temperature",
            description="Minimum annual temperature (°C)",
            value_type=ValueType.FLOAT,
            default=15.0,
            min_value=-10.0,
            max_value=45.0,
            unit="°C",
        )
        schema.add_admin_zone_field(
            name="temp_max",
            label="Max Temperature",
            description="Maximum annual temperature (°C)",
            value_type=ValueType.FLOAT,
            default=30.0,
            min_value=-10.0,
            max_value=50.0,
            unit="°C",
        )
        schema.add_admin_zone_field(
            name="temp_mean",
            label="Mean Temperature",
            description="Mean annual temperature (°C)",
            value_type=ValueType.FLOAT,
            default=25.0,
            min_value=-10.0,
            max_value=45.0,
            unit="°C",
        )

        # ---- Interventions ----
        schema.add_intervention(
            id="physical",
            label="Bite Reduction",
            description="Physical measures to reduce mosquito biting rate (bed nets, repellent)",
            adherence=50.0,
            transmission_reduction=50.0,
        )

        schema.add_intervention(
            id="chemical",
            label="Vector Control",
            description="Chemical vector control reducing mosquito survival (spraying, larvicide)",
            adherence=50.0,
            transmission_reduction=50.0,
        )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def __init__(self, config):
        self.population_matrix = np.array(config["initial_population"])
        self.compartment_list = list(self.COMPARTMENTS)

        self.start_date = config["start_date"]
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = config["time_steps"]
        self.admin_units = config["admin_units"]

        # Travel
        self.travel_matrix = np.fill_diagonal(
            np.array(config["travel_matrix"]), 1.0, inplace=False
        )
        self.sigma = config["travel_volume"]["leaving"]

        # Demographics
        self.demographics = config["case_file"]["demographics"]
        self.age_stratification = list(self.demographics.values())
        self.age_groups = list(self.demographics.keys())

        # Temperature
        self.hemisphere = config["hemisphere"]
        self.temp_min = config["temperature"]["temp_min"]
        self.temp_max = config["temperature"]["temp_max"]
        self.temp_average = config["temperature"]["temp_mean"]

        # Interventions
        self.intervention_dict = config.get("intervention_dict", {})
        self.intervention_statuses = {"physical": False, "chemical": False}

        self.payload = config

        # ---- Dengue parameters ----
        disease_cfg = config.get("Disease", {})

        def _get(key, default):
            v = disease_cfg.get(key, default)
            return default if v is None else v

        immunity = _get("immunity_period", 240)
        self.zeta_cross = 1.0 / immunity if immunity > 0 else 0.0

        self.delta_H = 1.0 / _get("latent_period", 5.9)
        self.eta_recovery = 1.0 / _get("infectious_period", 5.0)
        self.theta_hosp = _get("hospitalization_rate", 0.01)
        self.omega_hosp = 1.0 / _get("hospital_stay", 4.9)
        self.E_a = _get("vector_activation_energy", 0.05)
        self.k = _get("boltzmann_constant", 8.617e-5)
        self.N_v_m = _get("max_vector_capacity", 1.5)
        self.T_0 = _get("reference_temperature", 29.0)
        self.kappa = _get("vector_seed", 1e-5)

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        column_mapping = {v: i for i, v in enumerate(compartment_list)}
        initial_population = onp.zeros((len(admin_zones), len(compartment_list)))

        for i, zone in enumerate(admin_zones):
            # Mosquito population - starts at 0, dynamics handled in model
            initial_population[i, column_mapping["SV"]] = 0

            population = zone["population"]
            seroprevalence = zone.get("seroprevalence", 0) or 0
            infected_population = zone.get("infected_population", 0) or 0

            # Equal distribution across 4 serotypes
            serotype_weights = [0.25] * 4

            # Distribute seroprevalence across Snot1-4
            total_snot = 0
            for idx, weight in enumerate(serotype_weights, 1):
                snot_pop = round(weight * seroprevalence / 100 * population, 2)
                initial_population[i, column_mapping[f"Snot{idx}"]] = snot_pop
                total_snot += snot_pop

            # Distribute infected across I1-4
            infected_assigned = 0
            for idx, weight in enumerate(serotype_weights, 1):
                i_pop = round(weight * infected_population / 100 * population, 2)
                initial_population[i, column_mapping[f"I{idx}"]] = i_pop
                infected_assigned += i_pop

            # Remaining → fully susceptible
            initial_population[i, column_mapping["S0"]] = (
                population - total_snot - infected_assigned
            )

        return initial_population

    def get_params(self):
        """Pack temperature/calendar params for the ODE solver."""
        numeric_day = int(self.start_date.strftime("%j"))
        hemisphere_flag = 0 if self.hemisphere == "North" else 1
        return (
            self.temp_max,
            self.temp_min,
            self.temp_average,
            numeric_day,
            hemisphere_flag,
        )

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    def prepare_initial_state(self):
        # Transpose: rows=compartments, columns=regions
        self.population_matrix = self.population_matrix.T
        return self.population_matrix, self.compartment_list

    # ------------------------------------------------------------------
    # ODE derivative
    # ------------------------------------------------------------------

    def derivative(self, y, t, p):
        y = np.clip(y, 0.0, 1e9)
        Tmax, Tmin, Tmean, numeric_day, hemisphere = p

        current_day = numeric_day + t
        n_regions = self.travel_matrix.shape[0]

        # ---- Unpack all compartments ----
        error_val = 1e-6
        (
            SV, EV1, EV2, EV3, EV4, IV1, IV2, IV3, IV4,
            S0, E1, E2, E3, E4, I1, I2, I3, I4,
            C1, C2, C3, C4, Snot1, Snot2, Snot3, Snot4,
            E12, E13, E14, E21, E23, E24, E31, E32, E34, E41, E42, E43,
            I12, I13, I14, I21, I23, I24, I31, I32, I34, I41, I42, I43,
            H1, H2, H3, H4, R1, R2, R3, R4,
            E_total, I_total, C_total, Snot_total,
            E2_total, I2_total, H_total, R_total,
        ) = y

        NV = error_val + sum([SV, EV1, EV2, EV3, EV4, IV1, IV2, IV3, IV4])
        N = error_val + sum([
            S0, E1, E2, E3, E4, I1, I2, I3, I4,
            C1, C2, C3, C4, Snot1, Snot2, Snot3, Snot4,
            E12, E13, E14, E21, E23, E24, E31, E32, E34, E41, E42, E43,
            I12, I13, I14, I21, I23, I24, I31, I32, I34, I41, I42, I43,
            H1, H2, H3, H4, R1, R2, R3, R4,
        ])

        mu = 1 / (73 * 365)
        births = mu * N

        # ---- Temperature-driven vector parameters ----
        temperature = temperature_seasonality_jax(
            Tmax, Tmin, Tmean, current_day, hemisphere
        )
        (l_V_T, s_V_T, d_V_T, epsilon_V_T,
         delta_V_T, gamma_T, b_V_T, mu_V_T) = calculate_thermal_responses(temperature)
        vector_surviving_offspring, mu_V_T0 = calculate_surviving_offspring()
        carrying_capacity = get_carrying_capacity(
            temperature, vector_surviving_offspring, mu_V_T0,
            self.N_v_m, self.E_a, N, T0=self.T_0, k=self.k,
        )

        I_H_total = sum([
            I1, I2, I3, I4,
            I12, I13, I14, I21, I23, I24,
            I31, I32, I34, I41, I42, I43,
        ])
        I_H_prop_infected = np.sum(I_H_total) / np.sum(N)

        # ---- Interventions (shared intervention logic) ----
        current_ordinal_day = self.start_date_ordinal + t
        rates = {"b_V_T": b_V_T, "s_V_T": s_V_T}
        rates, self.intervention_statuses, contact_matrix = jax_timestep_intervention(
            self.intervention_dict, current_ordinal_day, rates,
            self.intervention_statuses, self.travel_matrix,
        )
        rates, self.intervention_statuses, contact_matrix = jax_prop_intervention(
            self.intervention_dict, I_H_prop_infected, rates,
            self.intervention_statuses, self.travel_matrix,
        )
        b_V_T = rates["b_V_T"]
        s_V_T = rates["s_V_T"]

        # Broadcast scalars to region arrays
        l_V_T = np.repeat(l_V_T, n_regions)
        d_V_T = np.repeat(d_V_T, n_regions)
        epsilon_V_T = np.repeat(epsilon_V_T, n_regions)
        mu_V_T = np.repeat(mu_V_T, n_regions)

        # ---- Force of infection (host) ----
        transmission_rate_H = b_V_T * gamma_T
        BETA_H = transmission_rate_H * contact_matrix
        I_V_total = sum([IV1, IV2, IV3, IV4])
        LAMBDA_H_all = BETA_H @ (I_V_total / N)
        LAMBDA_H_1 = BETA_H @ (IV1 / N)
        LAMBDA_H_2 = BETA_H @ (IV2 / N)
        LAMBDA_H_3 = BETA_H @ (IV3 / N)
        LAMBDA_H_4 = BETA_H @ (IV4 / N)
        LAMBDA_H_Snot1 = BETA_H @ (sum([IV2, IV3, IV4]) / N)
        LAMBDA_H_Snot2 = BETA_H @ (sum([IV1, IV3, IV4]) / N)
        LAMBDA_H_Snot3 = BETA_H @ (sum([IV1, IV2, IV4]) / N)
        LAMBDA_H_Snot4 = BETA_H @ (sum([IV1, IV2, IV3]) / N)

        # ---- Force of infection (vector) ----
        transmission_rate_V = b_V_T * delta_V_T
        BETA_V = transmission_rate_V * contact_matrix
        LAMBDA_V_all = BETA_V @ (I_H_total / N)
        LAMBDA_V_1 = BETA_V @ (sum([I1, I21, I31, I41]) / N)
        LAMBDA_V_2 = BETA_V @ (sum([I2, I12, I32, I42]) / N)
        LAMBDA_V_3 = BETA_V @ (sum([I3, I13, I23, I43]) / N)
        LAMBDA_V_4 = BETA_V @ (sum([I4, I14, I24, I34]) / N)

        # ---- Local aliases for model parameters ----
        delta_H = self.delta_H
        eta = self.eta_recovery
        zeta = self.zeta_cross
        theta = self.theta_hosp
        omega = self.omega_hosp

        # ---- Vector ODEs ----
        SV_dot = (
            (l_V_T * s_V_T * d_V_T) / mu_V_T
        ) * NV * (1 - NV / carrying_capacity) - SV * LAMBDA_V_all - mu_V_T * SV

        EV1_dot = SV * LAMBDA_V_1 - (epsilon_V_T + mu_V_T) * EV1
        EV2_dot = SV * LAMBDA_V_2 - (epsilon_V_T + mu_V_T) * EV2
        EV3_dot = SV * LAMBDA_V_3 - (epsilon_V_T + mu_V_T) * EV3
        EV4_dot = SV * LAMBDA_V_4 - (epsilon_V_T + mu_V_T) * EV4

        IV1_dot = epsilon_V_T * EV1 - mu_V_T * IV1 + self.kappa
        IV2_dot = epsilon_V_T * EV2 - mu_V_T * IV2 + self.kappa
        IV3_dot = epsilon_V_T * EV3 - mu_V_T * IV3 + self.kappa
        IV4_dot = epsilon_V_T * EV4 - mu_V_T * IV4 + self.kappa

        # ---- Host primary infection ODEs ----
        S0_dot = -S0 * LAMBDA_H_all + births - mu * S0

        E1_dot = S0 * LAMBDA_H_1 - delta_H * E1 - mu * E1
        E2_dot = S0 * LAMBDA_H_2 - delta_H * E2 - mu * E2
        E3_dot = S0 * LAMBDA_H_3 - delta_H * E3 - mu * E3
        E4_dot = S0 * LAMBDA_H_4 - delta_H * E4 - mu * E4
        E_total_dot = S0 * (LAMBDA_H_1 + LAMBDA_H_2 + LAMBDA_H_3 + LAMBDA_H_4)

        I1_dot = delta_H * E1 - eta * I1 - mu * I1
        I2_dot = delta_H * E2 - eta * I2 - mu * I2
        I3_dot = delta_H * E3 - eta * I3 - mu * I3
        I4_dot = delta_H * E4 - eta * I4 - mu * I4
        I_total_dot = delta_H * (E1 + E2 + E3 + E4)

        C1_dot = eta * I1 - zeta * C1 - mu * C1
        C2_dot = eta * I2 - zeta * C2 - mu * C2
        C3_dot = eta * I3 - zeta * C3 - mu * C3
        C4_dot = eta * I4 - zeta * C4 - mu * C4
        C_total_dot = eta * (I1 + I2 + I3 + I4)

        Snot1_dot = zeta * C1 - Snot1 * LAMBDA_H_Snot1 - mu * Snot1
        Snot2_dot = zeta * C2 - Snot2 * LAMBDA_H_Snot2 - mu * Snot2
        Snot3_dot = zeta * C3 - Snot3 * LAMBDA_H_Snot3 - mu * Snot3
        Snot4_dot = zeta * C4 - Snot4 * LAMBDA_H_Snot4 - mu * Snot4
        Snot_total_dot = zeta * (C1 + C2 + C3 + C4)

        # ---- Host secondary infection ODEs ----
        E12_dot = Snot1 * LAMBDA_H_2 - delta_H * E12 - mu * E12
        E13_dot = Snot1 * LAMBDA_H_3 - delta_H * E13 - mu * E13
        E14_dot = Snot1 * LAMBDA_H_4 - delta_H * E14 - mu * E14

        E21_dot = Snot2 * LAMBDA_H_1 - delta_H * E21 - mu * E21
        E23_dot = Snot2 * LAMBDA_H_3 - delta_H * E23 - mu * E23
        E24_dot = Snot2 * LAMBDA_H_4 - delta_H * E24 - mu * E24

        E31_dot = Snot3 * LAMBDA_H_1 - delta_H * E31 - mu * E31
        E32_dot = Snot3 * LAMBDA_H_2 - delta_H * E32 - mu * E32
        E34_dot = Snot3 * LAMBDA_H_4 - delta_H * E34 - mu * E34

        E41_dot = Snot4 * LAMBDA_H_1 - delta_H * E41 - mu * E41
        E42_dot = Snot4 * LAMBDA_H_2 - delta_H * E42 - mu * E42
        E43_dot = Snot4 * LAMBDA_H_3 - delta_H * E43 - mu * E43

        E2_total_dot = (
            Snot1 * (LAMBDA_H_2 + LAMBDA_H_3 + LAMBDA_H_4)
            + Snot2 * (LAMBDA_H_1 + LAMBDA_H_3 + LAMBDA_H_4)
            + Snot3 * (LAMBDA_H_1 + LAMBDA_H_2 + LAMBDA_H_4)
            + Snot4 * (LAMBDA_H_1 + LAMBDA_H_2 + LAMBDA_H_3)
        )

        I12_dot = delta_H * E12 - (theta + eta) * I12 - mu * I12
        I13_dot = delta_H * E13 - (theta + eta) * I13 - mu * I13
        I14_dot = delta_H * E14 - (theta + eta) * I14 - mu * I14

        I21_dot = delta_H * E21 - (theta + eta) * I21 - mu * I21
        I23_dot = delta_H * E23 - (theta + eta) * I23 - mu * I23
        I24_dot = delta_H * E24 - (theta + eta) * I24 - mu * I24

        I31_dot = delta_H * E31 - (theta + eta) * I31 - mu * I31
        I32_dot = delta_H * E32 - (theta + eta) * I32 - mu * I32
        I34_dot = delta_H * E34 - (theta + eta) * I34 - mu * I34

        I41_dot = delta_H * E41 - (theta + eta) * I41 - mu * I41
        I42_dot = delta_H * E42 - (theta + eta) * I42 - mu * I42
        I43_dot = delta_H * E43 - (theta + eta) * I43 - mu * I43
        I2_total_dot = delta_H * (
            E12 + E13 + E14 + E21 + E23 + E24
            + E31 + E32 + E34 + E41 + E42 + E43
        )

        # ---- Hospitalized & Recovered ----
        H1_dot = theta * (I21 + I31 + I41) - omega * H1 - mu * H1
        H2_dot = theta * (I12 + I32 + I42) - omega * H2 - mu * H2
        H3_dot = theta * (I13 + I23 + I43) - omega * H3 - mu * H3
        H4_dot = theta * (I14 + I24 + I34) - omega * H4 - mu * H4
        H_total_dot = theta * (
            I21 + I31 + I41 + I12 + I32 + I42
            + I13 + I23 + I43 + I14 + I24 + I34
        )

        R1_dot = eta * (I21 + I31 + I41) + omega * H1 - mu * R1
        R2_dot = eta * (I12 + I32 + I42) + omega * H2 - mu * R2
        R3_dot = eta * (I13 + I23 + I43) + omega * H3 - mu * R3
        R4_dot = eta * (I14 + I24 + I34) + omega * H4 - mu * R4
        R_total_dot = (
            eta * (I21 + I31 + I41 + I12 + I32 + I42
                   + I13 + I23 + I43 + I14 + I24 + I34)
            + omega * (H1 + H2 + H3 + H4)
        )

        return np.stack([
            SV_dot, EV1_dot, EV2_dot, EV3_dot, EV4_dot,
            IV1_dot, IV2_dot, IV3_dot, IV4_dot,
            S0_dot, E1_dot, E2_dot, E3_dot, E4_dot,
            I1_dot, I2_dot, I3_dot, I4_dot,
            C1_dot, C2_dot, C3_dot, C4_dot,
            Snot1_dot, Snot2_dot, Snot3_dot, Snot4_dot,
            E12_dot, E13_dot, E14_dot, E21_dot, E23_dot, E24_dot,
            E31_dot, E32_dot, E34_dot, E41_dot, E42_dot, E43_dot,
            I12_dot, I13_dot, I14_dot, I21_dot, I23_dot, I24_dot,
            I31_dot, I32_dot, I34_dot, I41_dot, I42_dot, I43_dot,
            H1_dot, H2_dot, H3_dot, H4_dot,
            R1_dot, R2_dot, R3_dot, R4_dot,
            E_total_dot, I_total_dot, C_total_dot, Snot_total_dot,
            E2_total_dot, I2_total_dot, H_total_dot, R_total_dot,
        ])
