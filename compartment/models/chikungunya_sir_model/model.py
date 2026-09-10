import jax.numpy as jnp
import logging

from compartment.model import Model, ValueType


logger = logging.getLogger(__name__)


class ChikungunyaSirModel(Model):
    """
    Deterministic SIR model for chikungunya transmission.

    The model supports two optional interventions:

    1. Non-pharmaceutical intervention (NPI)
       - Gradual reduction in beta.
       - Example: starts on day 17 and reaches an 80% reduction
         on day 28.

    2. Vaccination
       - Restricted to individuals aged 12 years or older.
       - Target coverage: 40%.
       - Vaccine efficacy against infection: 40%.
       - Protection delay: 14 days.
       - Vaccination starts on simulation day 0.

    The vaccination effect follows the original R/odin model logic.
    Vaccination does not introduce an additional epidemiological
    compartment. Its effect is represented through the time-varying
    effective transmission rate.
    """

    DISEASE_TYPE = "chikungunya_sir"

    # =========================================================================
    # Parameter definitions
    # =========================================================================
    @classmethod
    def define_parameters(cls, schema):

        # ---------------------------------------------------------------------
        # Model information
        # ---------------------------------------------------------------------
        schema.set_model_info(
            disease_type="chikungunya_sir",
            label="Chikungunya SIR Model",
            description=(
                "A deterministic SIR model of chikungunya transmission "
                "with optional NPI and vaccination interventions."
            ),
        )

        # ---------------------------------------------------------------------
        # Compartments
        # ---------------------------------------------------------------------
        schema.add_compartment(
            "S",
            "Susceptible",
            "Population susceptible to infection",
        )

        schema.add_compartment(
            "I",
            "Infected",
            "Currently infectious population",
            infective=True,
        )

        schema.add_compartment(
            "R",
            "Recovered",
            "Recovered and immune",
        )

        # ---------------------------------------------------------------------
        # Transmission parameters
        # ---------------------------------------------------------------------
        schema.add_transmission_parameter(
            source="susceptible",
            target="infected",
            variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (S->I)",
            description="Baseline transmission rate before intervention",
            default=0.323,
            default_min=0.323,
            default_max=0.323,
            min_value=0.01,
            max_value=2.0,
            unit="per day",
        )

        schema.add_transmission_parameter(
            source="infected",
            target="recovered",
            variable_name="gamma",
            label="Recovery Period (I->R)",
            description="Average infectious period",
            default=11.0,
            default_min=11.0,
            default_max=11.0,
            min_value=1.0,
            max_value=100.0,
            value_type=ValueType.DAYS,
            unit="days",
        )

        # =====================================================================
        # NPI parameters
        # =====================================================================
        schema.add_parameter(
            name="npi_start_day",
            label="NPI start day",
            description="Simulation day when the NPI begins",
            default=17,
            min_value=0,
            max_value=365,
            value_type=ValueType.INTEGER,
        )

        schema.add_parameter(
            name="npi_end_day",
            label="NPI full-effect day",
            description=(
                "Simulation day when the NPI reaches maximum effect"
            ),
            default=28,
            min_value=0,
            max_value=365,
            value_type=ValueType.INTEGER,
        )

        schema.add_parameter(
            name="npi_max_reduction",
            label="Maximum NPI transmission reduction",
            description="Maximum proportional reduction in beta",
            default=0.80,
            min_value=0.0,
            max_value=1.0,
            value_type=ValueType.RATE,
        )

        # =====================================================================
        # Vaccination parameters
        # =====================================================================

        # Original R:
        # prop_pop_under12 <- 0.13
        schema.add_parameter(
            name="prop_under12",
            label="Population aged under 12",
            description=(
                "Proportion of total population aged under 12 years"
            ),
            default=0.13,
            min_value=0.0,
            max_value=1.0,
            value_type=ValueType.RATE,
        )

        # Original R:
        # coverage = 0.40
        schema.add_parameter(
            name="vaccine_coverage",
            label="Vaccine coverage among age 12+",
            description=(
                "Target vaccination coverage among individuals aged 12+"
            ),
            default=0.40,
            min_value=0.0,
            max_value=1.0,
            value_type=ValueType.RATE,
        )

        # Original R:
        # vaccine_efficacy <- 0.40
        schema.add_parameter(
            name="vaccine_efficacy",
            label="Vaccine efficacy against infection",
            description="Vaccine efficacy against infection",
            default=0.40,
            min_value=0.0,
            max_value=1.0,
            value_type=ValueType.RATE,
        )

        # Original R:
        # vacc_delay <- 14
        schema.add_parameter(
            name="vacc_delay",
            label="Vaccine protection delay",
            description=(
                "Delay in days before vaccine protection becomes effective"
            ),
            default=14,
            min_value=0,
            max_value=100,
            value_type=ValueType.INTEGER,
        )

        # Original R:
        # vacc_start_day <- 0
        schema.add_parameter(
            name="vacc_start_day",
            label="Vaccination start day",
            description=(
                "Vaccination start day relative to simulation start"
            ),
            default=0,
            min_value=0,
            max_value=365,
            value_type=ValueType.INTEGER,
        )

        # Original R:
        # sim_days <- length(interp_ts)
        schema.add_parameter(
            name="n_days",
            label="Number of simulation days",
            description=(
                "Number of time points used for vaccination rollout"
            ),
            default=57,
            min_value=1,
            max_value=500,
            value_type=ValueType.INTEGER,
        )

        # =====================================================================
        # Interventions
        # =====================================================================

        # NPI switch
        schema.add_intervention(
            id="npi",
            label="Non-pharmaceutical intervention",
            description=(
                "Gradual reduction in chikungunya transmission."
            ),
            target_rates=["beta"],
            adherence=100.0,
            transmission_reduction=80.0,
        )

        # Vaccination switch
        #
        # The standard transmission_reduction field is deliberately set
        # to zero because the vaccine effect is calculated manually below.
        schema.add_intervention(
            id="vaccination",
            label="Vaccination",
            description=(
                "Vaccination among individuals aged 12 years or older."
            ),
            target_rates=["beta"],
            adherence=100.0,
            transmission_reduction=0.0,
        )

    # =========================================================================
    # Initialisation
    # =========================================================================
    def __init__(self, config):

        super().__init__(config)

        disease_config = self.payload["Disease"]

        # ---------------------------------------------------------------------
        # NPI parameters
        # ---------------------------------------------------------------------
        self.npi_start_day = float(
            disease_config["npi_start_day"]
        )

        self.npi_end_day = float(
            disease_config["npi_end_day"]
        )

        self.npi_max_reduction = float(
            disease_config["npi_max_reduction"]
        )

        # ---------------------------------------------------------------------
        # Vaccination parameters
        # ---------------------------------------------------------------------
        self.prop_under12 = float(
            disease_config["prop_under12"]
        )

        self.vaccine_coverage = float(
            disease_config["vaccine_coverage"]
        )

        self.vaccine_efficacy = float(
            disease_config["vaccine_efficacy"]
        )

        self.vacc_delay = float(
            disease_config["vacc_delay"]
        )

        self.vacc_start_day = float(
            disease_config["vacc_start_day"]
        )

        self.n_days = float(
            disease_config["n_days"]
        )

    # =========================================================================
    # Initial state
    # =========================================================================
    def prepare_initial_state(self):
        """
        Return the population matrix generated by the base Model class.
        """
        return self.population_matrix

    # =========================================================================
    # Intervention helper
    # =========================================================================
    def _intervention_enabled(self, intervention_id):
        """
        Check whether an intervention is enabled for this model instance.

        The framework removes interventions from the control instance,
        allowing the same model to generate intervention and control runs.
        """

        if not hasattr(self, "intervention_dict"):
            return False

        return intervention_id in self.intervention_dict

    # =========================================================================
    # Custom NPI
    # =========================================================================
    def custom_npi(self, t, beta):
        """
        Apply the gradual NPI reduction in beta.

        If NPI is disabled, beta remains unchanged.
        """

        if not self._intervention_enabled("npi"):
            return beta

        duration = (
            self.npi_end_day
            - self.npi_start_day
        )

        safe_duration = jnp.maximum(
            duration,
            1e-10,
        )

        progress = (
            (t - self.npi_start_day)
            / safe_duration
        )

        progress = jnp.clip(
            progress,
            0.0,
            1.0,
        )

        reduction = (
            self.npi_max_reduction
            * progress
        )

        effective_beta = (
            beta
            * (1.0 - reduction)
        )

        return effective_beta

    # =========================================================================
    # Custom vaccination
    # =========================================================================
    def custom_vaccination(self, t, beta):
        """
        Apply vaccination exactly following the original R/odin logic.

        Original R logic:

        effective_vacc_days =
            n_days - vacc_start_day - vacc_delay

        can_vaccinate =
            effective_vacc_days > 0

        daily_vacc_prop =
            if can_vaccinate:
                vaccine_coverage / effective_vacc_days
            else:
                0

        effective_days =
            max(
                0,
                t - (vacc_start_day + vacc_delay) + 1
            )

        vaccine_cov_current =
            min(
                vaccine_coverage,
                daily_vacc_prop *
                effective_days *
                vaccine_efficacy
            )

        vaccine_cov_total_population =
            vaccine_cov_current *
            (1 - prop_under12)

        transmission_reduction =
            1 - vaccine_cov_total_population

        effective_beta =
            beta * transmission_reduction
        """

        # ---------------------------------------------------------------------
        # Vaccination OFF:
        # equivalent to coverage_adult = 0 in the original R analysis
        # ---------------------------------------------------------------------
        if not self._intervention_enabled("vaccination"):
            return beta

        # ---------------------------------------------------------------------
        # Original R:
        #
        # effective_vacc_days <-
        #     n_days - vacc_start_day - vacc_delay
        # ---------------------------------------------------------------------
        effective_vacc_days = (
            self.n_days
            - self.vacc_start_day
            - self.vacc_delay
        )

        can_vaccinate = (
            effective_vacc_days > 0.0
        )

        safe_effective_vacc_days = jnp.maximum(
            effective_vacc_days,
            1e-10,
        )

        # ---------------------------------------------------------------------
        # Original R:
        #
        # daily_vacc_prop <-
        #     if (can_vaccinate)
        #         vacc_coverage_adult_target /
        #         effective_vacc_days
        #     else 0
        # ---------------------------------------------------------------------
        daily_vacc_prop = jnp.where(
            can_vaccinate,
            self.vaccine_coverage
            / safe_effective_vacc_days,
            0.0,
        )

        # ---------------------------------------------------------------------
        # Original R:
        #
        # effective_days <-
        #     max(
        #         0,
        #         t - (vacc_start_day + vacc_delay) + 1
        #     )
        #
        # The +1 is intentionally retained.
        # ---------------------------------------------------------------------
        effective_days = jnp.maximum(
            0.0,
            t
            - (
                self.vacc_start_day
                + self.vacc_delay
            )
            + 1.0,
        )

        # ---------------------------------------------------------------------
        # Original R:
        #
        # vaccine_cov_current <-
        #     min(
        #         vacc_coverage_adult_target,
        #         daily_vacc_prop *
        #         effective_days *
        #         vaccine_efficacy
        #     )
        # ---------------------------------------------------------------------
        vaccine_cov_current = jnp.minimum(
            self.vaccine_coverage,
            daily_vacc_prop
            * effective_days
            * self.vaccine_efficacy,
        )

        # ---------------------------------------------------------------------
        # Original R:
        #
        # vaccine_cov_total_population <-
        #     vaccine_cov_current *
        #     (1 - prop_under12)
        # ---------------------------------------------------------------------
        vaccine_cov_total_population = (
            vaccine_cov_current
            * (1.0 - self.prop_under12)
        )

        # ---------------------------------------------------------------------
        # Original R:
        #
        # transmission_reduction <-
        #     1 - vaccine_cov_total_population
        # ---------------------------------------------------------------------
        transmission_reduction = (
            1.0
            - vaccine_cov_total_population
        )

        # ---------------------------------------------------------------------
        # Original R:
        #
        # effective_beta <-
        #     sir_beta * transmission_reduction
        # ---------------------------------------------------------------------
        effective_beta = (
            beta
            * transmission_reduction
        )

        return effective_beta

    # =========================================================================
    # Differential equations
    # =========================================================================
    def equation(self, y, t, p):

        C = self.COMPARTMENTS

        # ---------------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------------
        params = self._unpack_params(p)

        # ---------------------------------------------------------------------
        # Current states
        # ---------------------------------------------------------------------
        states = {
            c: y[i]
            for i, c in enumerate(self.compartment_list)
        }

        S = states[C.S]
        I = states[C.I]
        R = states[C.R]

        # ---------------------------------------------------------------------
        # Population total
        #
        # In the original R model, population_total is fixed.
        #
        # Here S + I + R is conserved:
        #
        # dS/dt + dI/dt + dR/dt = 0
        #
        # Therefore this remains equal to the initial total population.
        # ---------------------------------------------------------------------
        population_total = (
            S
            + I
            + R
        )

        # ---------------------------------------------------------------------
        # Baseline beta
        # ---------------------------------------------------------------------
        beta = params["beta"]

        # ---------------------------------------------------------------------
        # NPI
        #
        # For the vaccine-only JSON, NPI is not enabled,
        # so this simply returns baseline beta.
        # ---------------------------------------------------------------------
        beta = self.custom_npi(
            t,
            beta,
        )

        # ---------------------------------------------------------------------
        # Vaccination
        # ---------------------------------------------------------------------
        effective_beta = self.custom_vaccination(
            t,
            beta,
        )

        gamma = params["gamma"]

        # ---------------------------------------------------------------------
        # Original R:
        #
        # s_to_i <-
        #     effective_beta *
        #     state_s *
        #     state_i /
        #     population_total
        # ---------------------------------------------------------------------
        s_to_i = (
            effective_beta
            * S
            * I
            / (
                population_total
                + 1e-10
            )
        )

        # ---------------------------------------------------------------------
        # Recovery
        # ---------------------------------------------------------------------
        i_to_r = (
            gamma
            * I
        )

        # ---------------------------------------------------------------------
        # Initialise derivatives, including framework-generated totals
        # ---------------------------------------------------------------------
        derivs = {
            c: jnp.zeros_like(I)
            for c in self.compartment_list
        }

        # ---------------------------------------------------------------------
        # Original S/I dynamics
        # ---------------------------------------------------------------------
        derivs[C.S] = (
            -s_to_i
        )

        derivs[C.I] = (
            s_to_i
            - i_to_r
        )

        derivs[C.R] = (
            i_to_r
        )

        # ---------------------------------------------------------------------
        # Framework cumulative totals
        # ---------------------------------------------------------------------
        if f"{C.I}_total" in derivs:
            derivs[
                f"{C.I}_total"
            ] = s_to_i

        if f"{C.R}_total" in derivs:
            derivs[
                f"{C.R}_total"
            ] = i_to_r

        # ---------------------------------------------------------------------
        # Preserve framework-defined compartment order
        # ---------------------------------------------------------------------
        return jnp.stack(
            [
                derivs[c]
                for c in self.compartment_list
            ]
        )