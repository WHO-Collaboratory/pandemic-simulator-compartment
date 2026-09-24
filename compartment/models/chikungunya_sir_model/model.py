import jax.numpy as jnp
import logging

from compartment.model import Model, ValueType


logger = logging.getLogger(__name__)


class ChikungunyaSirModel(Model):
    """
    Deterministic SIR model for chikungunya transmission.

    Model assumptions:
    - Closed population with no births, deaths, or imported infections.
    - Homogeneous mixing within the population.
    - Deterministic transmission dynamics.
    - The epidemiological state structure is S -> I -> R.
    - NPI and vaccination effects are represented through modifications
      of the effective transmission rate rather than through additional
      epidemiological compartments.

    The model supports two optional interventions:

    1. Non-pharmaceutical intervention (NPI)
       - NPI effects are represented through a gradual reduction in beta.
       - The intervention begins on simulation day 17 and reaches its
         maximum effect on day 28.
       - The default maximum reduction in beta is 80%.
       - This 80% reduction is treated as the strongest representation
         of the NPI intensity implemented during the outbreak.
       - Additional NPI scenario or sensitivity analyses should therefore
         examine weaker intervention intensities, corresponding to
         reductions in beta smaller than 80%.

    2. Vaccination
       - Vaccination eligibility is restricted to individuals aged
         12 years or older.
       - Target coverage: 40%.
       - Vaccine efficacy against infection: 40%.
       - Protection delay: 14 days.
       - Vaccination starts on simulation day 0.
       - Age eligibility is represented through the proportion of the
         population aged under 12 years; the S, I, and R compartments
         themselves are not age-stratified.

    The vaccination effect follows the original R/odin model logic.
    Vaccination does not introduce an additional epidemiological
    compartment. Instead, the effective vaccination coverage is used
    to reduce the effective transmission rate over time.

    The implementation is intended for intervention-scenario exploration
    in the Pandemic Simulator and preserves the mathematical structure
    of the original R/odin implementation.
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
        # The epidemiological state structure is a standard SIR model.
        #
        # No separate vaccinated compartment is included because vaccination
        # affects transmission through the effective beta only, following
        # the original R/odin implementation.
        schema.add_compartment(
            "S",
            "Susceptible",
            "Population susceptible to infection",
        )

        # infective=True tells the framework that the I compartment
        # contributes to the force of infection.
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
        # beta is the baseline transmission rate before any intervention
        # effects are applied.
        #
        # The default value 0.323 is used for the example intervention
        # scenarios in this implementation.
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

        # gamma is specified as a duration in DAYS.
        #
        # Therefore, a value of 11.0 represents an average infectious
        # period of 11 days. The framework handles the conversion from
        # duration to the corresponding recovery rate internally.
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

        # Simulation day on which the NPI begins to affect transmission.
        #
        # With the example simulation starting on 2 July 2025,
        # day 17 corresponds to 19 July 2025.
        schema.add_parameter(
            name="npi_start_day",
            label="NPI start day",
            description="Simulation day when the NPI begins",
            default=17,
            min_value=0,
            max_value=365,
            value_type=ValueType.INTEGER,
        )

        # Simulation day on which the NPI reaches its full effect.
        #
        # With the example simulation starting on 2 July 2025,
        # day 28 corresponds to 30 July 2025.
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

        # Maximum NPI effect on transmission.
        #
        # The default value of 0.80 represents the strongest NPI intensity
        # considered to correspond to the intervention implemented during
        # the outbreak.
        #
        # Therefore, additional NPI scenario analyses should use values
        # below 0.80 to represent weaker intervention intensities.
        # Values above 0.80 are not intended for the current scenario
        # interpretation, although the schema technically allows them.
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

        # ---------------------------------------------------------------------
        # Age eligibility
        # ---------------------------------------------------------------------
        # Original R:
        # prop_pop_under12 <- 0.13
        #
        # This parameter is used only to represent eligibility for
        # vaccination. The model does NOT stratify S, I, or R by age.
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

        # ---------------------------------------------------------------------
        # Vaccination coverage
        # ---------------------------------------------------------------------
        # Original R:
        # coverage = 0.40
        #
        # Coverage refers to the target vaccination coverage among the
        # vaccine-eligible population aged 12 years or older.
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

        # ---------------------------------------------------------------------
        # Vaccine efficacy
        # ---------------------------------------------------------------------
        # Original R:
        # vaccine_efficacy <- 0.40
        #
        # Vaccine efficacy is interpreted as efficacy against infection.
        schema.add_parameter(
            name="vaccine_efficacy",
            label="Vaccine efficacy against infection",
            description="Vaccine efficacy against infection",
            default=0.40,
            min_value=0.0,
            max_value=1.0,
            value_type=ValueType.RATE,
        )

        # ---------------------------------------------------------------------
        # Delay to protection
        # ---------------------------------------------------------------------
        # Original R:
        # vacc_delay <- 14
        #
        # Protection is assumed to become effective 14 days after
        # vaccination starts.
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

        # ---------------------------------------------------------------------
        # Vaccination start time
        # ---------------------------------------------------------------------
        # Original R:
        # vacc_start_day <- 0
        #
        # Day 0 means vaccination begins at the simulation start date.
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

        # ---------------------------------------------------------------------
        # Vaccination rollout duration
        # ---------------------------------------------------------------------
        # Original R:
        # sim_days <- length(interp_ts)
        #
        # For the example simulation from 2 July to 27 August 2025,
        # the number of output time points is 57.
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

        # ---------------------------------------------------------------------
        # NPI switch
        # ---------------------------------------------------------------------
        # The NPI is registered with the Pandemic Simulator so that the
        # framework can distinguish intervention and no-intervention runs.
        #
        # The framework-level intervention entry acts as an ON/OFF switch.
        # The detailed time-varying reduction in beta is implemented in
        # custom_npi() below.
        #
        # The 80% value represents the strongest NPI intensity used in the
        # current outbreak-based scenario. Weaker NPI scenarios should use
        # smaller reductions in beta.
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

        # ---------------------------------------------------------------------
        # Vaccination switch
        # ---------------------------------------------------------------------
        # The vaccination intervention is also registered with the framework
        # primarily as an ON/OFF switch.
        #
        # The standard framework transmission_reduction field is deliberately
        # set to zero because the actual vaccination effect on beta is
        # calculated manually in custom_vaccination() according to the
        # original R/odin model logic.
        #
        # This avoids applying the vaccination effect twice.
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
        """
        Initialise the model and load model-specific intervention parameters.

        The base Model initialiser first handles the standard framework setup,
        including transmission parameters, population state, and configured
        interventions. Model-specific NPI and vaccination parameters are then
        read from the Disease block of the configuration.
        """

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
        Return the initial population matrix created by the base Model class.

        The example configuration specifies the total population and the
        initial infected proportion. No custom age-stratified initial state is
        required because age eligibility is used only for vaccination coverage
        calculations and does not create separate age-specific compartments.
        """

        return self.population_matrix

    # =========================================================================
    # Intervention helper
    # =========================================================================
    def _intervention_enabled(self, intervention_id):
        """
        Check whether a specified intervention is enabled for this model run.

        The Pandemic Simulator uses the same model class for intervention
        and control simulations. The control model has its intervention
        dictionary cleared by the framework.

        Therefore:
        - if the intervention id is present, the intervention-specific logic
          is applied;
        - if it is absent, the corresponding intervention has no effect.
        """

        if not hasattr(self, "intervention_dict"):
            return False

        return intervention_id in self.intervention_dict

    # =========================================================================
    # Custom NPI
    # =========================================================================
    def custom_npi(self, t, beta):
        """
        Apply a gradual NPI-related reduction in the transmission rate beta.

        The intervention begins on npi_start_day and increases linearly
        until npi_end_day, after which the maximum reduction is maintained.

        In the default configuration:
        - npi_start_day = 17
        - npi_end_day = 28
        - npi_max_reduction = 0.80

        The 80% reduction is treated as the strongest representation of
        the NPI intensity implemented during the outbreak. Therefore,
        additional NPI scenario or sensitivity analyses should represent
        weaker interventions by specifying reductions in beta below 80%.

        If the NPI intervention is disabled, the baseline beta is returned
        unchanged.
        """

        # NPI OFF:
        # return the baseline transmission rate unchanged.
        if not self._intervention_enabled("npi"):
            return beta

        # Duration of the linear NPI ramp.
        duration = (
            self.npi_end_day
            - self.npi_start_day
        )

        # Avoid division by zero if start and end days were ever configured
        # to be identical.
        safe_duration = jnp.maximum(
            duration,
            1e-10,
        )

        # Linear progress of the NPI effect:
        #
        # t < start day     -> progress = 0
        # start <= t < end  -> progress increases from 0 to 1
        # t >= end day      -> progress = 1
        progress = (
            (t - self.npi_start_day)
            / safe_duration
        )

        progress = jnp.clip(
            progress,
            0.0,
            1.0,
        )

        # Current proportional reduction in beta.
        reduction = (
            self.npi_max_reduction
            * progress
        )

        # Effective transmission rate after applying the NPI.
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
        Apply vaccination following the original R/odin model logic.

        Vaccination is represented through a time-varying reduction in beta.
        No separate vaccinated epidemiological compartment is introduced.

        Age eligibility is handled using prop_under12:
        only the population aged 12 years or older is considered eligible,
        but S, I, and R are not explicitly age-stratified.

        Original R/odin logic:

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

        The placement of vaccine efficacy inside vaccine_cov_current and the
        +1 term in effective_days are retained exactly from the original
        R/odin implementation to preserve the original model dynamics.
        """

        # ---------------------------------------------------------------------
        # Vaccination OFF
        # ---------------------------------------------------------------------
        # When vaccination is disabled, this is equivalent to setting
        # vaccination coverage to zero in the original R analysis.
        #
        # beta is therefore returned unchanged.
        if not self._intervention_enabled("vaccination"):
            return beta

        # ---------------------------------------------------------------------
        # Number of days available for effective vaccination
        # ---------------------------------------------------------------------
        # Original R:
        #
        # effective_vacc_days <-
        #     n_days - vacc_start_day - vacc_delay
        #
        # The protection delay is subtracted because vaccine effects are not
        # assumed to begin immediately after vaccination starts.
        effective_vacc_days = (
            self.n_days
            - self.vacc_start_day
            - self.vacc_delay
        )

        can_vaccinate = (
            effective_vacc_days > 0.0
        )

        # Numerical safeguard for the denominator.
        #
        # This does not alter the intended logic because daily_vacc_prop is
        # set to zero through jnp.where when effective_vacc_days <= 0.
        safe_effective_vacc_days = jnp.maximum(
            effective_vacc_days,
            1e-10,
        )

        # ---------------------------------------------------------------------
        # Daily vaccination proportion
        # ---------------------------------------------------------------------
        # Original R:
        #
        # daily_vacc_prop <-
        #     if (can_vaccinate)
        #         vacc_coverage_adult_target /
        #         effective_vacc_days
        #     else 0
        #
        # This spreads the target vaccination coverage uniformly across the
        # available vaccination period.
        daily_vacc_prop = jnp.where(
            can_vaccinate,
            self.vaccine_coverage
            / safe_effective_vacc_days,
            0.0,
        )

        # ---------------------------------------------------------------------
        # Number of days for which vaccine protection has become effective
        # ---------------------------------------------------------------------
        # Original R:
        #
        # effective_days <-
        #     max(
        #         0,
        #         t - (vacc_start_day + vacc_delay) + 1
        #     )
        #
        # The +1 follows the original R/odin implementation exactly and is
        # retained to preserve its timing convention.
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
        # Current effective vaccination coverage among the eligible population
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
        #
        # Vaccine efficacy is intentionally multiplied within this effective
        # coverage term, exactly as in the original R/odin implementation.
        #
        # The current effective coverage cannot exceed the configured target
        # vaccination coverage.
        vaccine_cov_current = jnp.minimum(
            self.vaccine_coverage,
            daily_vacc_prop
            * effective_days
            * self.vaccine_efficacy,
        )

        # ---------------------------------------------------------------------
        # Convert effective coverage among age 12+ to the total population
        # ---------------------------------------------------------------------
        # Original R:
        #
        # vaccine_cov_total_population <-
        #     vaccine_cov_current *
        #     (1 - prop_under12)
        #
        # This is how the age restriction is represented without explicitly
        # creating age-stratified S, I, and R compartments.
        vaccine_cov_total_population = (
            vaccine_cov_current
            * (1.0 - self.prop_under12)
        )

        # ---------------------------------------------------------------------
        # Remaining transmission fraction after vaccination
        # ---------------------------------------------------------------------
        # Original R:
        #
        # transmission_reduction <-
        #     1 - vaccine_cov_total_population
        #
        # Despite the original variable name, this quantity is the remaining
        # fraction of baseline transmission after accounting for vaccination.
        transmission_reduction = (
            1.0
            - vaccine_cov_total_population
        )

        # ---------------------------------------------------------------------
        # Effective transmission rate after vaccination
        # ---------------------------------------------------------------------
        # Original R:
        #
        # effective_beta <-
        #     sir_beta * transmission_reduction
        #
        # Vaccination affects infection dynamics only through effective_beta.
        effective_beta = (
            beta
            * transmission_reduction
        )

        return effective_beta

    # =========================================================================
    # Differential equations
    # =========================================================================
    def equation(self, y, t, p):
        """
        Compute the compartment derivatives for one integration step.

        The derivatives are written explicitly rather than relying entirely
        on the framework's automatic equation builder. This preserves the
        custom force of infection in which beta is modified sequentially by
        the optional NPI and vaccination mechanisms.

        Args:
            y:
                Current compartment values in framework-defined order.
            t:
                Current simulation time in days since the simulation start.
            p:
                Packed model parameter vector.

        Returns:
            JAX array containing derivatives in self.compartment_list order.
        """

        C = self.COMPARTMENTS

        # ---------------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------------
        # Convert the framework parameter vector into a named dictionary.
        params = self._unpack_params(p)

        # ---------------------------------------------------------------------
        # Current states
        # ---------------------------------------------------------------------
        # Never hardcode compartment positions. Mapping through
        # self.compartment_list preserves compatibility with the framework's
        # automatically generated cumulative compartments.
        states = {
            c: y[i]
            for i, c in enumerate(self.compartment_list)
        }

        S = states[C.S]
        I = states[C.I]
        R = states[C.R]

        # ---------------------------------------------------------------------
        # Population total
        # ---------------------------------------------------------------------
        # In the original R model, population_total is fixed.
        #
        # Here:
        #
        # dS/dt + dI/dt + dR/dt = 0
        #
        # so S + I + R remains constant and is equal to the initial total
        # population throughout the simulation.
        #
        # This preserves the fixed population_total assumption of the
        # original R/odin implementation.
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
        # Optional NPI
        # ---------------------------------------------------------------------
        # The custom NPI modifies beta only when the NPI intervention is
        # enabled.
        #
        # In the default NPI scenario, beta is gradually reduced by up to
        # 80%, representing the strongest implemented NPI intensity
        # considered in this model.
        #
        # Weaker NPI scenarios should use smaller reductions in beta.
        #
        # For the vaccine-only example configuration, NPI is not enabled,
        # so custom_npi() simply returns the baseline beta.
        beta = self.custom_npi(
            t,
            beta,
        )

        # ---------------------------------------------------------------------
        # Optional vaccination
        # ---------------------------------------------------------------------
        # Vaccination is applied after the optional NPI modification.
        #
        # In a vaccine-only run, beta entering this function is the baseline
        # beta. In a combined NPI + vaccination run, beta already contains
        # the NPI-related reduction.
        effective_beta = self.custom_vaccination(
            t,
            beta,
        )

        # gamma has already been converted by the framework from the
        # configured DAYS representation to the corresponding recovery rate.
        gamma = params["gamma"]

        # ---------------------------------------------------------------------
        # Force of infection
        # ---------------------------------------------------------------------
        # Original R:
        #
        # s_to_i <-
        #     effective_beta *
        #     state_s *
        #     state_i /
        #     population_total
        #
        # This is a frequency-dependent SIR force of infection.
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
        # Recovery flow
        # ---------------------------------------------------------------------
        i_to_r = (
            gamma
            * I
        )

        # ---------------------------------------------------------------------
        # Initialise derivatives
        # ---------------------------------------------------------------------
        # self.compartment_list also includes framework-generated cumulative
        # compartments such as I_total and R_total.
        derivs = {
            c: jnp.zeros_like(I)
            for c in self.compartment_list
        }

        # ---------------------------------------------------------------------
        # Epidemiological SIR dynamics
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
        # I_total records cumulative incident infections.
        # R_total records cumulative recoveries.
        #
        # These are bookkeeping/output states only and do not contribute
        # to the force of infection or active population total.
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
        # The framework expects derivatives to be returned in exactly the
        # same order as self.compartment_list.
        return jnp.stack(
            [
                derivs[c]
                for c in self.compartment_list
            ]
        )