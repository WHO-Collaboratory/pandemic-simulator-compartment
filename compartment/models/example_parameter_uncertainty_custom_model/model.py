import jax.numpy as jnp
import numpy as np
import logging
from compartment.model import Model, ValueType

logger = logging.getLogger(__name__)


class ExampleParameterUncertaintyCustomModel(Model):
    """A simple SIR compartmental model for Example Disease with Parameter Uncertainty and Custom Equation."""

    @classmethod
    def define_parameters(cls, schema):
        """Declare the model's compartments, transmission edges, and parameters.

        Called once by the framework to build the model schema, from which the
        config validator and parameter set are generated.

        Args:
            schema (ParameterSchemaBuilder): Schema builder to populate with
                model info, compartments, transmission edges, disease
                parameters, and the intervention consumed by
                ``custom_intervention``.
        """
        schema.set_model_info(
            disease_type="example_parameter_uncertainty_custom",
            label="Example Disease with Parameter Uncertainty and Custom Equation",
            description="A SIR model for an example disease with parameter uncertainty, a custom written equation and a ramped intervention",
        )
        schema.set_model_metadata(
            authors=[
                {
                    "name": "Example Author",
                    "email": "example@example.com",
                    "affiliation": "Example LLC",
                }
            ],
            license="MIT",
            model_type="Compartmental",
            diseases=["Example disease"],
            transmission_routes=["Airborne"],
            questions_answered=[
                "How does an intervention that phases in gradually compare with one that switches on at full strength?",
                "How much do the speeds at which adherence builds and decays change outbreak size and timing?",
                "How wide are the outcome intervals when the transmission rate is only known within a range?",
            ],
            key_assumptions=[
                "Closed population — no births or deaths.",
                "Frequency-dependent transmission (force of infection scales with the proportion infectious).",
                "Recovered individuals are fully immune with no waning (no R→S transition).",
                "The intervention ramps linearly: it climbs from no effect to full effect over ramp_up_days from the start date, holds through the window, then falls back over ramp_down_days after the end date.",
                "At full effect the reduction matches the framework's built-in intervention (adherence x transmission reduction applied to beta).",
                "Parameter uncertainty is uniform: fields marked with variance in the config are drawn from their min/max range by Latin Hypercube Sampling.",
                "No inter-zone mobility — the travel matrix is the identity, so zones evolve independently.",
            ],
            applicability=(
                "A teaching example for writing equation() and intervention "
                "behaviour by hand. Copy it as the starting point when the "
                "built-in on/off intervention is too abrupt for your question."
            ),
            not_for=(
                "Real-world forecasting or policy advice — the compartments and "
                "parameter ranges are illustrative, not fitted to any disease."
            ),
        )

        # --- Compartments ---
        # Mark infective=True on compartments that contribute to force of infection.
        schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
        schema.add_compartment("I", "Infected", "Currently infectious population", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered and immune")

        # --- Transmission edges ---
        # The framework auto-generates I_total, R_total cumulative compartments
        # for the targets of these edges — do not declare them by hand.
        schema.add_transmission_parameter(
            source="susceptible",
            target="infected",
            variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (S->I)",
            description="Rate at which susceptibles become infected through contact",
            default=0.3,
            default_min=0.1,
            default_max=0.5,
            min_value=0.01,
            max_value=2.0,
            unit="per day",
        )
        schema.add_transmission_parameter(
            source="infected",
            target="recovered",
            variable_name="gamma",
            label="Recovery Period (I->R)",
            description="Average number of days to recover",
            default=10.0,
            default_min=5.0,
            default_max=20.0,
            min_value=1.0,
            max_value=100.0,
            value_type=ValueType.DAYS,
            unit="days",
        )

        schema.add_parameter(
            name="ramp_up_days",
            label="Intervention Ramp-Up (days)",
            description="Days for the intervention to climb from baseline to full adherence.",
            value_type=ValueType.DAYS,
            default=14.0,
            min_value=1.0,
            max_value=180.0,
            unit="days",
            required=False,
            enable_variance=True,
        )
        schema.add_parameter(
            name="ramp_down_days",
            label="Intervention Ramp-Down (days)",
            description="Days for adherence to fall back to baseline after the intervention ends.",
            value_type=ValueType.DAYS,
            default=21.0,
            min_value=1.0,
            max_value=180.0,
            unit="days",
            required=False,
            enable_variance=True,
        )

        # --- Optional: spatial travel support ---
        # Declare your mobility parameters as custom fields, then define how
        # they build the matrix in build_travel_matrix() below. Without this,
        # the base class supplies an identity matrix (no inter-zone travel).
        # schema.add_parameter(
        #     name="travel_sigma",
        #     label="Travel Rate (σ)",
        #     description="Percentage of each zone's population away from home per day.",
        #     value_type=ValueType.PERCENTAGE,
        #     default=20.0,
        #     min_value=0.0,
        #     max_value=100.0,
        #     unit="%",
        # )

        # --- Interventions ---
        # Declared here so the base class builds a runtime Intervention object
        # (self.interventions) from the config. custom_intervention() then reads
        # its adherence / transmission_reduction / date window and ramps it.
        schema.add_intervention(
            id="my_intervention",
            label="My Intervention",
            description="Reduces transmission while active",
            target_rates=["beta"],
            adherence=50.0,
            transmission_reduction=50.0,
        )

        # --- Optional: age-stratified demographics + contact matrix ---
        # schema.add_demographic_group("age_0_17",  "Children", default_weight=33.3, age_range=(0, 17))
        # schema.add_demographic_group("age_18_55", "Adults",   default_weight=44.4, age_range=(18, 55))
        # schema.add_demographic_group("age_56_plus","Elderly", default_weight=22.3, age_range=(56, 120))

    def __init__(self, config):
        """Initialize the model from a validated simulation config.

        Args:
            config (dict): Validated simulation configuration produced by the
                framework's config loader.
        """
        super().__init__(config)
        # Add any model-specific initialisation here (e.g. temperature).

    # --- Optional: spatial travel support ---
    # The framework calls this before prepare_initial_state() and stores the
    # result on self.travel_matrix. The default returns the identity matrix,
    # so only override it if your model has inter-zone mobility.
    #
    # def build_travel_matrix(self, admin_zones):
    #     # PERCENTAGE params arrive as 20.0, not 0.2 — convert first.
    #     sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
    #     return get_gravity_model_travel_matrix(admin_zones, sigma)

    def prepare_initial_state(self):
        """Return the initial compartment populations for the solver.

        Returns:
            jnp.ndarray: Population matrix (compartments x admin zones) used as
                the solver's initial state.
        """
        return self.population_matrix


    def custom_intervention(self, t, beta):
        """Scale ``beta`` by a hand-written intervention that ramps gradually.

        Replaces the built-in ``_apply_interventions`` helper. Reads the
        ``my_intervention`` adherence, transmission reduction, and active date
        window from the config-loaded intervention object, then instead of
        switching on and off instantly it climbs linearly from baseline to full
        effect over ``ramp_up_days`` from the start date, holds through the
        window, and falls back to baseline over ``ramp_down_days`` after the end
        date (both read from the config's "Disease" block). The full effect
        matches the framework's ``beta * (1 - adherence *
        transmission_reduction)``, scaled by the ramp; ``jnp.clip`` keeps this
        JAX-traceable since ``t`` is a traced value.

        Args:
            t (float): Current time in days since the simulation start date (a
                traced JAX value).
            beta (float): Baseline transmission rate to scale.

        Returns:
            jnp.ndarray: Transmission rate after applying the ramped
                intervention, or ``beta`` unchanged when the intervention is not
                configured (as in the control run).
        """
        # Look up the intervention loaded from config by its schema id. The
        # ``id in self.intervention_dict`` check matches the built-in helper so
        # the control ("without interventions") run correctly skips it.
        intv = next(
            (
                i
                for i in self.interventions
                if i.id == "my_intervention" and i.id in self.intervention_dict
            ),
            None,
        )
        if intv is None or intv.start_date_ordinal is None:
            return beta  # intervention not configured — leave beta unchanged

        # t is the day offset from the simulation start date, so convert it back
        # to an absolute ordinal day to compare against the intervention window.
        current_ordinal_day = self.start_date_ordinal + t

        # Phase-in: 0 before the start date, climbing linearly to 1 over
        # ramp_up_days, then holding at 1.
        ramp_in = jnp.clip(
            (current_ordinal_day - intv.start_date_ordinal) / self.ramp_up_days,
            0.0,
            1.0,
        )

        if intv.end_date_ordinal is None:
            # Open-ended intervention: phase in and stay, no phase-out.
            ramp = ramp_in
        else:
            # Phase-out: 1 until the end date, falling linearly to 0 over
            # ramp_down_days. The product yields a trapezoid over time.
            ramp_out = jnp.clip(
                (intv.end_date_ordinal + self.ramp_down_days - current_ordinal_day)
                / self.ramp_down_days,
                0.0,
                1.0,
            )
            ramp = ramp_in * ramp_out

        full_reduction = intv.adherence * intv.transmission_reduction
        scale = 1.0 - ramp * full_reduction
        return beta * scale

    def equation(self, y, t, p):
        """Compute the compartment derivatives for a single integration step.

        Builds the SIR derivatives by hand and applies ``custom_intervention``
        to the transmission rate instead of the built-in intervention helper.

        Args:
            y (jnp.ndarray): Current compartment values, ordered by
                ``compartment_list``.
            t (float): Current time in days since the simulation start date.
            p (tuple): Packed parameter tuple, unpacked via ``_unpack_params``.

        Returns:
            jnp.ndarray: Stacked per-compartment derivatives (dy/dt).
        """
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(self.compartment_list)}

        I = states[C.I]  
        non_total = [c for c in C if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)
        prop_infective = I.sum() / (N_total.sum() + 1e-10)

        # Apply our custom intervention to beta instead of _apply_interventions.
        beta = self.custom_intervention(t, params["beta"])
        gamma = params["gamma"]
        S = states[C.S]

        new_infections = beta * S * prop_infective  # S -> I (frequency-dependent)
        new_recoveries = gamma * I                   # I -> R

        # Start every compartment at zero so the auto-generated _total
        # cumulative compartments are always present before stacking.
        derivs = {c: jnp.zeros_like(I) for c in self.compartment_list}
        derivs[C.S] = -new_infections
        derivs[C.I] = new_infections - new_recoveries
        derivs[C.R] = new_recoveries

        # _total compartments accumulate inflows only.
        if f"{C.I}_total" in derivs:
            derivs[f"{C.I}_total"] = new_infections
        if f"{C.R}_total" in derivs:
            derivs[f"{C.R}_total"] = new_recoveries

        return jnp.stack([derivs[c] for c in self.compartment_list])
