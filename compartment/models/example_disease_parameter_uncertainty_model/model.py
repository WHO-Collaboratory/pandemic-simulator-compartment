import jax.numpy as jnp
import numpy as np
import logging
from compartment.model import Model, ValueType

logger = logging.getLogger(__name__)


class ExampleDiseaseParameterUncertaintyModel(Model):
    """A simple SIR compartmental model for Example Disease with Parameter Uncertainty."""

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="example_disease_parameter_uncertainty",
            label="Example Disease with Parameter Uncertainty",
            description="A SIR model for an example disease with parameter uncertainty",
        )

        # --- Compartments ---
        # Mark infective=True on compartments that contribute to force of infection.
        schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
        schema.add_compartment("I", "Infected", "Currently infectious population", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered and immune")

        # --- Transmission edges ---
        # The framework auto-generates I_total, R_total cumulative compartments
        # for the targets of these edges — do not declare them by hand.
        schema.add_transmission_edge(
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
        schema.add_transmission_edge(
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

        # --- Optional: spatial travel support ---
        # Declare your mobility parameters as custom fields, then define how
        # they build the matrix in build_travel_matrix() below. Without this,
        # the base class supplies an identity matrix (no inter-zone travel).
        # schema.add_disease_parameter(
        #     name="travel_sigma",
        #     label="Travel Rate (σ)",
        #     description="Percentage of each zone's population away from home per day.",
        #     value_type=ValueType.PERCENTAGE,
        #     default=20.0,
        #     min_value=0.0,
        #     max_value=100.0,
        #     unit="%",
        # )

        # --- Optional: interventions ---
        schema.add_intervention(
            id="my_intervention",
            label="My Intervention",
            description="Reduces transmission while active",
            target_rates=["beta"],
            adherence=50.0,
            transmission_reduction=50.0,
        )

        # --- Optional: age-stratified demographics + contact matrix ---
        schema.add_demographic_group("age_0_4",    "Young children", default_weight=6.0,  age_range=(0, 4))
        schema.add_demographic_group("age_5_17",   "School-age",     default_weight=16.0, age_range=(5, 17))
        schema.add_demographic_group("age_18_49",  "Young adults",   default_weight=42.0, age_range=(18, 49))
        schema.add_demographic_group("age_50_64",  "Older adults",   default_weight=19.0, age_range=(50, 64))
        schema.add_demographic_group("age_65_plus","Seniors",        default_weight=17.0, age_range=(65, 120))

    def __init__(self, config):
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
        return self.population_matrix

    def equation(self, y, t, p):
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(self.compartment_list)}

        I = states[C.I]  
        non_total = [c for c in C if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)
        prop_infective = I.sum() / (N_total.sum() + 1e-10)

        # _apply_interventions scales target_rates and returns the updated travel
        # matrix. It is a no-op when no interventions are configured.
        rates, self.travel_matrix = self._apply_interventions(
            t, {"beta": params["beta"]}, prop_infective
        )
        rates["gamma"] = params["gamma"]

        # _compute_equations handles mass-action / frequency-dependent FOI,
        # _total accumulation, and skips compartments not active in this variant.
        derivs = self._compute_equations(states, rates)
        return jnp.stack([derivs[c] for c in self.compartment_list])
