import jax.numpy as jnp
import numpy as np
import logging
from compartment.model import Model, ValueType

logger = logging.getLogger(__name__)


class MyDiseaseJaxModel(Model):
    """A simple SIR compartmental model for My Disease."""

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="MY_DISEASE",
            label="My Disease",
            description="A simple SIR model for My Disease",
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
        # schema.set_travel_volume(leaving_default=0.2)

        # --- Optional: interventions ---
        # schema.add_intervention(
        #     id="my_intervention",
        #     label="My Intervention",
        #     description="Reduces transmission while active",
        #     target_rates=["beta"],
        #     adherence=50.0,
        #     transmission_reduction=50.0,
        # )

        # --- Optional: age-stratified demographics + contact matrix ---
        # schema.add_demographic_group("age_0_17",  "Children", default_weight=33.3, age_range=(0, 17))
        # schema.add_demographic_group("age_18_55", "Adults",   default_weight=44.4, age_range=(18, 55))
        # schema.add_demographic_group("age_56_plus","Elderly", default_weight=22.3, age_range=(56, 120))

    def __init__(self, config):
        super().__init__(config)
        # Add any model-specific initialisation here (e.g. travel matrix, temperature).

    def prepare_initial_state(self):
        R = self.population_matrix.shape[1]
        # No inter-region travel: identity matrix keeps each region self-contained.
        # Replace with a gravity-model travel matrix if you add travel support.
        self.travel_matrix = np.eye(R)
        return self.population_matrix, list(self.compartment_list)

    def derivative(self, y, t, p):
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(self.compartment_list)}

        I = states[C.I]  # noqa: E741
        non_total = [c for c in C if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)
        prop_infective = I.sum() / (N_total.sum() + 1e-10)

        # _apply_interventions scales target_rates and returns the updated travel
        # matrix. It is a no-op when no interventions are configured.
        rates, self.travel_matrix = self._apply_interventions(
            t, {"beta": params["beta"]}, prop_infective
        )
        rates["gamma"] = params["gamma"]

        # _compute_derivatives handles mass-action / frequency-dependent FOI,
        # _total accumulation, and skips compartments not active in this variant.
        derivs = self._compute_derivatives(states, rates)
        return jnp.stack([derivs[c] for c in self.compartment_list])
