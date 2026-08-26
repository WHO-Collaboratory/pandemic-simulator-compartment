import jax.numpy as jnp
import logging
from compartment.model import Model, ValueType

logger = logging.getLogger(__name__)


class ExampleSirModel(Model):
    """A simple deterministic SIR model.

    Source uses beta * S * I with a normalized population (N = 1), so the
    frequency-dependent form (beta * S * I / N) coincides with the source's
    density-dependent term.
    """

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="EXAMPLE_SIR",
            label="Example SIR",
            description="A simple SIR model with frequency-dependent transmission",
        )

        schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
        schema.add_compartment("I", "Infected", "Currently infectious population", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered and immune")

        # S->I transmission. Source writes beta * S * I with N normalized to 1,
        # which is the frequency-dependent form.
        schema.add_transmission_parameter(
            source="susceptible",
            target="infected",
            variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (S->I)",
            description="Rate at which susceptibles become infected through contact",
            default=0.1,
            default_min=0.05,
            default_max=0.2,
            min_value=0.001,
            max_value=2.0,
            unit="per day",
        )

        # I->R recovery, expressed as a per-day rate in the source (g = 0.05).
        schema.add_transmission_parameter(
            source="infected",
            target="recovered",
            variable_name="gamma",
            label="Recovery Rate (I->R)",
            description="Per-capita recovery rate",
            default=0.05,
            default_min=0.02,
            default_max=0.1,
            min_value=0.001,
            max_value=1.0,
            unit="per day",
        )

    def __init__(self, config):
        super().__init__(config)

    def prepare_initial_state(self):
        return self.population_matrix

    def equation(self, y, t, p):
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(self.compartment_list)}

        I = states[C.I]  # noqa: E741
        non_total = [c for c in C if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)
        prop_infective = I.sum() / (N_total.sum() + 1e-10)

        rates, _ = self._apply_interventions(
            t, {"beta": params["beta"]}, prop_infective
        )
        rates["gamma"] = params["gamma"]

        derivs = self._compute_equations(states, rates)
        return jnp.stack([derivs[c] for c in self.compartment_list])