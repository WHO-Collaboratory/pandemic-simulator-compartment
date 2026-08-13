import jax.numpy as jnp
from compartment.model import ValueType
from compartment.models.example_parameter_uncertainty_declarative_model.model import ExampleParameterUncertaintyDeclarativeModel


class ExampleDiseaseHospitalized(ExampleParameterUncertaintyDeclarativeModel):
    """SIHR variant: adds a Hospitalized compartment to the declarative SIR model.

    Infected individuals either recover directly (I->R) or are hospitalized
    (I->H) and then recover from hospital (H->R). Only I is infectious; H does
    not contribute to the force of infection.
    """

    @classmethod
    def define_parameters(cls, schema):
        """Extend the base SIR schema with a Hospitalized compartment.

        Inherits the parent's compartments, edges, demographics, and
        intervention, gives this variant its own identity, then adds the ``H``
        compartment and the ``I->H`` / ``H->R`` transmission edges.

        Args:
            schema: The schema builder, pre-populated by the parent model's
                ``define_parameters``.
        """
        # Inherit the base S/I/R compartments, beta/gamma edges, demographics,
        # and my_intervention from the declarative model.
        super().define_parameters(schema)

        # Give this variant its own identity. The identity must live on the
        # schema (not just a DISEASE_TYPE class attribute), because the config
        # validator is auto-generated from the schema and pins disease_type to
        # schema.disease_type. set_model_info() may only be called once and the
        # parent already called it, so clear the inherited disease_type first.
        schema._disease_type = None
        schema.set_model_info(
            disease_type="example_parameter_uncertainty_declarative_hospitalized",
            label="Example Disease (Declarative, with Hospitalization)",
            description=(
                "SIHR variant of the declarative parameter-uncertainty example: "
                "infected individuals recover directly (I->R) or are hospitalized "
                "(I->H) before recovering (H->R)."
            ),
        )

        schema.add_compartment(
            "H", "Hospitalized", "Infected individuals requiring hospitalization"
        )
        schema.add_transmission_edge(
            source="infected",
            target="hospitalized",
            variable_name="zeta",
            label="Hospitalization Rate (I->H)",
            description="Rate at which infected individuals are hospitalized",
            default=0.05,
            default_min=0.01,
            default_max=0.1,
            min_value=0.0,
            max_value=1.0,
            unit="per day",
        )
        schema.add_transmission_edge(
            source="hospitalized",
            target="recovered",
            variable_name="eta",
            label="Hospital Stay (H->R)",
            description="Average number of days hospitalized before recovery",
            default=14.0,
            default_min=7.0,
            default_max=28.0,
            min_value=1.0,
            max_value=100.0,
            value_type=ValueType.DAYS,
            unit="days",
        )

    def equation(self, y, t, p):
        """Compute the compartment derivatives for one integration step.

        Wires the hospitalization rates (``zeta``, ``eta``) into the framework's
        derivative builder alongside the inherited ``beta`` / ``gamma`` rates.

        Args:
            y: Current compartment values, ordered by ``compartment_list``.
            t: Current time in days since the simulation start date.
            p: Packed parameter tuple, unpacked via ``_unpack_params``.

        Returns:
            The stacked per-compartment derivatives (dy/dt).
        """
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(self.compartment_list)}

        I = states[C.I]
        non_total = [c for c in C if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)
        prop_infective = I.sum() / (N_total.sum() + 1e-10)

        rates, self.travel_matrix = self._apply_interventions(
            t, {"beta": params["beta"]}, prop_infective
        )
        rates["gamma"] = params["gamma"]
        rates["zeta"] = params["zeta"]  # I -> H (hospitalization)
        rates["eta"] = params["eta"]    # H -> R (recovery from hospital)

        # Let the framework build S/I/H/R derivatives from the declared edges,
        # including the frequency-dependent force of infection into I.
        derivs = self._compute_equations(states, rates)
        return jnp.stack([derivs[c] for c in self.compartment_list])
