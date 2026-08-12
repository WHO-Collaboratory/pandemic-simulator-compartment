import jax.numpy as jnp

from compartment.model import ValueType
from compartment.models.example_disease_parameter_uncertainty_model.model import ExampleDiseaseParameterUncertaintyModel

class ExampleDiseaseChangeEquation(ExampleDiseaseParameterUncertaintyModel):
    @classmethod
    def define_parameters(cls, schema):
        # Inherit the base SIR schema, then add config-driven controls for how
        # gradually the custom intervention phases in and out. Declaring them as
        # disease parameters means they can be tuned from the config's "Disease"
        # block (as self.ramp_up_days / self.ramp_down_days) with no code change.
        super().define_parameters(schema)

        # Give this variant its own identity. set_model_info() may only be called
        # once, and the parent already called it, so clear the inherited
        # disease_type before setting the new info.
        schema._disease_type = None
        schema.set_model_info(
            disease_type="example_disease_ramped_intervention",
            label="Example Disease (Ramped Custom Intervention)",
            description=(
                "SIR model demonstrating a hand-written custom intervention that "
                "ramps adherence up and down over time instead of switching on "
                "and off instantly."
            ),
        )

        schema.add_disease_parameter(
            name="ramp_up_days",
            label="Intervention Ramp-Up (days)",
            description="Days for the intervention to climb from baseline to full adherence.",
            value_type=ValueType.DAYS,
            default=14.0,
            min_value=1.0,
            max_value=180.0,
            unit="days",
            required=False,
            enable_variance=False,
        )
        schema.add_disease_parameter(
            name="ramp_down_days",
            label="Intervention Ramp-Down (days)",
            description="Days for adherence to fall back to baseline after the intervention ends.",
            value_type=ValueType.DAYS,
            default=21.0,
            min_value=1.0,
            max_value=180.0,
            unit="days",
            required=False,
            enable_variance=False,
        )

    def custom_intervention(self, t, beta):
        """Example of a custom intervention written by hand.

        This replaces the built-in ``_apply_interventions`` helper so you can
        see exactly what an intervention does. It reads the ``my_intervention``
        settings (adherence, transmission reduction, and active date window)
        from the config-loaded intervention object, but instead of switching on
        and off instantly it **ramps gradually**:

        - adherence climbs linearly from baseline to full over ``ramp_up_days``
          starting on the intervention's start date,
        - holds at full effect through the window, then
        - falls linearly back to baseline over ``ramp_down_days`` after the end
          date.

        ``ramp_up_days`` and ``ramp_down_days`` are read from the config's
        "Disease" block (see ``define_parameters``).

        The full effect matches the framework's formula,
        ``beta * (1 - adherence * transmission_reduction)``, scaled by the ramp.
        ``jnp.clip`` keeps this JAX-traceable since ``t`` is a traced value.
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
