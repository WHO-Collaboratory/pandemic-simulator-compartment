"""Ebola SEIHFR model with community / hospital / burial transmission routes.

Port of the deterministic mean-field limit of the stochastic compartmental
model in Legrand J, Grais RF, Boelle PY, Valleron AJ, Flahault A.
"Understanding the dynamics of Ebola epidemics." Epidemiol Infect.
2007;135(4):610-621. doi:10.1017/S0950268806007217 (PMC2870608).

The paper splits the infectious period into three phases with distinct
transmission coefficients — illness in the community (I), hospitalisation
(H), and traditional burial of the dead (F) — and fits the model to the
1995 DRC (Kikwit) and 2000 Uganda (Gulu) outbreaks.

Compartments: S, E, I, H, F, R.

Two structural features of the source model don't fit the framework's
standard "declarative transmission edge" pattern and are therefore
computed manually in ``equation()`` (see the Pitfalls section of
``.claude/MODEL_AUTHORING_REFERENCE.md`` for the general pattern, and
``hantavirus_jax_model`` for a precedent):

1. The force of infection mixes three distinct betas across three
   distinct infectious compartments (community/hospital/funeral) into a
   single S -> E flow. ``_compute_equations`` only supports one shared
   ``infective_sum`` per frequency-dependent edge, so this can't be
   expressed as ordinary transmission edges.
2. The three-way split of exits from I (to H, R, or F) and the two-way
   split of exits from H (to R or F) use rates theta1, delta1, delta2
   that are themselves *derived* at every step from the raw
   epidemiological parameters (gamma_h, gamma_i, gamma_d, and the target
   hospitalisation/case-fatality proportions theta, delta) via the
   algebraic relationships in the paper's Table 2 / Appendix. These are
   not standalone "rate * source" edges.

The E -> I (incubation, alpha) and F -> R (burial, gamma_f) transitions
*are* simple single-rate flows and are declared as ordinary transmission
edges.

WARNING: Like the other models in this repo, this implementation is
intended for local experimentation; it is not yet supported by the
pandemic simulator app.
"""

import logging

import jax.numpy as jnp

from compartment.model import Model, ValueType
from compartment.parameters import CompartmentDef

logger = logging.getLogger(__name__)


class EbolaSeihfrBurialLegrandModel(Model):
    """SEIHFR Ebola model with community/hospital/burial transmission (Legrand et al. 2007)."""

    # Cumulative inflow trackers.  Declared here rather than in
    # define_parameters() (and computed explicitly in equation()) because most
    # of the inflows they track are manual multi-rate flows, not transmission
    # edges — the framework's per-edge auto-generation would only produce
    # I_total and R_total.  Same pattern as dengue_jax_model.
    _TOTAL_COMPARTMENTS = (
        ("E_total", "Exposed Total",
         "Cumulative number of individuals ever exposed (cumulative infections)."),
        ("I_total", "Infectious (community) Total",
         "Cumulative number of individuals who ever became infectious."),
        ("H_total", "Hospitalised Total",
         "Cumulative number of individuals ever hospitalised."),
        ("F_total", "Deaths Total",
         "Cumulative number of individuals who died of Ebola."),
        ("R_total", "Removed Total",
         "Cumulative number of individuals removed (recovered or safely buried)."),
    )

    @classmethod
    def _add_total_compartments(cls, schema):
        """Append this model's fixed cumulative ``_total`` compartments.

        Replaces the framework's per-edge auto-generation, which would only
        produce ``I_total`` and ``R_total``; the remaining trackers follow manual
        multi-rate flows and are assigned explicitly in ``equation``.

        Args:
            schema (ModelParameterSchema): Schema whose ``compartments`` list
                receives one entry per ``_TOTAL_COMPARTMENTS`` definition.
        """
        # Replaces the framework's per-edge auto-generation entirely: every
        # _total this model needs is declared below, in a fixed order, and its
        # derivative is assigned explicitly at the end of equation().
        for cid, label, description in cls._TOTAL_COMPARTMENTS:
            schema.compartments.append(
                CompartmentDef(id=cid, label=label, description=description)
            )

    @classmethod
    def define_parameters(cls, schema):
        """Declare the SEIHFR compartments, parameters, and control interventions.

        Declares the two simple transmission edges (E->I incubation, F->R burial),
        the community/hospital/burial transmission coefficients as day-scaled
        Legrand et al. 2007 DRC 1995 estimates, the raw durations feeding the
        derived exit-split rates, and one intervention per transmission route.

        Args:
            schema (ParameterSchemaBuilder): Schema builder to populate.
        """
        schema.set_model_info(
            disease_type="ebola-seihfr-burial-legrand",
            label="Ebola (SEIHFR, Community/Hospital/Burial) — Legrand et al. 2007",
            description=(
                "SEIHFR compartmental model of Ebola virus disease with explicit "
                "community, hospital, and traditional-burial transmission routes, "
                "fitted to the 1995 DRC and 2000 Uganda outbreaks (Legrand et al., "
                "Epidemiol. Infect. 2007)."
            ),
        )

        schema.set_model_metadata(
            authors=[
                {
                    "name": "Judith Legrand, R F Grais, P Y Boelle, A J Valleron, A Flahault",
                    "affiliation": "INSERM UMR-S 707 / Universite Pierre et Marie Curie",
                }
            ],
            license="N/A (public research model; see paper for terms)",
            citations=[
                "Legrand J, Grais RF, Boelle PY, Valleron AJ, Flahault A. "
                "Understanding the dynamics of Ebola epidemics. "
                "Epidemiol Infect. 2007;135(4):610-621. "
                "doi:10.1017/S0950268806007217"
            ],
            model_type="Compartmental",
            diseases=["Ebola virus disease"],
            transmission_routes=["Direct contact (community)", "Nosocomial (hospital)", "Traditional burial"],
            questions_answered=[
                "How much do community, hospital, and burial transmission each contribute to R0?",
                "How does the timing and strength of control interventions change epidemic size?",
            ],
            key_assumptions=[
                "Homogeneous mixing within the population (no age/spatial structure).",
                "Entire population initially susceptible.",
                "Interventions are fully efficient from their start date onward "
                "(step-function activation, not gradual).",
                "After interventions, hospital and burial transmission are eliminated "
                "and community transmission is reduced by a fixed factor.",
            ],
        )

        # --- Compartments ---------------------------------------------------
        schema.add_compartment(
            "S", "Susceptible", "Population susceptible to Ebola infection",
        )
        schema.add_compartment(
            "E", "Exposed",
            "Infected but not yet infectious (incubating).",
        )
        schema.add_compartment(
            "I", "Infectious (community)",
            "Symptomatic and infectious in the community.",
            infective=True,
        )
        schema.add_compartment(
            "H", "Hospitalised",
            "Hospitalised (including isolation ward); infectious.",
            infective=True,
        )
        schema.add_compartment(
            "F", "Dead, awaiting burial",
            "Died of Ebola but not yet buried; infectious during traditional burial.",
            infective=True,
        )
        schema.add_compartment(
            "R", "Removed",
            "Removed from the chain of transmission (recovered or safely buried).",
        )

        # --- Transmission edges (simple single-rate flows only) ------------
        schema.add_transmission_parameter(
            source="E", target="I",
            variable_name="alpha",
            label="Incubation Period (E->I)",
            description="Mean duration of the incubation (exposed, non-infectious) period.",
            default=7.0,
            default_min=4.0, default_max=10.0,
            min_value=1.0, max_value=30.0,
            unit="days",
            value_type=ValueType.DAYS,
        )
        schema.add_transmission_parameter(
            source="F", target="R",
            variable_name="gamma_f",
            label="Death-to-Burial Period (F->R)",
            description="Mean duration between death and (safe or traditional) burial.",
            default=2.0,
            default_min=1.0, default_max=7.0,
            min_value=1.0, max_value=14.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

        # --- Disease parameters (multi-rate flows computed manually) -------
        # Transmission coefficients. DRC 1995 point estimates from Table 4 of
        # the paper: beta_I=0.588, beta_H=0.794, beta_F=7.653 week^-1,
        # converted here to day^-1 (/7) since the framework's solver runs in
        # days. 95% CIs (also /7) become default_min/default_max.
        schema.add_parameter(
            name="betaI",
            label="Transmission Rate — Community (βI)",
            description=(
                "Transmission coefficient for contact with symptomatic cases in "
                "the community, before interventions."
            ),
            value_type=ValueType.RATE,
            default=0.084,
            default_min=0.060, default_max=0.313,
            min_value=0.0, max_value=5.0,
            unit="per day",
        )
        schema.add_parameter(
            name="betaH",
            label="Transmission Rate — Hospital (βH)",
            description=(
                "Transmission coefficient for contact with hospitalised cases "
                "(including isolation ward), before interventions."
            ),
            value_type=ValueType.RATE,
            default=0.113,
            default_min=0.0001, default_max=0.584,
            min_value=0.0, max_value=5.0,
            unit="per day",
        )
        schema.add_parameter(
            name="betaF",
            label="Transmission Rate — Traditional Burial (βF)",
            description=(
                "Transmission coefficient for contact with the body during "
                "traditional burial, before interventions."
            ),
            value_type=ValueType.RATE,
            default=1.093,
            default_min=0.0001, default_max=1.428,
            min_value=0.0, max_value=5.0,
            unit="per day",
        )

        # Raw duration/rate parameters that feed the derived theta1/delta1/
        # delta2/gamma_ih/gamma_dh quantities used to split exits from I and H.
        # NOTE: these three use ValueType.FLOAT rather than ValueType.DAYS.
        # ValueType.DAYS on add_parameter() (unlike on add_transmission_parameter())
        # maps to a Pydantic *int* field in the auto-generated disease config
        # (see schema_generator.py's _VALUE_TYPE_TO_PYTHON), which rejects the
        # fractional published estimate gamma_d=9.6 and non-integer bounds.
        # FLOAT keeps native-unit (days) semantics; the mean-duration ->
        # per-day-rate conversion is still done manually via
        # self._to_rate(value, ValueType.DAYS) in equation().
        schema.add_parameter(
            name="gamma_h",
            label="Onset to Hospitalisation (γh⁻¹, days)",
            description=(
                "Mean duration from symptom onset to hospitalisation, for cases "
                "who are hospitalised."
            ),
            value_type=ValueType.FLOAT,
            default=5.0,
            default_min=1.0, default_max=10.0,
            min_value=0.5, max_value=30.0,
            unit="days",
        )
        schema.add_parameter(
            name="gamma_i",
            label="Infectious Period, Survivors (γi⁻¹, days)",
            description=(
                "Mean duration from symptom onset to end of infectiousness for "
                "cases who survive. Must exceed γh⁻¹ (patients are hospitalised "
                "before their infectious period would otherwise end)."
            ),
            value_type=ValueType.FLOAT,
            default=10.0,
            default_min=7.0, default_max=15.0,
            min_value=1.0, max_value=40.0,
            unit="days",
        )
        schema.add_parameter(
            name="gamma_d",
            label="Onset to Death, Non-Hospitalised (γd⁻¹, days)",
            description=(
                "Mean duration from symptom onset to death for cases who die "
                "without being hospitalised. Must exceed γh⁻¹."
            ),
            value_type=ValueType.FLOAT,
            default=9.6,
            default_min=7.0, default_max=12.0,
            min_value=1.0, max_value=40.0,
            unit="days",
        )
        schema.add_parameter(
            name="theta_target",
            label="Hospitalisation Proportion (θ)",
            description=(
                "Target proportion of infectious cases who are hospitalised. "
                "The per-step hospitalisation rate (theta1) is derived from this "
                "target together with γh, γi, γd, and δ — it is not set directly."
            ),
            value_type=ValueType.PERCENTAGE,
            default=80.0,
            default_min=50.0, default_max=100.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )
        schema.add_parameter(
            name="delta_target",
            label="Case-Fatality Ratio (δ)",
            description=(
                "Target overall case-fatality ratio across both hospitalised and "
                "non-hospitalised cases. The per-compartment death-split rates "
                "(delta1, delta2) are derived from this target."
            ),
            value_type=ValueType.PERCENTAGE,
            default=81.0,
            default_min=60.0, default_max=95.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )

        # --- Interventions ---------------------------------------------------
        # The paper assumes interventions are fully efficient once active
        # (assumption b) and, from that date on: hospital and burial
        # transmission are eliminated, community transmission is reduced by a
        # fixed factor (assumption d). Three separate interventions are used
        # (rather than one shared one) because the reduction magnitude differs
        # per rate, and add_intervention applies one reduction % to all of its
        # target_rates.
        schema.add_intervention(
            id="community_intervention",
            label="Community Transmission Control",
            description=(
                "Community education, household protective equipment, and "
                "contact-tracing measures that reduce (but do not eliminate) "
                "community transmission."
            ),
            target_rates=["betaI"],
            adherence=100.0,
            transmission_reduction=12.0,
        )
        schema.add_intervention(
            id="hospital_intervention",
            label="Hospital Isolation / Barrier Nursing",
            description=(
                "Isolation ward and barrier-nursing procedures at the hospital. "
                "The paper assumes these eliminate hospital transmission once "
                "active."
            ),
            target_rates=["betaH"],
            adherence=100.0,
            transmission_reduction=100.0,
        )
        schema.add_intervention(
            id="funeral_intervention",
            label="Safe Burial Practices",
            description=(
                "Replacement of traditional burial with safe burial practices "
                "(e.g. by trained response teams). The paper assumes this "
                "eliminates funeral transmission once active."
            ),
            target_rates=["betaF"],
            adherence=100.0,
            transmission_reduction=100.0,
        )

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, config):
        """Initialise the model from the validated configuration.

        The source model has no inter-zone travel and no demographic
        stratification, so no extra setup is needed beyond the base class.

        Args:
            config (dict): Validated simulation configuration.
        """
        super().__init__(config)
        # No inter-zone travel and no demographic stratification in the
        # source model — nothing further to set up here.

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    def prepare_initial_state(self):
        """Return the initial compartment populations for the solver.

        Returns:
            jnp.ndarray: Population matrix of shape (n_compartments, n_zones).
        """
        return self.population_matrix

    # ------------------------------------------------------------------
    # ODE
    # ------------------------------------------------------------------

    def equation(self, y, t, p):
        """Compute the SEIHFR compartment derivatives for one integration step.

        Mixes the community, hospital, and burial transmission coefficients into a
        single S->E force of infection, then derives the theta1/delta1/delta2 exit
        splits for I and H from gamma_h, gamma_i, gamma_d and the target
        hospitalisation and case-fatality proportions (Legrand et al. 2007).

        Args:
            y (jnp.ndarray): Current compartment values, ordered by
                ``compartment_list``.
            t (float): Current time in days since the simulation start date.
            p (tuple): Packed transmission-edge parameter tuple (alpha, gamma_f),
                unpacked via ``_unpack_params``.

        Returns:
            jnp.ndarray: Stacked per-compartment derivatives (dy/dt).
        """
        states = {c: y[i] for i, c in enumerate(self.compartment_list)}
        params = self._unpack_params(p)

        S, E, I, H, F = (
            states["S"], states["E"], states["I"], states["H"], states["F"],
        )

        non_total = [c for c in self.compartment_list if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)
        prop_infective = (I + H + F).sum() / (N_total.sum() + 1e-10)

        # --- Interventions on the three transmission coefficients ----------
        disease_rates = {"betaI": self.betaI, "betaH": self.betaH, "betaF": self.betaF}
        disease_rates, self.travel_matrix = self._apply_interventions(
            t, disease_rates, prop_infective
        )
        betaI, betaH, betaF = (
            disease_rates["betaI"], disease_rates["betaH"], disease_rates["betaF"],
        )

        # --- Framework-handled simple edges: E->I (alpha), F->R (gamma_f) --
        derivs = self._compute_equations(states, params)

        # --- Manual multi-source FOI: S -> E --------------------------------
        foi = (betaI * I + betaH * H + betaF * F) / (N_total + 1e-10)
        flow_S_to_E = S * foi
        self._apply_flow(derivs, "S", "E", flow_S_to_E)

        # --- Derived split rates for I's and H's exits ----------------------
        gamma_h = self._to_rate(self.gamma_h, ValueType.DAYS)
        gamma_i = self._to_rate(self.gamma_i, ValueType.DAYS)
        gamma_d = self._to_rate(self.gamma_d, ValueType.DAYS)
        theta = self._to_rate(self.theta_target, ValueType.PERCENTAGE)
        delta = self._to_rate(self.delta_target, ValueType.PERCENTAGE)

        eps = 1e-10
        # 1/gamma_ih = 1/gamma_i - 1/gamma_h ; 1/gamma_dh = 1/gamma_d - 1/gamma_h
        gamma_ih = 1.0 / jnp.maximum(1.0 / gamma_i - 1.0 / gamma_h, eps)
        gamma_dh = 1.0 / jnp.maximum(1.0 / gamma_d - 1.0 / gamma_h, eps)

        delta1 = (delta * gamma_i) / (delta * gamma_i + (1.0 - delta) * gamma_d + eps)
        delta2 = (delta * gamma_ih) / (delta * gamma_ih + (1.0 - delta) * gamma_dh + eps)

        hosp_weight = theta * (gamma_i * (1.0 - delta1) + gamma_d * delta1)
        theta1 = hosp_weight / (hosp_weight + (1.0 - theta) * gamma_h + eps)

        # --- Manual exits from I: to H, R (survivors), F (deaths) ----------
        flow_I_to_H = gamma_h * theta1 * I
        flow_I_to_R = gamma_i * (1.0 - theta1) * (1.0 - delta1) * I
        flow_I_to_F = gamma_d * (1.0 - theta1) * delta1 * I
        self._apply_flow(derivs, "I", "H", flow_I_to_H)
        self._apply_flow(derivs, "I", "R", flow_I_to_R)
        self._apply_flow(derivs, "I", "F", flow_I_to_F)

        # --- Manual exits from H: to R (survivors), F (deaths) -------------
        flow_H_to_R = gamma_ih * (1.0 - delta2) * H
        flow_H_to_F = gamma_dh * delta2 * H
        self._apply_flow(derivs, "H", "R", flow_H_to_R)
        self._apply_flow(derivs, "H", "F", flow_H_to_F)

        # --- Cumulative inflow trackers -------------------------------------
        # Assigned (not incremented) after every flow above, so these
        # deliberately overwrite the implicit _total bookkeeping done inside
        # _compute_equations / _apply_flow instead of double-counting it.
        # Each is the sum of all inflows to the corresponding compartment.
        derivs["E_total"] = flow_S_to_E
        derivs["I_total"] = params["alpha"] * E
        derivs["H_total"] = flow_I_to_H
        derivs["F_total"] = flow_I_to_F + flow_H_to_F
        derivs["R_total"] = flow_I_to_R + flow_H_to_R + params["gamma_f"] * F

        return jnp.stack([derivs[c] for c in self.compartment_list])
