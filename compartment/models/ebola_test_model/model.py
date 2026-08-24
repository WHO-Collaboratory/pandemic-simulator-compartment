"""Ebola SEIHFR model with community / hospital / burial transmission routes.

Port of the deterministic SEIHFR compartmental model in

    Legrand J, Grais RF, Boelle PY, Valleron AJ, Flahault A.
    "Understanding the dynamics of Ebola epidemics."
    Epidemiol Infect. 2007;135(4):610-621. doi:10.1017/S0950268806007217

as implemented in the ``ebola_SEIHFR_model.ipynb`` source notebook. The paper
splits the infectious period into three phases with distinct transmission
coefficients — illness in the community (I), hospitalisation (H), and
traditional burial of the dead (F). The default parameter values below are the
notebook's ``DRC`` preset (Zaire EBOV, Kikwit 1995).

Compartments: S, E, I, H, F, R.

Two structural features of the source model do not fit the framework's standard
"declarative transmission edge" pattern and are therefore computed manually in
``equation()`` (see the Pitfalls section of ``.claude/MODEL_AUTHORING_REFERENCE.md``
for the general pattern, and ``hantavirus_jax_model`` for a precedent):

1. The force of infection mixes three distinct betas (betaI, betaH, betaF)
   across three distinct infectious compartments (I, H, F) into a single
   S -> E flow. ``_compute_equations`` supports only one shared infective sum
   per frequency-dependent edge, so this multi-rate FOI is applied by hand.
2. The three-way split of exits from I (to H, R, or F) and the two-way split of
   exits from H (to R or F) use rates theta1, delta1, delta2 that are themselves
   *derived* at every step from the raw epidemiological parameters (gamma_h,
   gamma_i, gamma_d and the target hospitalisation/case-fatality proportions
   theta, delta) via the algebraic relationships in the paper's Appendix. These
   are not standalone ``rate * source`` edges.

The E -> I (incubation, alpha) and F -> R (burial, gamma_f) transitions *are*
simple single-rate flows and are declared as ordinary transmission edges.

WARNING: Like the other models in this repo, this implementation is intended for
local experimentation and scenario exploration in the Pandemic Simulator.
"""

import logging

import jax.numpy as jnp

from compartment.model import Model, ValueType
from compartment.parameters import CompartmentDef

logger = logging.getLogger(__name__)


class EbolaTestModel(Model):
    """SEIHFR Ebola model with community/hospital/burial transmission (Legrand et al. 2007)."""

    # Cumulative inflow trackers. Declared here (rather than relying on the
    # framework's per-edge auto-generation) and computed explicitly in
    # equation(), because most of the inflows they track are manual multi-rate
    # flows, not transmission edges — auto-generation would only produce
    # I_total and R_total. Same pattern as dengue_jax_model /
    # ebola_seihfr_burial_legrand_model.
    _TOTAL_COMPARTMENTS = (
        ("E_total", "Exposed Total",
         "Cumulative number of individuals ever exposed (cumulative infections)."),
        ("I_total", "Infectious (community) Total",
         "Cumulative number of individuals who ever became infectious "
         "(cumulative symptom onsets). Accumulates the alpha*E inflow only, so "
         "the initial seed cases — placed directly into I — are NOT counted. "
         "This is the source notebook's C."),
        ("H_total", "Hospitalised Total",
         "Cumulative number of individuals ever hospitalised. Unlike I_total "
         "this does include the seed cases; on resolution it converges to "
         "theta * (I_total + I0)."),
        ("F_total", "Deaths Total",
         "Cumulative number of individuals who have died of Ebola *by the "
         "current time step* — the realised flow into F, not a projection of "
         "eventual deaths. Includes the seed cases, which I_total excludes. "
         "Once the outbreak has fully resolved this converges exactly to "
         "delta * (I_total + I0); before then it is strictly lower, because "
         "cases still in E/I/H have not died yet. See the 'Reading the deaths "
         "output' section of model.md."),
        ("R_total", "Removed Total",
         "Cumulative number of individuals removed (recovered or safely buried)."),
    )

    @classmethod
    def _add_total_compartments(cls, schema):
        """Declare the cumulative ``_total`` compartments this model tracks.

        Replaces the framework's per-edge auto-generation entirely: every
        ``_total`` this model needs is declared below, in a fixed order, and its
        derivative is assigned explicitly at the end of ``equation()``.

        Args:
            schema: The schema builder whose compartment list is appended to.
        """
        for cid, label, description in cls._TOTAL_COMPARTMENTS:
            schema.compartments.append(
                CompartmentDef(id=cid, label=label, description=description)
            )

    @classmethod
    def define_parameters(cls, schema):
        """Declare the SEIHFR compartments, edges, disease parameters, and interventions.

        Args:
            schema: The schema builder to populate with model info, metadata,
                compartments, transmission edges, disease parameters, and the
                three route-specific interventions.
        """
        schema.set_model_info(
            disease_type="ebola_test",
            label="Ebola (SEIHFR, Community/Hospital/Burial) — Legrand 2007 [test]",
            description=(
                "SEIHFR compartmental model of Ebola virus disease with explicit "
                "community, hospital, and traditional-burial transmission routes, "
                "based on Legrand et al. (Epidemiol. Infect. 2007) and ported from "
                "the ebola_SEIHFR notebook. Defaults are the DRC (Kikwit) 1995 preset."
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
            transmission_routes=[
                "Direct contact (community)",
                "Nosocomial (hospital)",
                "Traditional burial",
            ],
            questions_answered=[
                "How much do community, hospital, and burial transmission each contribute to R0?",
                "How does the timing and strength of control interventions change epidemic size?",
            ],
            key_assumptions=[
                "Homogeneous mixing within a closed population (no age/spatial structure).",
                "No background births or deaths; the population is conserved (Ebola "
                "deaths pass through F to the removed class R).",
                "Entire population initially susceptible apart from the seed cases.",
                "Interventions activate as a step function on their start date and, "
                "with a null end date, remain active thereafter.",
                "Frequency-dependent force of infection.",
            ],
        )

        # --- Compartments --------------------------------------------------
        # NOTE on infective=True below: this model computes its force of
        # infection by hand in equation() (three betas, three compartments), so
        # the framework never builds an infective_sum for it. The flags are set
        # for descriptive accuracy and for any future declarative edge; they do
        # not currently drive any flow. Changing them will not change results.
        schema.add_compartment(
            "S", "Susceptible", "Population susceptible to Ebola infection",
        )
        schema.add_compartment(
            "E", "Exposed",
            "Infected but not yet infectious (incubating).",
        )
        schema.add_compartment(
            "I", "Infectious (community)",
            "Symptomatic and infectious in the community (not yet hospitalised).",
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
        # E -> I incubation (alpha = 1/d_E) and F -> R burial (gamma_f = 1/d_f).
        # Declared as DAYS so the framework converts the mean duration to a
        # per-day rate at load; do not pre-divide.
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
        # Transmission coefficients. Notebook DRC preset (Table 4, Legrand 2007):
        # betaI=0.588, betaH=0.794, betaF=7.653 week^-1. The framework solver runs
        # in days, so these are stored in day^-1 (weekly value / 7):
        #   0.588 / 7 = 0.084        (exact)
        #   0.794 / 7 = 0.1134286
        #   7.653 / 7 = 1.0932857
        # Keep these at full precision. Rounding them to 3 decimals (0.113,
        # 1.093) understates betaH by 0.38% and shifts pre-intervention R0 from
        # 2.694 to 2.692, which compounds over a long uncontrolled run.
        schema.add_parameter(
            name="betaI",
            label="Transmission Rate — Community (betaI)",
            description=(
                "Transmission coefficient for contact with symptomatic cases in "
                "the community, before interventions (per day)."
            ),
            value_type=ValueType.RATE,
            default=0.084,
            default_min=0.060, default_max=0.313,
            min_value=0.0, max_value=5.0,
            unit="per day",
        )
        schema.add_parameter(
            name="betaH",
            label="Transmission Rate — Hospital (betaH)",
            description=(
                "Transmission coefficient for contact with hospitalised cases "
                "(including isolation ward), before interventions (per day)."
            ),
            value_type=ValueType.RATE,
            default=0.1134286,
            default_min=0.0001, default_max=0.584,
            min_value=0.0, max_value=5.0,
            unit="per day",
        )
        schema.add_parameter(
            name="betaF",
            label="Transmission Rate — Traditional Burial (betaF)",
            description=(
                "Transmission coefficient for contact with the body during "
                "traditional burial, before interventions (per day)."
            ),
            value_type=ValueType.RATE,
            default=1.0932857,
            default_min=0.0001, default_max=1.428,
            min_value=0.0, max_value=5.0,
            unit="per day",
        )

        # Raw duration parameters feeding the derived theta1/delta1/delta2/
        # gamma_ih/gamma_dh quantities that split exits from I and H.
        # NOTE: these three use ValueType.FLOAT rather than ValueType.DAYS.
        # ValueType.DAYS on add_parameter() (unlike on add_transmission_parameter())
        # maps to a Pydantic *int* field in the auto-generated disease config
        # (see schema_generator.py's _VALUE_TYPE_TO_PYTHON), which would reject
        # the fractional published estimate gamma_d=9.6. FLOAT keeps native
        # (days) units; the days -> per-day-rate conversion is done manually via
        # self._to_rate(value, ValueType.DAYS) in equation().
        schema.add_parameter(
            name="gamma_h",
            label="Onset to Hospitalisation (1/gamma_h, days)",
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
            label="Infectious Period, Survivors (1/gamma_i, days)",
            description=(
                "Mean duration from symptom onset to end of infectiousness for "
                "cases who survive in the community. Must exceed 1/gamma_h."
            ),
            value_type=ValueType.FLOAT,
            default=10.0,
            default_min=7.0, default_max=15.0,
            min_value=1.0, max_value=40.0,
            unit="days",
        )
        schema.add_parameter(
            name="gamma_d",
            label="Onset to Death, Non-Hospitalised (1/gamma_d, days)",
            description=(
                "Mean duration from symptom onset to death for cases who die "
                "without being hospitalised. Must exceed 1/gamma_h."
            ),
            value_type=ValueType.FLOAT,
            default=9.6,
            default_min=7.0, default_max=12.0,
            min_value=1.0, max_value=40.0,
            unit="days",
        )
        schema.add_parameter(
            name="theta_target",
            label="Hospitalisation Proportion (theta)",
            description=(
                "Target proportion of infectious cases who are hospitalised. The "
                "per-step hospitalisation rate (theta1) is derived from this target "
                "together with gamma_h, gamma_i, gamma_d, and delta — not set directly."
            ),
            value_type=ValueType.PERCENTAGE,
            default=80.0,
            default_min=50.0, default_max=100.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )
        schema.add_parameter(
            name="delta_target",
            label="Case-Fatality Ratio (delta)",
            description=(
                "Target overall case-fatality ratio across hospitalised and "
                "non-hospitalised cases. The per-compartment death-split rates "
                "(delta1, delta2) are derived from this target."
            ),
            value_type=ValueType.PERCENTAGE,
            default=81.0,
            default_min=60.0, default_max=95.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )

        # --- Interventions -------------------------------------------------
        # The notebook scales each route independently by a factor z in [0, 1]
        # from that route's start date onward (0 = route eliminated,
        # 1 = unchanged). The framework's step intervention applies
        # rate * (1 - adherence * transmission_reduction); with adherence = 100%
        # this is rate * (1 - reduction), so transmission_reduction = 1 - z.
        # Three separate interventions (one per beta) are used because each
        # route has its own start date and reduction magnitude.
        # DRC preset defaults: z_community = 0.50 (reduction 50%),
        # z_hospital = 0.0 (reduction 100%), z_funeral = 0.0 (reduction 100%).
        schema.add_intervention(
            id="community_intervention",
            label="Community Transmission Control",
            description=(
                "Community education, household protective equipment, and "
                "contact-tracing measures that reduce (but need not eliminate) "
                "community transmission."
            ),
            target_rates=["betaI"],
            adherence=100.0,
            transmission_reduction=50.0,
        )
        schema.add_intervention(
            id="hospital_intervention",
            label="Hospital Isolation / Barrier Nursing",
            description=(
                "Isolation ward and barrier-nursing procedures at the hospital. "
                "The DRC preset assumes these eliminate hospital transmission once "
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
                "(e.g. by trained response teams). The DRC preset assumes this "
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
        """Initialize the model from a validated simulation config.

        Args:
            config: The validated simulation configuration produced by the
                framework's config loader.
        """
        super().__init__(config)
        # No inter-zone travel and no demographic stratification in the source
        # model — nothing further to set up here.

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    def prepare_initial_state(self):
        """Return the initial compartment populations for the solver.

        The framework seeds the ``infected_population`` percentage into the ``I``
        compartment (matching the notebook's ``I0`` community seed), leaving E,
        H, F, R at zero.

        ``infected_population`` is a **percentage**, not a fraction or a count:
        ``get_initial_population`` computes
        ``round(infected_population / 100 * population, 2)``. The example config's
        ``0.0015`` therefore means 0.0015 % of 200,000 = **3** seed cases, which is
        the notebook's ``I0``. Writing ``3`` there would seed 3 % = 6,000 cases.

        Because the seeds enter ``I`` directly they never pass through the
        ``alpha*E`` inflow, so they are absent from ``I_total`` but present in
        ``H_total``/``F_total``/``R_total``.

        Returns:
            The population matrix used as the solver's initial state.
        """
        return self.population_matrix

    # ------------------------------------------------------------------
    # ODE
    # ------------------------------------------------------------------

    def equation(self, y, t, p):
        """Compute the SEIHFR compartment derivatives for one integration step.

        Args:
            y: Current compartment values, ordered by ``compartment_list``.
            t: Current time in days since the simulation start date.
            p: Packed transmission-edge parameter tuple (alpha, gamma_f),
                unpacked via ``_unpack_params``.

        Returns:
            The stacked per-compartment derivatives (dy/dt).
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

        # --- Manual multi-source FOI: S -> E -------------------------------
        # lambda = (betaI*I + betaH*H + betaF*F) / N ; frequency-dependent.
        foi = (betaI * I + betaH * H + betaF * F) / (N_total + 1e-10)
        flow_S_to_E = S * foi
        self._apply_flow(derivs, "S", "E", flow_S_to_E)

        # --- Derived split rates for I's and H's exits ---------------------
        gamma_h = self._to_rate(self.gamma_h, ValueType.DAYS)
        gamma_i = self._to_rate(self.gamma_i, ValueType.DAYS)
        gamma_d = self._to_rate(self.gamma_d, ValueType.DAYS)
        theta = self._to_rate(self.theta_target, ValueType.PERCENTAGE)
        delta = self._to_rate(self.delta_target, ValueType.PERCENTAGE)

        eps = 1e-10
        # In-hospital residual rates: 1/gamma_ih = 1/gamma_i - 1/gamma_h ;
        #                             1/gamma_dh = 1/gamma_d - 1/gamma_h.
        gamma_ih = 1.0 / jnp.maximum(1.0 / gamma_i - 1.0 / gamma_h, eps)
        gamma_dh = 1.0 / jnp.maximum(1.0 / gamma_d - 1.0 / gamma_h, eps)

        # Routing probabilities derived from the observed theta / delta targets.
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

        # --- Cumulative inflow trackers ------------------------------------
        # Assigned (not incremented) after all flows above, deliberately
        # overwriting the implicit _total bookkeeping done inside
        # _compute_equations / _apply_flow rather than double-counting it. Each
        # is the sum of all inflows to the corresponding compartment. I_total
        # tracks cumulative symptom onsets (alpha*E), i.e. the notebook's C.
        derivs["E_total"] = flow_S_to_E
        derivs["I_total"] = params["alpha"] * E
        derivs["H_total"] = flow_I_to_H
        derivs["F_total"] = flow_I_to_F + flow_H_to_F
        derivs["R_total"] = flow_I_to_R + flow_H_to_R + params["gamma_f"] * F

        return jnp.stack([derivs[c] for c in self.compartment_list])
