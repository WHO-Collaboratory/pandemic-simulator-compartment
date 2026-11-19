"""Stochastic SEIHFR Ebola model — community / hospital / burial routes.

Author: Yingsi Lai.

Stochastic (demographic-noise) counterpart of ``ebola_seihfr_burial_legrand_model``, both ported
from the notebook ``ebola_SEIHFR_model.ipynb``. Model structure and DRC 1995
parameter values are based on

    Legrand J, Grais RF, Boelle PY, Valleron AJ, Flahault A.
    "Understanding the dynamics of Ebola epidemics."
    Epidemiol Infect. 2007;135(4):610-621. doi:10.1017/S0950268806007217

The source of each default parameter value is documented in ``model.md``
("Parameters" table, and "Differences from Legrand 2007"). The paper's model is
stochastic — *"Simulations of the model were performed using Gillespie's first
reaction method"* — with the eight transitions of its Table 2.

The Pandemic Simulator integrates on a **fixed daily step** (JAX ``odeint`` or,
for stochastic models, fixed-step Euler), so it cannot run the paper's exact
event-driven Gillespie algorithm. This model therefore uses the framework's
stochastic path (``STOCHASTIC = True`` -> Euler + per-step random event draws):
a **chain-binomial tau-leap**. Each day, the number of individuals making each
transition is a binomial draw with the exact competing-risk probability
``1 - exp(-hazard)``; individuals leaving a compartment with several possible
destinations (I -> H/R/F, H -> R/F) are split multinomially in the ratio of the
destination hazards. This

  * conserves the population exactly (draws are bounded by the source count),
  * reproduces the target hospitalisation proportion ``theta`` and
    case-fatality ratio ``delta`` on expectation (the routing split preserves
    ``theta1``/``delta1``/``delta2``), and
  * matches the deterministic model's mean trajectory.

For an *exact* Gillespie first-reaction implementation (event-driven, continuous
time — truest to the paper, and the right tool for fine-grained early-extinction
probability), see the standalone ``ebola_seihfr_gillespie.py`` /
``ebola_SEIHFR_stochastic.ipynb``. See the "Tau-leap vs. exact Gillespie" note in
``model.md``.

Compartments: S, E, I, H, F, R (+ cumulative ``_total`` trackers).
Defaults are the DRC (Kikwit) 1995 preset (Zaire EBOV), matching ``ebola_seihfr_burial_legrand_model``.
"""

import logging

import numpy as np

from compartment.model import Model, ValueType
from compartment.parameters import CompartmentDef

logger = logging.getLogger(__name__)


class EbolaSeihfrStochasticLaiModel(Model):
    """Stochastic SEIHFR Ebola model by Yingsi Lai (chain-binomial tau-leap; based on Legrand et al. 2007)."""

    # Fixed-step Euler + multi-trajectory median/interval output.
    STOCHASTIC = True

    # Number of stochastic trajectories reported as a median + 95% band. This is
    # the fallback default; Model.__init_subclass__ overwrites it with
    # schema.num_runs, i.e. the default of the ``num_runs`` parameter declared in
    # define_parameters() (30; allowed range 10-100). Change the trajectory count
    # via that parameter / the config, not here.
    NUM_RUNS = 30

    # Cumulative inflow trackers. Declared explicitly (rather than relying on the
    # framework's per-edge auto-generation) and assigned in equation(), because
    # most of the inflows they track are manual multi-rate event draws, not
    # transmission edges. Same pattern as ebola_seihfr_burial_legrand_model.
    _TOTAL_COMPARTMENTS = (
        ("E_total", "Exposed Total",
         "Cumulative number of individuals ever exposed (cumulative infections)."),
        ("I_total", "Infectious (community) Total",
         "Cumulative number of individuals who ever became infectious "
         "(cumulative symptom onsets). Counts the E->I events only, so the "
         "initial seed cases — placed directly into I — are NOT included. "
         "This is the notebook's C."),
        ("H_total", "Hospitalised Total",
         "Cumulative number of individuals ever hospitalised (includes seeds)."),
        ("F_total", "Deaths Total",
         "Cumulative number of individuals who have died of Ebola by the current "
         "step (realised flow into F, including seed cases). Converges to "
         "delta * (I_total + I0) once the outbreak resolves."),
        ("R_total", "Removed Total",
         "Cumulative number of individuals removed (recovered or safely buried)."),
    )

    @classmethod
    def _add_total_compartments(cls, schema):
        """Declare the cumulative ``_total`` compartments this model tracks."""
        for cid, label, description in cls._TOTAL_COMPARTMENTS:
            schema.compartments.append(
                CompartmentDef(id=cid, label=label, description=description)
            )

    @classmethod
    def define_parameters(cls, schema):
        """Declare compartments, edges, disease parameters, and interventions.

        Args:
            schema (ParameterSchemaBuilder): builder to fill in.
        """
        schema.set_model_info(
            disease_type="ebola_seihfr_stochastic_lai",
            label="Ebola – SEIHFR, stochastic (Lai 2026)",
            description=(
                "Stochastic (chain-binomial tau-leap) SEIHFR model of Ebola virus "
                "disease with explicit community, hospital, and traditional-burial "
                "transmission routes, built by Yingsi Lai. Model structure and "
                "parameter values are based on Legrand et al. (Epidemiol. Infect. "
                "2007). Demographic-noise counterpart of the deterministic "
                "ebola_seihfr_burial_legrand_model. Defaults are the DRC (Kikwit) 1995 preset; see "
                "model.md for the source of each default and where they differ "
                "from the paper."
            ),
        )

        schema.set_model_metadata(
            authors=[
                {
                    "name": "Yingsi Lai",
                    "affiliation": "NA",
                }
            ],
            license="Apache-2.0",
            citations=[
                "Legrand J, Grais RF, Boelle PY, Valleron AJ, Flahault A. "
                "Understanding the dynamics of Ebola epidemics. "
                "Epidemiol Infect. 2007;135(4):610-621. "
                "https://doi.org/10.1017/S0950268806007217 "
                "-- model structure and DRC 1995 parameter values (Tables 3-4).",
                "Khan AS, et al. The reemergence of Ebola hemorrhagic fever, "
                "Democratic Republic of the Congo, 1995. "
                "J Infect Dis. 1999;179(Suppl 1):S76-S86. "
                "https://doi.org/10.1086/514306 "
                "-- primary data source for the DRC 1995 defaults, via Legrand et al.",
            ],
            model_type="Compartmental (SEIHFR, stochastic chain-binomial tau-leap)",
            diseases=["Ebola virus disease"],
            transmission_routes=[
                "Direct contact (community)",
                "Nosocomial (hospital)",
                "Traditional burial",
            ],
            questions_answered=[
                "How much do community, hospital, and burial transmission each "
                "contribute to R0?",
                "How do the timing and strength of hospital isolation, safe burial, "
                "and community measures change outbreak size and deaths?",
            ],
            key_assumptions=[
                "Homogeneous mixing within a single closed population (no age or "
                "spatial structure, no importations after the seed cases).",
                "No background births or deaths; the population is conserved (Ebola "
                "deaths pass through F to the removed class R).",
                "Entire population initially susceptible apart from the seed cases.",
                "Frequency-dependent force of infection with three routes: "
                "community (I), hospital (H), and traditional burial (F).",
                "Hospitalised proportion (theta) and case-fatality ratio (delta) are "
                "fixed for the whole simulation; interventions change only the "
                "transmission coefficients.",
                "Interventions activate as a step function on their start date (fully "
                "effective immediately) and, with a null end date, remain active "
                "thereafter.",
                "Demographic stochasticity via a chain-binomial tau-leap on a daily "
                "step; per-transition counts are binomial with probability "
                "1 - exp(-hazard), multinomially split across competing destinations.",
                "Default parameters are the Kikwit, DRC 1995 values from Legrand et al. "
                "2007, except the case-fatality ratio (default 50%; the paper reports "
                "81%). See model.md.",
            ],
            applicability=(
                "Scenario exploration for Ebola outbreaks in a single, well-mixed "
                "population (e.g. a town or health zone) where community, nosocomial, "
                "and traditional-burial transmission all matter. Suited to comparing "
                "the timing and strength of hospital isolation, safe burial, and "
                "community measures, and to showing the spread of outbreak sizes that "
                "demographic noise produces. Defaults reproduce the Kikwit, DRC 1995 "
                "(Zaire ebolavirus) setting of Legrand et al. 2007; other outbreaks "
                "need their own parameter values."
            ),
            not_for=(
                "Real-time forecasting or operational case projections. Precise "
                "early-extinction (fade-out) probabilities of small introductions -- "
                "the daily tau-leap understates them; use the exact Gillespie "
                "implementation instead. Multi-district, cross-border, or "
                "mobility-driven spread; age-specific questions; vaccination, "
                "therapeutics, or contact-tracing capacity, none of which are "
                "modelled. The no-intervention control run is a relative baseline, "
                "not a forecast."
            ),
            constraints=(
                "Fixed one-day step (chain-binomial tau-leap), not continuous-time "
                "Gillespie. Interventions are step changes on the three transmission "
                "coefficients only. The derived split rates require the survivor "
                "infectious period (1/gamma_i) and onset-to-death period (1/gamma_d) "
                "to exceed onset-to-hospitalisation (1/gamma_h). Number of "
                "trajectories (num_runs) is 10-100, default 30 (about 1 s per "
                "trajectory per scenario locally). Each trajectory is seeded from "
                "operating-system entropy, so runs are not reproducible. "
                "Cumulative _total series are stripped from the per-timestep output "
                "and appear only as final values in compartment_deltas."
            ),
            biases=(
                "The default case-fatality ratio (50%) is lower than the 81% observed "
                "in Kikwit 1995, so with defaults deaths and the burial contribution "
                "to R0 are lower than in the paper's DRC setting. Transmission "
                "coefficients come from one outbreak and have wide confidence "
                "intervals (Legrand et al. 2007, Table 4). Step-function "
                "interventions that act at full strength from day one overstate how "
                "quickly control takes hold. The daily tau-leap understates fade-out "
                "of small introductions, so results lean towards outbreaks that take "
                "off. Homogeneous mixing ignores household and healthcare-worker "
                "clustering. The unbounded control run gives a much larger attack "
                "rate than any observed Ebola outbreak."
            ),
            validation=(
                "Internal consistency checks: population is conserved exactly and no "
                "compartment goes negative on any trajectory; the realised "
                "case-fatality ratio and hospitalised fraction converge to delta and "
                "theta; the ensemble mean matches the deterministic ebola_seihfr_burial_legrand_model; "
                "the mean size of outbreaks that take off agrees with a standalone "
                "exact-Gillespie implementation. NumPy binomial draws are used because "
                "jax.random.binomial was found to inflate outbreak size by about 20%. "
                "The model is not re-fitted: transmission parameters are taken from "
                "Legrand et al. 2007, who fitted them to the 1995 DRC data, and no "
                "comparison with the observed Kikwit incidence is part of this model."
            ),
        )

        # --- Compartments --------------------------------------------------
        # infective=True is descriptive here: the force of infection is computed
        # by hand in equation() (three betas, three compartments), so the flags
        # do not drive any framework-generated flow.
        schema.add_compartment(
            "S", "Susceptible", "Population susceptible to Ebola infection",
        )
        schema.add_compartment(
            "E", "Exposed", "Infected but not yet infectious (incubating).",
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

        # --- Transmission edges (define alpha and gamma_f rates) -----------
        # E -> I incubation (alpha = 1/d_E) and F -> R burial (gamma_f = 1/d_f).
        # Declared as DAYS so the framework converts the mean duration to a
        # per-day rate at load. The flows themselves are drawn manually in
        # equation(); these edges exist so the rate parameters are defined.
        # Defaults: Legrand et al. 2007, Table 3 (DRC 1995) — incubation 7 d,
        # death-to-burial 2 d. Slider ranges are assumptions.
        schema.add_transmission_parameter(
            source="E", target="I",
            variable_name="alpha",
            label="Incubation Period (E->I)",
            description="Mean duration of the incubation (exposed, non-infectious) period.",
            default=7.0,
            default_min=5.0, default_max=9.0,
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
            default_min=1.0, default_max=3.0,
            min_value=1.0, max_value=14.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

        # --- Disease parameters (multi-rate flows computed manually) -------
        # Transmission coefficients (per day) = weekly Legrand et al. 2007
        # Table 4 value (DRC 1995, maximum-likelihood estimates) / 7, at full
        # precision: 0.588/7=0.084, 0.794/7=0.1134286, 7.653/7=1.0932857. Do not
        # round to 3 dp (that shifts pre-intervention R0 from 2.694 to 2.692).
        schema.add_parameter(
            name="betaI",
            label="Transmission Rate — Community (betaI)",
            description=(
                "Transmission coefficient for contact with symptomatic cases in "
                "the community, before interventions (per day)."
            ),
            value_type=ValueType.RATE,
            default=0.084,
            default_min=0.064, default_max=0.104,
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
            default_min=0.07, default_max=0.15,
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
            default_min=0.8, default_max=1.4,
            min_value=0.0, max_value=5.0,
            unit="per day",
        )

        # Raw duration parameters feeding the derived theta1/delta1/delta2/
        # gamma_ih/gamma_dh split rates. FLOAT (not DAYS) so the fractional
        # gamma_d=9.6 is accepted; days -> per-day-rate conversion is done via
        # self._to_rate(value, ValueType.DAYS) in equation().
        # Defaults: Legrand et al. 2007, Table 3 (DRC 1995) — onset to
        # hospitalisation 5 d, onset to end of infectiousness (survivors) 10 d,
        # onset to death 9.6 d, hospitalised proportion 80%. The case-fatality
        # default (50%) is NOT the paper's DRC value (81%); see model.md,
        # "Differences from Legrand 2007". Slider ranges are assumptions.
        schema.add_parameter(
            name="gamma_h",
            label="Onset to Hospitalisation (1/gamma_h, days)",
            description=(
                "Mean duration from symptom onset to hospitalisation, for cases "
                "who are hospitalised."
            ),
            value_type=ValueType.FLOAT,
            default=5.0,
            default_min=1.0, default_max=9.0,
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
            default=50.0,
            default_min=10.0, default_max=90.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )

        # Number of stochastic trajectories. run_simulation reads this to decide
        # how many runs to average for the median + 95% interval band. The
        # framework derives the class attribute NUM_RUNS from this parameter's
        # default (Model.__init_subclass__ sets cls.NUM_RUNS = schema.num_runs);
        # min_value/max_value limit a configured value to 10-100. Declared
        # exactly as in example_stochastic_model.
        schema.add_parameter(
            name="num_runs",
            label="Number of Stochastic Trajectories",
            description=(
                "How many independent stochastic realisations to simulate. The "
                "reported series is the per-timestep median with a 95% interval "
                "band across the trajectories."
            ),
            value_type=ValueType.INTEGER,
            default=30,
            min_value=10,
            max_value=100,
            enable_variance=False,
        )

        # --- Interventions -------------------------------------------------
        # Each route is scaled independently from its start date. The framework
        # step intervention applies rate * (1 - adherence * transmission_reduction).
        # In Legrand et al.'s notation the remaining fraction is z, so
        # adherence * transmission_reduction = 1 - z. Defaults for all three:
        # adherence 50%, transmission_reduction 100% -> each targeted beta is
        # halved once active (z = 0.5). Legrand et al. instead set hospital and
        # burial transmission to 0 (z = 0) and estimated a 12% community
        # reduction for DRC 1995; see model.md, "Differences from Legrand 2007".
        schema.add_intervention(
            id="community_intervention",
            label="Community Transmission Control",
            description=(
                "Community education, household protective equipment, and "
                "contact-tracing measures that reduce (but need not eliminate) "
                "community transmission."
            ),
            target_rates=["betaI"],
            adherence=50.0,
            transmission_reduction=100.0,
        )
        schema.add_intervention(
            id="hospital_intervention",
            label="Hospital Isolation / Barrier Nursing",
            description=(
                "Isolation ward and barrier-nursing procedures at the hospital. "
                "With the defaults (50% adherence, 100% reduction) they halve "
                "hospital transmission once active."
            ),
            target_rates=["betaH"],
            adherence=50.0,
            transmission_reduction=100.0,
        )
        schema.add_intervention(
            id="funeral_intervention",
            label="Safe Burial Practices",
            description=(
                "Replacement of traditional burial with safe burial practices "
                "(e.g. by trained response teams). With the defaults (50% "
                "adherence, 100% reduction) this halves funeral transmission "
                "once active."
            ),
            target_rates=["betaF"],
            adherence=50.0,
            transmission_reduction=100.0,
        )

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, config):
        """Initialize the model and seed its PRNG for the stochastic draws.

        Each trajectory in a stochastic batch is built by re-running ``__init__``
        (see ``BatchSimulationManager._single_run``), so seeding from operating-
        system entropy here gives every trajectory an independent random stream.

        Args:
            config: The validated simulation configuration.
        """
        super().__init__(config)
        # NumPy Generator gives exact binomial draws. (jax.random.binomial
        # carries a ~1% per-draw bias that compounds over the ~10^3 draws in a
        # run and inflates outbreak size by ~20%; numpy matches the deterministic
        # ODE and the exact-Gillespie reference. The framework's Euler path runs
        # this model in numpy end-to-end, so no JAX is needed here.)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    def prepare_initial_state(self):
        """Return the initial compartment populations for the solver.

        The framework seeds the ``infected_population`` percentage into ``I``
        (matching the notebook's ``I0`` community seed), leaving E, H, F, R at
        zero. ``infected_population`` is a percentage: the config's ``0.0015``
        means 0.0015% of 200,000 = 3 seed cases.

        Returns:
            The population matrix used as the solver's initial state.
        """
        return self.population_matrix

    # ------------------------------------------------------------------
    # ODE / tau-leap step
    # ------------------------------------------------------------------

    def equation(self, y, t, p):
        """One daily chain-binomial tau-leap step.

        The Euler integrator applies ``y_{t+1} = y_t + dt * equation(...)`` with
        ``dt = 1`` day (the framework step size is 1 for any horizon <= 365 d,
        which covers the DRC config), so this returns the integer per-day change
        in each compartment.

        Args:
            y: Current compartment values, ordered by ``compartment_list``.
            t: Current time in days since the simulation start date.
            p: Packed transmission-edge parameter tuple (alpha, gamma_f).

        Returns:
            The stacked per-compartment deltas for this step.
        """
        eps = 1e-10
        states = {c: y[i] for i, c in enumerate(self.compartment_list)}
        params = self._unpack_params(p)
        alpha = float(params["alpha"])
        gamma_f = float(params["gamma_f"])

        S = np.asarray(states["S"], dtype=float)
        E = np.asarray(states["E"], dtype=float)
        I = np.asarray(states["I"], dtype=float)
        H = np.asarray(states["H"], dtype=float)
        F = np.asarray(states["F"], dtype=float)

        non_total = [c for c in self.compartment_list if not c.endswith("_total")]
        N_total = sum(np.asarray(states[c], dtype=float) for c in non_total)
        prop_infective = float((I + H + F).sum() / (N_total.sum() + eps))

        # --- Interventions on the three transmission coefficients ----------
        disease_rates = {"betaI": self.betaI, "betaH": self.betaH, "betaF": self.betaF}
        disease_rates, self.travel_matrix = self._apply_interventions(
            t, disease_rates, prop_infective
        )
        betaI = float(disease_rates["betaI"])
        betaH = float(disease_rates["betaH"])
        betaF = float(disease_rates["betaF"])

        # --- Derived split rates for I's and H's exits (same as deterministic) --
        gamma_h = float(self._to_rate(self.gamma_h, ValueType.DAYS))
        gamma_i = float(self._to_rate(self.gamma_i, ValueType.DAYS))
        gamma_d = float(self._to_rate(self.gamma_d, ValueType.DAYS))
        theta = float(self._to_rate(self.theta_target, ValueType.PERCENTAGE))
        delta = float(self._to_rate(self.delta_target, ValueType.PERCENTAGE))

        gamma_ih = 1.0 / max(1.0 / gamma_i - 1.0 / gamma_h, eps)
        gamma_dh = 1.0 / max(1.0 / gamma_d - 1.0 / gamma_h, eps)

        delta1 = (delta * gamma_i) / (delta * gamma_i + (1.0 - delta) * gamma_d + eps)
        delta2 = (delta * gamma_ih) / (delta * gamma_ih + (1.0 - delta) * gamma_dh + eps)
        hosp_weight = theta * (gamma_i * (1.0 - delta1) + gamma_d * delta1)
        theta1 = hosp_weight / (hosp_weight + (1.0 - theta) * gamma_h + eps)

        # --- Per-compartment hazards (per day) -----------------------------
        foi = (betaI * I + betaH * H + betaF * F) / (N_total + eps)   # frequency-dependent
        h_IH = gamma_h * theta1                              # I -> H
        h_IR = gamma_i * (1.0 - theta1) * (1.0 - delta1)     # I -> R (survive)
        h_IF = gamma_d * (1.0 - theta1) * delta1             # I -> F (community death)
        h_I = h_IH + h_IR + h_IF
        h_HF = gamma_dh * delta2                             # H -> F (hospital death)
        h_HR = gamma_ih * (1.0 - delta2)                     # H -> R (hospital recover)
        h_H = h_HF + h_HR

        # Competing-risk per-individual probabilities over one day.
        p_SE = 1.0 - np.exp(-foi)
        p_EI = 1.0 - np.exp(-alpha)
        p_Ileave = 1.0 - np.exp(-h_I)
        p_Hleave = 1.0 - np.exp(-h_H)
        p_FR = 1.0 - np.exp(-gamma_f)

        # --- Draw event counts (chain-binomial; bounded => population-conserving) --
        # Exact NumPy binomials. Counts are drawn from integer compartment sizes;
        # individuals leaving a multi-destination compartment are split
        # multinomially by the destination-hazard ratios.
        rng = self._rng

        def _binom(n, prob):
            n_int = np.rint(np.asarray(n, dtype=float)).astype(np.int64)
            return rng.binomial(n_int, prob).astype(float)

        new_inf = _binom(S, p_SE)                                    # S -> E
        new_onset = _binom(E, p_EI)                                  # E -> I
        leave_I = _binom(I, p_Ileave)                                # I -> (H/R/F)
        n_IH = _binom(leave_I, h_IH / (h_I + eps))
        rem_I = leave_I - n_IH
        n_IR = _binom(rem_I, h_IR / (h_IR + h_IF + eps))
        n_IF = rem_I - n_IR
        leave_H = _binom(H, p_Hleave)                                # H -> (F/R)
        n_HF = _binom(leave_H, h_HF / (h_H + eps))
        n_HR = leave_H - n_HF
        new_burial = _binom(F, p_FR)                                 # F -> R

        # --- Assemble per-compartment deltas -------------------------------
        derivs = {c: np.zeros_like(S) for c in self.compartment_list}
        derivs["S"] = -new_inf
        derivs["E"] = new_inf - new_onset
        derivs["I"] = new_onset - leave_I
        derivs["H"] = n_IH - leave_H
        derivs["F"] = n_IF + n_HF - new_burial
        derivs["R"] = n_IR + n_HR + new_burial

        # --- Cumulative inflow trackers ------------------------------------
        derivs["E_total"] = new_inf
        derivs["I_total"] = new_onset            # cumulative symptom onsets (notebook's C)
        derivs["H_total"] = n_IH
        derivs["F_total"] = n_IF + n_HF          # realised deaths
        derivs["R_total"] = n_IR + n_HR + new_burial

        return np.stack([derivs[c] for c in self.compartment_list])
