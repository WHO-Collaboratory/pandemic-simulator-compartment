"""Stochastic SEIHFR Ebola model — community / hospital / burial routes.

Stochastic (demographic-noise) counterpart of ``ebola_test_model``, both ported
from

    Legrand J, Grais RF, Boelle PY, Valleron AJ, Flahault A.
    "Understanding the dynamics of Ebola epidemics."
    Epidemiol Infect. 2007;135(4):610-621. doi:10.1017/S0950268806007217

and its source notebook ``ebola_SEIHFR_model.ipynb``. The paper's model is
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
  * reproduces the observed hospitalisation proportion ``theta`` and
    case-fatality ratio ``delta`` on expectation (the routing split preserves
    ``theta1``/``delta1``/``delta2``), and
  * matches the deterministic model's mean trajectory.

For an *exact* Gillespie first-reaction implementation (event-driven, continuous
time — truest to the paper, and the right tool for fine-grained early-extinction
probability), see the standalone ``ebola_seihfr_gillespie.py`` /
``ebola_SEIHFR_stochastic.ipynb``. See the "Tau-leap vs. exact Gillespie" note in
``model.md``.

Compartments: S, E, I, H, F, R (+ cumulative ``_total`` trackers).
Defaults are the DRC (Kikwit) 1995 preset (Zaire EBOV), matching ``ebola_test_model``.
"""

import logging
import time

import numpy as np

from compartment.model import Model, ValueType
from compartment.parameters import CompartmentDef

logger = logging.getLogger(__name__)


class EbolaTestStochasticModel(Model):
    """Stochastic SEIHFR Ebola model (chain-binomial tau-leap; Legrand et al. 2007)."""

    # Fixed-step Euler + multi-trajectory median/interval output.
    STOCHASTIC = True

    # Number of stochastic trajectories reported as a median + 95% band. This is
    # the fallback default; Model.__init_subclass__ overwrites it with
    # schema.num_runs, i.e. the default of the ``num_runs`` parameter declared in
    # define_parameters() (also 100). The framework caps the effective count at
    # 100. Change the trajectory count via that parameter / the config, not here.
    NUM_RUNS = 100

    # Cumulative inflow trackers. Declared explicitly (rather than relying on the
    # framework's per-edge auto-generation) and assigned in equation(), because
    # most of the inflows they track are manual multi-rate event draws, not
    # transmission edges. Same pattern as ebola_test_model.
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
        """Declare compartments, edges, disease parameters, and interventions."""
        schema.set_model_info(
            disease_type="ebola_seihfr_stochastic_lai_v1",
            label="ebola_seihfr_stochastic_lai_v1",
            description=(
                "Stochastic (chain-binomial tau-leap) SEIHFR model of Ebola virus "
                "disease with explicit community, hospital, and traditional-burial "
                "transmission routes, based on Legrand et al. (Epidemiol. Infect. "
                "2007). Demographic-noise counterpart of the deterministic "
                "ebola_test model. Defaults are the DRC (Kikwit) 1995 preset."
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
                "How likely is a small introduction to fade out, and how variable is outbreak size?",
            ],
            key_assumptions=[
                "Homogeneous mixing within a closed population (no age/spatial structure).",
                "No background births or deaths; the population is conserved (Ebola "
                "deaths pass through F to the removed class R).",
                "Entire population initially susceptible apart from the seed cases.",
                "Interventions activate as a step function on their start date and, "
                "with a null end date, remain active thereafter.",
                "Frequency-dependent force of infection.",
                "Demographic stochasticity via a chain-binomial tau-leap on a daily "
                "step; per-transition counts are binomial with probability "
                "1 - exp(-hazard), multinomially split across competing destinations.",
            ],
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
        # Transmission coefficients (per day) = weekly Table 4 value / 7, at full
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
        # gamma_ih/gamma_dh split rates. FLOAT (not DAYS) so the fractional
        # gamma_d=9.6 is accepted; days -> per-day-rate conversion is done via
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

        # Number of stochastic trajectories. run_simulation reads this to decide
        # how many runs to average for the median + 95% interval band. The
        # framework derives the class attribute NUM_RUNS from this parameter's
        # default (Model.__init_subclass__ sets cls.NUM_RUNS = schema.num_runs),
        # and the framework caps the effective count at 100. Declared exactly as
        # in example_stochastic_model.
        schema.add_parameter(
            name="num_runs",
            label="Number of Stochastic Trajectories",
            description=(
                "How many independent stochastic realisations to simulate. The "
                "reported series is the per-timestep median with a 95% interval "
                "band across the trajectories."
            ),
            value_type=ValueType.INTEGER,
            default=100,
            min_value=10,
            max_value=100,
            enable_variance=False,
        )

        # --- Interventions -------------------------------------------------
        # Each route is scaled independently by z in [0, 1] from its start date.
        # The framework step intervention applies rate * (1 - adherence *
        # transmission_reduction); with adherence 100% this is rate * (1 - z),
        # so transmission_reduction = 1 - z. DRC preset: z_community = 0.50,
        # z_hospital = 0.0, z_funeral = 0.0.
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
        """Initialize the model and seed its PRNG for the stochastic draws.

        Each trajectory in a stochastic batch is built by re-running ``__init__``
        (see ``BatchSimulationManager._single_run``), so seeding from system
        entropy here gives every trajectory an independent random stream. Pass a
        ``seed`` in the config only for a single reproducible trajectory — a
        fixed seed makes every trajectory in a batch identical and collapses the
        interval band.

        Args:
            config: The validated simulation configuration.
        """
        super().__init__(config)
        seed = config.get("seed") if isinstance(config, dict) else None
        if seed is None:
            seed = int(time.time() * 1e6) % (2**31)
        # NumPy Generator gives exact binomial draws. (jax.random.binomial
        # carries a ~1% per-draw bias that compounds over the ~10^3 draws in a
        # run and inflates outbreak size by ~20%; numpy matches the deterministic
        # ODE and the exact-Gillespie reference. The framework's Euler path runs
        # this model in numpy end-to-end, so no JAX is needed here.)
        self._rng = np.random.default_rng(int(seed))

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
