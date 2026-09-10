"""Measles SEIR calibrated to the 2025 Texas outbreak (Gonzalez-Parra et al.).

A worked example of the *input-dataset* workflow. The model declares a recent,
real measles case series in ``datasets.yaml`` (4 weekly data points digitised
from Figure 1 of Gonzalez-Parra et al. 2025), reads it with
``self.dataset(...)``, and fits the S->E transmission rate ``beta`` to that
series inside ``__init__``. The simulation then runs with the fitted value.

``beta`` is deliberately NOT a schema parameter, so it never appears as a field
in the Simulator interface: on this platform every transmission edge becomes a
user control, and ``beta`` here is fully determined by the data, not something a
user should type. It is therefore computed at load time and applied through a
manual force-of-infection term in ``equation()`` rather than a declared edge.
The latent and infectious periods (``sigma``, ``gamma``) remain ordinary edges,
so they stay visible and adjustable.

Model basis: an SEIR measles model following Gonzalez-Parra, Vestrand & Mujynya
(2025), who analysed this same outbreak; latent 11 d and infectious 7 d are their
values. Population: 5-county affected region (N=128,924); initial immune fraction
set by the user-facing ``initial_immune_fraction`` parameter (default 92.15% =
0.95 × 0.97, all-or-nothing model, Anderson & May 1991). Beta calibration uses a
separate CALIBRATION_SUSCEPTIBLE_FRACTION (= 0.0785), locked to the outbreak
dataset and independent of ``initial_immune_fraction``. See model.md.
"""

import logging

import jax.numpy as jnp
import numpy as np

from compartment.model import Model, ValueType

logger = logging.getLogger(__name__)

DATASET_NAME = "measles-tx-2025-weekly"
BETA_FALLBACK = 0.4  # used only if the dataset/fit is unavailable

# ---------------------------------------------------------------------------
# Two conceptually separate constants that happen to yield the same f_s = 0.0785
# for the 2025 TX outbreak.  Keep them separate: each has a distinct role and
# would need to change independently for a different population or dataset.

# 1. CALIBRATION -- used ONLY in _calibrated_beta().
#    The effective susceptible fraction in the dataset's outbreak community at the
#    time the case series was collected.  This value is locked to the González-Parra
#    2025 data; changing it would invalidate the beta estimate for this dataset.
#    All-or-nothing vaccine model (Anderson & May 1991):
#      f_s = 1 - p*VE = 1 - 0.95*0.97 = 0.0785
#    Required so R_eff(0) = beta * f_s / gamma = R_t (self-consistency with the ODE).
CALIBRATION_SUSCEPTIBLE_FRACTION = 0.0785   # = 1 - 0.95 * 0.97 for the TX 2025 community

# 2. POPULATION INITIAL STATE -- controlled by the user-facing ``initial_immune_fraction``
#    schema parameter (ValueType.PERCENTAGE, 0-100).  Default 92.15 = 0.95 * 0.97
#    (MMR coverage × two-dose efficacy for the TX 2025 community; all-or-nothing model,
#    Anderson & May 1991).  For a different population the user sets this in the UI;
#    the calibrated beta stays fixed to the dataset regardless.
INITIAL_IMMUNE_FRACTION_DEFAULT = 92.15   # % (= 0.95 * 0.97 * 100 for TX 2025)
# ---------------------------------------------------------------------------


class MeaslesSeirCalibratedGonzalezparraModel(Model):
    """SEIR measles model with a data-calibrated (non-user-facing) transmission rate."""

    _beta_cache: dict = {}  # fit once per (config) per process; reused across trajectories

    # ------------------------------------------------------------------ schema
    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="measles_seir_calibrated_gonzalezparra",
            label="Measles SEIR calibrated to the 2025 Texas outbreak (Gonzalez-Parra et al.)",
            description=(
                "SEIR measles model whose transmission rate is calibrated to the 2025 Texas "
                "weekly case series (rising limb) via the dataset input, then simulated forward. "
                "The transmission rate is fixed by the data and is not a user input; the latent "
                "and infectious periods remain adjustable."
            ),
        )
        schema.set_model_metadata(
            authors=[{"name": "Yingsi", "affiliation": "WHO Pandemic Hub"}],
            license="MIT",
            model_type="Compartmental (SEIR)",
            diseases=["Measles"],
            transmission_routes=["Airborne"],
            citations=[
                "Gonzalez-Parra G, Vestrand A, Mujynya R. (2025) Modeling and Characterizing "
                "the Growth of the Texas-New Mexico Measles Outbreak of 2025. Epidemiologia "
                "6(4):60. doi:10.3390/epidemiologia6040060.",
                "Anderson RM, May RM. (1991) Infectious Diseases of Humans: Dynamics and Control. "
                "Oxford Univ. Press. -- all-or-nothing vaccine model: effective susceptible "
                "fraction = 1 - p*VE.",
                "Keeling MJ, Rohani P. (2008) Modeling Infectious Diseases in Humans and "
                "Animals. Princeton Univ. Press. -- SEIR structure and vaccination modelling.",
                "Wallinga J, Lipsitch M. (2007) How generation intervals shape the relationship "
                "between growth rates and reproductive numbers. Proc R Soc B 274:599-604. "
                "-- SEIR formula mapping exponential growth rate r to R_t.",
                "Guerra FM, et al. (2017) The basic reproduction number (R0) of measles: a "
                "systematic review. Lancet Infect Dis 17(12):e420-e428. -- R0 varies widely; "
                "the textbook 12-18 range is not universally supported.",
            ],
            validation=(
                "beta is calibrated from 3 data points (Figure 1, days 7/14/28; day 0 seed "
                "excluded) using the exponential growth rate method and Wallinga-Lipsitch SEIR "
                "formula. With sigma=11 d, gamma=7 d, N=128924, f_s=0.0785 (all-or-nothing model, "
                "p=0.95, VE=0.97; Anderson & May 1991), the fit yields r~0.083/day, R_t~3.0, "
                "R0_intrinsic = R_t/f_s ~ 38.6, beta ~ 5.5/day. R_eff(0) = beta*f_s/gamma ~ 3.0 "
                "(self-consistent). R0~38.6 is within Gonzalez-Parra 2025 (32-40). "
                "Day 0 (2 imported seed cases) is excluded from the OLS because it reflects "
                "index-case seeding, not community transmission growth. See model.md."
            ),
            key_assumptions=[
                "Closed population; no births, deaths, or waning immunity.",
                "Frequency-dependent transmission (FOI scales with the infectious proportion).",
                "Homogeneous mixing (an approximation for a clustered community outbreak).",
                "Latent (E) 11 d and infectious (I) 7 d fixed (Gonzalez-Parra 2025).",
                "5-county affected region: Lea NM, Dawson, Gaines, Terry, Yoakum TX; N=128,924.",
                "Initial immune fraction is user-defined via the ``initial_immune_fraction`` "
                "parameter (default 92.15% = 0.95 MMR coverage x 0.97 two-dose efficacy, "
                "all-or-nothing model, Anderson & May 1991). Set it to match the target population.",
                "Transmission rate beta is calibrated from data at load time and is not a user "
                "input; it is independent of ``initial_immune_fraction``.",
            ],
            not_for=(
                "Real-world forecasting or a definitive measles R0 -- a worked example of "
                "calibrating a model's transmission rate to a recent, real case series."
            ),
        )

        # --- Compartments (S -> E -> I -> R) ---
        schema.add_compartment("S", "Susceptible", "Susceptible to measles")
        schema.add_compartment("E", "Exposed", "Infected but not yet infectious (latent)")
        schema.add_compartment("I", "Infectious", "Currently infectious", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered with lifelong immunity")

        # --- Transmission edges ---
        # NOTE: beta (S->E) is intentionally NOT declared as an edge, so it is not a
        # user field. It is fitted in __init__ and applied manually in equation().
        # Only the latent and infectious periods are user-facing edges.
        schema.add_transmission_parameter(
            source="exposed", target="infectious", variable_name="sigma",
            label="Latent Period (E->I)",
            description="Mean days from infection to becoming infectious (Gonzalez-Parra 2025: 11 d).",
            default=11.0, default_min=8.0, default_max=14.0,
            min_value=1.0, max_value=21.0, value_type=ValueType.DAYS, unit="days",
        )
        schema.add_transmission_parameter(
            source="infectious", target="recovered", variable_name="gamma",
            label="Infectious Period (I->R)",
            description="Mean days infectious (Gonzalez-Parra 2025: 7 d).",
            default=7.0, default_min=5.0, default_max=10.0,
            min_value=1.0, max_value=21.0, value_type=ValueType.DAYS, unit="days",
        )

        # --- Initial conditions parameter (user-facing) ---
        # The user sets the fraction of the population already immune at t=0.
        # This is independent of the calibration; changing it adjusts S(0)/R(0)
        # but does NOT alter the fitted beta.
        schema.add_parameter(
            name="initial_immune_fraction",
            label="Initial Immune Fraction",
            description=(
                "Percentage of the population already immune at t=0 (from prior vaccination "
                "or infection). Default 92.15% = 95% MMR coverage × 97% two-dose efficacy for "
                "the TX 2025 5-county community (all-or-nothing vaccine model, Anderson & May "
                "1991). Adjust for a different population. The calibrated beta is unaffected."
            ),
            value_type=ValueType.PERCENTAGE,
            default=92.15,
            default_min=85.0,
            default_max=99.0,
            min_value=0.0,
            max_value=100.0,
            unit="%",
        )

        # --- Intervention: MMR vaccination reduces the (fitted) transmission rate ---
        schema.add_intervention(
            id="mmr_vaccination",
            label="MMR Vaccination Campaign",
            description="Reduces effective transmission through vaccine-derived immunity.",
            target_rates=["beta"], adherence=80.0, transmission_reduction=90.0,
        )

    # -------------------------------------------------------------------- init
    def __init__(self, config):
        super().__init__(config)  # sets self.sigma, self.gamma, population_matrix, and disease params
        # Defensive default for legacy configs that pre-date this parameter.
        self.initial_immune_fraction = getattr(self, "initial_immune_fraction", INITIAL_IMMUNE_FRACTION_DEFAULT)
        self.beta = self._calibrated_beta()  # fitted transmission rate (not from config)

    # --------------------------------------------------------- initial state
    def prepare_initial_state(self):
        """Redistribute S -> R to reflect pre-existing population immunity.

        Uses the user-defined ``initial_immune_fraction`` parameter (percentage 0-100)
        to set the immune count at t=0.  The default 92.15% = 0.95 x 0.97 (MMR
        coverage × two-dose efficacy) for the TX 2025 5-county community under the
        all-or-nothing vaccine model (Anderson & May 1991).

        This parameter is independent of CALIBRATION_SUSCEPTIBLE_FRACTION: changing
        it adjusts S(0) and R(0) but does NOT alter the fitted beta, which is locked
        to the outbreak dataset's f_s = 0.0785.

        The small seed of initially infectious individuals is applied by the platform
        on top of this redistribution via the case-file infected_population field.
        """
        y0 = np.array(self.population_matrix, dtype=float)
        s_idx = self.compartment_list.index("S")
        r_idx = self.compartment_list.index("R")
        N = float(y0.sum())
        immune_frac = float(self.initial_immune_fraction) / 100.0   # PERCENTAGE -> fraction
        immune_count = immune_frac * N
        # Guard: never move more than what is in S
        immune_count = min(immune_count, float(y0[s_idx]))
        y0[s_idx] -= immune_count
        y0[r_idx] += immune_count
        return y0

    # --------------------------------------------------------------- the fit
    def _calibrated_beta(self) -> float:
        """Estimate beta from the exponential growth rate of the rising limb.

        Method (Gonzalez-Parra et al. 2025; Wallinga & Lipsitch 2007):

          Step 1 -- fit the growth rate r.
            log-linear OLS: log(cases_t) = log(C0) + r*t over all non-zero
            data points (week-start day as t).

          Step 2 -- Wallinga-Lipsitch SEIR formula -> R_t.
            R_t = (1 + r*T_E)(1 + r*T_I)
            where T_E = 1/sigma, T_I = 1/gamma.
            R_t is the effective reproduction number in the observed (partially
            immune) population.

          Step 3 -- recover beta consistent with initial conditions.
            In the SEIR ODE: R_eff(0) = beta * S(0) / (N * gamma) = R_t
            => beta = R_t * gamma / f_s
            where f_s = CALIBRATION_SUSCEPTIBLE_FRACTION = S(0)/N.
            This is equivalent to: R0_intrinsic = R_t / f_s; beta = R0_intrinsic * gamma.
            (Anderson & May 1991; the intrinsic R0 = R_t / f_s ~ 38-39, consistent with
            Gonzalez-Parra's reported range of 32-40; day 0 seed excluded from OLS.)

        Result is cached at class level and reused across all deepcopy trajectories.
        Falls back to BETA_FALLBACK if the dataset is missing or the fit fails.
        """
        sigma, gamma = float(self.sigma), float(self.gamma)
        try:
            import pandas as pd

            table = pd.read_csv(self.dataset(DATASET_NAME))
            days   = table["day"].to_numpy(dtype=float)    # week-start days: 0, 7, 14, 28
            weekly = table["cases"].to_numpy(dtype=float)  # weekly incidence (Figure 1)
        except Exception as exc:
            logger.warning("measles_gp: could not load dataset (%s); using fallback beta=%.4f", exc, BETA_FALLBACK)
            return BETA_FALLBACK

        # Rising limb only (through the peak week), excluding zero-case weeks
        peak = int(np.argmax(weekly))
        mask  = weekly[: peak + 1] > 0
        wk    = days[: peak + 1][mask]
        obs   = weekly[: peak + 1][mask]

        # Exclude the first data point (day 0 / seed cases) from the growth-rate OLS.
        # Day 0 records the imported index cases that seeded the outbreak, not cases
        # generated within the community; including them inflates r because the jump from
        # 0 imports to 2 seed cases does not reflect the epidemic's intrinsic growth rate.
        # The established rising limb of community transmission starts from day 7.
        # This exclusion recovers r ~ 0.083/day and R0_intrinsic ~ 38-39, consistent
        # with Gonzalez-Parra et al.'s reported range of 32-40.
        if len(wk) > 1:
            wk  = wk[1:]
            obs = obs[1:]

        if len(wk) < 3:
            logger.warning("measles_gp: too few non-zero rising-limb points (%d); using fallback beta=%.4f",
                           len(wk), BETA_FALLBACK)
            return BETA_FALLBACK

        key = (round(sigma, 6), round(gamma, 6), int(wk[-1]), int(obs.sum()))
        if key in type(self)._beta_cache:
            return type(self)._beta_cache[key]

        try:
            # Step 1: log-linear fit -> growth rate r (per day)
            r, _ = np.polyfit(wk, np.log(obs), 1)
            if not np.isfinite(r) or r <= 0:
                raise ValueError(f"non-positive growth rate r={r:.4f}")

            # Step 2: Wallinga-Lipsitch SEIR formula -> R_t (effective R in observed population)
            T_E = 1.0 / sigma   # mean latent period (days)
            T_I = 1.0 / gamma   # mean infectious period (days)
            R_t = (1.0 + r * T_E) * (1.0 + r * T_I)

            # Step 3: recover beta consistent with S(0)/N = CALIBRATION_SUSCEPTIBLE_FRACTION
            # R0_intrinsic = R_t / f_s;  beta = R0_intrinsic * gamma = R_t * gamma / f_s
            R0_intrinsic = R_t / CALIBRATION_SUSCEPTIBLE_FRACTION
            beta = R0_intrinsic * gamma

            if not np.isfinite(beta) or beta <= 0:
                raise ValueError(f"invalid beta={beta:.4f}")
        except Exception as exc:
            logger.warning("measles_gp: calibration failed (%s); using fallback beta=%.4f", exc, BETA_FALLBACK)
            return BETA_FALLBACK

        logger.info(
            "measles_gp: r=%.5f/day, R_t=%.2f, R0_intrinsic=%.2f, beta=%.4f "
            "from '%s' rising limb (%d non-zero weeks; f_s=%.4f)",
            r, R_t, R0_intrinsic, beta, DATASET_NAME, len(obs), CALIBRATION_SUSCEPTIBLE_FRACTION,
        )
        type(self)._beta_cache[key] = beta
        return beta

    # ----------------------------------------------------------------- solver
    def equation(self, y, t, p):
        params = self._unpack_params(p)  # {"sigma": ..., "gamma": ...}
        states = {c: y[i] for i, c in enumerate(self.compartment_list)}

        non_total = [c for c in self.compartment_list if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)
        prop_infective = states["I"].sum() / (N_total.sum() + 1e-10)

        # The fitted beta is not a schema rate; feed it through the intervention
        # machinery so MMR vaccination can still scale it, then apply S->E manually.
        rates, self.travel_matrix = self._apply_interventions(
            t, {"beta": self.beta}, prop_infective
        )
        beta = rates["beta"]

        # E->I and I->R are ordinary edges handled by the framework.
        derivs = self._compute_equations(states, {"sigma": params["sigma"], "gamma": params["gamma"]})
        # Manual frequency-dependent S->E infection with the calibrated beta.
        self._apply_flow(derivs, "S", "E", beta * states["S"] * prop_infective)

        return jnp.stack([derivs[c] for c in self.compartment_list])
