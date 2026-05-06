"""Discrete-time stochastic Ebola virus disease model.

Port of ``model_ebola()`` from epiverse-trace/epidemics
(https://github.com/epiverse-trace/epidemics/blob/main/R/model_ebola.R),
based on the consensus compartment structure of Li et al. (2019) and the
Erlang boxcar passage-time formulation of Getz & Dougherty (2018).

Compartments: S, E (k_E boxcars), I (k_I boxcars), H (k_H boxcars), F, R.
Output is rolled up to the six public compartments via
``COMPARTMENT_DELTA_GROUPING``.

WARNING: Like the other models in this repo, this implementation is
intended for local experimentation; it is not yet supported by the
pandemic simulator app.
"""

import math
import time as _time
import logging

import jax
import jax.numpy as jnp
import numpy as onp

from compartment.helpers import setup_logging
from compartment.model import Model
from compartment.parameters import ValueType

setup_logging()
logger = logging.getLogger(__name__)


# Boxcar counts. Sized to capture > 99% of the discrete Erlang density
# at the schema default rates and enough headroom for the declared
# uncertainty bounds (incubation 3-10 days, infectious 7-20 days, k=2).
_N_E_BOXCARS = 25
_N_I_BOXCARS = 50
_N_H_BOXCARS = 50  # R model uses one rate for both I and H, so equal sizes


class EbolaJaxModel(Model):
    """Discrete-time, stochastic Ebola model with Erlang passage times."""

    # Use the framework's fixed-step Euler integrator; ``derivative()``
    # returns the per-timestep change rather than an instantaneous rate.
    STOCHASTIC = True

    # Erlang shape (k). The R model forbids varying this at runtime
    # because it changes the compartment count, so it is fixed here.
    ERLANG_K = 2

    N_E_BOXCARS = _N_E_BOXCARS
    N_I_BOXCARS = _N_I_BOXCARS
    N_H_BOXCARS = _N_H_BOXCARS

    # Roll boxcars up to the six public compartments for output.
    COMPARTMENT_DELTA_GROUPING = {
        "S": ["S"],
        "E": [f"E{i}" for i in range(1, _N_E_BOXCARS + 1)],
        "I": [f"I{i}" for i in range(1, _N_I_BOXCARS + 1)],
        "H": [f"H{i}" for i in range(1, _N_H_BOXCARS + 1)],
        "F": ["F"],
        "R": ["R"],
    }

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @classmethod
    def _add_total_compartments(cls, schema):
        # Suppress framework auto-generation of per-edge ``_total``
        # compartments. The boxcar slots aren't meaningful aggregation
        # targets, and the post-processor's grouping handles output
        # rollup. Cumulative incidence can be added later as a manual
        # ``R_total`` if needed.
        return

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="EBOLA",
            label="Ebola Virus Disease",
            description=(
                "Discrete-time stochastic SEIR model with hospitalisation "
                "and funeral transmission, with Erlang passage times "
                "(Li et al. 2019, Getz & Dougherty 2018)."
            ),
        )

        schema.add_compartment(
            "S", "Susceptible", "Population susceptible to Ebola infection",
        )
        for i in range(1, cls.N_E_BOXCARS + 1):
            schema.add_compartment(
                f"E{i}", f"Exposed boxcar {i}",
                f"Exposed individuals with {i} timesteps until becoming infectious",
            )
        for i in range(1, cls.N_I_BOXCARS + 1):
            schema.add_compartment(
                f"I{i}", f"Infectious boxcar {i}",
                f"Infectious in community with {i} timesteps until removal",
                infective=True,
            )
        for i in range(1, cls.N_H_BOXCARS + 1):
            schema.add_compartment(
                f"H{i}", f"Hospitalised boxcar {i}",
                f"Hospitalised with {i} timesteps until removal",
                infective=True,
            )
        schema.add_compartment(
            "F", "Funeral",
            "In funeral transmission stage (single timestep)",
            infective=True,
        )
        schema.add_compartment(
            "R", "Removed",
            "Removed from the dynamic system (recovered or safely buried)",
        )

        # Transmission-edge rates exposed to the UI/uncertainty layer.
        # The actual flow math is computed manually in ``derivative()``.
        schema.add_transmission_edge(
            source="susceptible", target="E1",
            variable_name="beta",
            label="Transmission Rate (S->E)",
            description=(
                "Baseline transmission rate β. Defaults to R0/infectious_period = "
                "1.5/12 ≈ 0.125 per day."
            ),
            default=0.125,
            default_min=0.05, default_max=0.4,
            min_value=0.001, max_value=2.0,
            unit="days",
        )
        schema.add_transmission_edge(
            source="E1", target="I1",
            variable_name="sigma",
            label="Incubation Period (E->I)",
            description="Mean pre-infectious (exposed) period in days.",
            default=5.0,
            default_min=3.0, default_max=10.0,
            min_value=2.0, max_value=20.0,
            unit="days",
            value_type=ValueType.DAYS,
        )
        schema.add_transmission_edge(
            source="I1", target="F",
            variable_name="gamma",
            label="Infectious / Hospitalised Period (I->F, H->R)",
            description=(
                "Mean duration in the infectious community or hospitalised "
                "compartment in days."
            ),
            default=12.0,
            default_min=7.0, default_max=20.0,
            min_value=2.0, max_value=40.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

        # Disease-specific scalars.
        schema.add_disease_parameter(
            name="prop_community",
            label="Proportion in Community",
            description=(
                "Proportion of infectious individuals who remain in the "
                "community and are not hospitalised."
            ),
            value_type=ValueType.PERCENTAGE,
            default=90.0,
            default_min=70.0, default_max=99.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )
        schema.add_disease_parameter(
            name="etu_risk",
            label="ETU Transmission Risk",
            description=(
                "Relative β for hospitalised individuals (Ebola Treatment Unit). "
                "0 = no onward transmission, 100 = same as community."
            ),
            value_type=ValueType.PERCENTAGE,
            default=70.0,
            default_min=10.0, default_max=90.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )
        schema.add_disease_parameter(
            name="funeral_risk",
            label="Funeral Transmission Risk",
            description=(
                "Relative β for funeral transmission. 0 = safe burials, "
                "100 = full community transmission."
            ),
            value_type=ValueType.PERCENTAGE,
            default=50.0,
            default_min=0.0, default_max=80.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )

        schema.add_intervention(
            id="social_distancing",
            label="Social Distancing",
            description=(
                "Community-level distancing measures that reduce the "
                "baseline transmission rate β."
            ),
            target_rates=["beta"],
            adherence=50.0,
            transmission_reduction=40.0,
        )

    # ------------------------------------------------------------------
    # Initial population
    # ------------------------------------------------------------------

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """Distribute each zone's population across S and the I boxcars.

        ``infected_population`` percentage is spread uniformly across the
        ``I1..I_NI`` boxcars. The exact slot distribution stabilises within
        a few simulated days as the boxcars relax to the steady-state
        Erlang shape, so a uniform initial seed is acceptable.
        """
        col = {v: i for i, v in enumerate(compartment_list)}
        pop = onp.zeros((len(admin_zones), len(compartment_list)))

        for z, zone in enumerate(admin_zones):
            N = float(zone["population"])
            inf_pct = float(zone.get("infected_population", 0.0) or 0.0)
            inf_pct = max(inf_pct, 0.0)
            infected = N * inf_pct / 100.0
            pop[z, col["S"]] = max(N - infected, 0.0)
            per_box = infected / max(cls.N_I_BOXCARS, 1)
            for i in range(1, cls.N_I_BOXCARS + 1):
                pop[z, col[f"I{i}"]] = per_box

        return pop

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, config):
        # Wire up the common state (population_matrix, schema rate attrs,
        # intervention runtime objects, etc.).
        super().__init__(config)

        n_regions = self.population_matrix.shape[1]
        # No spatial coupling — each admin zone evolves independently.
        self.travel_matrix = jnp.eye(n_regions)

        # ``_load_transmission_params`` converts DAYS edges via
        # ``_to_rate(days) = 1/days``. So self.sigma and self.gamma are
        # per-day rates (1 / mean period). The Erlang rate λ = k * (1/period)
        # gives mean stay = period in the Getz & Dougherty formulation.
        if self.beta is None:
            self.beta = 0.125
        sigma_per_day = self.sigma if self.sigma is not None else 1.0 / 5.0
        gamma_per_day = self.gamma if self.gamma is not None else 1.0 / 12.0
        self._erlang_lambda_E = self.ERLANG_K * sigma_per_day
        self._erlang_lambda_I = self.ERLANG_K * gamma_per_day

        # Disease-specific scalars (already validated by schema; defaults
        # mirror the R model).
        disease_cfg = config.get("Disease", {}) or {}
        self.prop_community = float(disease_cfg.get("prop_community", 90.0)) / 100.0
        self.etu_risk = float(disease_cfg.get("etu_risk", 70.0)) / 100.0
        self.funeral_risk = float(disease_cfg.get("funeral_risk", 50.0)) / 100.0

        # Pre-compute the discrete Erlang weights once per run.
        self._exposed_weights = jnp.array(
            _discrete_erlang_density(
                self.ERLANG_K, self._erlang_lambda_E, self.N_E_BOXCARS,
            )
        )
        self._infectious_weights = jnp.array(
            _discrete_erlang_density(
                self.ERLANG_K, self._erlang_lambda_I, self.N_I_BOXCARS,
            )
        )

        # PRNG seed — config seed for reproducibility, otherwise wall clock.
        seed = config.get("seed")
        if seed is None:
            seed = int(_time.time() * 1000) % (2**31)
        self._key = jax.random.PRNGKey(int(seed))

        # Population_size used as the FOI denominator. R model uses
        # ``sum(initial_state)`` — fixed throughout the run since this is
        # a closed population.
        self._population_size = jnp.sum(self.population_matrix, axis=0) + 1e-10

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    def prepare_initial_state(self):
        # population_matrix is already (K, R) from super().__init__.
        # Distribute initial E/I across boxcars per the discrete Erlang
        # density, mimicking the R model's rmultinom seeding.
        pm = onp.array(self.population_matrix)
        e_w = onp.array(self._exposed_weights)
        i_w = onp.array(self._infectious_weights)

        s_idx = self.compartment_list.index("S")  # noqa: F841 (sanity)
        e_start = self.compartment_list.index("E1")
        i_start = self.compartment_list.index("I1")

        # Collapse per-region E and I totals, then redistribute by weights.
        e_total = pm[e_start : e_start + self.N_E_BOXCARS].sum(axis=0)
        i_total = pm[i_start : i_start + self.N_I_BOXCARS].sum(axis=0)

        pm[e_start : e_start + self.N_E_BOXCARS] = e_w[:, None] * e_total[None, :]
        pm[i_start : i_start + self.N_I_BOXCARS] = i_w[:, None] * i_total[None, :]

        self.population_matrix = jnp.array(pm)
        return self.population_matrix, list(self.compartment_list)

    # ------------------------------------------------------------------
    # ODE / discrete-time derivative
    # ------------------------------------------------------------------

    def derivative(self, y, t, p):
        """Per-timestep change for the discrete stochastic model.

        With ``STOCHASTIC = True`` the framework uses Euler integration:
        ``y_{t+1} = y_t + 1 * derivative(y_t, t, p)``. So we return the
        full discrete-step change here, not an instantaneous rate.
        """
        params = self._unpack_params(p)
        # ``params`` has the schema-converted per-day rates: beta (rate),
        # sigma (1/days), gamma (1/days). The Erlang rates λ_E, λ_I are
        # already cached on self.
        beta = params["beta"]

        # Slice the state vector by compartment groups.
        e_start = 1
        e_end = e_start + self.N_E_BOXCARS
        i_start = e_end
        i_end = i_start + self.N_I_BOXCARS
        h_start = i_end
        h_end = h_start + self.N_H_BOXCARS
        f_idx = h_end
        r_idx = h_end + 1

        S = y[0]
        E = y[e_start:e_end]
        I = y[i_start:i_end]  # noqa: E741
        H = y[h_start:h_end]
        F = y[f_idx]
        R = y[r_idx]

        N = self._population_size

        # Schema-driven interventions on β. Travel matrix is identity so
        # _apply_interventions's travel handling is a no-op.
        I_total = I.sum(axis=0)
        H_total = H.sum(axis=0)
        prop_inf_scalar = (I_total.sum() + H_total.sum() + F.sum()) / N.sum()
        rates = {"beta": beta}
        rates, _ = self._apply_interventions(t, rates, prop_inf_scalar)
        beta = rates["beta"]

        # R model formula (per region):
        #   λ(t) = β * (I + etu*H + funeral*F) / N
        #   p_exposure = 1 - exp(-λ)
        infectious_pressure = (
            I_total + self.etu_risk * H_total + self.funeral_risk * F
        )
        current_rate = beta * infectious_pressure / N
        exposure_prob = 1.0 - jnp.exp(-jnp.maximum(current_rate, 0.0))
        exposure_prob = jnp.clip(exposure_prob, 0.0, 1.0)

        # Split keys for each random draw in this step.
        keys = jax.random.split(self._key, 4)
        self._key = keys[0]
        key_new_exposed = keys[1]
        key_exposed_split = keys[2]
        key_infectious_split = keys[3]

        # 1. New exposures per region.
        new_exposed = jax.random.binomial(
            key_new_exposed, jnp.maximum(S, 0.0), exposure_prob,
        ).astype(S.dtype)
        new_exposed = jnp.minimum(new_exposed, jnp.maximum(S, 0.0))

        # 2. Distribute new exposures across the E boxcars (multinomial
        #    per region). Output shape: (N_E, R).
        new_exposed_dist = _multinomial_per_region(
            key_exposed_split, new_exposed, self._exposed_weights,
        )

        # 3. Hospitalisation: a fraction of every I boxcar above slot 0
        #    moves into H at the same remaining-stay-time slot.
        hosp_admit = jnp.round(I[1:] * (1.0 - self.prop_community))
        hosp_admit = jnp.maximum(hosp_admit, 0.0)
        # Don't take more than what's in each I slot.
        hosp_admit = jnp.minimum(hosp_admit, jnp.maximum(I[1:], 0.0))

        # 4. New infectious are those exiting slot 0 of the E boxcars.
        new_infectious = E[0]
        new_infectious_dist = _multinomial_per_region(
            key_infectious_split, new_infectious, self._infectious_weights,
        )

        # 5. Boxcar advancement.
        zero_row = jnp.zeros((1, E.shape[1]), dtype=E.dtype)

        # E_new = shift(E left by 1) + new_exposed_dist
        E_shifted = jnp.concatenate([E[1:], zero_row], axis=0)
        E_new = E_shifted + new_exposed_dist

        # I_new = shift(I left by 1) - hosp_admit (in slots 0..NI-2)
        #         + new_infectious_dist
        I_shifted_minus_hosp = jnp.concatenate(
            [I[1:] - hosp_admit, zero_row], axis=0,
        )
        I_new = I_shifted_minus_hosp + new_infectious_dist

        # H_new = shift(H left by 1) + hospitalisations padded with a
        # leading zero (admissions go to slots 1..NH-1, matching the
        # remaining-stay-time of the I slot they came from).
        H_shifted = jnp.concatenate([H[1:], zero_row], axis=0)
        hosp_admit_padded = jnp.concatenate([zero_row, hosp_admit], axis=0)
        H_new = H_shifted + hosp_admit_padded

        # F is single-timestep: F_new = old I[0] (slot exiting infectious).
        F_new = I[0]

        # R accumulates exits from H (slot 0) and the previous F.
        R_new = R + H[0] + F

        # S decreases by new exposures.
        S_new = S - new_exposed

        # Return the per-step delta. Order must match self.compartment_list:
        #   [S, E1..E_NE, I1..I_NI, H1..H_NH, F, R]
        delta_S = (S_new - S)[None, :]
        delta_E = E_new - E
        delta_I = I_new - I
        delta_H = H_new - H
        delta_F = (F_new - F)[None, :]
        delta_R = (R_new - R)[None, :]

        return jnp.concatenate(
            [delta_S, delta_E, delta_I, delta_H, delta_F, delta_R], axis=0,
        )


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _discrete_erlang_density(shape, rate, n_bins):
    """Discrete Erlang(shape, rate) probability vector of length ``n_bins``.

    Direct port of ``prob_discrete_erlang`` from the R source. Returns the
    probability mass for waiting times T = 1, 2, ..., n_bins, renormalised
    so the truncated density sums to 1. ``rate`` is the Erlang rate λ;
    mean waiting time is shape/λ.
    """
    if rate <= 0 or n_bins <= 0:
        # Degenerate case: put all mass in the last slot so individuals
        # exit only after the maximum delay (better than NaN).
        v = onp.zeros(max(n_bins, 1))
        v[-1] = 1.0
        return v

    factorials = onp.array(
        [math.factorial(j) for j in range(shape + 1)], dtype=float,
    )
    # one_minus_cum[n] = 1 - P(T <= n) for n = 0..n_bins
    one_minus_cum = onp.zeros(n_bins + 1)
    one_minus_cum[0] = 1.0
    for n_bin in range(1, n_bins + 1):
        j = onp.arange(shape)
        terms = (
            onp.exp(-n_bin * rate)
            * (n_bin * rate) ** j
            / factorials[:shape]
        )
        one_minus_cum[n_bin] = terms.sum()

    density = one_minus_cum[:-1] - one_minus_cum[1:]
    s = density.sum()
    if s <= 0:
        density = onp.full(n_bins, 1.0 / n_bins)
    else:
        density = density / s
    return density


def _multinomial_per_region(key, total, probs):
    """Sample Multinomial(total, probs) independently per region.

    Implemented as a sequential cascade of binomials, which avoids the
    fixed ``n_max`` requirement of ``jax.random.multinomial`` and keeps
    the operation differentiable in shape.

    Args:
        key: PRNG key.
        total: array of shape ``(R,)`` — number of trials per region.
        probs: array of shape ``(K,)`` — multinomial probabilities.

    Returns:
        array of shape ``(K, R)`` with ``sum_K == total`` per region.
    """
    K = probs.shape[0]
    keys = jax.random.split(key, K)
    counts = []
    remaining = jnp.maximum(total, 0.0).astype(jnp.float32)
    cum_p = jnp.float32(0.0)
    for i in range(K):
        if i < K - 1:
            cond_p = probs[i] / jnp.maximum(1.0 - cum_p, 1e-10)
            cond_p = jnp.clip(cond_p, 0.0, 1.0)
            draw = jax.random.binomial(keys[i], remaining, cond_p)
            draw = jnp.minimum(draw, remaining)
            counts.append(draw)
            remaining = remaining - draw
            cum_p = cum_p + probs[i]
        else:
            counts.append(remaining)
    return jnp.stack(counts, axis=0).astype(total.dtype)
