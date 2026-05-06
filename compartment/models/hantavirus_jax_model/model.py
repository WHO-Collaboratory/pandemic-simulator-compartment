"""Sex-structured SEIR hantavirus model in a wild rodent host.

Implements the deterministic ODE system from Allen, Wesley, Owen, Goff
et al. (2009), "A habitat-based model for the spread of hantavirus
between reservoir and spillover species" / Allen, McCormack & Jonsson
(2006), "Mathematical models for hantavirus infection in rodents,"
PMC7472466. Eight compartments (Susceptible / Exposed / Infective /
Recovered for each sex) with asymmetric mass-action transmission, sex-
specific recovery, density-dependent mortality, and a harmonic-mean
birth function. R0 scales with carrying capacity K, so seeding levels
must be set realistically for a given habitat.

WARNING: Like the other models in this repo, this implementation is
intended for local experimentation; UI / app support is not guaranteed.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as onp

from compartment.helpers import setup_logging
from compartment.model import Model
from compartment.parameters import ValueType

setup_logging()
logger = logging.getLogger(__name__)


class HantavirusJaxModel(Model):
    """Sex-structured SEIR hantavirus model in a wild rodent host."""

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="HANTAVIRUS",
            label="Hantavirus (sex-structured rodent SEIR)",
            description=(
                "Sex-structured SEIR hantavirus model in a wild rodent host "
                "with asymmetric transmission, sex-specific recovery, "
                "density-dependent mortality, and harmonic-mean births "
                "(Allen et al., PMC7472466)."
            ),
        )

        # --- Compartments ---------------------------------------------------
        # Males
        schema.add_compartment(
            "Sm", "Susceptible (male)",
            "Susceptible male rodents",
        )
        schema.add_compartment(
            "Em", "Exposed (male)",
            "Male rodents infected but not yet shedding virus",
        )
        schema.add_compartment(
            "Im", "Infective (male)",
            "Male rodents actively shedding virus",
            infective=True,
        )
        schema.add_compartment(
            "Rm", "Recovered (male)",
            "Male rodents recovered (antibody-positive, non-infectious)",
        )
        # Females
        schema.add_compartment(
            "Sf", "Susceptible (female)",
            "Susceptible female rodents",
        )
        schema.add_compartment(
            "Ef", "Exposed (female)",
            "Female rodents infected but not yet shedding virus",
        )
        schema.add_compartment(
            "If", "Infective (female)",
            "Female rodents actively shedding virus",
            infective=True,
        )
        schema.add_compartment(
            "Rf", "Recovered (female)",
            "Female rodents recovered (antibody-positive, non-infectious)",
        )
        # Em / Ef are not the target of any framework-declared edge
        # (S->E flows are computed manually because the FOI is
        # sex-asymmetric and bilinear). Declare their cumulative trackers
        # by hand so derivative() can populate them via _apply_flow().
        schema.add_compartment(
            "Em_total", "Exposed (male) Total",
            "Cumulative male exposures",
        )
        schema.add_compartment(
            "Ef_total", "Exposed (female) Total",
            "Cumulative female exposures",
        )

        # --- Transmission edges --------------------------------------------
        # Latent and recovery rates. The S->E flows are NOT declared as
        # schema edges because their force of infection mixes male and
        # female infectives with three different beta rates; that math is
        # done manually in derivative().
        schema.add_transmission_edge(
            source="Em", target="Im",
            variable_name="delta_m",
            label="Incubation Period (male, Em->Im)",
            description="Mean exposed period for males before becoming infectious.",
            default=15.0,
            default_min=10.0, default_max=21.0,
            min_value=2.0, max_value=60.0,
            unit="days",
            value_type=ValueType.DAYS,
        )
        schema.add_transmission_edge(
            source="Im", target="Rm",
            variable_name="gamma_m",
            label="Infectious Period (male, Im->Rm)",
            description=(
                "Mean infectious period for males. Paper assumes males "
                "remain infectious longer than females (1/gamma_m > 1/gamma_f)."
            ),
            default=120.0,
            default_min=60.0, default_max=180.0,
            min_value=14.0, max_value=365.0,
            unit="days",
            value_type=ValueType.DAYS,
        )
        schema.add_transmission_edge(
            source="Ef", target="If",
            variable_name="delta_f",
            label="Incubation Period (female, Ef->If)",
            description="Mean exposed period for females before becoming infectious.",
            default=15.0,
            default_min=10.0, default_max=21.0,
            min_value=2.0, max_value=60.0,
            unit="days",
            value_type=ValueType.DAYS,
        )
        schema.add_transmission_edge(
            source="If", target="Rf",
            variable_name="gamma_f",
            label="Infectious Period (female, If->Rf)",
            description="Mean infectious period for females.",
            default=60.0,
            default_min=30.0, default_max=120.0,
            min_value=14.0, max_value=365.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

        # --- Disease parameters --------------------------------------------
        # The three asymmetric mass-action contact rates from the paper.
        # Units: per-rodent per-day. Defaults derive from the paper's
        # Bayou-virus example (per 2-month time step) divided by ~60 days.
        schema.add_disease_parameter(
            name="beta_mm",
            label="Male->Male Contact Rate",
            description=(
                "Mass-action transmission rate from infective males to "
                "susceptible males (per rodent per day)."
            ),
            value_type=ValueType.FLOAT,
            default=1.67e-4,
            default_min=5e-5, default_max=5e-4,
            min_value=0.0, max_value=1e-2,
            unit="per rodent per day",
        )
        schema.add_disease_parameter(
            name="beta_mf",
            label="Male->Female Contact Rate",
            description=(
                "Mass-action transmission rate from infective males to "
                "susceptible females (per rodent per day). Typically "
                "<= beta_mm."
            ),
            value_type=ValueType.FLOAT,
            default=8.3e-5,
            default_min=2e-5, default_max=3e-4,
            min_value=0.0, max_value=1e-2,
            unit="per rodent per day",
        )
        schema.add_disease_parameter(
            name="beta_f",
            label="Female->Either Contact Rate",
            description=(
                "Mass-action transmission rate from infective females to "
                "susceptible rodents of either sex (per rodent per day). "
                "Typically the smallest of the three betas."
            ),
            value_type=ValueType.FLOAT,
            default=3.3e-5,
            default_min=5e-6, default_max=2e-4,
            min_value=0.0, max_value=1e-2,
            unit="per rodent per day",
        )
        # Demographic rates.
        schema.add_disease_parameter(
            name="birth_rate",
            label="Per-Rodent Birth Rate",
            description=(
                "Per-individual birth rate b in the harmonic-mean birth "
                "function B = 2 b Nm Nf / (Nm + Nf). Units of per day. "
                "Paper uses b = 4 per 2-month gestation cycle."
            ),
            value_type=ValueType.FLOAT,
            default=6.67e-2,
            default_min=2e-2, default_max=1.5e-1,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="death_baseline",
            label="Baseline Death Rate (a)",
            description=(
                "Density-independent component of the per-rodent death "
                "rate d(N) = a + c N (per day)."
            ),
            value_type=ValueType.FLOAT,
            default=1.67e-4,
            default_min=5e-5, default_max=5e-3,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="density_death_coeff",
            label="Density-Dependent Death Coefficient (c)",
            description=(
                "Slope of the density-dependent death rate d(N) = a + c N. "
                "Equilibrium carrying capacity K = (b/2 - a) / c."
            ),
            value_type=ValueType.FLOAT,
            default=3.32e-5,
            default_min=1e-5, default_max=1e-4,
            min_value=0.0, max_value=1.0,
            unit="per rodent per day",
        )
        schema.add_disease_parameter(
            name="male_fraction",
            label="Initial Male Fraction",
            description=(
                "Fraction of the initial rodent population that is male. "
                "0.5 reflects the paper's symmetric seeding."
            ),
            value_type=ValueType.PERCENTAGE,
            default=50.0,
            default_min=40.0, default_max=60.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )

        # --- Optional intervention -----------------------------------------
        # Rodent control / culling acts as an extra mortality on every
        # compartment by raising the baseline death rate. We expose it as
        # a multiplier on beta_mm/beta_mf/beta_f so the framework's generic
        # intervention scaling reduces transmission when control is active.
        schema.add_intervention(
            id="rodent_control",
            label="Rodent Population Control",
            description=(
                "Habitat-based rodent control (trapping, exclusion) that "
                "reduces effective contact rates between rodents."
            ),
            target_rates=["delta_m", "delta_f", "gamma_m", "gamma_f"],
            adherence=50.0,
            transmission_reduction=30.0,
        )

    # ------------------------------------------------------------------
    # Initial population
    # ------------------------------------------------------------------

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """Distribute each zone's rodent population across the eight
        sex/state compartments.

        ``population`` is interpreted as the total rodent count (carrying
        capacity for that habitat). ``infected_population`` (percentage)
        is split evenly between male and female *infective* compartments;
        the remainder is split between Sm and Sf according to
        ``male_fraction`` (default 50/50). ``Em``/``Ef``/``Rm``/``Rf``
        and the cumulative ``_total`` columns start at zero.
        """
        col = {v: i for i, v in enumerate(compartment_list)}
        pop = onp.zeros((len(admin_zones), len(compartment_list)))

        # NOTE: schema-declared `male_fraction` lives under Disease, but
        # `get_initial_population` runs before the model is constructed
        # so we read it from kwargs if available, otherwise default 50%.
        male_frac = float(kwargs.get("male_fraction", 0.5))

        for z, zone in enumerate(admin_zones):
            N = float(zone["population"])
            inf_pct = max(float(zone.get("infected_population", 0.0) or 0.0), 0.0)
            infected = N * inf_pct / 100.0
            healthy = max(N - infected, 0.0)

            pop[z, col["Im"]] = infected * male_frac
            pop[z, col["If"]] = infected * (1.0 - male_frac)
            pop[z, col["Sm"]] = healthy * male_frac
            pop[z, col["Sf"]] = healthy * (1.0 - male_frac)

        return pop

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, config):
        super().__init__(config)

        # No spatial coupling — rodent habitats are treated as independent.
        n_regions = self.population_matrix.shape[1]
        self.travel_matrix = jnp.eye(n_regions)

        # Schema-converted per-day rates from `_load_transmission_params`.
        # Provide fallbacks for None values so default-only configs run.
        if self.delta_m is None:
            self.delta_m = 1.0 / 15.0
        if self.delta_f is None:
            self.delta_f = 1.0 / 15.0
        if self.gamma_m is None:
            self.gamma_m = 1.0 / 120.0
        if self.gamma_f is None:
            self.gamma_f = 1.0 / 60.0

        disease_cfg = config.get("Disease", {}) or {}

        def _f(key, default):
            v = disease_cfg.get(key, default)
            return default if v is None else float(v)

        # Asymmetric beta rates and demographic scalars.
        self.beta_mm = _f("beta_mm", 1.67e-4)
        self.beta_mf = _f("beta_mf", 8.3e-5)
        self.beta_f = _f("beta_f", 3.3e-5)
        self.b_birth = _f("birth_rate", 6.67e-2)
        self.a_death = _f("death_baseline", 1.67e-4)
        self.c_density = _f("density_death_coeff", 3.32e-5)

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    def prepare_initial_state(self):
        return self.population_matrix, list(self.compartment_list)

    # ------------------------------------------------------------------
    # ODE derivative
    # ------------------------------------------------------------------

    def derivative(self, y, t, p):
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(self.compartment_list)}
        Sm = states[C.Sm]
        Em = states[C.Em]
        Im = states[C.Im]
        Rm = states[C.Rm]
        Sf = states[C.Sf]
        Ef = states[C.Ef]
        If_ = states[C.If]
        Rf = states[C.Rf]

        # Live-population denominators (exclude cumulative _total columns).
        Nm = Sm + Em + Im + Rm
        Nf = Sf + Ef + If_ + Rf
        N = Nm + Nf

        prop_infective = (Im.sum() + If_.sum()) / (N.sum() + 1e-10)

        # Apply schema-driven interventions to the four framework-handled
        # rates. Travel matrix passes through unchanged (no travel).
        rates = {
            "delta_m": params["delta_m"],
            "gamma_m": params["gamma_m"],
            "delta_f": params["delta_f"],
            "gamma_f": params["gamma_f"],
        }
        rates, _ = self._apply_interventions(t, rates, prop_infective)

        # Framework computes Em->Im, Im->Rm, Ef->If, If->Rf and
        # auto-accumulates into Im_total, Rm_total, If_total, Rf_total.
        derivs = self._compute_derivatives(states, rates)

        # --- Force of infection (asymmetric mass action) -----------------
        # Paper's transmission structure:
        #   lambda_m = beta_mm * Im + beta_f  * If   (male susceptibles)
        #   lambda_f = beta_mf * Im + beta_f  * If   (female susceptibles)
        lambda_m = self.beta_mm * Im + self.beta_f * If_
        lambda_f = self.beta_mf * Im + self.beta_f * If_

        new_exposures_m = Sm * lambda_m
        new_exposures_f = Sf * lambda_f
        self._apply_flow(derivs, C.Sm, C.Em, new_exposures_m)
        self._apply_flow(derivs, C.Sf, C.Ef, new_exposures_f)

        # --- Births: harmonic mean, equally split between sexes ----------
        # B(Nm, Nf) = 2 * b * Nm * Nf / (Nm + Nf); B/2 enters each of Sm, Sf.
        births_per_sex = self.b_birth * Nm * Nf / (N + 1e-10)
        derivs[C.Sm] = derivs[C.Sm] + births_per_sex
        derivs[C.Sf] = derivs[C.Sf] + births_per_sex

        # --- Density-dependent deaths: d(N) = a + c * N ------------------
        # Applied to every live compartment (not _total trackers).
        d_N = self.a_death + self.c_density * N
        for cid in (C.Sm, C.Em, C.Im, C.Rm, C.Sf, C.Ef, C.If, C.Rf):
            derivs[cid] = derivs[cid] - states[cid] * d_N

        return jnp.stack([derivs[c] for c in self.compartment_list])
