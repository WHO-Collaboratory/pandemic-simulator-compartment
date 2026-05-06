"""Spatial rodent-to-human hantavirus spillover model.

Implements the three-sector spatial model from Cornejo-Donoso, Cornejo,
Wagner et al. (2023), "A new mathematical model of hantavirus spread
considering humans-rodents commuting between rural and urban
environments" (PMC10536976). The territory is partitioned into three
sectors:

    u — urban (humans only)
    a — rural populated (humans + rodents)
    f — rural non-populated (rodents only)

Rodents follow SEIR dynamics with one-way commuting f → a (some rodents
visit the populated sector and return). Humans follow SI dynamics — there
is no human-to-human transmission; all human cases come from rodent
spillover. Humans commute bidirectionally between u, a, and f, and each
human population is bookkept by *residence × current location* so that
"urban resident infected on a weekend trip to f" is distinct from
"rural resident infected at home".

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


# Compartment ids referenced in many places — define once for clarity.
_RODENT_COMPS = (
    "M_aaS", "M_aaE", "M_aaI", "M_aaR",
    "M_ffS", "M_ffE", "M_ffI", "M_ffR",
    "M_afS", "M_afE", "M_afI", "M_afR",
)


class HantavirusJaxModel(Model):
    """Spatial rodent-to-human hantavirus spillover (urban / rural-pop /
    rural-empty) following Cornejo-Donoso et al. 2023, PMC10536976."""

    # --- Output grouping ---------------------------------------------------
    # 23 raw compartments is too noisy for a UI. Aggregate rodents by
    # disease state across sectors and visiting status; aggregate
    # susceptible humans into one bucket and split cumulative infections
    # by where the spillover happened (a vs f).
    COMPARTMENT_DELTA_GROUPING = {
        "M_S": ["M_aaS", "M_ffS", "M_afS"],
        "M_E": ["M_aaE", "M_ffE", "M_afE"],
        "M_I": ["M_aaI", "M_ffI", "M_afI"],
        "M_R": ["M_aaR", "M_ffR", "M_afR"],
        "H_S": ["H_uuS", "H_auS", "H_fuS", "H_aaS", "H_uaS", "H_faS"],
        # Cumulative human spillover infections (residence × infection-site).
        "H_D_a": ["H_auD", "H_aaD"],   # spillover happened in sector a
        "H_D_f": ["H_fuD", "H_faD"],   # spillover happened in sector f
        "H_d": ["H_d"],                # cumulative deaths
    }

    @classmethod
    def _add_total_compartments(cls, schema):
        # Suppress framework auto-generation. Rodent cumulative incidence
        # isn't a model output; cumulative *human* infections are tracked
        # explicitly via H_*D, and cumulative deaths via H_d.
        return

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="HANTAVIRUS",
            label="Hantavirus (spatial spillover)",
            description=(
                "Three-sector (urban / rural-populated / rural-empty) "
                "rodent-to-human hantavirus spillover model with explicit "
                "human and rodent commuting and residence-by-infection-site "
                "case bookkeeping (Cornejo-Donoso et al., PMC10536976)."
            ),
        )

        # --- Rodent compartments -------------------------------------------
        # M_<residence><current><state>. Residence/current ∈ {a, f}; visiting
        # rodents go from f to a only ("af"). aa = resident in a (no commute).
        # ff = resident in f currently at home. af = resident in f visiting a.
        for prefix, label_loc in (
            ("M_aa", "rural-populated resident"),
            ("M_ff", "rural-empty resident at home"),
            ("M_af", "rural-empty resident visiting rural-populated"),
        ):
            schema.add_compartment(
                f"{prefix}S", f"Susceptible rodent ({label_loc})",
                f"Susceptible rodents — {label_loc}.",
            )
            schema.add_compartment(
                f"{prefix}E", f"Exposed rodent ({label_loc})",
                f"Latently infected rodents — {label_loc}.",
            )
            schema.add_compartment(
                f"{prefix}I", f"Infectious rodent ({label_loc})",
                f"Infectious (shedding) rodents — {label_loc}.",
                infective=True,
            )
            schema.add_compartment(
                f"{prefix}R", f"Recovered rodent ({label_loc})",
                f"Recovered rodents — {label_loc}.",
            )

        # --- Human compartments --------------------------------------------
        # H_<residence><current><state>. Residence ∈ {u, a}; current ∈
        # {u, a, f}. There are no rural-empty residents (uninhabited
        # sector). State ∈ {S, D}; D is the cumulative spillover-infection
        # tracker, indexed by residence × where-infected.
        schema.add_compartment(
            "H_uuS", "Susceptible (urban resident, at home)",
            "Susceptible urban residents currently in u.",
        )
        schema.add_compartment(
            "H_auS", "Susceptible (urban resident, visiting a)",
            "Susceptible urban residents commuting to rural-populated a.",
        )
        schema.add_compartment(
            "H_fuS", "Susceptible (urban resident, visiting f)",
            "Susceptible urban residents commuting to rural-empty f.",
        )
        schema.add_compartment(
            "H_aaS", "Susceptible (rural-pop resident, at home)",
            "Susceptible rural-populated residents currently in a.",
        )
        schema.add_compartment(
            "H_uaS", "Susceptible (rural-pop resident, visiting u)",
            "Susceptible rural-populated residents commuting to urban u.",
        )
        schema.add_compartment(
            "H_faS", "Susceptible (rural-pop resident, visiting f)",
            "Susceptible rural-populated residents commuting to rural-empty f.",
        )
        # Cumulative spillover infections — residence × infection-site.
        schema.add_compartment(
            "H_auD", "Cumulative cases (urban resident, infected in a)",
            "Cumulative urban residents who acquired infection in rural-populated a.",
        )
        schema.add_compartment(
            "H_fuD", "Cumulative cases (urban resident, infected in f)",
            "Cumulative urban residents who acquired infection in rural-empty f.",
        )
        schema.add_compartment(
            "H_aaD", "Cumulative cases (rural-pop resident, infected in a)",
            "Cumulative rural-populated residents who acquired infection at home in a.",
        )
        schema.add_compartment(
            "H_faD", "Cumulative cases (rural-pop resident, infected in f)",
            "Cumulative rural-populated residents who acquired infection in rural-empty f.",
        )
        schema.add_compartment(
            "H_d", "Cumulative deaths",
            "Cumulative human deaths from hantavirus (case fatality d × cumulative cases).",
        )

        # --- Schema edges (framework-handled) ------------------------------
        # The rodent E→I and I→R transitions are pure rate × source flows
        # and are the only flows that fit the framework's edge model
        # cleanly. Defaults are shared across the three location bins
        # (paper uses one δ and one γ for all rodents); the user is free
        # to vary them.
        for prefix, where in (
            ("M_aa", "rural-populated"),
            ("M_ff", "rural-empty (home)"),
            ("M_af", "rural-empty visiting populated"),
        ):
            schema.add_transmission_edge(
                source=f"{prefix}E", target=f"{prefix}I",
                variable_name=f"delta_{prefix.split('_')[1]}",
                label=f"Rodent Incubation ({where})",
                description=(
                    "Mean rodent latent (E) period before becoming "
                    f"infectious — {where}. Paper uses 1/δ ≈ 50 days "
                    "across all bins."
                ),
                default=50.0,
                default_min=21.0, default_max=90.0,
                min_value=2.0, max_value=180.0,
                unit="days", value_type=ValueType.DAYS,
            )
            schema.add_transmission_edge(
                source=f"{prefix}I", target=f"{prefix}R",
                variable_name=f"gamma_{prefix.split('_')[1]}",
                label=f"Rodent Infectious Period ({where})",
                description=(
                    "Mean rodent infectious (I) period — "
                    f"{where}. Paper uses 1/γ ≈ 5 days."
                ),
                default=5.0,
                default_min=3.0, default_max=14.0,
                min_value=1.0, max_value=60.0,
                unit="days", value_type=ValueType.DAYS,
            )

        # --- Disease parameters (rates not expressible as edges) -----------
        # Rodent FOI and human spillover all have bilinear `rate × source ×
        # infectious-pool` forms that don't match the framework's edge
        # model, so β, β_a, β_f live as disease parameters and the flows
        # are applied manually in derivative().
        schema.add_disease_parameter(
            name="beta",
            label="Rodent–Rodent Transmission (β)",
            description=(
                "Mass-action transmission rate among rodents (per "
                "rodent per day). Drives both home-sector and visiting-"
                "rodent FOI in M_aa, M_ff, and M_af."
            ),
            value_type=ValueType.FLOAT,
            default=0.3,
            default_min=0.05, default_max=0.6,
            min_value=0.0, max_value=10.0,
            unit="per rodent per day",
        )
        schema.add_disease_parameter(
            name="beta_a",
            label="Spillover Rate (sector a)",
            description=(
                "Rodent-to-human transmission rate in rural-populated "
                "sector a. Paper notes β_a < β_f because human presence "
                "in a disturbs habitat and reduces effective contact."
            ),
            value_type=ValueType.FLOAT,
            default=3e-5,
            default_min=1e-6, default_max=1e-4,
            min_value=0.0, max_value=1e-2,
            unit="per rodent per day",
        )
        schema.add_disease_parameter(
            name="beta_f",
            label="Spillover Rate (sector f)",
            description=(
                "Rodent-to-human transmission rate in rural-empty "
                "sector f."
            ),
            value_type=ValueType.FLOAT,
            default=4e-5,
            default_min=1e-6, default_max=1e-4,
            min_value=0.0, max_value=1e-2,
            unit="per rodent per day",
        )
        schema.add_disease_parameter(
            name="case_fatality",
            label="Case Fatality (d)",
            description="Fraction of human spillover cases that die.",
            value_type=ValueType.PERCENTAGE,
            default=30.0,
            default_min=10.0, default_max=50.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )

        # Rodent mobility f ↔ a (one-way out, then return).
        schema.add_disease_parameter(
            name="phi",
            label="Rodent Exit Rate from f",
            description=(
                "Per-rodent rate (per day) at which rural-empty rodents "
                "leave home to visit rural-populated a. Multiplied by a "
                "state-specific fraction λ_<state>."
            ),
            value_type=ValueType.FLOAT,
            default=0.01,
            default_min=0.001, default_max=0.05,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="sigma_visit",
            label="Rodent Visit Duration",
            description="Mean time a visiting rodent stays in a before returning home.",
            value_type=ValueType.FLOAT,
            default=7.0,
            default_min=2.0, default_max=21.0,
            min_value=0.5, max_value=90.0,
            unit="days",
        )
        for state, label in (
            ("s", "Susceptible"), ("e", "Exposed"),
            ("i", "Infectious"), ("r", "Recovered"),
        ):
            schema.add_disease_parameter(
                name=f"lambda_{state}",
                label=f"Visit Fraction ({label})",
                description=(
                    f"Fraction of {label.lower()} rural-empty rodents who "
                    "leave home (multiplies φ to give the per-state exit rate)."
                ),
                value_type=ValueType.FLOAT,
                default=0.25,
                default_min=0.05, default_max=1.0,
                min_value=0.0, max_value=1.0,
            )

        # Rodent demographics (uniform birth/death).
        schema.add_disease_parameter(
            name="b_ma",
            label="Rodent Birth Rate (sector a)",
            description="Per-resident birth rate for rural-populated rodents.",
            value_type=ValueType.FLOAT,
            default=0.00139,
            default_min=5e-4, default_max=5e-3,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="b_mf",
            label="Rodent Birth Rate (sector f)",
            description="Per-resident birth rate for rural-empty rodents.",
            value_type=ValueType.FLOAT,
            default=0.00139,
            default_min=5e-4, default_max=5e-3,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="d_m",
            label="Rodent Mortality",
            description="Per-rodent natural mortality rate (uniform across states).",
            value_type=ValueType.FLOAT,
            default=0.00139,
            default_min=5e-4, default_max=5e-3,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )

        # Human mobility (between sectors).
        schema.add_disease_parameter(
            name="nu_u",
            label="Urban Exit Rate",
            description="Per-day rate at which urban residents leave u to commute.",
            value_type=ValueType.FLOAT,
            default=0.09,
            default_min=0.01, default_max=0.5,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="nu_a",
            label="Rural-Populated Exit Rate",
            description="Per-day rate at which rural-populated residents leave a to commute.",
            value_type=ValueType.FLOAT,
            default=0.04,
            default_min=0.005, default_max=0.5,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="tau_u",
            label="Urban Visit Duration",
            description="Mean time an urban resident spends visiting before returning home.",
            value_type=ValueType.FLOAT,
            default=5.0,
            default_min=1.0, default_max=14.0,
            min_value=0.5, max_value=90.0,
            unit="days",
        )
        schema.add_disease_parameter(
            name="tau_a",
            label="Rural-Populated Visit Duration",
            description="Mean time a rural-populated resident spends visiting before returning home.",
            value_type=ValueType.FLOAT,
            default=5.0,
            default_min=1.0, default_max=14.0,
            min_value=0.5, max_value=90.0,
            unit="days",
        )
        schema.add_disease_parameter(
            name="alpha_au",
            label="Urban → a Visit Fraction",
            description="Fraction of departing urban residents who go to rural-populated a.",
            value_type=ValueType.FLOAT,
            default=0.88,
            default_min=0.5, default_max=1.0,
            min_value=0.0, max_value=1.0,
        )
        schema.add_disease_parameter(
            name="alpha_fu",
            label="Urban → f Visit Fraction",
            description=(
                "Fraction of departing urban residents who go to rural-empty f. "
                "α_au + α_fu should sum to 1."
            ),
            value_type=ValueType.FLOAT,
            default=0.12,
            default_min=0.0, default_max=0.5,
            min_value=0.0, max_value=1.0,
        )
        schema.add_disease_parameter(
            name="alpha_ua",
            label="Rural-pop → u Visit Fraction",
            description="Fraction of departing rural-populated residents who go to urban u.",
            value_type=ValueType.FLOAT,
            default=0.7,
            default_min=0.2, default_max=1.0,
            min_value=0.0, max_value=1.0,
        )
        schema.add_disease_parameter(
            name="alpha_fa",
            label="Rural-pop → f Visit Fraction",
            description=(
                "Fraction of departing rural-populated residents who go to rural-empty f. "
                "Note: paper has α_ua + α_fa = 0.74 (not 1) — the unaccounted "
                "fraction effectively leaves the modelled system."
            ),
            value_type=ValueType.FLOAT,
            default=0.04,
            default_min=0.0, default_max=0.3,
            min_value=0.0, max_value=1.0,
        )

        # --- Custom admin-zone fields (sector populations) -----------------
        # Each admin zone represents a *territory* containing all three
        # sectors. The framework's required `population` is interpreted as
        # the total *human* population (urban + rural-populated). The
        # rural_population_fraction splits it; rodents are seeded
        # independently via the four fields below.
        schema.add_admin_zone_field(
            name="rural_population_fraction",
            label="Rural-Populated Fraction",
            description=(
                "Fraction of human population living in rural-populated "
                "sector a (the rest live in urban u; nobody resides in "
                "rural-empty f)."
            ),
            value_type=ValueType.PERCENTAGE,
            default=20.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )
        schema.add_admin_zone_field(
            name="rodents_in_a",
            label="Rodent Population (sector a)",
            description="Total rodents resident in rural-populated sector a.",
            value_type=ValueType.COUNT,
            default=2000,
            min_value=0, max_value=10**9,
        )
        schema.add_admin_zone_field(
            name="rodents_in_f",
            label="Rodent Population (sector f)",
            description="Total rodents resident in rural-empty sector f.",
            value_type=ValueType.COUNT,
            default=8000,
            min_value=0, max_value=10**9,
        )
        schema.add_admin_zone_field(
            name="infected_rodent_fraction",
            label="Infected Rodent Fraction",
            description=(
                "Initial fraction of rodents who are infectious (split "
                "evenly between sectors a and f). Exposed and recovered "
                "rodents start at zero."
            ),
            value_type=ValueType.PERCENTAGE,
            default=1.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )

        # --- Optional intervention -----------------------------------------
        # Habitat-restoration / sanitation reduces effective rodent–human
        # spillover. Implemented as a multiplier on β_a and β_f. Schema
        # interventions can only target schema-declared edge variable
        # names, so we expose the spillover through "shadow" edge
        # variables and apply them in derivative() — see _spillover_scale.
        schema.add_intervention(
            id="environmental_management",
            label="Environmental Management",
            description=(
                "Habitat restoration / housing sanitation that reduces "
                "rodent–human spillover (β_a and β_f) without changing "
                "rodent dynamics."
            ),
            target_rates=["spillover_scale"],
            adherence=50.0,
            transmission_reduction=40.0,
        )

        # Shadow edge so the intervention has a target_rate to scale. The
        # source/target compartments are nominal — the edge is *always*
        # skipped in _compute_derivatives() and only its rate value
        # (modified by the intervention) is read in derivative() as a
        # multiplier on the spillover βs.
        schema.add_transmission_edge(
            source="H_uuS", target="H_uuS",
            variable_name="spillover_scale",
            label="Spillover Scaling Factor",
            description=(
                "Internal sentinel edge — its rate is the spillover "
                "multiplier modified by the environmental_management "
                "intervention. Self-loop; no flow effect."
            ),
            default=1.0,
            min_value=0.0, max_value=1.0,
            unit="multiplier",
        )

    # ------------------------------------------------------------------
    # Initial population
    # ------------------------------------------------------------------

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """Distribute each zone's population across the 23 compartments.

        The four custom admin-zone fields drive seeding:
          - ``rural_population_fraction``: % of humans living in a (rest in u).
          - ``rodents_in_a`` / ``rodents_in_f``: rodent populations per sector.
          - ``infected_rodent_fraction``: % of rodents initially infectious
            (split evenly between sector a and sector f rodents; visitors
            ``M_af*`` and exposed/recovered start at zero).

        At t = 0 every human is at home (visiting compartments = 0) and
        cumulative trackers (H_*D, H_d) are zero.
        """
        col = {v: i for i, v in enumerate(compartment_list)}
        pop = onp.zeros((len(admin_zones), len(compartment_list)))

        for z, zone in enumerate(admin_zones):
            humans = float(zone["population"])
            rural_frac = float(zone.get("rural_population_fraction", 20.0)) / 100.0
            rural_frac = min(max(rural_frac, 0.0), 1.0)

            rodents_a = float(zone.get("rodents_in_a", 0) or 0)
            rodents_f = float(zone.get("rodents_in_f", 0) or 0)
            inf_frac = max(float(zone.get("infected_rodent_fraction", 0) or 0), 0.0) / 100.0

            # Humans: start at home; visitors and cumulative trackers = 0.
            pop[z, col["H_uuS"]] = humans * (1.0 - rural_frac)
            pop[z, col["H_aaS"]] = humans * rural_frac

            # Rodents: split each sector into S and I per infected fraction;
            # E and R start at 0; visiting rodents (M_af*) start at 0.
            pop[z, col["M_aaI"]] = rodents_a * inf_frac
            pop[z, col["M_aaS"]] = max(rodents_a * (1.0 - inf_frac), 0.0)
            pop[z, col["M_ffI"]] = rodents_f * inf_frac
            pop[z, col["M_ffS"]] = max(rodents_f * (1.0 - inf_frac), 0.0)

        return pop

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, config):
        super().__init__(config)

        # No region-to-region travel — each admin zone is an independent
        # territory containing all three sectors internally.
        n_regions = self.population_matrix.shape[1]
        self.travel_matrix = jnp.eye(n_regions)

        # Schema edges set self.delta_aa/ff/af and self.gamma_aa/ff/af
        # via _load_transmission_params; provide fallbacks for None so
        # the model runs without an explicit TransmissionEdges block.
        for var, fallback in (
            ("delta_aa", 1.0 / 50.0), ("delta_ff", 1.0 / 50.0), ("delta_af", 1.0 / 50.0),
            ("gamma_aa", 1.0 / 5.0), ("gamma_ff", 1.0 / 5.0), ("gamma_af", 1.0 / 5.0),
            ("spillover_scale", 1.0),
        ):
            if getattr(self, var, None) is None:
                setattr(self, var, fallback)

        disease_cfg = config.get("Disease", {}) or {}

        def _f(key, default):
            v = disease_cfg.get(key, default)
            return default if v is None else float(v)

        # Transmission and case-fatality.
        self.beta = _f("beta", 0.3)
        self.beta_a = _f("beta_a", 3e-5)
        self.beta_f = _f("beta_f", 4e-5)
        self.case_fatality = _f("case_fatality", 30.0) / 100.0
        # Rodent mobility.
        self.phi = _f("phi", 0.01)
        self.sigma_visit = _f("sigma_visit", 7.0)
        self.lambda_s = _f("lambda_s", 0.25)
        self.lambda_e = _f("lambda_e", 0.25)
        self.lambda_i = _f("lambda_i", 0.25)
        self.lambda_r = _f("lambda_r", 0.25)
        # Rodent demographics.
        self.b_ma = _f("b_ma", 0.00139)
        self.b_mf = _f("b_mf", 0.00139)
        self.d_m = _f("d_m", 0.00139)
        # Human mobility.
        self.nu_u = _f("nu_u", 0.09)
        self.nu_a = _f("nu_a", 0.04)
        self.tau_u = _f("tau_u", 5.0)
        self.tau_a = _f("tau_a", 5.0)
        self.alpha_au = _f("alpha_au", 0.88)
        self.alpha_fu = _f("alpha_fu", 0.12)
        self.alpha_ua = _f("alpha_ua", 0.7)
        self.alpha_fa = _f("alpha_fa", 0.04)

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

        # --- Apply schema interventions ----------------------------------
        # All eight rate parameters (six rodent E→I, I→R rates plus the
        # spillover_scale sentinel) are exposed to the framework's
        # generic intervention scaling. Travel matrix is identity (no
        # travel), so its modification is a no-op.
        all_infectious = (
            states["M_aaI"] + states["M_ffI"] + states["M_afI"]
        )
        N_total = sum(
            states[c] for c in self.compartment_list
            if not c.endswith("_total") and c not in (
                # Cumulative trackers don't represent live population.
                "H_auD", "H_fuD", "H_aaD", "H_faD", "H_d",
            )
        )
        prop_inf = all_infectious.sum() / (N_total.sum() + 1e-10)

        rate_keys = [
            "delta_aa", "delta_ff", "delta_af",
            "gamma_aa", "gamma_ff", "gamma_af",
            "spillover_scale",
        ]
        rates = {k: params[k] for k in rate_keys}
        rates, _ = self._apply_interventions(t, rates, prop_inf)

        # --- Framework-handled rodent E→I and I→R ------------------------
        # Skip the spillover_scale sentinel (self-loop, no flow).
        derivs = self._compute_derivatives(
            states, rates, skip_edges={"spillover_scale"},
        )

        # Spillover multiplier (default 1.0; reduced when the
        # environmental_management intervention is active).
        s_scale = rates["spillover_scale"]

        # --- Aliases (rodents) -------------------------------------------
        M_aaS = states["M_aaS"]; M_aaE = states["M_aaE"]
        M_aaI = states["M_aaI"]; M_aaR = states["M_aaR"]
        M_ffS = states["M_ffS"]; M_ffE = states["M_ffE"]
        M_ffI = states["M_ffI"]; M_ffR = states["M_ffR"]
        M_afS = states["M_afS"]; M_afE = states["M_afE"]
        M_afI = states["M_afI"]; M_afR = states["M_afR"]

        # --- Aliases (humans) --------------------------------------------
        H_uuS = states["H_uuS"]; H_auS = states["H_auS"]; H_fuS = states["H_fuS"]
        H_aaS = states["H_aaS"]; H_uaS = states["H_uaS"]; H_faS = states["H_faS"]

        # Resident totals (births are proportional to the home population).
        N_ya = M_aaS + M_aaE + M_aaI + M_aaR
        N_yf = (
            M_ffS + M_ffE + M_ffI + M_ffR
            + M_afS + M_afE + M_afI + M_afR
        )

        I_in_a = M_aaI + M_afI   # infectious in sector a (residents + visitors)
        I_in_f = M_ffI           # infectious in sector f

        # --- Rodent FOI (manual; frequency-dependent) --------------------
        # The paper writes the FOI as `β·S·I` (raw mass action) but the
        # baseline value β = 0.3 only produces sensible dynamics if S, I
        # are proportions or the rodent population is on the order of 1
        # (the paper does not specify the unit scale). We use the
        # frequency-dependent form `β·S·I/N_in_sector`, which keeps R₀
        # ≈ β/γ ≈ 1.5 independent of the absolute rodent count and
        # behaves robustly across the population scales typical of
        # admin-zone seeding.
        N_in_a = M_aaS + M_aaE + M_aaI + M_aaR + M_afS + M_afE + M_afI + M_afR
        N_in_f = M_ffS + M_ffE + M_ffI + M_ffR
        foi_aa = self.beta * M_aaS * I_in_a / (N_in_a + 1e-10)
        foi_ff = self.beta * M_ffS * I_in_f / (N_in_f + 1e-10)
        foi_af = self.beta * M_afS * I_in_a / (N_in_a + 1e-10)
        self._apply_flow(derivs, "M_aaS", "M_aaE", foi_aa)
        self._apply_flow(derivs, "M_ffS", "M_ffE", foi_ff)
        self._apply_flow(derivs, "M_afS", "M_afE", foi_af)

        # --- Rodent mobility f ↔ a (per-state out / shared return) -------
        # Outflow: λ_<state>·φ moves M_ff* into M_af*.
        # Return:  (1/σ_visit) moves M_af* back into M_ff*.
        for state, lam in (
            ("S", self.lambda_s), ("E", self.lambda_e),
            ("I", self.lambda_i), ("R", self.lambda_r),
        ):
            ff = states[f"M_ff{state}"]
            af = states[f"M_af{state}"]
            out_flow = lam * self.phi * ff
            return_flow = af / self.sigma_visit
            # f-resident leaves home toward a (no _total to update).
            derivs[f"M_ff{state}"] = derivs[f"M_ff{state}"] - out_flow + return_flow
            derivs[f"M_af{state}"] = derivs[f"M_af{state}"] + out_flow - return_flow

        # --- Rodent births (inflow only into S of resident sector) -------
        # b_ma·N_ya into M_aaS; b_mf·N_yf into M_ffS (visitors are still
        # f-residents, so their offspring are born at home in f).
        derivs["M_aaS"] = derivs["M_aaS"] + self.b_ma * N_ya
        derivs["M_ffS"] = derivs["M_ffS"] + self.b_mf * N_yf

        # --- Rodent natural mortality (uniform on all 12 compartments) ---
        for cid in _RODENT_COMPS:
            derivs[cid] = derivs[cid] - self.d_m * states[cid]

        # --- Human mobility (linear; manual to share split-fraction params)
        # Urban residents: leave at ν_u, split between a (α_au) and
        # f (α_fu). We renormalize the split fractions to sum to 1 so the
        # population is conserved if a user enters values that don't add up.
        alpha_u_total = self.alpha_au + self.alpha_fu
        alpha_au_eff = self.alpha_au / (alpha_u_total + 1e-10)
        alpha_fu_eff = self.alpha_fu / (alpha_u_total + 1e-10)
        derivs["H_uuS"] = derivs["H_uuS"] - self.nu_u * H_uuS \
            + (H_auS + H_fuS) / self.tau_u
        derivs["H_auS"] = derivs["H_auS"] + alpha_au_eff * self.nu_u * H_uuS \
            - H_auS / self.tau_u
        derivs["H_fuS"] = derivs["H_fuS"] + alpha_fu_eff * self.nu_u * H_uuS \
            - H_fuS / self.tau_u

        # Rural-pop residents: leave at ν_a, split between u (α_ua) and
        # f (α_fa). Paper's defaults (α_ua=0.7, α_fa=0.04) only sum to
        # 0.74; we renormalize so the split sums to 1 and the population
        # is conserved. To recover the paper's literal behaviour
        # (where 26% of departing residents leak out of the model) set
        # both α values so they don't get rescaled below.
        alpha_a_total = self.alpha_ua + self.alpha_fa
        alpha_ua_eff = self.alpha_ua / (alpha_a_total + 1e-10)
        alpha_fa_eff = self.alpha_fa / (alpha_a_total + 1e-10)
        derivs["H_aaS"] = derivs["H_aaS"] - self.nu_a * H_aaS \
            + (H_uaS + H_faS) / self.tau_a
        derivs["H_uaS"] = derivs["H_uaS"] + alpha_ua_eff * self.nu_a * H_aaS \
            - H_uaS / self.tau_a
        derivs["H_faS"] = derivs["H_faS"] + alpha_fa_eff * self.nu_a * H_aaS \
            - H_faS / self.tau_a

        # --- Human spillover (S × I, location-dependent β) ----------------
        # Susceptibles in a (residents H_aaS or visitors H_auS) are exposed
        # to I_in_a at rate β_a; susceptibles in f (visitors only — H_fuS
        # for urban residents, H_faS for rural-pop residents) are exposed
        # to I_in_f at rate β_f.
        # NOTE: the paper's dH_faS/dt equation uses H_fuS in the spillover
        # term — almost certainly a typo for H_faS, which is the
        # biologically correct subject. We use H_faS.
        beta_a_eff = self.beta_a * s_scale
        beta_f_eff = self.beta_f * s_scale
        spill_au = beta_a_eff * H_auS * I_in_a   # urban → a → cases H_auD
        spill_aa = beta_a_eff * H_aaS * I_in_a   # rural → a → cases H_aaD
        spill_fu = beta_f_eff * H_fuS * I_in_f   # urban → f → cases H_fuD
        spill_fa = beta_f_eff * H_faS * I_in_f   # rural → f → cases H_faD

        # Subtract from each susceptible compartment, accumulate into the
        # corresponding cumulative tracker.
        derivs["H_auS"] = derivs["H_auS"] - spill_au
        derivs["H_aaS"] = derivs["H_aaS"] - spill_aa
        derivs["H_fuS"] = derivs["H_fuS"] - spill_fu
        derivs["H_faS"] = derivs["H_faS"] - spill_fa
        derivs["H_auD"] = derivs["H_auD"] + spill_au
        derivs["H_aaD"] = derivs["H_aaD"] + spill_aa
        derivs["H_fuD"] = derivs["H_fuD"] + spill_fu
        derivs["H_faD"] = derivs["H_faD"] + spill_fa

        # Cumulative deaths: case-fatality fraction of total spillover.
        derivs["H_d"] = derivs["H_d"] + self.case_fatality * (
            spill_au + spill_aa + spill_fu + spill_fa
        )

        return jnp.stack([derivs[c] for c in self.compartment_list])
