"""Hantavirus SEIR model with person-to-person transmission and risk perception.

Implements the model from Gutiérrez Jara & Muñoz Quezada (2022),
"Modeling of hantavirus cardiopulmonary syndrome", Medwave 22(3):e002526
(DOI 10.5867/medwave.2022.03.002526).

The model couples a human SEIR (S_h, E_h, I_h, R_h plus a cumulative
death tracker D_h) with a rodent SEIR (S_m, E_m, I_m, R_m) and a
risk-perception state P. Two transmission routes drive human infection:

    rodent -> human:  beta * S_h * I_m / N_h
    human  -> human:  beta_h * S_h * I_h / N_h     (Andes virus only)

Risk perception P responds to the proportion of infected humans and in
turn modulates beta and beta_h via beta_eff = beta_star * (P_star / P)
— so a population that perceives more risk than the average effectively
suppresses both spillover and person-to-person spread.

Differs from ``hantavirus_jax_model`` (Cornejo-Donoso et al. 2023, which
is rodent-focused with three-sector spatial structure and *no* human-
to-human transmission). This model is well-mixed in space but adds the
human-to-human pathway and the dynamic risk-perception feedback.

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


_HUMAN_LIVE = ("S_h", "E_h", "I_h", "R_h")
_RODENT_LIVE = ("S_m", "E_m", "I_m", "R_m")


class HantavirusHumanJaxModel(Model):
    """SEIR hantavirus model with person-to-person transmission and an
    endogenous risk-perception state (Gutiérrez Jara & Muñoz Quezada,
    Medwave 2022)."""

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="HANTAVIRUS_HUMAN_TRANSMISSION",
            label="Hantavirus (person-to-person + risk perception)",
            description=(
                "SEIR hantavirus model coupling rodent-to-human spillover "
                "with Andes-virus person-to-person transmission and a "
                "dynamic risk-perception state that modulates both betas "
                "(Gutiérrez Jara & Muñoz Quezada, Medwave 2022)."
            ),
        )

        # --- Human compartments -------------------------------------------
        schema.add_compartment(
            "S_h", "Susceptible humans",
            "Humans susceptible to hantavirus infection.",
        )
        schema.add_compartment(
            "E_h", "Exposed humans",
            "Latently infected humans (incubation period).",
        )
        schema.add_compartment(
            "I_h", "Infectious humans",
            "Symptomatic, infectious humans capable of person-to-person "
            "spread of the Andes virus.",
            infective=True,
        )
        schema.add_compartment(
            "R_h", "Recovered humans",
            "Humans recovered from hantavirus and assumed immune.",
        )
        schema.add_compartment(
            "D_h", "Cumulative human deaths",
            "Cumulative human deaths attributable to hantavirus "
            "(case-fatality fraction of cumulative I_h flow).",
        )

        # --- Rodent compartments ------------------------------------------
        schema.add_compartment(
            "S_m", "Susceptible rodents",
            "Rodents susceptible to hantavirus infection.",
        )
        schema.add_compartment(
            "E_m", "Exposed rodents",
            "Latently infected rodents.",
        )
        schema.add_compartment(
            "I_m", "Infectious rodents",
            "Infectious rodents shedding virus that can spill over to "
            "humans.",
            infective=True,
        )
        schema.add_compartment(
            "R_m", "Recovered rodents",
            "Rodents recovered from hantavirus.",
        )

        # --- Risk perception (continuous state in [0, 1]) -----------------
        # Treated as a compartment so it integrates with the framework's
        # state machinery; not a population.
        schema.add_compartment(
            "P", "Risk perception",
            "Population-level risk perception in [0, 1]. Higher values "
            "suppress both rodent-to-human and person-to-person "
            "transmission via beta_eff = beta_star * (P_star / P).",
        )

        # --- Schema edges (framework-handled) ------------------------------
        # Pure rate * source flows for E -> I and I -> R (humans + rodents).
        # Force-of-infection terms (S -> E) are multi-rate and applied
        # manually in derivative().
        schema.add_transmission_edge(
            source="E_h", target="I_h",
            variable_name="delta_h",
            label="Human Incubation Period (E_h -> I_h)",
            description=(
                "Mean human incubation period. Paper uses 1/delta_h "
                "in the range 7-42 days."
            ),
            default=21.0,
            default_min=7.0, default_max=42.0,
            min_value=1.0, max_value=120.0,
            unit="days", value_type=ValueType.DAYS,
        )
        schema.add_transmission_edge(
            source="I_h", target="R_h",
            variable_name="gamma_h",
            label="Human Infectious Period (I_h -> R_h)",
            description=(
                "Mean human infectious period. Paper uses 1/gamma_h "
                "in the range 1-5 days."
            ),
            default=3.0,
            default_min=1.0, default_max=5.0,
            min_value=1.0, max_value=60.0,
            unit="days", value_type=ValueType.DAYS,
        )
        schema.add_transmission_edge(
            source="E_m", target="I_m",
            variable_name="delta_m",
            label="Rodent Incubation Period (E_m -> I_m)",
            description=(
                "Mean rodent incubation period. Paper uses 1/delta_m "
                "in the range 7-30 days."
            ),
            default=18.0,
            default_min=7.0, default_max=30.0,
            min_value=1.0, max_value=90.0,
            unit="days", value_type=ValueType.DAYS,
        )
        schema.add_transmission_edge(
            source="I_m", target="R_m",
            variable_name="gamma_m",
            label="Rodent Infectious Period (I_m -> R_m)",
            description=(
                "Mean rodent infectious period. Paper uses 1/gamma_m "
                "in the range 7-60 days."
            ),
            default=30.0,
            default_min=7.0, default_max=60.0,
            min_value=1.0, max_value=120.0,
            unit="days", value_type=ValueType.DAYS,
        )

        # --- Disease parameters (manual flows) -----------------------------
        # Force-of-infection rates. The two human-FOI rates (beta_star and
        # beta_h_star) act on different infectious sources (rodents and
        # humans) so a single schema edge cannot represent S_h -> E_h. The
        # rodent FOI is also applied manually for symmetry and because P
        # only modulates the human-side betas.
        schema.add_disease_parameter(
            name="beta_star",
            label="Avg Rodent-to-Human Transmission (beta*)",
            description=(
                "Average rodent-to-human transmission rate at the "
                "average risk perception P*. Effective rate in the "
                "model is beta = beta* * (P* / P)."
            ),
            value_type=ValueType.FLOAT,
            default=0.01,
            default_min=1e-3, default_max=5e-2,
            min_value=0.0, max_value=10.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="beta_h_star",
            label="Avg Human-to-Human Transmission (beta_h*)",
            description=(
                "Average person-to-person transmission rate (Andes "
                "virus) at the average risk perception P*. Calibrated "
                "from R0 = beta_h* / (gamma_h + d_h) ~= 2.12 "
                "(Argentina 2018-19 outbreak). Set to 0 for Sin Nombre."
            ),
            value_type=ValueType.FLOAT,
            default=1.55,
            default_min=0.5, default_max=3.0,
            min_value=0.0, max_value=10.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="beta_m",
            label="Rodent-to-Rodent Transmission (beta_m)",
            description=(
                "Mass-action rodent transmission rate (frequency-"
                "dependent in the implementation). Paper notes "
                "beta_m ~= 1.38 * (gamma_m + d_m)."
            ),
            value_type=ValueType.FLOAT,
            default=0.048,
            default_min=0.01, default_max=0.2,
            min_value=0.0, max_value=10.0,
            unit="per day",
        )

        # Disease lethality — competing hazards from I_h.
        # I_h exits either to R_h (recovery) or D_h (death) at the same
        # total rate gamma_h, split by cfr_eff. cfr_eff rises from cfr
        # toward cfr_max as concurrent cases exceed hospital_capacity,
        # modelling the real increase in fatality when ICUs are overwhelmed.
        schema.add_disease_parameter(
            name="cfr",
            label="Baseline Case Fatality Rate (%)",
            description=(
                "Fraction of infectious humans who die when hospital "
                "capacity is not exceeded. US hantavirus historical "
                "average ~36%."
            ),
            value_type=ValueType.PERCENTAGE,
            default=36.0,
            default_min=20.0, default_max=60.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )
        schema.add_disease_parameter(
            name="cfr_max",
            label="Saturated Case Fatality Rate (%)",
            description=(
                "CFR when hospital capacity is fully exceeded. "
                "Transitions smoothly from cfr toward cfr_max as "
                "concurrent I_h surpasses hospital_capacity * N_h."
            ),
            value_type=ValueType.PERCENTAGE,
            default=70.0,
            default_min=40.0, default_max=90.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )
        schema.add_disease_parameter(
            name="hospital_capacity",
            label="Hospital Capacity (fraction of population)",
            description=(
                "Fraction of the zone population that can be "
                "simultaneously hospitalised before CFR begins rising "
                "toward cfr_max. US baseline ~0.003 (2.7 beds per "
                "1,000 people, ~half acute occupancy available)."
            ),
            value_type=ValueType.FLOAT,
            default=0.003,
            default_min=0.001, default_max=0.02,
            min_value=0.0, max_value=1.0,
        )

        # Seasonality of rodent-to-human spillover.
        schema.add_disease_parameter(
            name="seasonal_amplitude",
            label="Seasonal Spillover Amplitude",
            description=(
                "Strength of the cosine seasonal modulation on the "
                "rodent-to-human spillover rate beta. At amplitude=0.6 "
                "the rate ranges from 40% (winter) to 160% (peak summer) "
                "of its mean. Set to 0 to disable seasonality."
            ),
            value_type=ValueType.FLOAT,
            default=0.6,
            default_min=0.0, default_max=1.0,
            min_value=0.0, max_value=1.0,
        )
        schema.add_disease_parameter(
            name="seasonal_peak_day",
            label="Seasonal Peak Day of Year",
            description=(
                "Day of year (1-365) when rodent-to-human spillover "
                "peaks. US hantavirus peaks around day 150 (late May / "
                "early June) when people open winter-closed buildings "
                "and have maximum outdoor rodent contact."
            ),
            value_type=ValueType.INTEGER,
            default=150,
            min_value=1, max_value=365,
        )

        # Demographics (births and natural mortality).
        schema.add_disease_parameter(
            name="b_h",
            label="Human Birth Rate",
            description="Per-capita human birth rate (per day).",
            value_type=ValueType.FLOAT,
            default=0.0001243,
            default_min=5e-5, default_max=5e-4,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="d_h",
            label="Human Natural Mortality",
            description="Per-capita non-disease human mortality rate (per day).",
            value_type=ValueType.FLOAT,
            default=0.0001243,
            default_min=5e-5, default_max=5e-4,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="b_m",
            label="Rodent Birth Rate",
            description="Per-capita rodent birth rate (per day).",
            value_type=ValueType.FLOAT,
            default=0.00139,
            default_min=5e-4, default_max=5e-3,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="d_m",
            label="Rodent Natural Mortality",
            description="Per-capita rodent mortality rate (per day).",
            value_type=ValueType.FLOAT,
            default=0.00139,
            default_min=5e-4, default_max=5e-3,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )

        # Risk perception parameters.
        schema.add_disease_parameter(
            name="P_star",
            label="Average Risk Perception (P*)",
            description=(
                "Long-run population-mean risk perception (0-1). "
                "Acts as the equilibrium target of P and as the "
                "numerator in the beta scaling P*/P."
            ),
            value_type=ValueType.FLOAT,
            default=0.5,
            default_min=0.1, default_max=0.9,
            min_value=0.01, max_value=1.0,
        )
        schema.add_disease_parameter(
            name="lambda_1",
            label="Resistance to Change (lambda_1)",
            description=(
                "Rate at which risk perception relaxes back toward "
                "P* in absence of new cases. Paper baseline 0.65; "
                "lower values represent a population that updates "
                "perception faster."
            ),
            value_type=ValueType.FLOAT,
            default=0.65,
            default_min=0.001, default_max=1.0,
            min_value=0.0, max_value=1.0,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="lambda_2",
            label="Reaction Speed (lambda_2)",
            description=(
                "Sensitivity of risk perception to the current human "
                "infectious fraction I_h / N_h. Paper baseline 0.05; "
                "raising it together with lowering lambda_1 cuts the "
                "peak by ~25% (Fig. 4 in the paper)."
            ),
            value_type=ValueType.FLOAT,
            default=0.05,
            default_min=0.01, default_max=2.0,
            min_value=0.0, max_value=10.0,
            unit="per day",
        )

        # --- Custom admin-zone fields --------------------------------------
        # Standard `population` is the human population; standard
        # `infected_population` seeds I_h. Rodents are seeded separately.
        schema.add_admin_zone_field(
            name="rodent_population",
            label="Rodent Population",
            description=(
                "Total rodent reservoir population (N_m) for the zone. "
                "Paper uses N_m = 100 alongside N_h = 100,000."
            ),
            value_type=ValueType.COUNT,
            default=100,
            min_value=0, max_value=10**9,
        )
        schema.add_admin_zone_field(
            name="infected_rodent_count",
            label="Initial Infected Rodents",
            description=(
                "Initial number of infectious rodents (I_m at t = 0). "
                "Paper seeds 1 infectious rodent."
            ),
            value_type=ValueType.COUNT,
            default=1,
            min_value=0, max_value=10**9,
        )
        schema.add_admin_zone_field(
            name="exposed_fraction",
            label="Rodent-Exposed Population Fraction (%)",
            description=(
                "Percentage of the zone's human population with meaningful "
                "rodent-excreta exposure risk (agricultural workers, rural "
                "residents, barn-cleaners, hikers). Only this fraction "
                "contributes to the spillover FOI. Urban zones: ~1-3%; "
                "rural/agricultural zones: ~8-15%."
            ),
            value_type=ValueType.PERCENTAGE,
            default=5.0,
            min_value=0.0, max_value=100.0,
            unit="%",
        )
        schema.add_admin_zone_field(
            name="initial_risk_perception",
            label="Initial Risk Perception (P_0)",
            description=(
                "Initial value of P at t = 0 (0-1). Defaults to P* if "
                "left at the default."
            ),
            value_type=ValueType.PERCENTAGE,
            default=50.0,
            min_value=1.0, max_value=100.0,
            unit="%",
        )

        # --- Travel volume -------------------------------------------------
        # Exposes the `leaving` fraction in the config travel_volume block.
        # When > 0, the gravity model in prepare_initial_state() replaces
        # the identity matrix with a population-distance weighted matrix.
        schema.set_travel_volume(leaving_default=0.05)

        # --- Optional intervention -----------------------------------------
        # Argentina 2018-19 outbreak: R0 fell from 2.12 to 0.96 once
        # symptomatic patients were isolated and high-risk contacts
        # quarantined. We expose this as a generic transmission scaler on
        # the human-to-human and spillover routes via a sentinel edge —
        # same pattern as compartment/models/hantavirus_jax_model.
        schema.add_intervention(
            id="case_isolation",
            label="Case Isolation & Risk Communication",
            description=(
                "Symptomatic-case isolation and risk-communication "
                "campaign that scales both spillover (beta) and "
                "person-to-person (beta_h) transmission while active."
            ),
            target_rates=["transmission_scale"],
            adherence=70.0,
            transmission_reduction=55.0,
        )
        schema.add_transmission_edge(
            source="S_h", target="S_h",
            variable_name="transmission_scale",
            label="Transmission Scaling Factor",
            description=(
                "Internal sentinel edge — its rate is the multiplier "
                "applied to beta and beta_h while case_isolation is "
                "active. Self-loop; no flow effect."
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
        """Seed humans, rodents, and risk perception per zone.

        Standard fields:
          - ``population`` — total human population N_h.
          - ``infected_population`` — initial I_h.
        Custom fields:
          - ``rodent_population`` — total rodents N_m.
          - ``infected_rodent_count`` — initial I_m.
          - ``initial_risk_perception`` — initial P (percent, 0-100).
        """
        col = {v: i for i, v in enumerate(compartment_list)}
        pop = onp.zeros((len(admin_zones), len(compartment_list)))

        for z, zone in enumerate(admin_zones):
            humans = float(zone["population"])
            inf_h = float(zone.get("infected_population", 0) or 0)
            inf_h = max(min(inf_h, humans), 0.0)

            rodents = float(zone.get("rodent_population", 0) or 0)
            inf_m = float(zone.get("infected_rodent_count", 0) or 0)
            inf_m = max(min(inf_m, rodents), 0.0)

            p0 = float(zone.get("initial_risk_perception", 50.0) or 0.0) / 100.0
            p0 = min(max(p0, 0.01), 1.0)

            # Humans: split into S/I; E and R start empty.
            pop[z, col["S_h"]] = humans - inf_h
            pop[z, col["I_h"]] = inf_h

            # Rodents: split into S/I; E and R start empty.
            pop[z, col["S_m"]] = rodents - inf_m
            pop[z, col["I_m"]] = inf_m

            # Risk perception state (single scalar per zone).
            pop[z, col["P"]] = p0

        return pop

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, config):
        super().__init__(config)

        # Admin zones + sigma — used by gravity() in prepare_initial_state().
        case_file = config.get("case_file") or {}
        self._admin_zones = case_file.get("admin_zones", [])
        travel_volume = config.get("travel_volume") or {}
        self._sigma = (
            travel_volume.get("leaving", 0.0)
            if isinstance(travel_volume, dict) else 0.0
        )
        # Placeholder; overwritten in prepare_initial_state().
        n_regions = self.population_matrix.shape[1]
        self.travel_matrix = jnp.eye(n_regions)

        # Schema edges populate self.delta_h / gamma_h / delta_m / gamma_m
        # and self.transmission_scale. Provide fallbacks so the model can
        # run without an explicit TransmissionEdges block.
        for var, fallback in (
            ("delta_h", 1.0 / 21.0),
            ("gamma_h", 1.0 / 3.0),
            ("delta_m", 1.0 / 18.0),
            ("gamma_m", 1.0 / 30.0),
            ("transmission_scale", 1.0),
        ):
            if getattr(self, var, None) is None:
                setattr(self, var, fallback)

        disease_cfg = config.get("Disease", {}) or {}

        def _f(key, default):
            v = disease_cfg.get(key, default)
            return default if v is None else float(v)

        # Per-zone rodent-exposed fraction — shape (R,) broadcast against I_m.
        self.exposed_fraction = jnp.array([
            min(max(float(z.get("exposed_fraction", 5.0) or 5.0) / 100.0, 0.0), 1.0)
            for z in self._admin_zones
        ]) if self._admin_zones else jnp.array([0.05])

        # Force-of-infection rates.
        self.beta_star = _f("beta_star", 0.01)
        self.beta_h_star = _f("beta_h_star", 1.55)
        self.beta_m = _f("beta_m", 0.048)

        # Competing-hazard CFR and hospital saturation.
        self.cfr_base = _f("cfr", 36.0) / 100.0
        self.cfr_max = _f("cfr_max", 70.0) / 100.0
        self.hospital_capacity = _f("hospital_capacity", 0.003)

        # Seasonal spillover.
        self.seasonal_amplitude = _f("seasonal_amplitude", 0.6)
        self.seasonal_peak_day = _f("seasonal_peak_day", 150.0)

        # Demographics.
        self.b_h = _f("b_h", 0.0001243)
        self.d_h = _f("d_h", 0.0001243)
        self.b_m = _f("b_m", 0.00139)
        self.d_m = _f("d_m", 0.00139)

        # Risk perception.
        self.P_star = _f("P_star", 0.5)
        self.lambda_1 = _f("lambda_1", 0.65)
        self.lambda_2 = _f("lambda_2", 0.05)

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    def gravity(self, admin_zones, sigma, alpha=1.5):
        """Gravity travel matrix from admin-zone lat/lon/population.

        Entry T[i, j] = fraction of zone i's population present in zone j.

        Attraction from i to j is proportional to pop_j / dist_ij^alpha.
        Rows are normalised so the off-diagonal mass sums to sigma; the
        diagonal is 1 - sigma (the stay-home fraction).

        alpha=1.5 is a typical empirical value for inter-regional US
        mobility (steeper than the 1.0 used in simple gravity models,
        reflecting that very distant zones contribute little). Falls back
        to an identity matrix when there is only one zone or sigma == 0.
        """
        n = len(admin_zones)
        if n <= 1 or sigma == 0.0:
            return onp.eye(n)

        lats = onp.array([z["center_lat"] for z in admin_zones])
        lons = onp.array([z["center_lon"] for z in admin_zones])
        pops = onp.array([float(z["population"]) for z in admin_zones])

        # Pairwise great-circle distance (vectorised Haversine).
        R_earth = 6371.0
        lat_r = onp.radians(lats)
        lon_r = onp.radians(lons)
        dlat = lat_r[:, None] - lat_r[None, :]
        dlon = lon_r[:, None] - lon_r[None, :]
        a = (
            onp.sin(dlat / 2) ** 2
            + onp.cos(lat_r[:, None]) * onp.cos(lat_r[None, :]) * onp.sin(dlon / 2) ** 2
        )
        dist_km = 2 * R_earth * onp.arcsin(onp.sqrt(onp.clip(a, 0.0, 1.0)))

        # Gravity attraction: pop_j / dist_ij^alpha (clamp denominator).
        dist_clamped = onp.where(dist_km < 1.0, 1.0, dist_km)
        attraction = pops[None, :] / (dist_clamped ** alpha)
        onp.fill_diagonal(attraction, 0.0)

        # Normalise rows → off-diagonal sums to 1, then scale by sigma.
        row_sums = attraction.sum(axis=1, keepdims=True)
        row_sums = onp.where(row_sums == 0.0, 1.0, row_sums)
        T = sigma * (attraction / row_sums)
        onp.fill_diagonal(T, 1.0 - sigma)

        return T

    def prepare_initial_state(self):
        self.travel_matrix = jnp.array(
            self.gravity(self._admin_zones, self._sigma)
        )
        return self.population_matrix, list(self.compartment_list)

    # ------------------------------------------------------------------
    # ODE derivative
    # ------------------------------------------------------------------

    def derivative(self, y, t, p):
        params = self._unpack_params(p)
        states = {c: y[i] for i, c in enumerate(self.compartment_list)}

        # --- Apply schema interventions ----------------------------------
        # Only the sentinel transmission_scale rate is meaningfully driven
        # by the case_isolation intervention; the four SEIR rates pass
        # through unchanged unless a future intervention targets them.
        N_h = sum(states[c] for c in _HUMAN_LIVE)
        I_h = states["I_h"]
        I_m = states["I_m"]
        prop_inf = (I_h + I_m).sum() / (
            N_h.sum() + sum(states[c] for c in _RODENT_LIVE).sum() + 1e-10
        )

        rate_keys = ["delta_h", "gamma_h", "delta_m", "gamma_m", "transmission_scale"]
        rates = {k: params[k] for k in rate_keys}
        rates, _ = self._apply_interventions(t, rates, prop_inf)

        # --- Framework-handled E->I (humans + rodents); I->R is skipped -----
        # gamma_h is skipped here because I_h exits via competing hazards
        # (recovery vs death) rather than a single edge. Applied manually below.
        derivs = self._compute_derivatives(
            states, rates, skip_edges={"transmission_scale", "gamma_h"},
        )

        scale = rates["transmission_scale"]

        # --- Risk-perception modulation of betas -------------------------
        P = jnp.clip(states["P"], 1e-3, None)
        risk_factor = self.P_star / P

        # --- Seasonal spillover modulation --------------------------------
        # Cosine peaks at seasonal_peak_day (day ~150 = late May in the US),
        # reflecting spring barn-cleaning and outdoor contact. Clipped at 0.1
        # so winter beta never fully collapses.
        day_of_year = (self.start_date_ordinal + t) % 365.0
        seasonal = jnp.clip(
            1.0 + self.seasonal_amplitude * jnp.cos(
                2.0 * jnp.pi * (day_of_year - self.seasonal_peak_day) / 365.0
            ),
            0.1, None,
        )

        # Seasonality applies to spillover only — person-to-person spread
        # is not tied to outdoor rodent contact.
        beta_eff = self.beta_star * risk_factor * scale * seasonal
        beta_h_eff = self.beta_h_star * risk_factor * scale

        # --- Force of infection ------------------------------------------
        N_m = sum(states[c] for c in _RODENT_LIVE)
        S_h = states["S_h"]
        S_m = states["S_m"]

        # Rodent spillover is local and only reaches the at-risk fraction of
        # the human population (agricultural workers, rural residents, etc.).
        # exposed_fraction is per-zone, shape (R,), element-wise with I_m.
        spillover = beta_eff * self.exposed_fraction * I_m / (N_h + 1e-10)

        # Person-to-person FOI uses the gravity travel matrix so that
        # infectious humans in zone j exert pressure on susceptibles in
        # zone i weighted by how much time i-residents spend in j.
        # T[i,j] = fraction of zone-i population present in zone j.
        I_over_N_h = I_h / (N_h + 1e-10)
        coupled_hh = jnp.einsum("ij,j->i", self.travel_matrix, I_over_N_h)
        foi_h = spillover + beta_h_eff * coupled_hh
        new_exposures_h = S_h * foi_h
        self._apply_flow(derivs, "S_h", "E_h", new_exposures_h)

        # Rodents: frequency-dependent within rodent population.
        foi_m = self.beta_m * I_m / (N_m + 1e-10)
        new_exposures_m = S_m * foi_m
        self._apply_flow(derivs, "S_m", "E_m", new_exposures_m)

        # --- Demographic births (inflow into S only) ---------------------
        derivs["S_h"] = derivs["S_h"] + self.b_h * N_h
        derivs["S_m"] = derivs["S_m"] + self.b_m * N_m

        # --- Natural (non-disease) mortality on every live compartment ---
        for cid in _HUMAN_LIVE:
            derivs[cid] = derivs[cid] - self.d_h * states[cid]
        for cid in _RODENT_LIVE:
            derivs[cid] = derivs[cid] - self.d_m * states[cid]

        # --- Competing hazards: I_h exits to R_h (recovery) or D_h (death) --
        # Effective CFR rises from cfr_base toward cfr_max as concurrent
        # infectious cases exceed hospital_capacity * N_h (per-zone).
        capacity = self.hospital_capacity * N_h
        cfr_eff = jnp.clip(
            self.cfr_base
            + (self.cfr_max - self.cfr_base) * I_h / (I_h + capacity + 1e-10),
            0.0, 1.0,
        )
        gamma_h = rates["gamma_h"]
        recovery_flow = gamma_h * (1.0 - cfr_eff) * I_h
        fatality_flow = gamma_h * cfr_eff * I_h
        self._apply_flow(derivs, "I_h", "R_h", recovery_flow)
        self._apply_flow(derivs, "I_h", "D_h", fatality_flow)
        # R_h_total was not auto-accumulated (gamma_h edge was skipped).
        if "R_h_total" in derivs:
            derivs["R_h_total"] = derivs["R_h_total"] + recovery_flow

        # --- Risk perception ODE -----------------------------------------
        # dP/dt = -lambda_1 * (P - P*) + lambda_2 * I_h / N_h
        derivs["P"] = (
            -self.lambda_1 * (states["P"] - self.P_star)
            + self.lambda_2 * I_h / (N_h + 1e-10)
        )

        return jnp.stack([derivs[c] for c in self.compartment_list])
