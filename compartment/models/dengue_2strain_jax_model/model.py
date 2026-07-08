import logging

import jax.numpy as np
import numpy as onp

from compartment.helpers import setup_logging
from compartment.model import Model
from compartment.parameters import ValueType

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


class Dengue2StrainModel(Model):
    """Two-strain dengue model with seasonal forcing, temporary cross-immunity,
    and antibody-dependent enhancement (ADE) of secondary infections.

    Migrated from the legacy hardcoded ``VECTOR_BORNE_2STRAIN`` model to the
    schema-driven framework.  The compartment structure and the epidemiological
    parameters are now declared in :meth:`define_parameters` (so they appear in
    the artifact, validation config, and are configurable via the ``Disease``
    block), while the ODE dynamics in :meth:`derivative` are preserved exactly
    from the original model.

    The force of infection has a non-standard multi-term form (a constant
    cross-immunity "leak" scaled by population and an ADE enhancement of
    secondary infectiousness), so the flows are applied manually rather than
    through schema transmission edges — the same approach used by the
    ``dengue_jax_model`` and ``hantavirus_jax_model`` models.

    Compartments (per region):

    - ``S``                 fully susceptible to both strains
    - ``I1`` / ``I2``       primary infection with strain 1 / strain 2
    - ``R1`` / ``R2``       recovered from a primary infection (cross-protected)
    - ``S1`` / ``S2``       cross-immunity has waned; susceptible to the *other* strain
    - ``I12`` / ``I21``     secondary infection (strain 2 after 1 / strain 1 after 2)
    - ``R``                 recovered from a secondary infection (immune to both)
    """

    DISEASE_TYPE = "VECTOR_BORNE_2STRAIN"
    DISEASE_LABEL = "Dengue (2-Strain)"
    DISEASE_DESCRIPTION = (
        "A two-strain dengue model with seasonal transmission, temporary "
        "cross-protective immunity, and antibody-dependent enhancement of "
        "secondary infections."
    )

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type=cls.DISEASE_TYPE,
            label=cls.DISEASE_LABEL,
            description=cls.DISEASE_DESCRIPTION,
        )

        # ---- Core human compartments ----
        schema.add_compartment("S", "Susceptible", "Susceptible to both strains")
        schema.add_compartment(
            "I1", "Infected (Strain 1)",
            "Primary infection with strain 1", infective=True,
        )
        schema.add_compartment(
            "I2", "Infected (Strain 2)",
            "Primary infection with strain 2", infective=True,
        )
        schema.add_compartment(
            "R1", "Recovered (Strain 1)",
            "Recovered from strain 1; temporarily cross-protected against strain 2",
        )
        schema.add_compartment(
            "R2", "Recovered (Strain 2)",
            "Recovered from strain 2; temporarily cross-protected against strain 1",
        )
        schema.add_compartment(
            "S1", "Post-Strain-1 Susceptible",
            "Immune to strain 1 (cross-immunity waned); susceptible to strain 2",
        )
        schema.add_compartment(
            "S2", "Post-Strain-2 Susceptible",
            "Immune to strain 2 (cross-immunity waned); susceptible to strain 1",
        )
        schema.add_compartment(
            "I12", "Secondary Infection (1→2)",
            "Secondary infection with strain 2 after recovering from strain 1",
            infective=True,
        )
        schema.add_compartment(
            "I21", "Secondary Infection (2→1)",
            "Secondary infection with strain 1 after recovering from strain 2",
            infective=True,
        )
        schema.add_compartment(
            "R", "Recovered (Both Strains)",
            "Recovered from a secondary infection; immune to both strains",
        )

        # ---- Cumulative tracking compartments ----
        # This model declares no transmission edges (the multi-term force of
        # infection is applied manually in derivative()), so the framework's
        # automatic per-edge _total generation is a no-op.  We declare the
        # cumulative trackers explicitly to preserve the legacy output columns.
        schema.add_compartment(
            "I_total", "Primary Infections (Cumulative)",
            "Cumulative primary infections (strain 1 and strain 2)",
        )
        schema.add_compartment(
            "R1_total", "Primary Recoveries (Cumulative)",
            "Cumulative recoveries from a primary infection",
        )
        schema.add_compartment(
            "S2_total", "Post-Primary Susceptibles (Cumulative)",
            "Cumulative entries into post-primary (cross-immunity-waned) susceptibility",
        )
        schema.add_compartment(
            "I2_total", "Secondary Infections (Cumulative)",
            "Cumulative secondary infections",
        )
        schema.add_compartment(
            "R2_total", "Secondary Recoveries (Cumulative)",
            "Cumulative recoveries from a secondary infection",
        )

        # ---- Disease parameters (configurable via the Disease block) ----
        schema.add_disease_parameter(
            name="transmission_rate",
            label="Transmission Rate",
            description="Baseline force-of-infection coefficient (per day).",
            value_type=ValueType.RATE,
            default=0.03,
            min_value=0.0,
            max_value=2.0,
            default_min=0.02,
            default_max=0.05,
            unit="per day",
        )
        schema.add_disease_parameter(
            name="seasonality_amplitude",
            label="Seasonality Amplitude",
            description=(
                "Relative amplitude of the annual sinusoidal forcing applied to "
                "the transmission rate (0 = no seasonality)."
            ),
            value_type=ValueType.FLOAT,
            default=0.2,
            min_value=0.0,
            max_value=1.0,
        )
        schema.add_disease_parameter(
            name="ade_factor",
            label="ADE Enhancement Factor",
            description=(
                "Antibody-dependent enhancement multiplier on the infectiousness "
                "of secondary infections (1.0 = no enhancement)."
            ),
            value_type=ValueType.FLOAT,
            default=1.5,
            min_value=1.0,
            max_value=5.0,
        )
        schema.add_disease_parameter(
            name="cross_immunity_leak",
            label="Cross-Immunity Leak",
            description=(
                "Small constant force-of-infection term scaled by population, "
                "representing background seeding/importation."
            ),
            value_type=ValueType.FLOAT,
            default=1e-5,
            min_value=0.0,
            max_value=1.0,
        )
        schema.add_disease_parameter(
            name="infectious_period",
            label="Infectious Period",
            description="Mean duration of infectiousness for primary and secondary infections.",
            value_type=ValueType.DAYS,
            default=50.0,
            min_value=1.0,
            max_value=365.0,
            unit="days",
        )
        schema.add_disease_parameter(
            name="cross_immunity_period",
            label="Cross-Immunity Period",
            description=(
                "Mean duration of temporary cross-protective immunity following a "
                "primary infection (after which the host becomes susceptible to "
                "the other strain)."
            ),
            value_type=ValueType.DAYS,
            default=730.0,
            min_value=1.0,
            max_value=3650.0,
            unit="days",
        )
        schema.add_disease_parameter(
            name="life_expectancy",
            label="Life Expectancy",
            description="Mean host life expectancy; sets balanced birth and death rates.",
            value_type=ValueType.FLOAT,
            default=65.0,
            min_value=1.0,
            max_value=120.0,
            unit="years",
        )

        # ---- Admin-zone fields ----
        schema.add_admin_zone_field(
            name="seroprevalence",
            label="Seroprevalence",
            description=(
                "Percentage of the population with prior dengue exposure. Split "
                "evenly across the two strains into the post-primary susceptible "
                "compartments (S1, S2) at initialization."
            ),
            value_type=ValueType.PERCENTAGE,
            default=30.0,
            min_value=0.0,
            max_value=100.0,
            unit="%",
        )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def __init__(self, config):
        super().__init__(config)

        # Map the configurable disease parameters (native units, as declared in
        # the schema) onto the rate constants used by the ODE.  Done here rather
        # than in derivative() so uncertainty runs (which rebuild the model from
        # an overridden config) re-derive every constant from scratch.
        self.beta_0 = self.transmission_rate
        self.eta = self.seasonality_amplitude
        self.epsilon = self.ade_factor
        self.rho = self.cross_immunity_leak

        inf_period = self.infectious_period
        self.gamma = 1.0 / inf_period if inf_period > 0 else 0.0

        cross_period = self.cross_immunity_period
        self.alpha = 1.0 / cross_period if cross_period > 0 else 0.0

        life = self.life_expectancy
        self.mu = 1.0 / (life * 365.0) if life > 0 else 0.0
        self.b = self.mu  # balanced demography: births = deaths

        # Annual seasonal forcing, no phase shift.  Preserved from the original
        # model and intentionally not exposed as a configurable knob.
        self.omega = 0.5 * np.pi / 365.0
        self.phi = 0.0

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """Seed S / I1 / I2 / S1 / S2 per zone.

        ``infected_population`` and ``seroprevalence`` are percentages, each
        split evenly across the two strains (hence division by 200):
        infections seed I1/I2 and prior exposure seeds the post-primary
        susceptibles S1/S2.  All other compartments (including the cumulative
        ``_total`` trackers) start at zero.
        """
        column_mapping = {value: index for index, value in enumerate(compartment_list)}
        initial_population = onp.zeros((len(admin_zones), len(compartment_list)))

        for i, zone in enumerate(admin_zones):
            population = zone["population"]
            seroprevalence = zone.get("seroprevalence", 0) or 0
            infected_population = zone.get("infected_population", 0) or 0

            # Split infected_population 50/50 between I1 and I2 (percentage → count)
            I1 = round(infected_population / 200 * population, 2)
            I2 = round(infected_population / 200 * population, 2)

            # Split seroprevalence 50/50 between S1 and S2 (percentage → count)
            S1 = round(seroprevalence / 200 * population, 2)
            S2 = round(seroprevalence / 200 * population, 2)

            # Remainder is fully susceptible
            S = population - I1 - I2 - S1 - S2

            initial_population[i, column_mapping["S"]] = S
            initial_population[i, column_mapping["I1"]] = I1
            initial_population[i, column_mapping["I2"]] = I2
            initial_population[i, column_mapping["S1"]] = S1
            initial_population[i, column_mapping["S2"]] = S2

        return initial_population

    def prepare_initial_state(self):
        # Base __init__ already produced population_matrix as (compartments, regions)
        # (it transposes config["initial_population"]).  No demographic
        # stratification is used, so the state is passed through unchanged.
        return self.population_matrix, self.compartment_list

    def derivative(self, y, t, p):
        y = np.clip(y, 0.0, 1e9)  # clip to avoid infs/negatives feeding back in

        states = {comp: y[i] for i, comp in enumerate(self.compartment_list)}
        S = states["S"]
        I1, I2 = states["I1"], states["I2"]
        R1, R2 = states["R1"], states["R2"]
        S1, S2 = states["S1"], states["S2"]
        I12, I21 = states["I12"], states["I21"]
        R = states["R"]

        beta_0, eta, epsilon = self.beta_0, self.eta, self.epsilon
        rho, gamma, mu, b, alpha = self.rho, self.gamma, self.mu, self.b, self.alpha
        omega, phi = self.omega, self.phi

        error_val = 1e-6
        N = error_val + (S + I1 + I2 + R1 + R2 + S1 + S2 + I12 + I21 + R)

        # Seasonal forcing of the transmission rate
        beta = beta_0 * (1 + eta * np.cos(omega * (t + phi)))

        d = {}
        # NOTE: the dS outflow term below is carried over verbatim from the
        # legacy model.  It is *not* exactly the sum of the inflows to I1 and I2
        # (it uses a single rho*N leak and an `eta*I12` term where the per-strain
        # inflows use `epsilon*I12`), so the system is not strictly
        # mass-conserving.  This is preserved to keep the migration
        # behavior-identical; see the migration notes if this should be fixed.
        d["S"] = -beta / N * S * (I1 + I2 + rho * N + epsilon * I21 + eta * I12) + b * N - mu * S

        d["I1"] = beta / N * S * (I1 + rho * N + epsilon * I21) - (gamma + mu) * I1
        d["I2"] = beta / N * S * (I2 + rho * N + epsilon * I12) - (gamma + mu) * I2

        d["R1"] = gamma * I1 - (alpha + mu) * R1
        d["R2"] = gamma * I2 - (alpha + mu) * R2

        d["S1"] = -beta / N * S1 * (I2 + rho * N + epsilon * I12) + alpha * R1 - mu * S1
        d["S2"] = -beta / N * S2 * (I1 + rho * N + epsilon * I21) + alpha * R2 - mu * S2

        d["I12"] = beta / N * S1 * (I2 + rho * N + epsilon * I12) - (gamma + mu) * I12
        d["I21"] = beta / N * S2 * (I1 + rho * N + epsilon * I21) - (gamma + mu) * I21

        d["R"] = gamma * (I12 + I21) - mu * R

        # Cumulative trackers (inflow only)
        d["I_total"] = beta / N * S * (I1 + I2 + 2 * (rho * N) + epsilon * (I21 + I12))
        d["R1_total"] = gamma * (I1 + I2)
        d["S2_total"] = (
            -beta / N * S1 * (I2 + rho * N + epsilon * I12) + alpha * R1
            + -beta / N * S2 * (I1 + rho * N + epsilon * I21) + alpha * R2
        )
        d["I2_total"] = beta / N * (
            S1 * (I2 + rho * N + epsilon * I12) + S2 * (I1 + rho * N + epsilon * I21)
        )
        d["R2_total"] = gamma * (I12 + I21)

        return np.stack([d[comp] for comp in self.compartment_list])
