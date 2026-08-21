"""Discrete-time stochastic Ebola virus disease model.

Port of ``model_ebola()`` from epiverse-trace/epidemics
(https://github.com/epiverse-trace/epidemics/blob/main/R/model_ebola.R),
based on the consensus compartment structure of Li et al. (2019) and the
Erlang passage-time formulation of Getz & Dougherty (2018).

The original large-boxcar Erlang implementation is replaced here by the
standard **linear chain trick**: each Erlang(k=2, λ) period is represented
as two sequential sub-compartments each draining at rate λ per step. This
produces the same Erlang(2) sojourn time distribution as the boxcar model
with a fraction of the state — O(k) = 2 compartments per stage instead of
O(k * mean_period) boxcars, reducing ~133 compartments to 14 and making
parameter uncertainty runs practical.

Compartments: S, E1/E2, I1/I2, H1/H2, Funeral, R.
Output is rolled up to the six public compartments via
``COMPARTMENT_DELTA_GROUPING``.

WARNING: Like the other models in this repo, this implementation is
intended for local experimentation; it is not yet supported by the
pandemic simulator app.
"""

import time as _time
import logging

import jax
import jax.numpy as jnp
import numpy as onp

from compartment.helpers import get_gravity_model_travel_matrix, setup_logging
from compartment.model import Model
from compartment.parameters import ValueType

setup_logging()
logger = logging.getLogger(__name__)


class EbolaJaxModel(Model):
    """Discrete-time stochastic Ebola model with Erlang(2) linear chain."""

    STOCHASTIC = True
    ERLANG_K = 2

    COMPARTMENT_DELTA_GROUPING = {
        "S": ["S"],
        "E": ["E1", "E2"],
        "I": ["I1", "I2"],
        "H": ["H1", "H2"],
        "Funeral": ["Funeral"],
        "R": ["R"],
    }

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @classmethod
    def _add_total_compartments(cls, schema):
        """Suppress the framework's automatic per-edge ``_total`` compartments.

        This model declares its own aggregate cumulative compartments
        (``E_total`` ... ``R_total``) in ``define_parameters`` instead.

        Args:
            schema (ModelParameterSchema): Schema left deliberately unchanged.
        """
        # Suppress framework auto-generation; we declare our own aggregate
        # cumulative compartments below in define_parameters().
        return

    @classmethod
    def define_parameters(cls, schema):
        """Declare compartments, transmission edges, parameters, and interventions.

        Builds the Erlang(2) linear-chain structure (S, E1/E2, I1/I2, H1/H2,
        Funeral, R) plus aggregate cumulative trackers, the ETU and funeral
        transmission-risk parameters, gravity-model mobility, and the stochastic
        run count (Li et al. 2019; Getz & Dougherty 2018).

        Args:
            schema (ParameterSchemaBuilder): Schema builder to populate.
        """
        schema.set_model_info(
            disease_type="EBOLA",
            label="Ebola Virus Disease",
            description=(
                "Discrete-time stochastic SEIR model with hospitalisation "
                "and funeral transmission, Erlang(2) passage times via "
                "linear chain (Li et al. 2019, Getz & Dougherty 2018)."
            ),
        )

        schema.add_compartment(
            "S", "Susceptible", "Population susceptible to Ebola infection",
        )
        schema.add_compartment(
            "E1", "Exposed stage 1", "First Erlang(2) exposed sub-stage",
        )
        schema.add_compartment(
            "E2", "Exposed stage 2", "Second Erlang(2) exposed sub-stage",
        )
        schema.add_compartment(
            "I1", "Infectious stage 1",
            "First Erlang(2) community-infectious sub-stage",
            infective=True,
        )
        schema.add_compartment(
            "I2", "Infectious stage 2",
            "Second Erlang(2) community-infectious sub-stage",
            infective=True,
        )
        schema.add_compartment(
            "H1", "Hospitalised stage 1",
            "First Erlang(2) hospitalised sub-stage",
            infective=True,
        )
        schema.add_compartment(
            "H2", "Hospitalised stage 2",
            "Second Erlang(2) hospitalised sub-stage",
            infective=True,
        )
        schema.add_compartment(
            "Funeral", "Funeral",
            "In funeral transmission stage (single timestep)",
            infective=True,
        )
        schema.add_compartment(
            "R", "Removed",
            "Removed from the dynamic system (recovered or safely buried)",
        )

        # Aggregate cumulative trackers — accumulate inflows only so they
        # record total throughput rather than occupancy.
        schema.add_compartment(
            "E_total", "Exposed Total",
            "Cumulative number of individuals who entered the exposed stage",
        )
        schema.add_compartment(
            "I_total", "Infectious Total",
            "Cumulative number of individuals who became infectious",
        )
        schema.add_compartment(
            "H_total", "Hospitalised Total",
            "Cumulative number of hospital (ETU) admissions",
        )
        schema.add_compartment(
            "Funeral_total", "Funeral Total",
            "Cumulative number of individuals who entered the funeral-transmission stage",
        )
        schema.add_compartment(
            "R_total", "Removed Total",
            "Cumulative number of individuals removed (recovered or safely buried)",
        )

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
            source="I1", target="Funeral",
            variable_name="gamma",
            label="Infectious / Hospitalised Period (I->Funeral, H->R)",
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

        schema.add_parameter(
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
        schema.add_parameter(
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
        schema.add_parameter(
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

        # ---- Mobility ----
        # NOTE: deliberately named travel_sigma, not sigma — sigma is already
        # this model's E1->I1 incubation edge (see add_transmission_edge above).
        schema.add_parameter(
            name="travel_sigma",
            label="Travel Rate (σ)",
            description=(
                "Percentage of each zone's population away from home on a given day. "
                "Trips are distributed across destinations by an inverse-square "
                "gravity model weighted by population and distance, driving the "
                "spatial FOI mixing. 0 disables inter-zone travel. Note that if an "
                "admin 2 zone is selected, no travel takes place regardless of this "
                "value."
            ),
            value_type=ValueType.PERCENTAGE,
            default=5.0,
            min_value=0.0,
            max_value=100.0,
            unit="%",
        )

        schema.add_parameter(
            name="num_runs",
            label="Number of Runs",
            description="Number of stochastic trajectories to simulate.",
            value_type=ValueType.INTEGER,
            default=10,
            min_value=5,
            max_value=30,
            enable_variance=False,
        )

    # ------------------------------------------------------------------
    # Initial population
    # ------------------------------------------------------------------

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """Distribute each zone's population across S and I1/I2 (50/50 split).

        The 50/50 I1/I2 split approximates steady-state Erlang(2) seeding; the
        chain relaxes to the correct distribution within a few simulated days
        regardless of the initial split.

        Args:
            admin_zones (list[dict]): Admin-zone dicts providing ``population``
                and the ``infected_population`` percentage.
            compartment_list (list[str]): Ordered compartment names, used for
                column indexing.
            **kwargs (Any): Additional keyword arguments (unused).

        Returns:
            np.ndarray: Initial populations of shape (n_zones, n_compartments).
        """
        col = {v: i for i, v in enumerate(compartment_list)}
        pop = onp.zeros((len(admin_zones), len(compartment_list)))

        for z, zone in enumerate(admin_zones):
            N = float(zone["population"])
            inf_pct = float(zone.get("infected_population", 0.0) or 0.0)
            inf_pct = max(inf_pct, 0.0)
            infected = N * inf_pct / 100.0
            pop[z, col["S"]] = max(N - infected, 0.0)
            pop[z, col["I1"]] = infected * 0.5
            pop[z, col["I2"]] = infected * 0.5

        return pop

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, config):
        """Initialise the model, derive Erlang(2) step probabilities, and seed the PRNG.

        Converts the mean exposed and infectious periods into per-step Erlang(k=2)
        transition probabilities (``p = 1 - exp(-k / mean_period)`` at ``dt = 1``
        day), reads the community/ETU/funeral proportions from the ``Disease``
        config block, and fixes the FOI denominator to the initial population sum.

        Args:
            config (dict): Validated simulation configuration. A ``seed`` entry
                gives reproducible trajectories; otherwise the PRNG is seeded from
                the clock.
        """
        super().__init__(config)

        if self.beta is None:
            self.beta = 0.125
        sigma_per_day = self.sigma if self.sigma is not None else 1.0 / 5.0
        gamma_per_day = self.gamma if self.gamma is not None else 1.0 / 12.0

        # Erlang(k=2) per-step transition probabilities.
        # λ = k * (1/mean_period); p = 1 - exp(-λ * dt) with dt = 1 day.
        self._p_E = float(1.0 - onp.exp(-self.ERLANG_K * sigma_per_day))
        self._p_I = float(1.0 - onp.exp(-self.ERLANG_K * gamma_per_day))

        disease_cfg = config.get("Disease", {}) or {}
        self.prop_community = float(disease_cfg.get("prop_community", 90.0)) / 100.0
        self.etu_risk = float(disease_cfg.get("etu_risk", 70.0)) / 100.0
        self.funeral_risk = float(disease_cfg.get("funeral_risk", 50.0)) / 100.0

        seed = config.get("seed")
        if seed is None:
            seed = int(_time.time() * 1000) % (2**31)
        self._key = jax.random.PRNGKey(int(seed))

        # Fixed-population denominator for FOI (R model uses initial sum).
        self._population_size = jnp.sum(self.population_matrix, axis=0) + 1e-10

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    def build_travel_matrix(self, admin_zones):
        """Build the inverse-square gravity mobility matrix.

        Driven by the ``travel_sigma`` custom field and used for spatial FOI
        mixing in ``equation``.

        Args:
            admin_zones (list[dict]): Admin-zone dicts with ``center_lat``,
                ``center_lon``, and ``population``.

        Returns:
            np.ndarray: Travel matrix of shape (n_zones, n_zones) whose entry
                ``[i, j]`` is the fraction of zone ``i``'s population present in
                zone ``j``.
        """
        sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
        return get_gravity_model_travel_matrix(admin_zones, sigma)

    def prepare_initial_state(self):
        """Return the initial compartment populations for the integrator.

        Returns:
            jnp.ndarray: Population matrix of shape (n_compartments, n_zones).
        """
        return self.population_matrix

    # ------------------------------------------------------------------
    # ODE / discrete-time equation
    # ------------------------------------------------------------------

    def equation(self, y, t, p):
        """Compute the per-step deltas for the discrete stochastic Erlang(2) chain.

        With ``STOCHASTIC = True`` the framework uses Euler integration:
        ``y_{t+1} = y_t + 1 * equation(y_t, t, p)``. Transitions are binomial draws
        on the per-step probabilities, and the force of infection combines
        community, ETU, and funeral infectiousness mixed across zones by the
        travel matrix.

        Args:
            y (jnp.ndarray): Current compartment values, ordered by
                ``compartment_list``.
            t (float): Current time in days since the simulation start date.
            p (tuple): Packed parameter tuple, unpacked via ``_unpack_params``.

        Returns:
            jnp.ndarray: Stacked per-compartment deltas for this step.
        """
        params = self._unpack_params(p)
        beta = params["beta"]

        # Unpack state; each is shape (R,) for R regions.
        # Order matches define_parameters() compartment declaration order.
        S       = y[0]
        E1      = y[1]
        E2      = y[2]
        I1      = y[3]
        I2      = y[4]
        H1      = y[5]
        H2      = y[6]
        Funeral = y[7]
        R       = y[8]  # noqa: F841
        # y[9..13] are _total trackers; read-only here, accumulated below.

        N = self._population_size

        # Schema-driven interventions on β.
        prop_inf_scalar = (I1 + I2 + H1 + H2 + Funeral).sum() / N.sum()
        rates = {"beta": beta}
        rates, _ = self._apply_interventions(t, rates, prop_inf_scalar)
        beta = rates["beta"]

        # R model FOI formula with gravity-model spatial mixing.
        # Compute effective infectious fraction per zone, then mix across zones
        # via the travel matrix so susceptibles in zone i are exposed to the
        # weighted-average infectious pressure from all zones they visit.
        I_eff = (I1 + I2) + self.etu_risk * (H1 + H2) + self.funeral_risk * Funeral
        I_frac = I_eff / N                         # shape (R,)
        mixed_frac = self.travel_matrix @ I_frac   # shape (R,)
        current_rate = beta * mixed_frac
        p_exposure = 1.0 - jnp.exp(-jnp.maximum(current_rate, 0.0))
        p_exposure = jnp.clip(p_exposure, 0.0, 1.0)

        p_E = jnp.float32(self._p_E)
        p_I = jnp.float32(self._p_I)
        p_comm = jnp.float32(self.prop_community)

        # One PRNG key per binomial draw.
        keys = jax.random.split(self._key, 9)
        self._key = keys[0]

        def _binom(key, n, p):
            """Draw binomial transition counts, clamped to the available population.

            Args:
                key (jnp.ndarray): PRNG key for this draw.
                n (jnp.ndarray): Per-zone population available to transition.
                p (jnp.ndarray): Per-step transition probability.

            Returns:
                jnp.ndarray: Event counts, never negative and never exceeding ``n``.
            """
            n = jnp.maximum(n, 0.0)
            draw = jax.random.binomial(key, n, p).astype(n.dtype)
            return jnp.minimum(draw, n)

        # S → E1
        new_exposed = _binom(keys[1], S, p_exposure)

        # E1 → E2
        E1_exit = _binom(keys[2], E1, p_E)

        # E2 → new infectious, routed to I1 (community) or H1 (hospital)
        E2_exit = _binom(keys[3], E2, p_E)
        E2_community = _binom(keys[4], E2_exit, p_comm)
        E2_hosp = E2_exit - E2_community

        # I1 → I2
        I1_exit = _binom(keys[5], I1, p_I)

        # I2 → Funeral (single-timestep stage)
        I2_exit = _binom(keys[6], I2, p_I)

        # H1 → H2  (same rate γ as I; Getz & Dougherty use one rate for both)
        H1_exit = _binom(keys[7], H1, p_I)

        # H2 → R
        H2_exit = _binom(keys[8], H2, p_I)

        delta_S       = -new_exposed
        delta_E1      = new_exposed - E1_exit
        delta_E2      = E1_exit - E2_exit
        delta_I1      = E2_community - I1_exit
        delta_I2      = I1_exit - I2_exit
        delta_H1      = E2_hosp - H1_exit
        delta_H2      = H1_exit - H2_exit
        # Funeral holds people exactly one step: receives I2_exit, clears to R.
        delta_Funeral = I2_exit - Funeral
        delta_R       = H2_exit + Funeral

        # Cumulative inflow trackers (never decremented).
        delta_E_total       = new_exposed       # S → E1
        delta_I_total       = E2_exit           # all new infectious (community + hospital)
        delta_H_total       = E2_hosp           # E2 → H1 admissions
        delta_Funeral_total = I2_exit           # I2 → Funeral inflow
        delta_R_total       = H2_exit + Funeral # H2 → R and Funeral → R

        return jnp.stack(
            [
                delta_S,
                delta_E1, delta_E2,
                delta_I1, delta_I2,
                delta_H1, delta_H2,
                delta_Funeral, delta_R,
                delta_E_total, delta_I_total, delta_H_total,
                delta_Funeral_total, delta_R_total,
            ],
            axis=0,
        )
