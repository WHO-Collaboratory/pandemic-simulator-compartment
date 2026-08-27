import jax.numpy as jnp
import numpy as np
import logging
from compartment.model import Model, ValueType

logger = logging.getLogger(__name__)


class ExampleSpatialMobilityModel(Model):
    """SIR model with a *selectable* mechanistic inter-zone mobility matrix.

    The user picks the mobility mechanism with the numeric ``travel_model``
    parameter — **1 = gravity, 2 = exponential, 3 = radiation, 4 = uniform** —
    and ``build_travel_matrix`` synthesises the matrix from each admin zone's
    population and coordinates (no movement data needed). Every mechanism returns
    a row-stochastic *presence* matrix (``T[i,j]`` = fraction of zone i present in
    zone j; rows sum to 1; diagonal = stay-home), and ``equation()`` applies it so
    zones are genuinely coupled.
    """

    # Numeric choice -> mechanism name. The config sets an integer 1-4.
    TRAVEL_MODEL_CHOICES = {
        1: "gravity",       # pop_j / d_ij^alpha           (needs travel_alpha)
        2: "exp",           # pop_j * exp(-d_ij/scale_km)   (needs travel_scale_km)
        3: "radiation",     # Simini et al. 2012, parameter-free
        4: "uniform",       # equal spread to every other zone
    }

    @classmethod
    def define_parameters(cls, schema):
        """Declare compartments, transmission edges, mobility, and parameters."""
        schema.set_model_info(
            disease_type="example_spatial_mobility",
            label="Example Disease with Spatial Mobility",
            description="A spatial SIR model for an example disease with selectable inter-zone mobility and parameter uncertainty",
        )

        # --- Compartments ---
        schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
        schema.add_compartment("I", "Infected", "Currently infectious population", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered and immune")

        # --- Transmission edges ---
        schema.add_transmission_parameter(
            source="susceptible",
            target="infected",
            variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (S->I)",
            description="Rate at which susceptibles become infected through contact",
            default=0.3,
            default_min=0.1,
            default_max=0.5,
            min_value=0.01,
            max_value=2.0,
            unit="per day",
        )
        schema.add_transmission_parameter(
            source="infected",
            target="recovered",
            variable_name="gamma",
            label="Recovery Period (I->R)",
            description="Average number of days to recover",
            default=10.0,
            default_min=5.0,
            default_max=20.0,
            min_value=1.0,
            max_value=100.0,
            value_type=ValueType.DAYS,
            unit="days",
        )

        # --- Spatial mobility: choose a mechanism by NUMBER (1-4) -----------
        # No movement data required — every mechanism is built from the admin
        # zones' population + coordinates (great-circle distances).
        #
        #   1 = gravity     flow i->j proportional to  pop_j / distance^alpha.
        #                   Classic "big, nearby places pull hardest"; the
        #                   distance exponent `travel_alpha` controls how fast
        #                   pull falls with distance (2.0 = inverse-square).
        #   2 = exponential flow i->j proportional to  pop_j * exp(-distance/L),
        #                   with L = `travel_scale_km`. Pull decays smoothly with
        #                   a characteristic range L (few trips beyond ~2-3 L).
        #   3 = radiation   Simini et al. 2012, PARAMETER-FREE. Flow depends on
        #                   populations and the "intervening opportunities" (the
        #                   population living closer than the destination). No
        #                   decay knob to tune.
        #   4 = uniform     each zone spreads its travel equally to every other
        #                   zone, ignoring distance entirely. Simplest baseline.
        schema.add_parameter(
            name="travel_model",
            label="Mobility Mechanism (1=gravity, 2=exponential, 3=radiation, 4=uniform)",
            description=(
                "Which mechanism builds the inter-zone mobility matrix, chosen by "
                "number: 1 = gravity (pop_j / distance^alpha), 2 = exponential "
                "(pop_j * exp(-distance/scale_km)), 3 = radiation (parameter-free, "
                "Simini 2012), 4 = uniform (equal spread, distance-agnostic)."
            ),
            value_type=ValueType.INTEGER,
            default=1,
            min_value=1,
            max_value=4,
            required=False,
            enable_variance=False,
        )
        schema.add_parameter(
            name="travel_sigma",
            label="Travel Rate (sigma)",
            description=(
                "Percentage of each zone's population away from home per day. Sets "
                "the off-diagonal mass (diagonal = 1 - sigma). Set 0 for no travel."
            ),
            value_type=ValueType.PERCENTAGE,
            default=20.0,
            min_value=0.0,
            max_value=100.0,
            unit="%",
            required=False,
            enable_variance=False,
        )
        schema.add_parameter(
            name="travel_scale_km",
            label="Decay Length (exponential model)",
            description="Characteristic decay distance in km for mechanism 2 (exponential).",
            value_type=ValueType.FLOAT,
            default=150.0,
            min_value=1.0,
            max_value=20000.0,
            unit="km",
            required=False,
            enable_variance=False,
        )
        schema.add_parameter(
            name="travel_alpha",
            label="Distance Exponent (gravity model)",
            description="Distance exponent for mechanism 1 (gravity); 2.0 = inverse-square.",
            value_type=ValueType.FLOAT,
            default=2.0,
            min_value=0.1,
            max_value=5.0,
            required=False,
            enable_variance=False,
        )

        # --- Optional: interventions ---
        schema.add_intervention(
            id="my_intervention",
            label="My Intervention",
            description="Reduces transmission while active",
            target_rates=["beta"],
            adherence=50.0,
            transmission_reduction=50.0,
        )

        # --- Optional: age-stratified demographics + contact matrix ---
        schema.add_demographic_group("age_0_17",    "children", default_weight=22.0,  age_range=(0, 17))
        schema.add_demographic_group("age_18_49",  "Young adults",   default_weight=42.0, age_range=(18, 49))
        schema.add_demographic_group("age_50_64",  "Older adults",   default_weight=19.0, age_range=(50, 64))
        schema.add_demographic_group("age_65_plus","Seniors",        default_weight=17.0, age_range=(65, 120))

    def __init__(self, config):
        """Initialize the model from a validated simulation config."""
        super().__init__(config)

    # ------------------------------------------------------------------
    # Mobility matrix — dispatch on the chosen mechanism number
    # ------------------------------------------------------------------

    def build_travel_matrix(self, admin_zones):
        """Return the (R, R) row-stochastic presence matrix for the chosen model.

        Reads the numeric ``travel_model`` (1-4) and the relevant knobs
        (``travel_sigma``, ``travel_scale_km``, ``travel_alpha``) and synthesises
        the matrix from the admin zones' populations and coordinates.
        """
        n = len(admin_zones)
        choice = int(getattr(self, "travel_model", 1) or 1)
        model = self.TRAVEL_MODEL_CHOICES.get(choice)
        if model is None:
            raise ValueError(
                f"travel_model must be one of {sorted(self.TRAVEL_MODEL_CHOICES)} "
                f"(1=gravity, 2=exponential, 3=radiation, 4=uniform); got {choice}."
            )

        sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)  # -> fraction 0-1

        if n == 1:
            return np.array([[1.0]])
        if not sigma:
            logger.info("travel_sigma=0 -> identity (no inter-zone travel).")
            return np.eye(n)

        pops = np.array([float(z["population"]) for z in admin_zones])
        lats = np.array([float(z["center_lat"]) for z in admin_zones])
        lons = np.array([float(z["center_lon"]) for z in admin_zones])
        dist = self._haversine_km(lats, lons)

        if model == "gravity":
            w = self._w_gravity(pops, dist, alpha=float(self.travel_alpha))
        elif model == "exp":
            w = self._w_exp(pops, dist, scale_km=float(self.travel_scale_km))
        elif model == "radiation":
            w = self._w_radiation(pops, dist)
        elif model == "uniform":
            w = self._w_uniform(pops)

        T = self._finalise(w, float(sigma))
        logger.info("Built '%s' (choice %d) mobility matrix for %d zones (sigma=%.3f).",
                    model, choice, n, sigma)
        return T

    # ---- distance ----
    @staticmethod
    def _haversine_km(lats, lons):
        R = 6371.0
        lat = np.radians(lats)
        lon = np.radians(lons)
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    # ---- kernels: off-diagonal attraction weights (diagonal zero) ----
    @staticmethod
    def _w_gravity(pops, dist, alpha=2.0):
        """1 = GRAVITY. Pull of destination j on origin i is pop_j / d_ij^alpha:
        larger and nearer zones attract more trips. ``alpha`` controls how
        sharply attraction falls with distance (2.0 = inverse-square)."""
        with np.errstate(divide="ignore"):
            w = pops[None, :] / np.power(dist, alpha)
        w[~np.isfinite(w)] = 0.0
        np.fill_diagonal(w, 0.0)
        return w

    @staticmethod
    def _w_exp(pops, dist, scale_km=150.0):
        """2 = EXPONENTIAL. Pull is pop_j * exp(-d_ij / scale_km): attraction
        decays smoothly over a characteristic range ``scale_km`` (very few trips
        beyond ~2-3 x scale_km). Lighter long-range tail than gravity."""
        w = pops[None, :] * np.exp(-dist / scale_km)
        np.fill_diagonal(w, 0.0)
        return w

    @staticmethod
    def _w_uniform(pops):
        """4 = UNIFORM. Distance-agnostic: each origin spreads its travel equally
        across all other zones. A simple, assumption-light baseline."""
        n = len(pops)
        w = np.ones((n, n))
        np.fill_diagonal(w, 0.0)
        return w

    @staticmethod
    def _w_radiation(pops, dist):
        """3 = RADIATION (Simini et al. 2012), parameter-free.

        flux_ij ~ m_i n_j / [(m_i + s_ij)(m_i + n_j + s_ij)], where s_ij is the
        population living closer to i than j is (the "intervening opportunities",
        excluding i and j). Flows emerge from the population landscape with no
        distance-decay knob to tune.
        """
        n = len(pops)
        m = pops
        w = np.zeros((n, n))
        for i in range(n):
            order = np.argsort(dist[i])
            s = np.zeros(n)
            cum = 0.0
            for k in order:
                if k == i:
                    continue
                s[k] = cum
                cum += m[k]
            for j in range(n):
                if j == i:
                    continue
                denom = (m[i] + s[j]) * (m[i] + m[j] + s[j])
                w[i, j] = (m[i] * m[j] / denom) if denom > 0 else 0.0
        return w

    # ---- normalise: off-diagonal -> sigma, diagonal -> stay-home ----
    @staticmethod
    def _finalise(w, sigma):
        row = w.sum(axis=1, keepdims=True)
        frac = np.divide(w, row, out=np.zeros_like(w), where=row > 0)  # off-diag sums to 1 (or 0)
        T = frac * sigma                                              # off-diag sums to sigma
        np.fill_diagonal(T, 1.0 - T.sum(axis=1))                     # rows sum to 1
        return T

    def prepare_initial_state(self):
        """Return the initial compartment populations for the solver."""
        return self.population_matrix

    # ------------------------------------------------------------------
    # Dynamics — metapopulation presence force of infection
    # ------------------------------------------------------------------

    def equation(self, y, t, p):
        """SIR derivatives with a mobility-coupled force of infection.

        The travel matrix ``T`` couples zones: residents of zone i are exposed
        to the infection prevalence of every zone j they spend time in.

            N_present_j = sum_i N_i T[i,j]           (people present in j)
            I_present_j = sum_i I_i T[i,j]           (infectious present in j)
            phi_j       = I_present_j / N_present_j  (prevalence experienced in j)
            new_inf_i   = S_i * beta * sum_j T[i,j] phi_j

        With ``T = I`` (travel_sigma = 0) this reduces to standard per-zone SIR.
        Zone is the last axis of the state arrays, so this works for plain (R,)
        and age-stratified (A, R) states alike.
        """
        C = self.COMPARTMENTS
        params = self._unpack_params(p)
        states = {c: y[i] for i, c in enumerate(self.compartment_list)}

        S = states[C.S]
        I = states[C.I]
        non_total = [c for c in C if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)

        prop_infective = I.sum() / (N_total.sum() + 1e-10)
        rates, travel_matrix = self._apply_interventions(
            t, {"beta": params["beta"]}, prop_infective
        )
        beta = rates["beta"]
        gamma = params["gamma"]
        T = jnp.asarray(travel_matrix)

        # spatial presence FOI (zone = last axis): X_present = X @ T
        N_present = N_total @ T
        I_present = I @ T
        phi = I_present / (N_present + 1e-10)     # prevalence in each destination zone
        force = beta * (phi @ T.T)                # exposure of each origin zone's residents
        new_inf = S * force
        new_rec = gamma * I

        derivs = {c: jnp.zeros_like(S) for c in self.compartment_list}
        derivs[C.S] = -new_inf
        derivs[C.I] = new_inf - new_rec
        derivs[C.R] = new_rec
        if "I_total" in derivs:
            derivs["I_total"] = new_inf
        if "R_total" in derivs:
            derivs["R_total"] = new_rec
        return jnp.stack([derivs[c] for c in self.compartment_list])
