import jax.numpy as jnp
import numpy as np
import logging
from compartment.model import Model, ValueType

logger = logging.getLogger(__name__)


class ExampleSpatialMobilityModel(Model):
    """SIR model with selectable inter-zone mobility AND age-structured mixing.

    This extends the spatial-mobility SIR so the population is **age-stratified**
    and age groups mix through a **contact matrix**, while transmission is still
    governed by a **single scalar ``beta``** (no per-age transmission rate).

    Two couplings act together on the force of infection:

    * **Space** — a selectable, row-stochastic *presence* travel matrix ``T``
      (``T[i,j]`` = fraction of zone *i* present in zone *j*; rows sum to 1),
      built by ``build_travel_matrix`` from each admin zone's population and
      coordinates (1 = gravity, 2 = exponential, 3 = radiation, 4 = uniform).
    * **Age** — a contact matrix ``M`` (``M[a,b]`` = relative contact intensity
      between age groups *a* and *b*), auto-loaded by the framework from each
      demographic group's ``age_range``. ``M`` is normalised by its spectral
      radius so it only *redistributes* risk across ages; the overall
      transmission scale stays set by the single ``beta`` (i.e. ``beta`` keeps
      its usual R0-equivalent meaning, as in the non-age model).

    Force of infection for susceptibles of age *a* in zone *r* (state axes are
    ``(age, zone)``; the zone axis is last)::

        N_present[b,j] = sum_i N[b,i] T[i,j]      (age-b people present in j)
        I_present[b,j] = sum_i I[b,i] T[i,j]      (age-b infectious present in j)
        phi[b,j]       = I_present[b,j] / N_present[b,j]   (age-b prevalence in j)
        exposure[a,j]  = sum_b M[a,b] phi[b,j]    (age mixing within each zone)
        lambda[a,r]    = beta * sum_j T[r,j] exposure[a,j]

    With ``M = I`` (identity) and one age group this reduces exactly to the
    original zone-only spatial SIR; with ``travel_sigma = 0`` (``T = I``) it
    reduces to independent, age-mixed SIRs per zone.
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
        """Declare compartments, transmission edges, mobility, and demographics."""
        schema.set_model_info(
            disease_type="example_spatial_mobility",
            label="Example Disease with Spatial Mobility",
            description="A spatial, age-structured SIR model with selectable inter-zone mobility, contact-matrix age mixing, and a single transmission rate",
        )

        # --- Compartments ---
        schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
        schema.add_compartment("I", "Infected", "Currently infectious population", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered and immune")

        # --- Transmission edges (single scalar beta for everyone) ---
        schema.add_transmission_parameter(
            source="susceptible",
            target="infected",
            variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (S->I)",
            description="Single transmission rate applied to all age groups; the contact matrix redistributes contact intensity across ages.",
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

        # --- Age-stratified demographics + contact matrix ---
        # Declaring age_range on each group lets the framework auto-load the
        # country's Prem 2021 synthetic contact matrix, aggregated to these bands.
        schema.add_demographic_group("age_0_17",    "Children",     default_weight=22.0, age_range=(0, 17))
        schema.add_demographic_group("age_18_49",   "Young adults", default_weight=42.0, age_range=(18, 49))
        schema.add_demographic_group("age_50_64",   "Older adults", default_weight=19.0, age_range=(50, 64))
        schema.add_demographic_group("age_65_plus", "Seniors",      default_weight=17.0, age_range=(65, 120))

    def __init__(self, config):
        """Initialize the model from a validated simulation config."""
        super().__init__(config)

    # ------------------------------------------------------------------
    # Mobility matrix — dispatch on the chosen mechanism number
    # ------------------------------------------------------------------

    def build_travel_matrix(self, admin_zones):
        """Return the (R, R) row-stochastic presence matrix for the chosen model."""
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
        """1 = GRAVITY. Pull of destination j on origin i is pop_j / d_ij^alpha."""
        with np.errstate(divide="ignore"):
            w = pops[None, :] / np.power(dist, alpha)
        w[~np.isfinite(w)] = 0.0
        np.fill_diagonal(w, 0.0)
        return w

    @staticmethod
    def _w_exp(pops, dist, scale_km=150.0):
        """2 = EXPONENTIAL. Pull is pop_j * exp(-d_ij / scale_km)."""
        w = pops[None, :] * np.exp(-dist / scale_km)
        np.fill_diagonal(w, 0.0)
        return w

    @staticmethod
    def _w_uniform(pops):
        """4 = UNIFORM. Distance-agnostic equal spread across all other zones."""
        n = len(pops)
        w = np.ones((n, n))
        np.fill_diagonal(w, 0.0)
        return w

    @staticmethod
    def _w_radiation(pops, dist):
        """3 = RADIATION (Simini et al. 2012), parameter-free."""
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

    # ------------------------------------------------------------------
    # Age-stratified state + normalised contact matrix
    # ------------------------------------------------------------------
    def prepare_initial_state(self):
        """Expand the state across age bands and precompute the age-mixing matrix.

        ``_prepare_demographic_state`` reshapes the compartments to
        ``(compartments, age groups, zones)`` using the declared demographic
        weights, so results are tracked per age group. The framework-loaded
        contact matrix is normalised by its spectral radius once here (it is
        constant over the run) so the single ``beta`` keeps its overall
        transmission scale while ``M`` only sets the *relative* age mixing.
        """
        self._prepare_demographic_state()

        # Precompute a spectral-radius-normalised contact matrix (constant),
        # so equation() stays cheap and jit-friendly (no eig inside the solver).
        M = getattr(self, "contact_matrix", None)
        if M is None:
            n_age = self.population_matrix.shape[1]
            self._contact_norm = jnp.eye(n_age)
        else:
            M = np.asarray(M, dtype=float)
            rho = float(np.max(np.abs(np.linalg.eigvals(M))))
            if not np.isfinite(rho) or rho <= 0:
                rho = 1.0
            self._contact_norm = jnp.asarray(M / rho)
        return self.population_matrix

    # ------------------------------------------------------------------
    # Dynamics — spatial presence FOI with contact-matrix age mixing
    # ------------------------------------------------------------------
    def equation(self, y, t, p):
        """SIR derivatives with spatial (travel) and age (contact) mixing.

        State rows are shaped ``(age groups, zones)``. Space is coupled by the
        presence travel matrix ``T`` and age by the normalised contact matrix
        ``M``; a single scalar ``beta`` sets the transmission scale.
        """
        C = self.COMPARTMENTS
        params = self._unpack_params(p)
        states = {c: y[i] for i, c in enumerate(self.compartment_list)}

        S = states[C.S]                                   # (A, R)
        I = states[C.I]                                   # (A, R)
        non_total = [c for c in self.compartment_list if not c.endswith("_total")]
        N = sum(states[c] for c in non_total)             # (A, R)

        prop_infective = I.sum() / (N.sum() + 1e-10)
        rates, travel_matrix = self._apply_interventions(
            t, {"beta": params["beta"]}, prop_infective
        )
        beta = rates["beta"]
        gamma = params["gamma"]
        T = jnp.asarray(travel_matrix)                    # (R, R)
        M = self._contact_norm                            # (A, A)

        # Spatial presence, per age band (zone = last axis): X_present = X @ T
        N_present = N @ T                                 # (A, R)
        I_present = I @ T                                 # (A, R)
        phi = I_present / (N_present + 1e-10)             # (A, R) age-b prevalence in zone j
        exposure = M @ phi                                # (A, R) age mixing within each zone
        force = beta * (exposure @ T.T)                   # (A, R) exposure of residents by origin zone
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
