import jax.numpy as np
import jax
import numpy as onp
import logging
import pandas as pd
from compartment.helpers import setup_logging
from compartment.model import Model
from compartment.parameters import ValueType

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

""" 
WARNING: This model is not currently supported in the pandemic simulator app, 
but is available for testing and experimentation in the codebase. 
"""

class MpoxJaxModel(Model):
    """A simple SIRS compartmental model for MPOX with spatial mobility"""

    # ------------------------------------------------------------------
    # Declarative parameter schema (single source of truth)
    #
    # Everything below — COMPARTMENT_LIST, disease_type, transmission
    # param attributes (self.beta, self.gamma, self.omega), and
    # get_params() — is derived automatically from these declarations
    # by the base class.
    # ------------------------------------------------------------------

    @classmethod
    def define_parameters(cls, schema):
        """Declare the SIRS compartments, edges, mobility fields, and intervention.

        Declares the S->I, I->R, and waning R->S edges, the cumulative ``I_total``
        tracker, the ring-vaccination intervention, and the exponential
        distance-decay mobility fields consumed by ``build_travel_matrix``.

        Args:
            schema (ParameterSchemaBuilder): Schema builder to populate.
        """
        schema.set_model_info(
            disease_type="MPOX",
            label="MPOX",
            description="A simple SIR compartmental model for MPOX",
        )

        schema.add_compartment(
            "S",
            "Susceptible",
            "Population susceptible to MPOX infection",
        )
        schema.add_compartment(
            "I",
            "Infected",
            "Currently infected population",
            infective=True,
        )
        schema.add_compartment(
            "R",
            "Recovered",
            "Recovered and immune population",
        )
        schema.add_compartment(
            "I_total",
            "Infected Total",
            "Cumulative infected population",
        )

        schema.add_transmission_edge(
            source="susceptible",
            target="infected",
            variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (S->I)",
            description="Rate at which susceptible individuals become infected through contact with infected individuals",
            default=0.3,
            default_min=0.1,
            default_max=0.5,
            min_value=0.01,
            max_value=2.0,
            unit="per day",
        )

        schema.add_transmission_edge(
            source="infected",
            target="recovered",
            variable_name="gamma",
            label="Recovery Period (I->R)",
            description="Average number of days for an infected individual to recover",
            default=10.0,
            default_min=5.0,
            default_max=20.0,
            min_value=1.0,
            max_value=100.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

        schema.add_transmission_edge(
            source="recovered",
            target="susceptible",
            variable_name="omega",
            label="Waning Immunity Period (R->S)",
            description="Average number of days before a recovered individual loses immunity",
            default=60.0,
            min_value=14.0,
            max_value=365.0,
            default_min=30.0,
            default_max=90.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

        schema.add_intervention(
            id="ring_vaccination",
            label="Ring Vaccination",
            description="Targeted vaccination of confirmed case contacts to contain spread",
            adherence=70.0,
            transmission_reduction=75.0,
        )

        # ---- Mobility ----
        # This model uses exponential distance decay rather than a gravity
        # power law, so it declares its own decay length alongside sigma.
        # Both are consumed by build_travel_matrix() -> mobility() below.
        schema.add_parameter(
            name="travel_sigma",
            label="Travel Rate (σ)",
            description=(
                "Percentage of each zone's population away from home on a given day. "
                "0 disables inter-zone travel."
            ),
            value_type=ValueType.PERCENTAGE,
            default=20.0,
            min_value=0.0,
            max_value=100.0,
            unit="%",
        )
        schema.add_parameter(
            name="travel_scale_km",
            label="Travel Decay Length",
            description=(
                "Characteristic distance of the exponential mobility decay: trips "
                "to a zone are weighted by exp(-distance / this value). Larger "
                "values spread travel further afield."
            ),
            value_type=ValueType.FLOAT,
            default=500.0,
            min_value=1.0,
            max_value=20000.0,
            unit="km",
        )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def __init__(self, input):
        """Initialise the MPOX SIRS model from a configuration dictionary.

        Sets up the population matrix, transmission parameters, simulation window,
        mobility fields, ring-vaccination state, and the modeler-supplied
        transition-multiplier schedule.

        Args:
            input (dict): Simulation configuration with ``initial_population``,
                ``transmission_dict``, ``start_date``, ``time_steps``,
                ``admin_units``, and optional ``Disease`` and
                ``intervention_dict`` blocks.
        """
        # Population data
        self.population_matrix = np.array(input["initial_population"]).T
        self.compartment_list = list(self.COMPARTMENTS)

        # Transmission params (self.beta, self.gamma, self.omega) are set
        # automatically from the schema edge variable_names.
        self._load_transmission_params(input.get("transmission_dict", {}))

        if self.omega is None:
            self.omega = 1 / 60

        # Simulation parameters
        self.start_date = input["start_date"]
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = input["time_steps"]
        self.transition_schedule = self._load_transition_schedule()

        # Administrative units
        self.admin_units = input["admin_units"]

        # Mobility parameters (declared as custom fields in define_parameters)
        disease_cfg = input.get("Disease", {}) or {}
        self.travel_sigma = disease_cfg.get("travel_sigma", 20.0) or 0.0
        self.travel_scale_km = disease_cfg.get("travel_scale_km", 500.0) or 500.0

        # Interventions
        self.intervention_dict = input.get("intervention_dict", {})
        self.intervention_statuses = {"ring_vaccination": False}

        self.payload = input

    def _load_transition_schedule(self):
        """Load a region-independent event-rate schedule from modeler data.

        Validates that the dataset has ``day``, ``infection``, and ``recovery``
        columns, starts at day 0, has unique increasing days, and no negative
        multipliers.

        Returns:
            dict: Maps each of ``day``, ``infection``, and ``recovery`` to its
                column as a ``jnp.ndarray``.
        """
        table = pd.read_csv(self.dataset("mpox-transition-multipliers"))
        required = {"day", "infection", "recovery"}
        missing = required - set(table.columns)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Dataset is missing column(s): {names}")
        if table.empty or table["day"].iloc[0] != 0:
            raise ValueError("Dataset schedule must start at day 0")
        if not table["day"].is_monotonic_increasing or table["day"].duplicated().any():
            raise ValueError("Dataset days must be unique and increasing")
        if (table[["infection", "recovery"]] < 0).any().any():
            raise ValueError("Dataset multipliers cannot be negative")
        return {
            column: np.asarray(table[column].to_numpy())
            for column in ("day", "infection", "recovery")
        }

    def _transition_multiplier(self, transition, t):
        """Interpolate a transition multiplier for elapsed simulation day ``t``.

        Args:
            transition (str): Schedule column to read, ``"infection"`` or
                ``"recovery"``.
            t (float): Elapsed simulation time in days.

        Returns:
            jnp.ndarray: Multiplier interpolated from the transition schedule.
        """
        return np.interp(
            t,
            self.transition_schedule["day"],
            self.transition_schedule[transition],
        )

    # ------------------------------------------------------------------
    # Mobility model (defined on the disease class, built from case file)
    # ------------------------------------------------------------------

    def build_travel_matrix(self, admin_zones):
        """Build the exponential distance-decay mobility matrix.

        Driven by the ``travel_sigma`` and ``travel_scale_km`` custom fields.

        Args:
            admin_zones (list[dict]): Admin-zone dicts with ``center_lat``,
                ``center_lon``, and ``population``.

        Returns:
            np.ndarray: Travel matrix of shape (n_zones, n_zones).
        """
        sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
        return self.mobility(admin_zones, sigma, scale_km=self.travel_scale_km)

    def mobility(self, admin_zones, sigma, scale_km=500.0):
        """Build an exponential distance-decay travel matrix from admin zones.

        Flow from zone ``i`` to zone ``j`` is proportional to
        ``population_j * exp(-distance_ij / scale_km)``, using great-circle
        distances. Returns the identity matrix for a single zone or zero sigma.

        Args:
            admin_zones (list[dict]): Admin-zone dicts with ``center_lat``,
                ``center_lon``, and ``population``.
            sigma (float): Fraction (0-1) of each zone's population travelling out
                per timestep.
            scale_km (float): Characteristic decay distance in km.

        Returns:
            np.ndarray: Travel matrix of shape (n_zones, n_zones) whose entry
                ``[i, j]`` is the fraction of zone ``i``'s population present in
                zone ``j``.
        """
        n = len(admin_zones)
        if n <= 1 or sigma == 0.0:
            return onp.eye(n)

        lats = onp.array([z["center_lat"] for z in admin_zones])
        lons = onp.array([z["center_lon"] for z in admin_zones])
        pops = onp.array([z["population"] for z in admin_zones], dtype=float)

        # Pairwise great-circle distance in km (vectorised Haversine)
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

        # Attraction: destination population weighted by distance decay
        attraction = pops[None, :] * onp.exp(-dist_km / scale_km)
        onp.fill_diagonal(attraction, 0.0)  # exclude self-flow

        # Normalise rows so off-diagonal sums to 1
        row_sums = attraction.sum(axis=1, keepdims=True)
        row_sums = onp.where(row_sums == 0.0, 1.0, row_sums)
        T = attraction / row_sums

        # Apply sigma: fraction sigma leaves, fraction (1 - sigma) stays
        travel_matrix = sigma * T
        onp.fill_diagonal(travel_matrix, 1.0 - sigma)

        return travel_matrix

    # ------------------------------------------------------------------
    # Mpox-specific intervention (ring vaccination)
    # ------------------------------------------------------------------

    def ring_vaccination_intervention(self, beta, t, prop_infective):
        """Apply the mpox ring-vaccination reduction to the transmission rate.

        Activates either when the simulation date falls within the configured
        window (date-based) or when the proportion of infectives crosses the start
        threshold (threshold-based), and reduces beta by
        ``adherence * transmission_reduction`` while active. Returns beta unchanged
        when the intervention is not configured.

        Args:
            beta (jnp.ndarray): Baseline transmission rate before the reduction.
            t (float): Elapsed simulation time in days since ``start_date``.
            prop_infective (jnp.ndarray): Current proportion of the population
                that is infectious, used for threshold activation.

        Returns:
            tuple: The possibly reduced beta and the updated intervention-status
                dict.
        """
        cfg = self.intervention_dict.get("ring_vaccination")
        if cfg is None:
            return beta, self.intervention_statuses

        adh = cfg["adherence_min"]
        reduc = cfg["transmission_percentage"]

        status = self.intervention_statuses["ring_vaccination"]
        current_ordinal = self.start_date_ordinal + t

        # --- Date-based window ---
        start_ord = cfg.get("start_date_ordinal")
        end_ord = cfg.get("end_date_ordinal")
        if start_ord is not None:
            if end_ord is not None:
                in_date_window = np.logical_and(
                    current_ordinal >= start_ord, current_ordinal <= end_ord
                )
            else:
                in_date_window = current_ordinal >= start_ord
        else:
            in_date_window = np.bool_(False)

        # --- Threshold-based activation ---
        start_th = cfg.get("start_threshold")
        end_th = cfg.get("end_threshold")
        if start_th is not None:
            turn_on_thresh = np.logical_and(
                prop_infective >= start_th, np.logical_not(status)
            )
        else:
            turn_on_thresh = np.bool_(False)

        if end_th is not None:
            turn_off_thresh = np.logical_and(prop_infective <= end_th, status)
        else:
            turn_off_thresh = np.bool_(False)

        # Combine: active if date window OR threshold turn-on; deactivate on threshold turn-off
        new_status = np.where(
            np.logical_or(in_date_window, turn_on_thresh),
            True,
            np.where(turn_off_thresh, False, status),
        )

        reduced_beta = beta * (1.0 - adh * reduc)
        new_beta = np.where(new_status, reduced_beta, beta)

        new_statuses = {**self.intervention_statuses, "ring_vaccination": new_status}
        return new_beta, new_statuses

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    def prepare_initial_state(self):
        """Return the initial compartment populations for the solver.

        Returns:
            jnp.ndarray: Population matrix of shape (n_compartments, n_zones).
        """
        # The travel matrix is built by the framework via build_travel_matrix().
        return self.population_matrix

    def equation(self, y, t, p):
        """Compute the SIRS compartment derivatives for one integration step.

        The I->R and R->S edges are handled by the base class, while the S->I flow
        is applied manually so the force of infection can be coupled across zones
        through the travel matrix. Ring vaccination and the scheduled transition
        multipliers scale the rates before the flows are computed.

        Args:
            y (jnp.ndarray): Current compartment values, ordered by
                ``compartment_list``.
            t (float): Current time in days since the simulation start date.
            p (tuple): Packed parameter tuple, unpacked via ``_unpack_params``.

        Returns:
            jnp.ndarray: Stacked per-compartment derivatives (dy/dt).
        """
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        # Extract compartments by name from state vector
        states = {c: y[i] for i, c in enumerate(C)}
        S = states[C.S]
        I = states[C.I]  # noqa: E741

        non_total = [c for c in C if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)
        prop_infective = I.sum() / (N_total.sum() + 1e-10)

        # --- Mpox ring vaccination intervention (modifies beta) ---
        beta, self.intervention_statuses = self.ring_vaccination_intervention(
            params["beta"], t, prop_infective
        )
        beta = beta * self._transition_multiplier("infection", t)

        # Force of infection with spatial coupling via travel matrix.
        # T[i, j] = fraction of zone i's population present in zone j.
        # Susceptibles from zone i are exposed to infectious pressure across
        # all zones they visit, weighted by how much time they spend there.
        I_over_N = I / (N_total + 1e-10)
        lambda_force = beta * np.einsum("ij,j->i", self.travel_matrix, I_over_N)

        # Base class auto-handles gamma (I->R) and omega (R->S).
        # beta is skipped here — spatially-coupled S->I flow applied manually below.
        rates = {
            "gamma": params["gamma"] * self._transition_multiplier("recovery", t),
            "omega": params["omega"],
        }
        derivs = self._compute_equations(states, rates, skip_edges={"beta"})

        # Manually apply spatially-coupled S->I flow and accumulate into I_total
        foi_flow = S * lambda_force
        derivs[C.S] = derivs[C.S] - foi_flow
        derivs[C.I] = derivs[C.I] + foi_flow
        if C.I_total in derivs:
            derivs[C.I_total] = derivs[C.I_total] + foi_flow

        return np.stack([derivs[c] for c in C])
