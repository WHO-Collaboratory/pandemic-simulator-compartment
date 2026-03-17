import jax.numpy as np
import jax
import numpy as onp
import logging
from compartment.helpers import setup_logging
from compartment.model import Model

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


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
        schema.set_model_info(
            disease_type="MONKEYPOX",
            label="Monkeypox",
            description="A simple SIR compartmental model for Monkeypox",
        )

        schema.add_compartment(
            "S",
            "Susceptible",
            "Population susceptible to Monkeypox infection",
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
            label="Recovery Rate (I->R)",
            description="Rate at which infected individuals recover and gain immunity",
            default=0.1,
            default_min=0.05,
            default_max=0.2,
            min_value=0.01,
            max_value=1.0,
            unit="per day",
        )

        schema.add_transmission_edge(
            source="recovered",
            target="susceptible",
            variable_name="omega",
            label="Waning Immunity Rate (R->S)",
            description="Rate at which recovered individuals lose immunity and return to susceptible (1/immunity_duration)",
            default=1 / 60,
            min_value=1 / 365,
            max_value=1 / 14,
            default_min=1 / 90,
            default_max=1 / 30,
            unit="per day",
        )

        schema.add_intervention(
            id="ring_vaccination",
            label="Ring Vaccination",
            description="Targeted vaccination of confirmed case contacts to contain spread",
            adherence=70.0,
            transmission_reduction=75.0,
        )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def __init__(self, input):
        """Initialize the MPOX SIRS model with a configuration dictionary"""
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

        # Administrative units
        self.admin_units = input["admin_units"]

        # Admin zone data for mobility model (lat, lon, population per zone)
        case_file_dict = input.get("case_file") or {}
        self._admin_zones = case_file_dict.get("admin_zones", [])

        # Sigma: fraction of population that leaves each zone per timestep
        travel_volume = input.get("travel_volume") or {}
        self._sigma = travel_volume.get("leaving", 0.0) if isinstance(travel_volume, dict) else 0.0

        # Interventions
        self.intervention_dict = input.get("intervention_dict", {})
        self.intervention_statuses = {"ring_vaccination": False}

        self.payload = input

    # ------------------------------------------------------------------
    # Mobility model (defined on the disease class, built from case file)
    # ------------------------------------------------------------------

    def mobility(self, admin_zones, sigma, scale_km=500.0):
        """
        Simple exponential distance-decay mobility model built from case file admin zones.

        Flow from zone i to zone j is proportional to:
            population_j * exp(-distance_ij / scale_km)

        sigma controls the overall leaving rate (fraction of population that
        travels out of each zone per timestep). Returns an (n x n) travel
        matrix where entry [i, j] is the fraction of zone i's population
        present in zone j.
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
        """
        Mpox ring vaccination intervention defined on the disease class.

        Activates either when the simulation date falls within the configured
        window (date-based) or when the proportion of infectives crosses the
        start threshold (threshold-based). Reduces beta by
        adherence * transmission_reduction while active.
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
        # Build travel matrix from case file using the disease-level mobility model
        self.travel_matrix = np.array(
            self.mobility(self._admin_zones, self._sigma)
        )

        return (
            self.population_matrix,
            list(self.compartment_list),
        )

    def derivative(self, y, t, p):
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

        # Force of infection with spatial coupling via travel matrix.
        # T[i, j] = fraction of zone i's population present in zone j.
        # Susceptibles from zone i are exposed to infectious pressure across
        # all zones they visit, weighted by how much time they spend there.
        I_over_N = I / (N_total + 1e-10)
        lambda_force = beta * np.einsum("ij,j->i", self.travel_matrix, I_over_N)

        # Base class auto-handles gamma (I->R) and omega (R->S).
        # beta is skipped here — spatially-coupled S->I flow applied manually below.
        rates = {"gamma": params["gamma"], "omega": params["omega"]}
        derivs = self._compute_derivatives(states, rates, skip_edges={"beta"})

        # Manually apply spatially-coupled S->I flow and accumulate into I_total
        foi_flow = S * lambda_force
        derivs[C.S] = derivs[C.S] - foi_flow
        derivs[C.I] = derivs[C.I] + foi_flow
        if C.I_total in derivs:
            derivs[C.I_total] = derivs[C.I_total] + foi_flow

        return np.stack([derivs[c] for c in C])
