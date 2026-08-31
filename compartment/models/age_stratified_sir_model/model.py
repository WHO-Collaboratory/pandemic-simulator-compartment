"""Age-stratified SIR model with a per-age-group transmission rate.

This is a schema-based (declarative) model for the WHO Pandemic Simulator
compartment framework.  It extends the framework's SIR example so that **each
age group has its own transmission rate**.  Age groups are coupled through a
country contact matrix (Prem 2021, auto-loaded from each group's ``age_range``)
and regions are coupled through a gravity travel matrix.

There is no single "average" transmission rate: the four ``beta_age_*``
parameters are the only transmission rates, and interventions act directly on
each of them.

Force of infection (frequency-dependent), for susceptibles in age group *a*
and region *r*::

    lambda[a, r] = beta[a] * sum_b  M[a, b] * ( T @ (I_b / N_b) )[r]

where

    beta[a]   per-age-group transmission rate (the four ``beta_age_*`` params)
    M[a, b]   contact matrix — mean daily contacts a group-*a* person has
              with group-*b* people (age mixing)
    T         (R x R) travel matrix — spatial mixing of the infectious pool
    I_b / N_b proportion infectious in group *b*, per region

Because ``beta`` is indexed by the susceptible age group, changing one
``beta_age_*`` value changes the risk that that band acquires infection while
leaving the others untouched.
"""

import jax.numpy as jnp
import numpy as np
import logging

from compartment.model import Model, ValueType
from compartment.helpers import get_gravity_model_travel_matrix

logger = logging.getLogger(__name__)


# (group id, label, population weight %, inclusive age range, default beta/day)
# The age betas taper with age: younger bands mix more and transmit faster.
# The population weights and age bands mirror the framework's SIR example.
AGE_GROUPS = [
    ("age_0_17",    "Children (0-17)",      22.0, (0, 17),   0.45),
    ("age_18_49",   "Young adults (18-49)", 42.0, (18, 49),  0.35),
    ("age_50_64",   "Older adults (50-64)", 19.0, (50, 64),  0.30),
    ("age_65_plus", "Seniors (65+)",        17.0, (65, 120), 0.25),
]


class AgeStratifiedSIRModel(Model):
    """A schema-based SIR model whose transmission rate varies by age group."""

    # Class-level identifiers so the framework registry auto-discovers the model.
    DISEASE_TYPE = "AGE_STRATIFIED_SIR"
    DISEASE_LABEL = "Age-Stratified SIR (per-age transmission)"
    DISEASE_DESCRIPTION = (
        "An SIR model in which each age group has its own transmission rate, "
        "with age mixing through a contact matrix and spatial mixing through a "
        "travel matrix."
    )

    # ------------------------------------------------------------------
    # Schema declaration
    # ------------------------------------------------------------------
    @classmethod
    def define_parameters(cls, schema):
        """Declare compartments, transmission edges, per-age betas and demographics.

        Called once by the framework to build the model schema, from which the
        config validator and parameter set are generated.

        Args:
            schema (ParameterSchemaBuilder): Schema builder to populate.
        """
        schema.set_model_info(
            disease_type=cls.DISEASE_TYPE,
            label=cls.DISEASE_LABEL,
            description=cls.DISEASE_DESCRIPTION,
        )
        schema.set_model_metadata(
            model_type="Compartmental",
            diseases=["Generic directly-transmitted infection"],
            transmission_routes=["Direct contact / airborne"],
            questions_answered=[
                "How does a per-age-group transmission rate shape the outbreak "
                "in each age band?",
                "How does age mixing (contact matrix) spread infection from "
                "high-transmission bands to the rest of the population?",
                "How do transmission-reducing interventions change the "
                "age-specific attack rate?",
            ],
            key_assumptions=[
                "Closed population — no births or deaths.",
                "Four age bands (0-17, 18-49, 50-64, 65+), each with its own "
                "transmission rate.",
                "Age mixing follows the country's Prem 2021 synthetic contact "
                "matrix, aggregated to the four bands via each group's age_range.",
                "Homogeneous mixing within an (age group, region) cell; regions "
                "are coupled by a gravity travel matrix.",
                "Transmission is frequency-dependent (force of infection scales "
                "with the proportion infectious, not the raw count).",
                "Recovered individuals are fully immune (no R->S waning).",
            ],
        )

        # ---- Compartments (S-I-R) ----
        # Mark infective=True on compartments that drive the force of infection.
        schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
        schema.add_compartment("I", "Infected", "Currently infectious population", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered and immune")
        # Cumulative infections tracker. Normally the framework auto-generates a
        # ``<target>_total`` compartment for each transmission edge's target;
        # since the S->I force of infection is applied manually (no scalar beta
        # edge to declare), we declare I_total explicitly so cumulative incidence
        # is still tracked. ``_apply_flow`` accumulates into it automatically.
        schema.add_compartment("I_total", "Infected Total", "Cumulative infections")

        # ---- Transmission edges ----
        # Only the I->R recovery edge is declared. Its target (R) auto-generates
        # R_total. The S->I transmission is age-specific and handled manually in
        # equation() from the per-age beta parameters below, so there is no
        # single scalar/average beta edge.
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

        # ---- Per-age-group transmission rates (first-class parameters) ----
        # One editable transmission rate per age band -- the only transmission
        # rates in the model. Each is an absolute per-day rate with its own
        # default / bounds / uncertainty range and is surfaced as a custom field
        # in the platform UI.
        for group_id, label, _weight, _age_range, default_beta in AGE_GROUPS:
            schema.add_parameter(
                name=f"beta_{group_id}",
                label=f"Transmission Rate — {label}",
                description=(
                    f"Per-day transmission rate applied to susceptibles in the "
                    f"{label} band. This is the rate at which a susceptible in "
                    f"this age group acquires infection per unit of contact-"
                    f"weighted infectious pressure."
                ),
                value_type=ValueType.RATE,
                default=default_beta,
                min_value=0.01,
                max_value=0.9,
                default_min=0,
                default_max=1,
                unit="per day",
            )

        # ---- Spatial travel support ----
        schema.add_parameter(
            name="travel_sigma",
            label="Travel Rate (σ)",
            description="Percentage of each region's population away from home per day.",
            value_type=ValueType.PERCENTAGE,
            default=20.0,
            min_value=0.0,
            max_value=100.0,
            unit="%",
        )

        # ---- Intervention ----
        # A single transmission-reducing intervention that targets every per-age
        # transmission rate directly, so it scales each age group's beta while it
        # is active (see equation()).
        age_beta_rates = [f"beta_{g[0]}" for g in AGE_GROUPS]
        schema.add_intervention(
            id="social_isolation",
            label="Social Isolation",
            description="Reduces each age group's transmission rate while active.",
            target_rates=age_beta_rates,
            adherence=50.0,
            transmission_reduction=50.0,
        )

        # ---- Demographics + contact matrix ----
        # Declaring age_range on every group lets the framework auto-load the
        # country's Prem 2021 contact matrix, aggregated to these four bands.
        for group_id, label, weight, age_range, _default_beta in AGE_GROUPS:
            schema.add_demographic_group(
                group_id, label, default_weight=weight, age_range=age_range
            )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------
    def __init__(self, config):
        """Initialize the model from a validated simulation config.

        Args:
            config (dict): Validated simulation configuration produced by the
                framework's config loader.
        """
        super().__init__(config)

        # Effective age-group order for this run: config demographics if the run
        # supplied them, otherwise the schema's declared groups. This order
        # defines axis 1 of the state tensor and the rows/cols of the contact
        # matrix, so the per-age beta vector must follow it exactly.
        if getattr(self, "demographics", None):
            self._age_group_ids = list(self.demographics.keys())
        else:
            schema = type(self)._get_cached_schema()
            self._age_group_ids = [g.id for g in schema.demographic_groups]

        # Rate-dict keys the interventions target, one per age band, in state order.
        self._beta_names = [f"beta_{gid}" for gid in self._age_group_ids]

    def build_travel_matrix(self, admin_zones):
        """Build an inverse-square gravity travel matrix from ``travel_sigma``.

        Args:
            admin_zones (list[dict]): Admin-zone dicts with ``center_lat``,
                ``center_lon`` and ``population``.

        Returns:
            np.ndarray: The ``(n_zones, n_zones)`` travel matrix.
        """
        # PERCENTAGE params arrive as 20.0, not 0.2 — convert first.
        sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
        try:
            return get_gravity_model_travel_matrix(admin_zones, sigma)
        except ValueError:
            # On some pandas/numpy versions the framework helper pivots through
            # pandas and calls np.fill_diagonal on the result; copy-on-write can
            # make DataFrame.to_numpy() return a read-only array, raising
            # "underlying array is read-only". Fall back to an equivalent
            # construction built from writable numpy arrays.
            logger.warning(
                "Gravity travel helper returned a read-only array; using the "
                "writable fallback (results are identical)."
            )
            return self._writable_gravity_travel_matrix(admin_zones, sigma)

    @staticmethod
    def _writable_gravity_travel_matrix(admin_zones, sigma):
        """Writable equivalent of ``get_gravity_model_travel_matrix``.

        ``T[i, j]`` (i != j) is ``sigma`` shared across destinations by
        inverse-square gravity weight ``pop_i * pop_j / dist_km**2``; the
        diagonal holds the stay-home remainder ``1 - sum_off_diagonal`` (which
        is ``1 - sigma`` for any zone with a reachable destination, and ``1``
        for an isolated zone). Matches the framework helper's semantics exactly
        but never touches a read-only array.
        """
        from geopy.distance import geodesic

        n = len(admin_zones)
        if n == 1:
            return np.array([[1.0]])
        if not sigma:
            return np.eye(n)

        coords = [(float(z["center_lat"]), float(z["center_lon"])) for z in admin_zones]
        pops = np.array([float(z["population"]) for z in admin_zones], dtype=float)

        gravity = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist_km = geodesic(coords[i], coords[j]).km
                if dist_km > 0:
                    gravity[i, j] = pops[i] * pops[j] / dist_km ** 2

        off_totals = gravity.sum(axis=1, keepdims=True)
        travel = np.divide(
            gravity * sigma,
            off_totals,
            out=np.zeros_like(gravity),
            where=off_totals > 0,
        )
        np.fill_diagonal(travel, 1.0 - travel.sum(axis=1))
        return travel

    def prepare_initial_state(self):
        """Return the age-stratified initial populations for the solver.

        Returns:
            jnp.ndarray: Populations shaped (compartments, age groups, zones),
                including the appended ``_total`` rows.
        """
        # Expand (K, R) -> (K, A, R) using the demographic weights and append the
        # _total rows for active compartments.
        self._prepare_demographic_state()
        return self.population_matrix

    def _beta_by_age(self):
        """Assemble the per-age transmission-rate vector, aligned to the state.

        Reads each ``beta_<group_id>`` attribute in effective age-group order.
        Reading via ``getattr`` here — rather than caching in ``__init__`` —
        keeps the vector correct when parameter-uncertainty sampling overrides a
        ``beta_age_*`` value on the model instance. Any group without a
        dedicated parameter falls back to the mean of the declared per-age rates.

        Returns:
            jnp.ndarray: Shape ``(A,)`` per-age transmission rates (per day).
        """
        raw = [getattr(self, name, None) for name in self._beta_names]
        present = [v for v in raw if v is not None]
        fallback = sum(present) / len(present) if present else 0.0
        values = [
            self._to_rate(v if v is not None else fallback, ValueType.RATE)
            for v in raw
        ]
        return jnp.asarray(values)

    def equation(self, y, t, p):
        """Compute the compartment derivatives with an age-stratified force of infection.

        The standard I->R edge is handled by the framework; the S->I force of
        infection is built manually so it can mix age groups through the contact
        matrix and regions through the travel matrix, using the per-age beta
        vector, and is applied via :meth:`_apply_flow` (which also accumulates
        I_total). Interventions scale each per-age beta directly.

        Args:
            y (jnp.ndarray): Current compartment values, ordered by
                ``compartment_list``; each row is shaped (age groups, zones).
            t (float): Current time in days since the simulation start date.
            p (tuple): Packed parameter tuple, unpacked via ``_unpack_params``.

        Returns:
            jnp.ndarray: Stacked per-compartment derivatives (dy/dt).
        """
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(self.compartment_list)}
        S = states[C.S]
        I = states[C.I]  # noqa: E741

        # Per-region total population (exclude cumulative _total rows).
        non_total = [c for c in self.compartment_list if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total).sum(axis=0)  # (R,)
        I_frac = I / (N_total[None, :] + 1e-10)                   # (A, R)

        # Per-age transmission rates as a rate dict, so interventions can scale
        # each age band's beta by name (target_rates = the beta_age_* names).
        beta_vec = self._beta_by_age()                           # (A,)
        rates = {name: beta_vec[i] for i, name in enumerate(self._beta_names)}
        prop_infective_scalar = I.sum() / (N_total.sum() + 1e-10)
        rates, travel_matrix = self._apply_interventions(
            t, rates, prop_infective_scalar
        )
        # Reassemble the (post-intervention) per-age beta vector in state order.
        beta_vec = jnp.asarray([rates[name] for name in self._beta_names])  # (A,)
        rates["gamma"] = params["gamma"]

        # Standard edges via the framework (only the I->R gamma edge exists; the
        # extra beta_age_* keys in `rates` are ignored — they aren't edges).
        derivs = self._compute_equations(states, rates)

        # Manual force of infection: spatial mixing then age mixing.
        contact_matrix = (
            self.contact_matrix
            if self.contact_matrix is not None
            else jnp.eye(S.shape[0])
        )
        infectious_pressure = (travel_matrix @ I_frac.T).T        # (A, R) spatial mix
        mixed = contact_matrix @ infectious_pressure              # (A, R) age mix
        flow_foi = S * beta_vec[:, None] * mixed                  # (A, R)

        self._apply_flow(derivs, C.S, C.I, flow_foi)

        return jnp.stack([derivs[c] for c in self.compartment_list])
