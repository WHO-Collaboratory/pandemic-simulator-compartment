import time
import jax
import jax.numpy as jnp
import numpy as np
import logging
from compartment.model import Model, ValueType

logger = logging.getLogger(__name__)


class ExampleStochasticModel(Model):
    """A stochastic SIR compartmental model for Example Disease.

    Setting ``STOCHASTIC = True`` tells the SimulationManager to integrate with
    fixed-step Euler instead of the adaptive ODE solver, and to run several
    trajectories and report a median with an interval band.

    ``equation()`` is written by hand using **tau-leaping**: the number of
    infection and recovery events each step is drawn from a Poisson distribution
    whose mean equals the deterministic rate. This adds demographic randomness
    while keeping the same expected trajectory as the ODE version.
    """

    # Fixed-step Euler integration + multi-run median/interval output.
    STOCHASTIC = True

    # Collapse the two infectious compartments into one "Infected" series for
    # graphing, so the output shows S / I / R with I = asymptomatic +
    # symptomatic. (The cumulative combined tracker is I_total, added in
    # _add_total_compartments.)
    COMPARTMENT_DELTA_GROUPING = {
        "S": ["S"],
        "I": ["A", "Sym"],
        "R": ["R"],
    }

    @classmethod
    def define_parameters(cls, schema):
        """Declare the model's compartments, transmission edges, and parameters.

        Called once by the framework to build the model schema, from which the
        config validator and parameter set are generated. Also declares the
        aggregate ``I_total`` / ``R_total`` cumulative compartments and the
        stochastic run count.

        Args:
            schema: The schema builder to populate with model info, metadata,
                compartments, transmission edges, disease parameters, and the
                intervention.
        """
        schema.set_model_info(
            disease_type="example_stochastic",
            label="Example Disease with Stochasticity",
            description="A stochastic SIR model for an example disease",
        )
        schema.set_model_metadata(
            authors=[
                {
                    "name": "Jenny Blase",
                    "email": "jblase@ruvos.com",
                    "affiliation": "Ruvos",
                }
            ],
            license="MIT",
            model_type="Compartmental",
            diseases=["Example disease"],
            transmission_routes=["Airborne"],
            questions_answered=[
                "How much does demographic stochasticity change outbreak size and timing?",
                "How do asymptomatic and symptomatic infections jointly drive transmission?",
                "How does a transmission-reducing intervention change the trajectory?",
            ],
            key_assumptions=[
                "Closed population — no births or deaths.",
                "Tau-leaping: infection and recovery events are Poisson draws around the deterministic rates.",
                "Two infectious compartments (asymptomatic and symptomatic), both equally infectious.",
                "A fixed fraction of new infections are asymptomatic.",
                "Frequency-dependent transmission (force of infection scales with the proportion infectious).",
                "Recovered individuals are fully immune with no waning (no R→S transition).",
            ],
        )

        # --- Compartments ---
        # Two infectious compartments — asymptomatic and symptomatic — both
        # contribute to the force of infection, so both are marked infective.
        # They are combined into a single "Infected" curve for graphing via
        # COMPARTMENT_DELTA_GROUPING above.
        schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
        schema.add_compartment("A", "Asymptomatic", "Infectious but showing no symptoms", infective=True)
        schema.add_compartment("Sym", "Symptomatic", "Infectious and showing symptoms", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered and immune")

        # Cumulative tracker for the combined "Infected" group. Like DengueJaxModel,
        # we suppress the framework's per-edge _total compartments (see
        # _add_total_compartments) and declare our own aggregate total here. The
        # delta/"total infected" chart reads the I_total column (group name "I" ->
        # "I_total"); without it the total would collapse to the final-day
        # occupancy of A + Sym (~0 once the outbreak ends).
        schema.add_compartment(
            "I_total",
            "Infected Total",
            "Cumulative infections (asymptomatic + symptomatic combined)",
        )

        schema.add_compartment("R_total", "Recovered Total", "Cumulative recoveries")

        # --- Transmission edges ---
        # We suppress the framework's per-target _total compartments (see
        # _add_total_compartments) and hand-roll the equation, so these edges
        # mainly declare the tunable beta / gamma rates. New infections are split
        # between the asymptomatic and symptomatic compartments in equation().
        schema.add_transmission_edge(
            source="susceptible",
            target="asymptomatic",
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
        schema.add_transmission_edge(
            source="asymptomatic",
            target="recovered",
            variable_name="gamma",
            label="Recovery Period (I->R)",
            description="Average number of days to recover (applies to both infectious compartments)",
            default=10.0,
            default_min=5.0,
            default_max=20.0,
            min_value=1.0,
            max_value=100.0,
            value_type=ValueType.DAYS,
            unit="days",
        )

        # Fraction of new infections that are asymptomatic; the rest are
        # symptomatic. Read as self.asymptomatic_fraction (a percentage).
        schema.add_disease_parameter(
            name="asymptomatic_fraction",
            label="Asymptomatic Fraction",
            description="Percentage of new infections that never develop symptoms.",
            value_type=ValueType.PERCENTAGE,
            default=40.0,
            min_value=0.0,
            max_value=100.0,
            unit="%",
            required=False,
            enable_variance=False,
        )

        # Number of stochastic trajectories to simulate. run_simulation reads
        # this to decide how many runs to average for the median + interval.
        schema.add_disease_parameter(
            name="num_runs",
            label="Number of Runs",
            description="Number of stochastic trajectories to simulate.",
            value_type=ValueType.INTEGER,
            default=30,
            min_value=5,
            max_value=50,
            enable_variance=False,
        )

        # --- Optional: spatial travel support ---
        # Declare your mobility parameters as custom fields, then define how
        # they build the matrix in build_travel_matrix() below. Without this,
        # the base class supplies an identity matrix (no inter-zone travel).
        # schema.add_disease_parameter(
        #     name="travel_sigma",
        #     label="Travel Rate (σ)",
        #     description="Percentage of each zone's population away from home per day.",
        #     value_type=ValueType.PERCENTAGE,
        #     default=20.0,
        #     min_value=0.0,
        #     max_value=100.0,
        #     unit="%",
        # )

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
        # schema.add_demographic_group("age_0_17",  "Children", default_weight=33.3, age_range=(0, 17))
        # schema.add_demographic_group("age_18_55", "Adults",   default_weight=44.4, age_range=(18, 55))
        # schema.add_demographic_group("age_56_plus","Elderly", default_weight=22.3, age_range=(56, 120))

    @classmethod
    def _add_total_compartments(cls, schema):
        """Suppress the framework's automatic per-edge ``_total`` compartments.

        Overrides the base behavior so no per-target cumulative compartments
        are auto-added; this model declares its own aggregate ``I_total`` and
        ``R_total`` in ``define_parameters`` instead.

        Args:
            schema: The schema builder (intentionally left unchanged).
        """
        # Suppress the framework's per-edge _total compartments (same approach
        # as DengueJaxModel). We declare our own aggregate I_total in
        # define_parameters() instead, and accumulate into it in equation().
        pass

    def __init__(self, config):
        """Initialize the model and seed its PRNG for stochastic draws.

        Args:
            config: The validated simulation configuration. If it is a dict
                containing a ``seed``, that seed is used for reproducible
                trajectories; otherwise the PRNG is seeded from system entropy.
        """
        super().__init__(config)
        # Stochastic models need a PRNG key for the random draws in equation().
        # Default to system entropy so each run differs; pass "seed" in the
        # config for reproducible trajectories.
        seed = config.get("seed") if isinstance(config, dict) else None
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)
        self._key = jax.random.PRNGKey(seed)

    # --- Optional: spatial travel support ---
    # The framework calls this before prepare_initial_state() and stores the
    # result on self.travel_matrix. The default returns the identity matrix,
    # so only override it if your model has inter-zone mobility.
    #
    # def build_travel_matrix(self, admin_zones):
    #     # PERCENTAGE params arrive as 20.0, not 0.2 — convert first.
    #     sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
    #     return get_gravity_model_travel_matrix(admin_zones, sigma)

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """Seed the initial infected across the two infectious compartments.

        The base implementation seeds a single ``I`` compartment, which this
        model does not have. Infectious individuals are split between ``A``
        (asymptomatic) and ``Sym`` (symptomatic) using the schema's default
        asymptomatic fraction.

        Args:
            admin_zones: Admin zone dicts providing ``population`` and the
                ``infected_population`` percentage.
            compartment_list: Ordered compartment names, used for column
                indexing.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A (zones x compartments) array of initial populations.
        """
        # The base implementation seeds a single "I" compartment, but this model
        # has no "I" — infectious individuals live in A (asymptomatic) and Sym
        # (symptomatic). Split the initial infected between them using the
        # schema's default asymptomatic fraction.
        col = {name: i for i, name in enumerate(compartment_list)}
        initial_population = np.zeros((len(admin_zones), len(compartment_list)))

        asymp_pct = 40.0
        schema = cls._get_cached_schema()
        if schema:
            for p in schema.disease_parameters:
                if p.name == "asymptomatic_fraction":
                    asymp_pct = p.default
                    break
        asymp_frac = asymp_pct / 100.0

        for i, zone in enumerate(admin_zones):
            infected = round(zone["infected_population"] / 100 * zone["population"], 2)
            initial_population[i, col["S"]] = zone["population"] - infected
            initial_population[i, col["A"]] = infected * asymp_frac
            initial_population[i, col["Sym"]] = infected * (1.0 - asymp_frac)

        return initial_population

    def prepare_initial_state(self):
        """Return the initial compartment populations for the solver.

        Returns:
            The population matrix (admin zones x compartments) used as the
            solver's initial state.
        """
        return self.population_matrix

    def equation(self, y, t, p):
        """Tau-leaping stochastic step with two infectious compartments.

        Returns the per-step change (delta); the Euler integrator applies
        ``y_{t+1} = y_t + dt * equation(...)``. We build the derivatives by hand
        so new infections can be split into an asymptomatic (``A``) and a
        symptomatic (``Sym``) compartment, which are combined into one
        "Infected" curve for graphing via COMPARTMENT_DELTA_GROUPING.

        Args:
            y: Current compartment values, ordered by ``compartment_list``.
            t: Current time in days since the simulation start date.
            p: Packed parameter tuple, unpacked via ``_unpack_params``.

        Returns:
            The stacked per-compartment deltas for this step.
        """
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(self.compartment_list)}
        S = states[C.S]
        A = states[C.A]
        Sym = states[C.Sym]

        # Both infectious compartments drive the force of infection.
        non_total = [c for c in C if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)
        infective = A + Sym
        prop_infective = infective.sum() / (N_total.sum() + 1e-10)

        # _apply_interventions scales target_rates and returns the updated travel
        # matrix. It is a no-op when no interventions are configured.
        rates, self.travel_matrix = self._apply_interventions(
            t, {"beta": params["beta"]}, prop_infective
        )
        beta = rates["beta"]
        gamma = params["gamma"]
        # PERCENTAGE params arrive as e.g. 40.0 — convert to a 0-1 fraction.
        asymp_frac = self._to_rate(self.asymptomatic_fraction, ValueType.PERCENTAGE)

        # Expected events per day (frequency-dependent force of infection).
        expected_infections = beta * S * prop_infective
        expected_recoveries_A = gamma * A
        expected_recoveries_Sym = gamma * Sym

        # Draw the actual event counts from Poisson distributions. Split the key
        # each step so successive draws are independent; Euler runs this in a
        # plain Python loop, so mutating self._key here is safe.
        self._key, k_inf, k_rec_a, k_rec_s = jax.random.split(self._key, 4)
        new_infections = jax.random.poisson(k_inf, expected_infections).astype(S.dtype)
        new_infections = jnp.minimum(new_infections, S)

        # Split total new infections between asymptomatic and symptomatic.
        new_asymp = new_infections * asymp_frac
        new_sym = new_infections - new_asymp

        # Recoveries drawn separately for each infectious compartment.
        new_rec_a = jax.random.poisson(k_rec_a, expected_recoveries_A).astype(A.dtype)
        new_rec_s = jax.random.poisson(k_rec_s, expected_recoveries_Sym).astype(Sym.dtype)
        new_rec_a = jnp.minimum(new_rec_a, A)
        new_rec_s = jnp.minimum(new_rec_s, Sym)

        # Build derivatives for every compartment (start at zero, add flows).
        derivs = {c: jnp.zeros_like(S) for c in self.compartment_list}
        derivs[C.S] = -new_infections
        derivs[C.A] = new_asymp - new_rec_a
        derivs[C.Sym] = new_sym - new_rec_s
        derivs[C.R] = new_rec_a + new_rec_s

        # Cumulative combined infected tracker (inflow only) so the
        # "total infected" chart reads cumulative incidence
        derivs["I_total"] = new_infections
        derivs["R_total"] = new_rec_a + new_rec_s

        return jnp.stack([derivs[c] for c in self.compartment_list])
