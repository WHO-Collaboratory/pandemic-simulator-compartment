import jax.numpy as np
import jax
import logging
from compartment.helpers import setup_logging
from compartment.model import Model
from compartment.parameters import ValueType

setup_logging()
logger = logging.getLogger(__name__)

""" 
WARNING: This model is not currently supported in the pandemic simulator app, 
but is available for testing and experimentation in the codebase. 
"""

class CovidSirStochasticModel(Model):
    """A simple stochastic SIR compartmental model for COVID-19.

    Uses tau-leaping: at each timestep the number of infection and
    recovery events is drawn from a Poisson distribution whose mean
    equals the deterministic rate.  This gives demographic stochasticity
    while keeping the same expected trajectory as the ODE model.

    Sets STOCHASTIC = True so the SimulationManager uses fixed-step
    Euler integration instead of the adaptive ODE solver.
    """

    DISEASE_TYPE = "COVID_SIR_STOCHASTIC"
    STOCHASTIC = True

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="COVID_SIR_STOCHASTIC",
            label="COVID-19 Stochastic SIR",
            description="A simple stochastic SIR model for COVID-19",
        )

        schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
        schema.add_compartment("I", "Infected", "Currently infected population", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered and immune population")

        schema.add_transmission_edge(
            source="susceptible",
            target="infected",
            variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (S->I)",
            description="Rate at which susceptible individuals become infected",
            default=0.4,
            default_min=0.2,
            default_max=0.6,
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
            default=7.0,
            default_min=5.0,
            default_max=14.0,
            min_value=1.0,
            max_value=30.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def __init__(self, input):
        self.population_matrix = np.array(input["initial_population"]).T
        self.compartment_list = list(self.COMPARTMENTS)

        self._load_transmission_params(input.get("transmission_dict", {}))

        if self.beta is None:
            self.beta = 0.4
        if self.gamma is None:
            self.gamma = 1.0 / 7.0

        self.start_date = input["start_date"]
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = input["time_steps"]
        self.admin_units = input["admin_units"]

        self.intervention_dict = {}
        self.intervention_statuses = {}
        self.payload = input

        # PRNG key for stochastic draws — use system entropy by default
        # so every run produces a different trajectory.  Pass "seed" in
        # the config to get reproducible results.
        import time
        seed = input.get("seed", None)
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)
        self._key = jax.random.PRNGKey(seed)

    def prepare_initial_state(self):
        self.travel_matrix = np.eye(self.population_matrix.shape[1])
        return (self.population_matrix, list(self.compartment_list))

    def derivative(self, y, t, p):
        """Tau-leaping stochastic step.

        Returns the *change* (delta) for one timestep so the Euler
        integrator y_{t+1} = y_t + dt * derivative gives the correct
        stochastic update.
        """
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(C)}
        S = states[C.S]
        I = states[C.I]  # noqa: E741

        non_total = [c for c in C if not c.endswith("_total")]
        N = sum(states[c] for c in non_total)

        beta = params["beta"]
        gamma = params["gamma"]

        # Expected events per day
        infection_rate = beta * S * I / (N + 1e-10)
        recovery_rate = gamma * I

        # Poisson draws
        self._key, k1, k2 = jax.random.split(self._key, 3)
        new_infections = jax.random.poisson(k1, infection_rate).astype(S.dtype)
        new_recoveries = jax.random.poisson(k2, recovery_rate).astype(I.dtype)

        # Clamp so we don't move more people than exist
        new_infections = np.minimum(new_infections, S)
        new_recoveries = np.minimum(new_recoveries, I)

        # Build derivatives for all compartments (including auto-generated _totals)
        zero = np.zeros_like(S)
        derivs = {c: zero for c in C}

        derivs[C.S] = -new_infections
        derivs[C.I] = new_infections - new_recoveries
        derivs[C.R] = new_recoveries

        # Auto-generated cumulative tracking
        if C.I_total in derivs:
            derivs[C.I_total] = new_infections
        if C.R_total in derivs:
            derivs[C.R_total] = new_recoveries

        return np.stack([derivs[c] for c in C])
