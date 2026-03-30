import logging
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

from compartment.helpers import get_simulation_step_size, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class SimulationManager:
    def __init__(self, model):
        self.model = model

    def run_simulation(self):
        logger.info("run_simulation function STARTING")

        logger.info(f"n_timesteps: {self.model.n_timesteps}")
        step = get_simulation_step_size(self.model.n_timesteps)
        logger.info(f"step size (days): {step}")
        ts = np.arange(0.0, float(self.model.n_timesteps), step)

        init_state, comp_list = self.model.prepare_initial_state()
        params = self.model.get_params()

        # Default: adaptive ODE solver (JAX).
        # Models can set SOLVER = "euler" to use fixed-step Euler
        # integration instead — required for stochastic models and
        # compatible with plain numpy models.
        # Legacy: STOCHASTIC = True also triggers Euler.
        solver = getattr(self.model, "SOLVER", None)
        if solver is None:
            solver = "euler" if getattr(self.model, "STOCHASTIC", False) else "odeint"

        if solver == "euler":
            logger.info("solver: euler")
            pred = self._euler_integrate(init_state, ts, params)
            if isinstance(pred, jax.Array):
                pred = jax.device_get(pred)
            out = np.asarray(pred)
        else:
            logger.info("solver: odeint")
            pred = odeint(self.model.derivative, init_state, ts, params)
            out = jax.device_get(pred)

        logger.info("run_simulation function ENDING")
        return np.array(out)

    def _euler_integrate(self, y0, ts, params):
        """Fixed-step Euler integration.

        Works with both JAX and plain numpy arrays — the array module
        is detected from *y0* so numpy-only models never touch JAX.
        """
        if isinstance(y0, jax.Array):
            xp = jnp
        else:
            xp = np

        results = [y0]
        y = y0
        for i in range(1, len(ts)):
            dt = ts[i] - ts[i - 1]
            dy = self.model.derivative(y, ts[i - 1], params)
            y = y + dt * dy
            y = xp.maximum(y, 0.0)
            results.append(y)
        return xp.stack(results)
