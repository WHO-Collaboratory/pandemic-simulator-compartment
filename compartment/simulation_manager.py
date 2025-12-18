import logging
import numpy as np
import jax
from jax.experimental.ode import odeint

from compartment.helpers import get_simulation_step_size, setup_logging, log_mem

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

        log_mem("before odeint")
        pred = odeint(self.model.derivative, init_state, ts, params)

        # IMPORTANT: wait for async work to finish before measuring
        pred.block_until_ready()
        log_mem("after odeint")

        # Transfer to host (can be a big extra spike)
        log_mem("before device_get")
        out = jax.device_get(pred)   # returns numpy arrays
        log_mem("after device_get")

        logger.info("run_simulation function ENDING")
        return np.array(out)  # optional; out is already host-side
