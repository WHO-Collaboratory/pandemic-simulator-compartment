import logging
import numpy as np
from compartment.helpers import get_simulation_step_size, setup_logging
from jax.experimental.ode import odeint

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class SimulationManager:
    def __init__(self, model):
        self.model = model

    def run_simulation(self):
        """ Run the simulation over the specified number of timesteps """
        logger.info(f"run_simulation function STARTING")

        # Set up timesteps, and step size
        logger.info(f"n_timesteps: {self.model.n_timesteps}")
        step = get_simulation_step_size(self.model.n_timesteps)
        logger.info(f"step size (days): {step}")
        ts = np.arange(0.0,float(self.model.n_timesteps),step)

        # Prepare the initial state 
        init_state, comp_list = self.model.prepare_initial_state()
        params = self.model.get_params()

        # Solve the ODE
        pred = odeint(self.model.derivative, init_state, ts, params)

        logger.info(f"run_simulation function ENDING")
        return np.array(pred)