import jax.numpy as np
import jax
import numpy as onp
import logging
from compartment.helpers import setup_logging
from compartment.interventions import jax_timestep_intervention, jax_prop_intervention
from compartment.model import Model

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


class MpoxJaxModel(Model):
    """A simple SIR compartmental model for MPOX"""

    # ------------------------------------------------------------------
    # Declarative parameter schema (single source of truth)
    #
    # Everything below — COMPARTMENT_LIST, disease_type, transmission
    # param attributes (self.beta, self.gamma), and get_params() — is
    # derived automatically from these declarations by the base class.
    # ------------------------------------------------------------------

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="MONKEYPOX",
            label="Monkeypox",
            description="A simple SIR compartmental model for Monkeypox",
        )

        schema.add_compartment(
            "S", "Susceptible", "Population susceptible to Monkeypox infection"
        )
        schema.add_compartment("I", "Infected", "Currently infected population")
        schema.add_compartment("R", "Recovered", "Recovered and immune population")
        schema.add_compartment("I_total", "Infected Total", "Total infected population")

        schema.add_transmission_edge(
            source="susceptible",
            target="infected",
            variable_name="beta",
            label="Transmission Rate (S->I)",
            description="Rate at which susceptible individuals become infected through contact with infected individuals",
            default=0.3,
            min_value=0.01,
            max_value=2.0,
            default_min=0.1,
            default_max=0.5,
            unit="per day",
        )

        schema.add_transmission_edge(
            source="infected",
            target="recovered",
            variable_name="gamma",
            label="Recovery Rate (I->R)",
            description="Rate at which infected individuals recover and gain immunity",
            default=0.1,
            min_value=0.01,
            max_value=1.0,
            default_min=0.05,
            default_max=0.2,
            unit="per day",
        )

        schema.add_intervention(
            id="mask_wearing",
            label="Mask Wearing",
            description="Reduces transmission rate through mask usage in the population",
            adherence=30.0,
            transmission_reduction=35.0,
        )

        schema.add_intervention(
            id="vaccination",
            label="Vaccination",
            description="Reduces transmission rate through immunization of susceptible population",
            adherence=60.0,
            transmission_reduction=80.0,
        )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def __init__(self, config):
        """Initialize the MPOX SIR model with a configuration dictionary"""
        # Population data
        self.population_matrix = np.array(config["initial_population"]).T
        self.compartment_list = list(self.COMPARTMENTS)

        # Transmission params (self.beta, self.gamma) are set
        # automatically from the schema edge variable_names.
        self._load_transmission_params(config.get("transmission_dict", {}))

        # Simulation parameters
        self.start_date = config["start_date"]
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = config["time_steps"]

        # Administrative units
        self.admin_units = config["admin_units"]

        # Interventions
        self.intervention_dict = config.get("intervention_dict", {})
        self.intervention_statuses = {
            "mask_wearing": False,
            "vaccination": False,
        }

        self.payload = config

    # disease_type  — derived from schema (set_model_info)
    # COMPARTMENTS   — derived from schema (add_compartment)
    # get_params()  — derived from schema (edge order: beta, gamma)
    # All inherited from base class; no need to override.

    def prepare_initial_state(self):
        # self._add_cumulative_compartments()

        # Identity travel matrix for intervention function signatures.
        # Mpox doesn't model inter-zone travel, but the shared intervention
        # functions require a travel_matrix argument.
        n_regions = len(self.admin_units)
        self.travel_matrix = np.eye(n_regions)

        return (
            self.population_matrix,
            list(self.compartment_list),
        )

    def derivative(self, y, t, p):
        C = self.COMPARTMENTS
        # Unpack parameters (order matches add_transmission_edge order)
        beta, gamma = p

        # Extract compartments by name from state vector
        states = {c: y[i] for i, c in enumerate(C)}
        S = states[C.S]
        I = states[C.I]  # noqa: E741
        R = states[C.R]

        # Compute total population
        N_total = S + I + R

        # --- Interventions ---
        current_ordinal_day = self.start_date_ordinal + t
        rates = {"beta": beta}
        prop_infective_scalar = I.sum() / (N_total.sum() + 1e-10)

        rates, self.intervention_statuses, _ = jax_timestep_intervention(
            self.intervention_dict,
            current_ordinal_day,
            rates,
            self.intervention_statuses,
            self.travel_matrix,
        )
        rates, self.intervention_statuses, _ = jax_prop_intervention(
            self.intervention_dict,
            prop_infective_scalar,
            rates,
            self.intervention_statuses,
            self.travel_matrix,
        )

        beta = rates["beta"]

        # Force of infection (simple frequency-dependent transmission)
        lambda_force = (
            beta * I / (N_total + 1e-10)
        )  # Add small epsilon to avoid division by zero

        # SIR derivatives
        derivs = {}
        derivs[C.S] = -S * lambda_force
        derivs[C.I] = S * lambda_force - gamma * I
        derivs[C.R] = gamma * I
        derivs[C.I_total] = S * lambda_force

        return np.stack([derivs[c] for c in C])
