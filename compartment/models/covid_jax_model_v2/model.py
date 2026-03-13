import jax.numpy as np
import logging
import numpy as onp
from compartment.helpers import setup_logging, prepare_covid_initial_state
from compartment.interventions import jax_timestep_intervention, jax_prop_intervention
from compartment.model import Model

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


class CovidJaxModelV2(Model):
    """A class representing a compartmental model with dynamic travel and intervention mechanisms"""

    def __init__(self, config):
        """Initialize the CompartmentalModel with a configuration dictionary"""
        # Load population and travel data
        self.population_matrix = np.array(config["initial_population"])
        self.travel_matrix = np.fill_diagonal(
            np.array(config["travel_matrix"]), 1.0, inplace=False
        )
        self.compartment_list = config["compartment_list"]
        self.sigma = config["travel_volume"]["leaving"]

        # Load disease transmission parameters from schema edge declarations
        self._load_transmission_params(config.get("transmission_dict", {}))
        self.original_rates = {"beta": self.beta}

        # Simulation parameters
        self.start_date = config["start_date"]
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = config["time_steps"]

        # Administrative units & demographics
        self.admin_units = config["admin_units"]
        self.demographics = config["case_file"]["demographics"]
        self.age_stratification = list(self.demographics.values())
        self.age_groups = list(self.demographics.keys())
        self.interaction_matrix = np.array(
            [[5.46, 5.18, 0.93], [1.70, 9.18, 1.68], [0.83, 5.90, 3.80]]
        )

        # Interventions
        self.intervention_dict = config["intervention_dict"]
        self.intervention_statuses = {
            "lock_down": False,
            "mask_wearing": False,
            "social_isolation": False,
            "vaccination": False,
        }

        self.payload = config

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="RESPIRATORY",
            label="Novel Respiratory (Advanced)",
            description="An SEIHDR compartmental model for novel respiratory diseases with age-stratified transmission",
        )

        # ---- Compartments (S-E-I-H-D-R) ----
        schema.add_compartment(
            "S", "Susceptible", "Population susceptible to infection"
        )
        schema.add_compartment(
            "E", "Exposed", "Population exposed but not yet infectious"
        )
        schema.add_compartment(
            "I",
            "Infected",
            "Currently infected and infectious population",
            infective=True,
        )
        schema.add_compartment(
            "H", "Hospitalized", "Infected individuals requiring hospitalization"
        )
        schema.add_compartment(
            "D", "Deceased", "Population that has died from the disease"
        )
        schema.add_compartment("R", "Recovered", "Recovered and immune population")

        # ---- Transmission edges ----
        schema.add_transmission_edge(
            source="susceptible",
            target="exposed",
            variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (S->E)",
            description="Rate at which susceptible individuals become exposed through contact with infected individuals",
            default=0.25,
            min_value=0.01,
            max_value=2.0,
            default_min=0.2,
            default_max=0.3,
            unit="per day",
        )

        schema.add_transmission_edge(
            source="exposed",
            target="infected",
            variable_name="theta",
            label="Incubation Rate (E->I)",
            description="Rate at which exposed individuals become infectious",
            default=0.2,
            min_value=0.01,
            max_value=1.0,
            unit="per day",
        )

        schema.add_transmission_edge(
            source="infected",
            target="hospitalized",
            variable_name="zeta",
            label="Hospitalization Rate (I->H)",
            description="Rate at which infected individuals require hospitalization",
            default=0.0004,
            min_value=0.0001,
            max_value=0.01,
            default_min=0.0004,
            default_max=0.004,
            unit="per day",
        )

        schema.add_transmission_edge(
            source="infected",
            target="deceased",
            variable_name="delta",
            label="Infection Fatality Rate (I->D)",
            description="Rate at which infected individuals die from the disease",
            default=0.0001,
            min_value=0.00001,
            max_value=0.01,
            default_min=0.0001,
            default_max=0.001,
            unit="per day",
        )

        schema.add_transmission_edge(
            source="hospitalized",
            target="deceased",
            variable_name="epsilon",
            label="Hospital Fatality Rate (H->D)",
            description="Rate at which hospitalized individuals die",
            default=0.001,
            min_value=0.0001,
            max_value=0.01,
            unit="per day",
        )

        schema.add_transmission_edge(
            source="infected",
            target="recovered",
            variable_name="gamma",
            label="Recovery Rate (I->R)",
            description="Rate at which infected individuals recover and gain immunity",
            default=0.14,
            min_value=0.01,
            max_value=1.0,
            unit="per day",
        )

        schema.add_transmission_edge(
            source="hospitalized",
            target="recovered",
            variable_name="eta",
            label="Hospital Recovery Rate (H->R)",
            description="Rate at which hospitalized individuals recover",
            default=0.14,
            min_value=0.01,
            max_value=1.0,
            unit="per day",
        )

        # ---- Interventions ----
        schema.add_intervention(
            id="mask_wearing",
            label="Mask Wearing",
            description="Reduces transmission rate through mask usage in the population",
            adherence=20.0,
            transmission_reduction=35.0,
        )

        schema.add_intervention(
            id="social_isolation",
            label="Social Isolation",
            description="Reduces transmission rate through social distancing and isolation measures",
            adherence=40.0,
            transmission_reduction=50.0,
        )

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """
        Standard SEIHDR initial population for respiratory diseases.
        Uses base class implementation (S-I split).
        """
        return super().get_initial_population(admin_zones, compartment_list, **kwargs)

    # disease_type  — derived from schema (set_model_info)
    # COMPARTMENTS  — derived from schema (add_compartment)
    # get_params()  — derived from schema (edge order: beta, theta, zeta, delta, epsilon, gamma, eta)
    # All inherited from base class; no need to override.

    def _sync_total_compartments(self):
        """Add ``_total`` tracking compartments for active base compartments.

        Iterates over the schema's ``COMPARTMENT_LIST`` (populated by
        ``build()``) and appends a zero row to the population matrix for
        every ``_total`` compartment whose base compartment is present in
        the runtime ``self.compartment_list``.
        """
        zero_row = onp.zeros((1, *self.population_matrix[0].shape))
        for cid in list(self.COMPARTMENTS):
            if cid.endswith("_total"):
                base_id = cid.removesuffix("_total")
                if (
                    base_id in self.compartment_list
                    and cid not in self.compartment_list
                ):
                    self.population_matrix = onp.vstack(
                        (self.population_matrix, zero_row)
                    )
                    self.compartment_list = self.compartment_list + [cid]

    def prepare_initial_state(self):
        self.population_matrix, self.interaction_matrix = prepare_covid_initial_state(
            self.population_matrix, self.interaction_matrix, self.demographics
        )
        self._sync_total_compartments()

        return self.population_matrix, self.compartment_list

    def derivative(self, y, t, p):
        """
        Calculates how each compartment changes over time, used by ODE solver.

        Uses the escape-hatch pattern: ``_compute_derivatives()`` handles
        the 6 standard ``rate * source`` edges (E→I, I→H, I→D, H→D, I→R,
        H→R), while the S→E force of infection — which requires an
        age-stratified contact matrix — is computed manually and applied
        via ``_apply_flow()``.

        Edges whose source or target compartment is absent from the
        runtime ``compartment_list`` are automatically skipped by the
        framework, so optional compartments (E, H, D) work without
        conditional rate zeroing.
        """
        params = self._unpack_params(p)
        age_trans = self.interaction_matrix

        states = {comp: y[i] for i, comp in enumerate(self.compartment_list)}
        S = states["S"]
        I = states["I"]  # noqa: E741

        # Population for FOI (exclude deaths and _total compartments)
        non_total = [c for c in self.compartment_list if not c.endswith("_total")]
        pop_terms = [states[c] for c in non_total if c != "D"]
        N_total = sum(pop_terms).sum(axis=0)
        I_frac = I / N_total[None, :]

        # --- Interventions (only beta is subject to interventions) ---
        current_ordinal_day = self.start_date_ordinal + t
        rates = {"beta": params["beta"]}
        prop_infective_scalar = I.sum() / N_total.sum()
        rates, self.intervention_statuses, contact_matrix = jax_timestep_intervention(
            self.intervention_dict,
            current_ordinal_day,
            rates,
            self.intervention_statuses,
            self.travel_matrix,
        )
        rates, self.intervention_statuses, contact_matrix = jax_prop_intervention(
            self.intervention_dict,
            prop_infective_scalar,
            rates,
            self.intervention_statuses,
            self.travel_matrix,
        )

        # Non-intervention rates (all standard edges)
        for name in ["theta", "zeta", "delta", "epsilon", "gamma", "eta"]:
            rates[name] = params[name]

        # --- Standard edges via framework (skip manual FOI edge) ---
        derivs = self._compute_derivatives(states, rates, skip_edges={"beta"})

        # --- Manual FOI: age-stratified contact matrix (S→E or S→I) ---
        BETA = ((rates["beta"] * contact_matrix) @ I_frac.T).T
        omega = age_trans @ BETA
        flow_foi = S * omega

        # Route FOI to E if present, otherwise directly to I
        target = "E" if "E" in states else "I"
        self._apply_flow(derivs, "S", target, flow_foi)

        return np.stack([derivs[comp] for comp in self.compartment_list])
