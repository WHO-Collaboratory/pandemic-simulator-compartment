import jax.numpy as np
import logging
import numpy as onp
from compartment.helpers import setup_logging, prepare_covid_initial_state
from compartment.model import Model
from compartment.parameters import ValueType

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

"""
WARNING: This model is not currently supported in the pandemic simulator app,
but is available for testing and experimentation in the codebase.
"""

class CovidJaxModelV2(Model):
    """SEIHDR compartmental model with age-stratified transmission and spatial mobility."""
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
            label="Incubation Period (E->I)",
            description="Average number of days from exposure to becoming infectious",
            default=5.0,
            min_value=1.0,
            max_value=100.0,
            default_min=2.0,
            default_max=14.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

        schema.add_transmission_edge(
            source="infected",
            target="hospitalized",
            variable_name="zeta",
            label="Hospitalization Rate (I->H)",
            description="Percentage of infected individuals who require hospitalization",
            default=0.04,
            min_value=0.01,
            max_value=1.0,
            default_min=0.04,
            default_max=0.4,
            unit="%",
            value_type=ValueType.PERCENTAGE,
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
            default_min=0.001,
            default_max=0.005,
            unit="per day",
        )

        schema.add_transmission_edge(
            source="infected",
            target="recovered",
            variable_name="gamma",
            label="Recovery Period (I->R)",
            description="Average number of days for an infected individual to recover",
            default=7.14,
            min_value=1.0,
            max_value=100.0,
            default_min=4.0,
            default_max=10.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

        schema.add_transmission_edge(
            source="hospitalized",
            target="recovered",
            variable_name="eta",
            label="Hospital Recovery Period (H->R)",
            description="Average number of days for a hospitalized individual to recover",
            default=7.14,
            min_value=1.0,
            max_value=100.0,
            default_min=3.0,
            default_max=14.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

        # ---- Interventions ----
        schema.add_intervention(
            id="mask_wearing",
            label="Mask Wearing",
            description="Reduces transmission rate through mask usage in the population",
            target_rates=["beta"],
            adherence=20.0,
            transmission_reduction=35.0,
        )

        schema.add_intervention(
            id="social_isolation",
            label="Social Isolation",
            description="Reduces transmission rate through social distancing and isolation measures",
            target_rates=["beta"],
            adherence=40.0,
            transmission_reduction=50.0,
        )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def __init__(self, config):
        super().__init__(config)
        self.population_matrix = np.array(config["initial_population"])

        # Travel
        self.travel_matrix = np.fill_diagonal(
            np.array(config["travel_matrix"]), 1.0, inplace=False
        )
        self.sigma = config["travel_volume"]["leaving"]

        # Demographics & age stratification
        self.demographics = config["case_file"]["demographics"]
        self.age_stratification = list(self.demographics.values())
        self.age_groups = list(self.demographics.keys())
        self.interaction_matrix = np.array(
            [[5.46, 5.18, 0.93], [1.70, 9.18, 1.68], [0.83, 5.90, 3.80]]
        )

    def prepare_initial_state(self):
        self.population_matrix, self.interaction_matrix = prepare_covid_initial_state(
            self.population_matrix, self.interaction_matrix, self.demographics
        )

        # Append zero rows for auto-generated _total compartments
        n_base = len([c for c in self.compartment_list if not c.endswith("_total")])
        n_total_comps = len(self.compartment_list) - n_base
        if n_total_comps > 0 and self.population_matrix.shape[0] < len(self.compartment_list):
            zero_rows = onp.zeros((n_total_comps, *self.population_matrix[0].shape))
            self.population_matrix = onp.vstack((self.population_matrix, zero_rows))

        return self.population_matrix, self.compartment_list

    def derivative(self, y, t, p):
        """Compute derivatives with age-stratified force of infection.

        Uses the escape-hatch pattern: ``_compute_derivatives()`` handles
        the 6 standard edges (E->I, I->H, I->D, H->D, I->R, H->R),
        while S->E uses an age-stratified contact matrix and is applied
        manually via ``_apply_flow()``.
        """
        C = self.COMPARTMENTS
        params = self._unpack_params(p)
        age_trans = self.interaction_matrix

        states = {comp: y[i] for i, comp in enumerate(self.compartment_list)}
        S = states[C.S]
        I = states[C.I]  # noqa: E741

        # Population for FOI (exclude deaths and _total compartments)
        non_total = [c for c in self.compartment_list if not c.endswith("_total")]
        pop_terms = [states[c] for c in non_total if c != "D"]
        N_total = sum(pop_terms).sum(axis=0)
        I_frac = I / N_total[None, :]

        # --- Interventions (schema-driven) ---
        rates = {"beta": params["beta"]}
        prop_infective_scalar = I.sum() / N_total.sum()
        rates, contact_matrix = self._apply_interventions(
            t, rates, prop_infective_scalar
        )

        # Non-intervention rates (all standard edges)
        for name in ["theta", "zeta", "delta", "epsilon", "gamma", "eta"]:
            rates[name] = params[name]

        # --- Standard edges via framework (skip manual FOI edge) ---
        derivs = self._compute_derivatives(states, rates, skip_edges={"beta"})

        # --- Manual FOI: age-stratified contact matrix (S->E or S->I) ---
        BETA = ((rates["beta"] * contact_matrix) @ I_frac.T).T
        omega = age_trans @ BETA
        flow_foi = S * omega

        # Route FOI to E if present, otherwise directly to I
        target = "E" if "E" in states else "I"
        self._apply_flow(derivs, "S", target, flow_foi)

        return np.stack([derivs[comp] for comp in self.compartment_list])
