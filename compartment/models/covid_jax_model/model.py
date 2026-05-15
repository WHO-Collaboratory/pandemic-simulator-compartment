import jax.numpy as np
import logging
from compartment.helpers import setup_logging
from compartment.model import Model
from compartment.parameters import ValueType

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

"""
WARNING: This model is not currently supported in the pandemic simulator app,
but is available for testing and experimentation in the codebase.
"""

class CovidJaxModel(Model):
    """SEIHDR compartmental model with age-stratified transmission and spatial mobility."""

    DISEASE_TYPE = "COVID_SEIHDR"
    DISEASE_LABEL = "Novel Respiratory (SEIHDR)"
    DISEASE_DESCRIPTION = "An SEIHDR compartmental model for novel respiratory diseases with age-stratified transmission"

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type=cls.DISEASE_TYPE,
            label=cls.DISEASE_LABEL,
            description=cls.DISEASE_DESCRIPTION,
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

        # ---- Demographics ----
        # age_range enables auto-loading of the country's Prem 2021 contact
        # matrix (aggregated to these bands) when no explicit overrides are
        # declared.  The explicit overrides below still take precedence; the
        # age_range tags are forward-compatible for when the overrides are
        # removed in favor of country-aware defaults.
        schema.add_demographic_group("age_0_17",    "Children (0-17)", default_weight=33.3, age_range=(0, 17))
        schema.add_demographic_group("age_18_55",   "Adults (18-55)",  default_weight=44.4, age_range=(18, 55))
        schema.add_demographic_group("age_56_plus", "Elderly (56+)",   default_weight=22.3, age_range=(56, 120))

        # Contact matrix — age-structured interaction rates (POLYMOD-derived).
        # All 9 entries are declared; diagonal values override the identity default.
        schema.set_contact_override("age_0_17",    "age_0_17",    5.46)
        schema.set_contact_override("age_0_17",    "age_18_55",   5.18)
        schema.set_contact_override("age_0_17",    "age_56_plus", 0.93)
        schema.set_contact_override("age_18_55",   "age_0_17",    1.70)
        schema.set_contact_override("age_18_55",   "age_18_55",   9.18)
        schema.set_contact_override("age_18_55",   "age_56_plus", 1.68)
        schema.set_contact_override("age_56_plus", "age_0_17",    0.83)
        schema.set_contact_override("age_56_plus", "age_18_55",   5.90)
        schema.set_contact_override("age_56_plus", "age_56_plus", 3.80)

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def __init__(self, config):
        super().__init__(config)

        # Override with the compartment list from config (may be a variant subset).
        self.compartment_list = config["compartment_list"]

        # base __init__ sets population_matrix to jnp.array(initial_population).T
        # which is already (K, R). No override needed.

        # Travel
        self.travel_matrix = np.fill_diagonal(
            np.array(config["travel_matrix"]), 1.0, inplace=False
        )
        self.sigma = config["travel_volume"]["leaving"]

        # Demographic weights come from case file for population tensor construction.
        # Group definitions and contact matrix are declared in define_parameters().
        self.demographics = config["case_file"]["demographics"]

    def prepare_initial_state(self):
        # Expand (K, R) → (K, A, R) and append _total rows for active compartments.
        self._prepare_demographic_state()
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
        rates, travel_matrix = self._apply_interventions(
            t, rates, prop_infective_scalar
        )

        # Add all active (non-None) rates for standard edges
        for name, value in params.items():
            if name != "beta" and value is not None:
                rates[name] = value

        # --- Standard edges via framework (skip manual FOI edge) ---
        derivs = self._compute_derivatives(states, rates, skip_edges={"beta"})

        # --- Manual FOI: spatial travel mixing + demographic contact matrix ---
        BETA = ((rates["beta"] * travel_matrix) @ I_frac.T).T
        omega = self.contact_matrix @ BETA
        flow_foi = S * omega

        # Route FOI to E if present, otherwise directly to I
        target = "E" if "E" in states else "I"
        self._apply_flow(derivs, "S", target, flow_foi)

        return np.stack([derivs[comp] for comp in self.compartment_list])
