import jax.numpy as np
import logging
from compartment.helpers import setup_logging
from compartment.model import Model
from compartment.parameters import ValueType

setup_logging()
logger = logging.getLogger(__name__)


class SEIcIscRModel(Model):
    """Age-structured SEIcIscR model of the Wuhan COVID-19 outbreak.

    Faithful translation of the SEIcIscR variant of Prem et al. (2020):
    a deterministic, discrete-time (dt = 1 day) age-structured model with

        S  -> E                 (contact-matrix mediated force of infection)
        E  -> Ic  (clinical,    fraction rho, age-specific)
        E  -> Isc (subclinical, fraction 1 - rho)
        Ic -> R
        Isc-> R

    Clinical (Ic) cases are fully infectious; subclinical (Isc) cases are
    only 25% as infectious. Transitions use the discrete-time hazard
    1 - exp(-1/duration) exactly as in the source (rather than 1/duration),
    so a fixed-step Euler solver reproduces the difference equations.

    NOTE: the source's location-decomposed contact matrix (home/work/school/
    others) and its time-varying pWorkOpen intervention schedule cannot be
    represented exactly; the framework's single aggregated (Prem 2021) contact
    matrix is used and the interventions are approximated as reductions on beta.
    """

    SOLVER = "euler"

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="SEICISCR_WUHAN",
            label="Age-structured SEIcIscR (clinical/subclinical)",
            description=(
                "Age-structured SEIcIscR model of the Wuhan COVID-19 outbreak with "
                "clinical and subclinical infectious compartments and contact-matrix "
                "mediated transmission."
            ),
        )
        schema.set_model_metadata(
            citations=["https://doi.org/10.1016/S2468-2667(20)30073-6"],
            key_assumptions=[
                "Discrete-time daily updates with 1-exp(-1/duration) transition hazards",
                "Subclinical infections are 25% as infectious as clinical infections",
                "Clinical fraction rho is age-specific (0.4 for ages 0-19, 0.8 otherwise)",
                "Mean latent period 6.4 days; mean infectious period 7 days",
            ],
        )

        # ---- Compartments ----
        schema.add_compartment("S", "Susceptible", "Susceptible population")
        schema.add_compartment("E", "Exposed", "Latently infected, not yet infectious")
        schema.add_compartment("Ic", "Infectious (clinical)",
                               "Clinical infectious cases (fully infectious)", infective=True)
        schema.add_compartment("Isc", "Infectious (subclinical)",
                               "Subclinical infectious cases (reduced infectiousness)", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered / removed")

        # ---- Transmission edges ----
        # S->E infection: frequency-dependent, applied manually via the age contact matrix.
        schema.add_transmission_parameter(
            source="susceptible", target="exposed", variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (S->E)",
            description="Per-contact transmission probability (normally reverse-engineered from R0)",
            default=0.025, min_value=0.001, max_value=1.0,
            default_min=0.01, default_max=0.05, unit="per day",
        )
        # E->Ic and E->Isc: latent progression, split by clinical fraction rho (applied manually).
        schema.add_transmission_parameter(
            source="exposed", target="infectious (clinical)", variable_name="alpha_c",
            label="Latent Period (E->Ic)",
            description="Mean latent period before becoming clinically infectious",
            default=6.4, min_value=1.0, max_value=30.0,
            default_min=4.0, default_max=9.0,
            unit="days", value_type=ValueType.DAYS,
        )
        schema.add_transmission_parameter(
            source="exposed", target="infectious (subclinical)", variable_name="alpha_sc",
            label="Latent Period (E->Isc)",
            description="Mean latent period before becoming subclinically infectious",
            default=6.4, min_value=1.0, max_value=30.0,
            default_min=4.0, default_max=9.0,
            unit="days", value_type=ValueType.DAYS,
        )
        # Ic->R and Isc->R: recovery / removal.
        schema.add_transmission_parameter(
            source="infectious (clinical)", target="recovered", variable_name="gamma_c",
            label="Infectious Period (Ic->R)",
            description="Mean duration of infectiousness for clinical cases",
            default=7.0, min_value=1.0, max_value=30.0,
            default_min=4.0, default_max=14.0,
            unit="days", value_type=ValueType.DAYS,
        )
        schema.add_transmission_parameter(
            source="infectious (subclinical)", target="recovered", variable_name="gamma_sc",
            label="Infectious Period (Isc->R)",
            description="Mean duration of infectiousness for subclinical cases",
            default=7.0, min_value=1.0, max_value=30.0,
            default_min=4.0, default_max=14.0,
            unit="days", value_type=ValueType.DAYS,
        )

        # ---- Interventions (approximations of the source's contact-matrix scaling) ----
        # NOTE: the source scales location-specific contact matrices over time; here we
        # approximate school closure and lockdown as reductions on the transmission rate.
        schema.add_intervention(
            id="school_closure", label="School closure",
            description="School winter-break / closure reducing transmission",
            target_rates=["beta"], adherence=100.0, transmission_reduction=20.0,
        )
        schema.add_intervention(
            id="lockdown", label="Intense intervention (lockdown)",
            description="Workplace distancing and lockdown reducing transmission",
            target_rates=["beta"], modifies_travel=True,
            adherence=100.0, transmission_reduction=60.0,
        )

        # ---- Extra (non-edge) parameters ----
        schema.add_parameter(
            "subclinical_infectiousness", "Subclinical relative infectiousness",
            "Infectiousness of subclinical (Isc) cases relative to clinical (Ic) cases",
            value_type=ValueType.FLOAT, default=0.25, min_value=0.0, max_value=1.0,
        )
        schema.add_parameter(
            "clinical_fraction_child", "Clinical fraction (ages 0-19)",
            "Fraction of infections that become clinical among ages 0-19",
            value_type=ValueType.FLOAT, default=0.4, min_value=0.0, max_value=1.0,
        )
        schema.add_parameter(
            "clinical_fraction_adult", "Clinical fraction (ages 20+)",
            "Fraction of infections that become clinical among ages 20 and older",
            value_type=ValueType.FLOAT, default=0.8, min_value=0.0, max_value=1.0,
        )

        # ---- Demographics: 16 five-year age bands (opts into Prem 2021 contact matrix) ----
        weights = [5.9, 6.0, 5.7, 5.4, 5.9, 7.6, 8.3, 6.8,
                   6.8, 7.9, 8.4, 6.1, 5.4, 4.7, 3.1, 3.9]
        ranges = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29),
                  (30, 34), (35, 39), (40, 44), (45, 49), (50, 54), (55, 59),
                  (60, 64), (65, 69), (70, 74), (75, 120)]
        for i, ((lo, hi), w) in enumerate(zip(ranges, weights)):
            label = f"Age {lo}-{hi}" if hi < 120 else "Age 75+"
            schema.add_demographic_group(f"age_{lo}_{hi}", label, default_weight=w, age_range=(lo, hi))

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """Seed each zone's infected percentage into the clinical infectious track.

        The framework default seeds a compartment literally named "I", which
        this model does not have. The seed goes into Ic (clinical infectious);
        the E/Ic/Isc balance relaxes to the model's own proportions within a
        few simulated days.
        """
        import numpy as onp

        col = {v: i for i, v in enumerate(compartment_list)}
        pop = onp.zeros((len(admin_zones), len(compartment_list)))
        for z, zone in enumerate(admin_zones):
            N = float(zone["population"])
            pct = max(float(zone.get("infected_population", 0.0) or 0.0), 0.0)
            infected = N * pct / 100.0
            pop[z, col["S"]] = max(N - infected, 0.0)
            pop[z, col["Ic"]] = infected
        return pop

    def __init__(self, config):
        super().__init__(config)
        # Extra parameters (fall back to source defaults if the framework did not set them).
        if not hasattr(self, "subclinical_infectiousness") or self.subclinical_infectiousness is None:
            self.subclinical_infectiousness = 0.25
        if not hasattr(self, "clinical_fraction_child") or self.clinical_fraction_child is None:
            self.clinical_fraction_child = 0.4
        if not hasattr(self, "clinical_fraction_adult") or self.clinical_fraction_adult is None:
            self.clinical_fraction_adult = 0.8

    def prepare_initial_state(self):
        # Expand (K, R) -> (K, A, R) using the declared age groups.
        self._prepare_demographic_state()
        return self.population_matrix

    def equation(self, y, t, p):
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(self.compartment_list)}
        S = states[C.S]
        E = states[C.E]
        Ic = states[C.Ic]
        Isc = states[C.Isc]

        non_total = [c for c in self.compartment_list if not c.endswith("_total")]
        N_age = sum(states[c] for c in non_total)          # (A, R) per-age-group population
        eps = 1e-10

        # Discrete-time hazards: 1 - exp(-1/duration). params[...] already = 1/duration.
        alpha_c = 1.0 - np.exp(-params["alpha_c"])
        alpha_sc = 1.0 - np.exp(-params["alpha_sc"])
        gamma_c = 1.0 - np.exp(-params["gamma_c"])
        gamma_sc = 1.0 - np.exp(-params["gamma_sc"])

        # Interventions modify beta (and possibly the travel matrix).
        prop_infective = (Ic + Isc).sum() / (N_age.sum() + eps)
        rates, travel_matrix = self._apply_interventions(t, {"beta": params["beta"]}, prop_infective)
        beta = rates["beta"]

        # Age-specific clinical fraction rho (ages 0-19 -> child, else adult).
        rho = np.array([self.clinical_fraction_child] * 4
                       + [self.clinical_fraction_adult] * 12)[:, None]  # (A, 1)
        subclin = self.subclinical_infectiousness

        # Force of infection: lambda_a = beta * sum_b C[a,b] * (Ic_b + 0.25*Isc_b)/N_b
        inf_frac = (Ic + subclin * Isc) / (N_age + eps)              # (A, R)
        BETA = ((beta * travel_matrix) @ inf_frac.T).T              # spatial mixing (identity here)
        lam = self.contact_matrix @ BETA                            # (A, R)

        num_S_to_E = lam * S
        num_E_to_Ic = alpha_c * rho * E
        num_E_to_Isc = alpha_sc * (1.0 - rho) * E
        num_Ic_to_R = gamma_c * Ic
        num_Isc_to_R = gamma_sc * Isc

        derivs = {c: np.zeros_like(S) for c in self.compartment_list}
        self._apply_flow(derivs, "S", "E", num_S_to_E)
        self._apply_flow(derivs, "E", "Ic", num_E_to_Ic)
        self._apply_flow(derivs, "E", "Isc", num_E_to_Isc)
        self._apply_flow(derivs, "Ic", "R", num_Ic_to_R)
        self._apply_flow(derivs, "Isc", "R", num_Isc_to_R)

        return np.stack([derivs[c] for c in self.compartment_list])