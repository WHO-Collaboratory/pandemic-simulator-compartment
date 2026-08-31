import jax.numpy as jnp
import logging
from compartment.helpers import setup_logging
from compartment.model import Model
from compartment.parameters import ValueType

setup_logging()
logger = logging.getLogger(__name__)


class EbolaSEI1I2RDModel(Model):
    """Ebola SEI1I2RD importation-risk model (DRC -> EU/EEA).

    Faithful translation of the deterministic ODE core of the source
    Monte-Carlo model. Exposed individuals progress to one of two infectious
    strata: those who will recover (IR) and those who will die (ID), with the
    branching governed by the case-fatality probability p_death:

        dS  = -beta * S * (IR+ID) / N
        dE  =  beta * S * (IR+ID) / N - sigma * E
        dIR =  (1-p_death) * sigma * E - gamma * IR
        dID =  p_death * sigma * E - mu * ID
        dR  =  gamma * IR
        dD  =  mu * ID

    The Monte-Carlo parameter sampling and eigenvalue-based initial-condition
    seeding from the source are not reproduced here (the framework supplies
    initial state and uncertainty handling). The S->E infection edge and the
    IR->R / ID->D recovery/death edges are standard schema edges; the branching
    E->IR / E->ID flows are applied manually because they share sigma and split
    on p_death.
    """

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="EBOLA_SEI1I2RD",
            label="Ebola SEI1I2RD",
            description=(
                "Ebola SEI(R)I(D)RD model with separate infectious strata for "
                "recovering and dying cases, used to estimate importation risk"
            ),
        )

        schema.set_model_metadata(
            authors=[
                {"name": "Eva Bons"},
                {"name": "Joana Gomes Dias"},
                {"name": "Disa Hansson"},
                {"name": "Sharon Kuhlmann Berenzon"},
                {"name": "Bastian Prasse"},
            ],
            key_assumptions=[
                "Well-mixed population of Ituri + Nord Kivu (~13,392,200)",
                "Frequency-dependent transmission from both infectious strata",
                "Exposed cases branch to recovery or death by the case-fatality probability",
            ],
        )

        # ---- Compartments ----
        schema.add_compartment("S", "Susceptible", "Susceptible population")
        schema.add_compartment("E", "Exposed", "Exposed (latent, not yet infectious)")
        schema.add_compartment(
            "IR", "Infectious (recovering)",
            "Infectious individuals who will recover", infective=True,
        )
        schema.add_compartment(
            "ID", "Infectious (dying)",
            "Infectious individuals who will die", infective=True,
        )
        schema.add_compartment("R", "Recovered", "Recovered and immune")
        schema.add_compartment("D", "Dead", "Deceased")

        # ---- Transmission edges ----
        # S->E: frequency-dependent infection (beta * S * (IR+ID) / N).
        # Default beta derived from mean parameters:
        #   R0_mean = (1.37+1.11)/2 = 1.24; p_death ~ 0.43; gamma=mu=0.1
        #   beta = R0 / ((1-p_death)/gamma + p_death/mu) ~ 0.124
        schema.add_transmission_parameter(
            source="susceptible", target="exposed", variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (S->E)",
            description="Frequency-dependent transmission rate from infectious cases",
            default=0.124, min_value=0.001, max_value=2.0,
            default_min=0.08, default_max=0.2, unit="per day",
        )
        # IR->R: recovery, expressed as an infectious period (days).
        schema.add_transmission_parameter(
            source="infectious (recovering)", target="recovered", variable_name="gamma",
            label="Infectious Period, recovering (IR->R)",
            description="Time from symptom onset to recovery",
            default=10.0, min_value=2.0, max_value=26.0,
            default_min=7.0, default_max=14.0,
            unit="days", value_type=ValueType.DAYS,
        )
        # ID->D: death, expressed as symptom-to-death period (days).
        schema.add_transmission_parameter(
            source="infectious (dying)", target="dead", variable_name="mu",
            label="Symptom-to-Death Period (ID->D)",
            description="Time from symptom onset to death given dying",
            default=10.0, min_value=3.0, max_value=21.0,
            default_min=7.0, default_max=14.0,
            unit="days", value_type=ValueType.DAYS,
        )

        # ---- Parameters for the branching E outflow (applied manually) ----
        schema.add_parameter(
            name="incubation_period",
            label="Incubation Period",
            description="Time from infection to symptom onset (E dwell time)",
            value_type=ValueType.DAYS, default=10.0,
            min_value=2.0, max_value=21.0, unit="days",
        )
        schema.add_parameter(
            name="p_death",
            label="Case-fatality probability",
            description="Probability of dying given infection (fraction)",
            value_type=ValueType.FLOAT, default=0.43,
            min_value=0.32, max_value=0.54,
        )

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """Seed each zone's infected percentage into IR/ID split by p_death.

        The framework default seeds a compartment literally named "I", which
        this model does not have. The initial infectious are divided between
        the recovering (IR) and dying (ID) strata using the case-fatality
        probability's schema default (0.43), matching the model's E-outflow
        branching.
        """
        import numpy as onp

        p_death = 0.43  # schema default for p_death
        col = {v: i for i, v in enumerate(compartment_list)}
        pop = onp.zeros((len(admin_zones), len(compartment_list)))
        for z, zone in enumerate(admin_zones):
            N = float(zone["population"])
            inf_pct = max(float(zone.get("infected_population", 0.0) or 0.0), 0.0)
            infected = N * inf_pct / 100.0
            pop[z, col["S"]] = max(N - infected, 0.0)
            pop[z, col["IR"]] = infected * (1.0 - p_death)
            pop[z, col["ID"]] = infected * p_death
        return pop

    def __init__(self, config):
        super().__init__(config)
        # Native-unit parameters; access defensively with source-mean fallbacks.
        self.incubation_period = getattr(self, "incubation_period", 10.0)
        self.p_death = getattr(self, "p_death", 0.43)

    def prepare_initial_state(self):
        # No inter-region travel — the framework supplies the identity matrix.
        return self.population_matrix

    def equation(self, y, t, p):
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(self.compartment_list)}
        E = states[C.E]
        IR = states[C.IR]
        ID = states[C.ID]  # noqa: E741

        infectives = IR + ID
        non_total = [c for c in self.compartment_list if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)
        prop_infective = infectives.sum() / (N_total.sum() + 1e-10)

        rates = {"beta": params["beta"]}
        rates, _ = self._apply_interventions(t, rates, prop_infective)
        rates["gamma"] = params["gamma"]
        rates["mu"] = params["mu"]

        # Standard edges: S->E (freq. dependent), IR->R, ID->D.
        derivs = self._compute_equations(states, rates)

        # Branching E outflow: E -> IR at (1-p_death)*sigma, E -> ID at p_death*sigma.
        sigma = 1.0 / self.incubation_period
        p_death = self.p_death
        to_IR = (1.0 - p_death) * sigma * E
        to_ID = p_death * sigma * E
        self._apply_flow(derivs, "E", "IR", to_IR)
        self._apply_flow(derivs, "E", "ID", to_ID)

        return jnp.stack([derivs[c] for c in self.compartment_list])