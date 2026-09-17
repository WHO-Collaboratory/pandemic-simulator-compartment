import jax.numpy as jnp
import numpy as np
import logging
from compartment.model import Model, ValueType

logger = logging.getLogger(__name__)


class RespxietyModel(Model):
    """A simple model of a respiratory disease with adaptive behavior determined by the number
    and rate of change of detected (reported or hospitalized) cases
    
    Transition from S (susceptible) to E (exposed) depends on the parameter ``beta``, which is 
    modified by the risk coefficient cr = exp(-sens * p_infection), 
    where ``sens`` is a model parameter representing risk sensitivity, and
    p_infection is an individual's perceived infection probability, based on a
    time-lagged assessment of reported case counts.

    The parameters ``lrLevel`` and ``lrSlope`` represent the behavioral time lag.
    ``lrLevel`` is the rate at which individuals assimilate information about case counts, and
    ``lrSlope`` is the rate at which individuals respond to changes in the trend.

    Transition from E occurs either to Id (detected infection) or Iu (undetected infection),
    depending on the detection rate, which is a function of time and model parameters as follows: 
    The detection rate increases from 0 at the start of the simulation to ``thMax`` at a rate of
    ``thInc`` following a logistic function, and optionally decreases to ``thLow`` at a rate of ``thDec``.
    """

    DISEASE_TYPE="RESPXIETY"

    # Collapse the two infectious compartments into one "Infected" series
    COMPARTMENT_DELTA_GROUPING = {
        "S": ["S"],
        "E": ["E"],
        "I": ["Iu", "Id"],
        "R": ["R"],
    }

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="RESPXIETY",
            label="Respiratory disease with adaptive behavior based on reported (or hospitalized) cases",
            description="A simple model of a respiratory disease with adaptive behavior determined by the number and rate of change of detected (reported or hospitalized) cases",
        )

        # --- Compartments ---
        # Mark infective=True on compartments that contribute to force of infection.
        schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
        schema.add_compartment("E", "Exposed", "Population exposed but not yet infectious")
        schema.add_compartment("Iu", "Undetected", "Currently undetected infectious population", infective=True)
        schema.add_compartment("Id", "Detected", "Currently detected infectious population", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered and immune")
        schema.add_compartment("Level", "Level", "Perceived count of detected infections")
        schema.add_compartment("Slope", "Slope", "Perceived rate of change in detected infections")
        ## manually update totals so we can track Iu + Id
        schema.add_compartment("I_total", "Infected Total", "Cumulative infections (detected + undetected)")
        schema.add_compartment("R_total", "Recovered Total", "Cumulative recoveries")
        schema.add_compartment("Iu_total", "Undetected Total", "Cumulative undetected infections")
        schema.add_compartment("Id_total", "Detected Total", "Cumulative detected infections")

        # --- Transmission edges ---
        # transmission and total are determined manually; these only declare the model parameters
        schema.add_transmission_parameter(
            source="Susceptible",
            target="Detected",
            variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (infection by Detected)",
            description="Rate at which detected cases infect susceptibles they contact",
            default=0.4,
            default_min=0.1,
            default_max=0.5,
            min_value=0.01,
            max_value=2.0,
            unit="per day",
        )
        schema.add_transmission_parameter(
            source="Susceptible",
            target="Undetected",
            variable_name="beta_u",
            frequency_dependent=True,
            label="Transmission Rate (infection by Undetected)",
            description="Rate at which undetected cases infect susceptibles they contact",
            default=0.4,
            default_min=0.1,
            default_max=0.5,
            min_value=0.01,
            max_value=2.0,
            unit="per day",
        )
        schema.add_transmission_parameter(
            source="Exposed",
            target="detected",
            variable_name="mu",
            label="Incubation period (E -> Iu or Id)",
            description="Average number of days for exposed case to become infectious",
            default=6.0,
            default_min=3.0,
            default_max=12.0,
            min_value=1.0,
            max_value=100.0,
            value_type=ValueType.DAYS,
            unit="days",
        )
        schema.add_transmission_parameter(
            source="Detected",
            target="Recovered",
            variable_name="gamma",
            label="Recovery Period for Detected case (Id->R)",
            description="Average number of days for detected infection to recover",
            default=5.0,
            default_min=2.0,
            default_max=10.0,
            min_value=1.0,
            max_value=100.0,
            value_type=ValueType.DAYS,
            unit="days",
        )
        schema.add_transmission_parameter(
            source="Undetected",
            target="Recovered",
            variable_name="gamma_u",
            label="Recovery Period for Undetected case (Iu->R)",
            description="Average number of days for undetected infection to recover",
            default=5.0,
            default_min=2.0,
            default_max=10.0,
            min_value=1.0,
            max_value=100.0,
            value_type=ValueType.DAYS,
            unit="days",
        )

        schema.add_parameter(
            name="sens",
            label="Risk sensitivity",
            description="Sensitivity to perceived probablility of infection",
            default=500.0,
            min_value=0.0,
            max_value=10000.0,
            value_type=ValueType.RATE,
        )
        schema.add_parameter(
            name="lrLevel",
            label="Level learning rate",
            description="Rate at which perceived number of infections changes in response to new detected cases",
            default=0.025,
            min_value=0.0,
            max_value=0.5,
            value_type=ValueType.RATE,
        )
        schema.add_parameter(
            name="lrSlope",
            label="Slope learning rate",
            description="Rate at which perceived rate of change in infections responds to new detected cases",
            default=0.05,
            min_value=0.0,
            max_value=0.5,
            value_type=ValueType.RATE,
        )
        schema.add_parameter(
            name="thInc",
            label="Detection increase rate",
            description="Rate of increase over time in the fraction of infections that are detected",
            default=0.02,
            min_value=0.0,
            max_value=1.0,
            value_type=ValueType.RATE,
        )
        schema.add_parameter(
            name="thDec",
            label="Detection decrease rate",
            description="Rate of decrease over time in the fraction of infections that are detected",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            value_type=ValueType.RATE,
        )
        schema.add_parameter(
            name="thMax",
            label="Detection rate upper asymptote",
            description="Detection rate upper asymptote",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            value_type=ValueType.RATE,
        )
        schema.add_parameter(
            name="thLow",
            label="Detection rate lower asymptote",
            description="Detection rate lower asymptote",
            default=0.1,
            min_value=0.0,
            max_value=1.0,
            value_type=ValueType.RATE,
        )

        # --- Optional: spatial travel support ---
        # Declare your mobility parameters as custom fields, then define how
        # they build the matrix in build_travel_matrix() below. Without this,
        # the base class supplies an identity matrix (no inter-zone travel).
        # schema.add_parameter(
        #     name="travel_sigma",
        #     label="Travel Rate (σ)",
        #     description="Percentage of each zone's population away from home per day.",
        #     value_type=ValueType.PERCENTAGE,
        #     default=20.0,
        #     min_value=0.0,
        #     max_value=100.0,
        #     unit="%",
        # )

        # --- Optional: interventions ---
        # schema.add_intervention(
        #     id="my_intervention",
        #     label="My Intervention",
        #     description="Reduces transmission while active",
        #     target_rates=["beta"],
        #     adherence=50.0,
        #     transmission_reduction=50.0,
        # )

        # --- Optional: age-stratified demographics + contact matrix ---
        # schema.add_demographic_group("age_0_17",  "Children", default_weight=33.3, age_range=(0, 17))
        # schema.add_demographic_group("age_18_55", "Adults",   default_weight=44.4, age_range=(18, 55))
        # schema.add_demographic_group("age_56_plus","Elderly", default_weight=22.3, age_range=(56, 120))

    def __init__(self, config):
        super().__init__(config)
        # Add any model-specific initialisation here (e.g. temperature).

    # --- Optional: your own data ---
    # To use a data file, declare it in a datasets.yaml next to this model and
    # read it with self.dataset(name). The same call works locally and in the
    # cloud — never build the path by hand.
    #
    #   # datasets.yaml
    #   datasets:
    #     - name: my-contact-matrix
    #       version: "1"
    #       file: data/contacts.csv
    #
    #   import pandas as pd
    #   contacts = pd.read_csv(self.dataset("my-contact-matrix"))
    #
    # Upload it once with `python -m compartment.datasets push`, and see
    # docs/guides/adding-datasets.md. Limit: 500 MB per dataset.

    # --- Optional: spatial travel support ---
    # The framework calls this before prepare_initial_state() and stores the
    # result on self.travel_matrix. The default returns the identity matrix,
    # so only override it if your model has inter-zone mobility.
    #
    # def build_travel_matrix(self, admin_zones):
    #     # PERCENTAGE params arrive as 20.0, not 0.2 — convert first.
    #     sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
    #     return get_gravity_model_travel_matrix(admin_zones, sigma)

    @classmethod
    def _add_total_compartments(cls, schema):
        """Suppress the framework's automatic per-edge ``_total`` compartments.

        Overrides the base behavior so no per-target cumulative compartments
        are auto-added; this model declares its own aggregate ``I_total`` and
        ``R_total`` instead.

        Args:
            schema (ModelParameterSchema): Model schema, intentionally left
                unchanged.
        """
        pass

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """Seed the initial infected across the two infectious compartments.

        The base implementation seeds a single ``I`` compartment; this model
        seeds two infectious compartments: U (unreported and/or unhospitalized)
        and I (reported and/or hospitalized)

        Args:
            admin_zones (list[dict]): Admin zone dicts providing ``population``
                and the ``infected_population`` percentage.
            compartment_list (list[str]): Ordered compartment names, used for
                column indexing.
            **kwargs (Any): Additional keyword arguments (unused).

        Returns:
            np.ndarray: Initial populations of shape
                ``(n_zones, n_compartments)``.
        """
        column_mapping = {value: index for index, value in enumerate(compartment_list)}
        initial_population = np.zeros((len(admin_zones), len(compartment_list)))

        for i, zone in enumerate(admin_zones):
            ## initial infections specified in config are unreported
            unreported = round(zone["infected_population"] / 100 * zone["population"], 2)
            ## simulation starts when 1 case is reported
            reported = 1
            susceptible = zone["population"] - (unreported + reported)
            initial_population[i, column_mapping["S"]] = susceptible
            initial_population[i, column_mapping["Iu"]] = unreported
            initial_population[i, column_mapping["Id"]] = reported

        return initial_population


    def prepare_initial_state(self):
        return self.population_matrix

    def equation(self, y, t, p):
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(self.compartment_list)}
        S = states[C.S]
        E = states[C.E]
        Iu = states[C.Iu]
        Id = states[C.Id]
        Level = states[C.Level]
        Slope = states[C.Slope]  

        non_total = [c for c in C if not c.endswith("_total")]
        N = sum(states[c] for c in non_total)
        prop_infective = (Iu.sum() + Id.sum()) / (N.sum() + 1e-10)

        # _apply_interventions scales target_rates and returns the updated travel
        # matrix. With no interventions configured it returns both unchanged.
        rates, self.travel_matrix = self._apply_interventions(
            t, {"beta": params["beta"], "beta_u": params["beta_u"]}, prop_infective
        )

        sens = self.sens
        ## hardcode the projection timeframe for now (has same effect as sens, avoid an extra param)
        lookahead = 3
        proj = jnp.sum(jnp.exp(Level + Slope * jnp.arange(1.0, 1.0+lookahead)))
        p_infection = proj / N
        cr = jnp.exp(-1.0 * sens * p_infection)
        beta = cr*rates["beta"]
        beta_u = cr*rates["beta_u"]

        mu = params["mu"]
        gamma = params["gamma"]
        gamma_u = params["gamma_u"]
        r = self.lrLevel
        m = self.lrSlope
        thInc = self.thInc
        thDec = self.thDec
        thMax = self.thMax
        thLow = self.thLow
        ## theta (detection or reporting or hospitalization fraction)
        ## time-varying function of th* params
        theta =  (2.0 * thMax / (1.0 + jnp.exp(-t * thInc))) - (2.0*(thMax-thLow) / (1.0 + jnp.exp(-t * thDec))) - thLow

        derivs = {c: jnp.zeros_like(S) for c in self.compartment_list}

        derivs[C.S] = -S*Iu*beta_u/N - S*Id*beta/N
        derivs[C.E] = S*Iu*beta_u/N + S*Id*beta/N - E*mu
        new_Iu = E*mu*(1-theta)
        new_Id = E*mu*theta
        derivs[C.Iu] = new_Iu - Iu*gamma_u
        derivs[C.Id] = new_Id - Id*gamma
        derivs[C.R] = Iu*gamma_u + Id*gamma

        ## perception of level and slope of reported (or hospitalized) cases
        ## on log scale
        ## updated by double exponential smoothing
        ##  ds/dt = b(t) + r*(x(t) - s(t))
        ## where x(t) = log(incident reported or hospitalized cases)
        ##  db/dt =  m*(ds/dt - b(t)) = m*r*(x(t) - s(t))
        delta = (jnp.log(1.0+new_Id) - Level)
        derivs[C.Level] = Slope + r*delta
        derivs[C.Slope] = m*r*delta

        ## put Iu + Id in "I_total"
        derivs["I_total"] = new_Iu + new_Id
        ## track these too
        derivs["R_total"] = derivs[C.R]
        derivs["Iu_total"] = new_Iu
        derivs["Id_total"] = new_Id

        return jnp.stack([derivs[c] for c in self.compartment_list])
