import jax.numpy as np
import numpy as onp
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

class KlebsiellaAmrModel(Model):
    """
    Compartmental model for multidrug-resistant Klebsiella pneumoniae.

    Based on Kachalov et al. (2021) "Identifying the drivers of multidrug-
    resistant Klebsiella pneumoniae at a European level", PLoS Computational
    Biology 17(1): e1008446.

    Models the spread of ESBL-producing and carbapenem-resistant K. pneumoniae
    across hospital and community settings with antibiotic treatment
    stratification.

    Three dimensions:
    - Setting: Community (c) / Hospital (h)
    - Colonization: Susceptible (S) / Colonized-WT (CW) / Colonized-ESBL (CE)
      / Colonized-CRK (CR) / Infected (I) [hospital only]
    - Treatment: Untreated (u) / Drug A cephalosporins (a) / Drug B
      carbapenems (b)

    27 core compartments + 2 cumulative tracking compartments.
    """

    DISEASE_TYPE = "KLEBSIELLA_AMR"

    # Compartment group look-ups for clean iteration in derivative().
    # Keys: (strain, setting) -> (untreated, drug_a, drug_b) compartment IDs
    _GROUPS = {
        ("S", "c"): ("Sc_u", "Sc_a", "Sc_b"),
        ("CW", "c"): ("CWc_u", "CWc_a", "CWc_b"),
        ("CE", "c"): ("CEc_u", "CEc_a", "CEc_b"),
        ("CR", "c"): ("CRc_u", "CRc_a", "CRc_b"),
        ("S", "h"): ("Sh_u", "Sh_a", "Sh_b"),
        ("CW", "h"): ("CWh_u", "CWh_a", "CWh_b"),
        ("CE", "h"): ("CEh_u", "CEh_a", "CEh_b"),
        ("CR", "h"): ("CRh_u", "CRh_a", "CRh_b"),
    }

    # Output grouping: aggregate treatment sub-states for display
    COMPARTMENT_DELTA_GROUPING = {
        "Susceptible Community": ["Sc_u", "Sc_a", "Sc_b"],
        "Susceptible Hospital": ["Sh_u", "Sh_a", "Sh_b"],
        "Colonized WT Community": ["CWc_u", "CWc_a", "CWc_b"],
        "Colonized WT Hospital": ["CWh_u", "CWh_a", "CWh_b"],
        "Colonized ESBL Community": ["CEc_u", "CEc_a", "CEc_b"],
        "Colonized ESBL Hospital": ["CEh_u", "CEh_a", "CEh_b"],
        "Colonized CRK Community": ["CRc_u", "CRc_a", "CRc_b"],
        "Colonized CRK Hospital": ["CRh_u", "CRh_a", "CRh_b"],
        "Infected WT": ["IW_h"],
        "Infected ESBL": ["IE_h"],
        "Infected CRK": ["IR_h"],
        "ESBL Infections Total": ["IE_total"],
        "CRK Infections Total": ["IR_total"],
    }

    # ------------------------------------------------------------------
    # Declarative parameter schema
    # ------------------------------------------------------------------

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="KLEBSIELLA_AMR",
            label="Klebsiella pneumoniae AMR",
            description=(
                "Compartmental model for ESBL and carbapenem-resistant "
                "K. pneumoniae transmission across hospital and community "
                "settings (Kachalov et al. 2021)"
            ),
        )

        # ---- Community compartments ----
        schema.add_compartment(
            "Sc_u", "Susceptible Community",
            "Susceptible, untreated, in community",
        )
        schema.add_compartment(
            "Sc_a", "Susceptible Community Drug A",
            "Susceptible, on cephalosporins, in community",
        )
        schema.add_compartment(
            "Sc_b", "Susceptible Community Drug B",
            "Susceptible, on carbapenems, in community",
        )

        schema.add_compartment(
            "CWc_u", "Colonized WT Community",
            "Colonized with wild-type K. pneumoniae, untreated, community",
        )
        schema.add_compartment(
            "CWc_a", "Colonized WT Community Drug A",
            "Colonized WT, on cephalosporins, community",
        )
        schema.add_compartment(
            "CWc_b", "Colonized WT Community Drug B",
            "Colonized WT, on carbapenems, community",
        )

        schema.add_compartment(
            "CEc_u", "Colonized ESBL Community",
            "Colonized with ESBL-producing K. pneumoniae, untreated, community",
        )
        schema.add_compartment(
            "CEc_a", "Colonized ESBL Community Drug A",
            "Colonized ESBL, on cephalosporins, community",
        )
        schema.add_compartment(
            "CEc_b", "Colonized ESBL Community Drug B",
            "Colonized ESBL, on carbapenems, community",
        )

        schema.add_compartment(
            "CRc_u", "Colonized CRK Community",
            "Colonized with carbapenem-resistant K. pneumoniae, untreated, community",
        )
        schema.add_compartment(
            "CRc_a", "Colonized CRK Community Drug A",
            "Colonized CRK, on cephalosporins, community",
        )
        schema.add_compartment(
            "CRc_b", "Colonized CRK Community Drug B",
            "Colonized CRK, on carbapenems, community",
        )

        # ---- Hospital compartments ----
        schema.add_compartment(
            "Sh_u", "Susceptible Hospital",
            "Susceptible, untreated, in hospital",
        )
        schema.add_compartment(
            "Sh_a", "Susceptible Hospital Drug A",
            "Susceptible, on cephalosporins, in hospital",
        )
        schema.add_compartment(
            "Sh_b", "Susceptible Hospital Drug B",
            "Susceptible, on carbapenems, in hospital",
        )

        schema.add_compartment(
            "CWh_u", "Colonized WT Hospital",
            "Colonized WT, untreated, in hospital",
        )
        schema.add_compartment(
            "CWh_a", "Colonized WT Hospital Drug A",
            "Colonized WT, on cephalosporins, in hospital",
        )
        schema.add_compartment(
            "CWh_b", "Colonized WT Hospital Drug B",
            "Colonized WT, on carbapenems, in hospital",
        )

        schema.add_compartment(
            "CEh_u", "Colonized ESBL Hospital",
            "Colonized ESBL, untreated, in hospital",
        )
        schema.add_compartment(
            "CEh_a", "Colonized ESBL Hospital Drug A",
            "Colonized ESBL, on cephalosporins, in hospital",
        )
        schema.add_compartment(
            "CEh_b", "Colonized ESBL Hospital Drug B",
            "Colonized ESBL, on carbapenems, in hospital",
        )

        schema.add_compartment(
            "CRh_u", "Colonized CRK Hospital",
            "Colonized CRK, untreated, in hospital",
        )
        schema.add_compartment(
            "CRh_a", "Colonized CRK Hospital Drug A",
            "Colonized CRK, on cephalosporins, in hospital",
        )
        schema.add_compartment(
            "CRh_b", "Colonized CRK Hospital Drug B",
            "Colonized CRK, on carbapenems, in hospital",
        )

        # ---- Infected (hospital only, no treatment sub-states) ----
        schema.add_compartment(
            "IW_h", "Infected WT",
            "Symptomatic bloodstream infection with wild-type K. pneumoniae",
            infective=True,
        )
        schema.add_compartment(
            "IE_h", "Infected ESBL",
            "Symptomatic bloodstream infection with ESBL K. pneumoniae",
            infective=True,
        )
        schema.add_compartment(
            "IR_h", "Infected CRK",
            "Symptomatic bloodstream infection with carbapenem-resistant K. pneumoniae",
            infective=True,
        )

        # ---- Cumulative tracking ----
        schema.add_compartment(
            "IE_total", "ESBL Infections Total",
            "Cumulative ESBL bloodstream infections",
        )
        schema.add_compartment(
            "IR_total", "CRK Infections Total",
            "Cumulative CRK bloodstream infections",
        )

        # ---- Transmission edges (user-configurable) ----
        # All dynamics are computed manually in derivative(); these edges
        # expose the key rates to the UI as configurable parameters.

        schema.add_transmission_edge(
            source="Sc_u",
            target="CWc_u",
            variable_name="beta",
            frequency_dependent=True,
            label="Colonization Rate",
            description=(
                "Base community colonization rate for K. pneumoniae "
                "via contact with colonized individuals"
            ),
            default=0.0098,
            default_min=0.005,
            default_max=0.02,
            min_value=0.001,
            max_value=0.1,
            unit="per day",
        )

        schema.add_transmission_edge(
            source="CWc_u",
            target="Sc_u",
            variable_name="clearance",
            label="Natural Clearance Period",
            description="Average duration of K. pneumoniae colonization before natural decolonization",
            default=60.0,
            default_min=30.0,
            default_max=120.0,
            min_value=14.0,
            max_value=365.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

        schema.add_transmission_edge(
            source="CWh_u",
            target="IW_h",
            variable_name="infection_rate",
            label="Infection Development Period",
            description=(
                "Average days for a colonized individual to develop "
                "symptomatic bloodstream infection (hospital only)"
            ),
            default=500.0,
            default_min=200.0,
            default_max=1000.0,
            min_value=50.0,
            max_value=5000.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

        schema.add_transmission_edge(
            source="IW_h",
            target="Sh_u",
            variable_name="recovery_rate",
            label="Recovery Period",
            description="Average duration of K. pneumoniae bloodstream infection",
            default=14.0,
            default_min=7.0,
            default_max=28.0,
            min_value=3.0,
            max_value=60.0,
            unit="days",
            value_type=ValueType.DAYS,
        )

        # ---- Disease-specific parameters ----

        schema.add_disease_parameter(
            name="hospital_transmission_mult",
            label="Hospital Transmission Multiplier",
            description=(
                "Factor by which nosocomial transmission exceeds "
                "community rate.  Varies by country: ~0.4 (Finland) "
                "to ~33 (Greece) in Kachalov et al."
            ),
            value_type=ValueType.FLOAT,
            default=14.0,
            default_min=5.0,
            default_max=30.0,
            min_value=0.1,
            max_value=50.0,
        )

        schema.add_disease_parameter(
            name="fitness_cost_esbl",
            label="Fitness Cost ESBL (%)",
            description="Relative fitness cost of ESBL strains vs. wild-type",
            value_type=ValueType.PERCENTAGE,
            default=1.92,
            default_min=0.5,
            default_max=5.0,
            min_value=0.0,
            max_value=20.0,
            unit="%",
        )

        schema.add_disease_parameter(
            name="fitness_cost_crk",
            label="Fitness Cost CRK (%)",
            description="Relative fitness cost of carbapenem-resistant strains vs. wild-type",
            value_type=ValueType.PERCENTAGE,
            default=2.25,
            default_min=0.5,
            default_max=5.0,
            min_value=0.0,
            max_value=20.0,
            unit="%",
        )

        schema.add_disease_parameter(
            name="super_colonization_coeff",
            label="Super-colonization Coefficient",
            description=(
                "Rate of super-colonization with horizontal gene transfer "
                "relative to primary colonization"
            ),
            value_type=ValueType.FLOAT,
            default=0.177,
            default_min=0.01,
            default_max=0.5,
            min_value=0.0,
            max_value=1.0,
        )

        schema.add_disease_parameter(
            name="treatment_susceptibility",
            label="Treatment Susceptibility Increase",
            description=(
                "Factor by which antibiotic treatment increases "
                "susceptibility to colonization"
            ),
            value_type=ValueType.FLOAT,
            default=0.10,
            default_min=0.01,
            default_max=1.0,
            min_value=0.0,
            max_value=5.0,
        )

        schema.add_disease_parameter(
            name="treatment_clearance_mult",
            label="Treatment Clearance Multiplier",
            description=(
                "How many times faster effective antibiotic treatment "
                "clears susceptible colonizing strains vs. natural clearance"
            ),
            value_type=ValueType.FLOAT,
            default=3.0,
            default_min=1.0,
            default_max=10.0,
            min_value=0.0,
            max_value=20.0,
        )

        schema.add_disease_parameter(
            name="plasmid_loss_rate",
            label="Plasmid Loss Rate",
            description=(
                "Rate of resistance plasmid loss/displacement, "
                "expressed relative to natural decolonization rate"
            ),
            value_type=ValueType.FLOAT,
            default=0.044,
            default_min=0.001,
            default_max=0.2,
            min_value=0.0,
            max_value=1.0,
        )

        schema.add_disease_parameter(
            name="import_esbl",
            label="ESBL Import (per 100k)",
            description=(
                "External introduction of ESBL strains as equivalent "
                "reservoir size per 100,000 population"
            ),
            value_type=ValueType.FLOAT,
            default=326.1,
            default_min=10.0,
            default_max=1000.0,
            min_value=0.0,
            max_value=5000.0,
        )

        schema.add_disease_parameter(
            name="import_crk",
            label="CRK Import (per 100k)",
            description=(
                "External introduction of CRK strains as equivalent "
                "reservoir size per 100,000 population"
            ),
            value_type=ValueType.FLOAT,
            default=5.6,
            default_min=0.1,
            default_max=50.0,
            min_value=0.0,
            max_value=1000.0,
        )

        schema.add_disease_parameter(
            name="admission_rate",
            label="Hospital Admission Rate",
            description="Daily fraction of community population admitted to hospital",
            value_type=ValueType.FLOAT,
            default=0.000274,
            default_min=0.0001,
            default_max=0.001,
            min_value=0.0,
            max_value=0.01,
            unit="per day",
        )

        schema.add_disease_parameter(
            name="discharge_rate",
            label="Hospital Discharge Rate",
            description="Daily fraction of hospital population discharged (1/mean LOS)",
            value_type=ValueType.FLOAT,
            default=0.1,
            default_min=0.05,
            default_max=0.2,
            min_value=0.01,
            max_value=1.0,
            unit="per day",
        )

        schema.add_disease_parameter(
            name="ceph_consumption_community",
            label="Community Cephalosporin Consumption",
            description=(
                "3rd/4th gen cephalosporin consumption in community "
                "(DDD per 1,000 inhabitants per day)"
            ),
            value_type=ValueType.FLOAT,
            default=1.0,
            default_min=0.2,
            default_max=2.5,
            min_value=0.0,
            max_value=5.0,
            unit="DDD/1000/day",
        )

        schema.add_disease_parameter(
            name="ceph_consumption_hospital",
            label="Hospital Cephalosporin Consumption",
            description=(
                "3rd/4th gen cephalosporin consumption in hospital "
                "(DDD per 1,000 patient-days)"
            ),
            value_type=ValueType.FLOAT,
            default=0.15,
            default_min=0.05,
            default_max=0.30,
            min_value=0.0,
            max_value=1.0,
            unit="DDD/1000/day",
        )

        schema.add_disease_parameter(
            name="carb_consumption_hospital",
            label="Hospital Carbapenem Consumption",
            description=(
                "Carbapenem consumption in hospital "
                "(DDD per 1,000 patient-days)"
            ),
            value_type=ValueType.FLOAT,
            default=0.08,
            default_min=0.02,
            default_max=0.25,
            min_value=0.0,
            max_value=0.5,
            unit="DDD/1000/day",
        )

        schema.add_disease_parameter(
            name="treatment_duration",
            label="Treatment Duration",
            description="Average duration of an antibiotic treatment course",
            value_type=ValueType.FLOAT,
            default=10.0,
            default_min=5.0,
            default_max=14.0,
            min_value=1.0,
            max_value=30.0,
            unit="days",
        )

        # ---- Interventions ----
        schema.add_intervention(
            id="antibiotic_stewardship",
            label="Antibiotic Stewardship",
            description=(
                "Hospital antibiotic stewardship program reducing "
                "inpatient antibiotic consumption"
            ),
            adherence=80.0,
            transmission_reduction=30.0,
        )

        schema.add_intervention(
            id="infection_control",
            label="Hospital Infection Control",
            description=(
                "Enhanced infection prevention and control measures "
                "reducing nosocomial transmission rate"
            ),
            adherence=90.0,
            transmission_reduction=50.0,
        )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def __init__(self, input):
        """Initialize the Klebsiella AMR model with a configuration dict."""
        self.population_matrix = np.array(input["initial_population"]).T
        self.compartment_list = list(self.COMPARTMENTS)

        # Transmission params: self.beta, self.clearance, etc.
        self._load_transmission_params(input.get("transmission_dict", {}))

        if self.beta is None:
            self.beta = 0.0098
        if self.clearance is None:
            self.clearance = 1.0 / 60.0
        if self.infection_rate is None:
            self.infection_rate = 1.0 / 500.0
        if self.recovery_rate is None:
            self.recovery_rate = 1.0 / 14.0

        # Disease-specific parameters
        dp = input.get("Disease", {})
        self.hospital_mult = float(dp.get("hospital_transmission_mult", 14.0))
        self.fitness_esbl = float(dp.get("fitness_cost_esbl", 1.92)) / 100.0
        self.fitness_crk = float(dp.get("fitness_cost_crk", 2.25)) / 100.0
        self.super_col = float(dp.get("super_colonization_coeff", 0.177))
        self.treat_suscept = float(dp.get("treatment_susceptibility", 0.10))
        self.treat_clear_mult = float(dp.get("treatment_clearance_mult", 3.0))
        self.plasmid_loss = float(dp.get("plasmid_loss_rate", 0.044))
        self.import_esbl = float(dp.get("import_esbl", 326.1)) / 100_000.0
        self.import_crk = float(dp.get("import_crk", 5.6)) / 100_000.0
        self.admission = float(dp.get("admission_rate", 0.000274))
        self.discharge = float(dp.get("discharge_rate", 0.1))

        # Consumption → treatment start rates (DDD/1000/day → per capita/day)
        self.tau_a_c = float(dp.get("ceph_consumption_community", 1.0)) / 1000.0
        self.tau_a_h = float(dp.get("ceph_consumption_hospital", 0.15)) / 1000.0
        self.tau_b_h = float(dp.get("carb_consumption_hospital", 0.08)) / 1000.0
        self.tau_b_c = 0.0  # carbapenems rarely used in community outpatient
        treat_dur = float(dp.get("treatment_duration", 10.0))
        self.tau_stop = 1.0 / treat_dur if treat_dur > 0 else 0.1

        # Simulation parameters
        self.start_date = input["start_date"]
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = input["time_steps"]
        self.admin_units = input["admin_units"]

        # Interventions
        self.intervention_dict = input.get("intervention_dict", {})
        self.intervention_statuses = {
            "antibiotic_stewardship": False,
            "infection_control": False,
        }

        self.payload = input

    # ------------------------------------------------------------------
    # Initial population
    # ------------------------------------------------------------------

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """Compute initial population across hospital/community and colonization states."""
        col = {v: i for i, v in enumerate(compartment_list)}
        n_zones = len(admin_zones)
        pop = onp.zeros((n_zones, len(compartment_list)))

        for z in range(n_zones):
            zone = admin_zones[z]
            N = zone["population"]
            # infected_population pct → initial colonization prevalence
            col_pct = max(zone.get("infected_population", 5.0), 1.0) / 100.0

            hosp_frac = 0.005  # 0.5% of population in hospital
            N_c = N * (1.0 - hosp_frac)
            N_h = N * hosp_frac

            # ---- Community ----
            col_c = N_c * col_pct
            sus_c = N_c - col_c

            # Susceptible (95% untreated, 4% Drug A, 1% Drug B)
            pop[z, col["Sc_u"]] = sus_c * 0.95
            pop[z, col["Sc_a"]] = sus_c * 0.04
            pop[z, col["Sc_b"]] = sus_c * 0.01

            # Colonized WT (80% of colonized)
            cw = col_c * 0.80
            pop[z, col["CWc_u"]] = cw * 0.90
            pop[z, col["CWc_a"]] = cw * 0.08
            pop[z, col["CWc_b"]] = cw * 0.02

            # Colonized ESBL (15% of colonized)
            ce = col_c * 0.15
            pop[z, col["CEc_u"]] = ce * 0.90
            pop[z, col["CEc_a"]] = ce * 0.08
            pop[z, col["CEc_b"]] = ce * 0.02

            # Colonized CRK (5% of colonized)
            cr = col_c * 0.05
            pop[z, col["CRc_u"]] = cr * 0.90
            pop[z, col["CRc_a"]] = cr * 0.08
            pop[z, col["CRc_b"]] = cr * 0.02

            # ---- Hospital ----
            col_h_frac = min(col_pct * 2.0, 0.30)  # higher colonization
            col_h = N_h * col_h_frac
            sus_h = N_h - col_h

            # Susceptible (higher treatment rates in hospital)
            pop[z, col["Sh_u"]] = sus_h * 0.60
            pop[z, col["Sh_a"]] = sus_h * 0.25
            pop[z, col["Sh_b"]] = sus_h * 0.15

            # Colonized WT (70%)
            hw = col_h * 0.70
            pop[z, col["CWh_u"]] = hw * 0.40
            pop[z, col["CWh_a"]] = hw * 0.35
            pop[z, col["CWh_b"]] = hw * 0.25

            # Colonized ESBL (20%)
            he = col_h * 0.20
            pop[z, col["CEh_u"]] = he * 0.40
            pop[z, col["CEh_a"]] = he * 0.35
            pop[z, col["CEh_b"]] = he * 0.25

            # Colonized CRK (10%)
            hr = col_h * 0.10
            pop[z, col["CRh_u"]] = hr * 0.40
            pop[z, col["CRh_a"]] = hr * 0.35
            pop[z, col["CRh_b"]] = hr * 0.25

            # Small seed of infected cases
            pop[z, col["IW_h"]] = max(N_h * 0.001, 1.0)
            pop[z, col["IE_h"]] = max(N_h * 0.0003, 0.5)
            pop[z, col["IR_h"]] = max(N_h * 0.0001, 0.1)

        return pop

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    def prepare_initial_state(self):
        # No spatial travel matrix — hospital/community coupling is
        # modelled internally via admission/discharge flows.
        self.travel_matrix = np.eye(self.population_matrix.shape[1])
        return (
            self.population_matrix,
            list(self.compartment_list),
        )

    # ------------------------------------------------------------------
    # Intervention helpers
    # ------------------------------------------------------------------

    def _intervention_multiplier(self, cfg, t):
        """Return JAX-traced multiplier (0–1) for an intervention config."""
        if cfg is None:
            return 1.0

        adh = cfg.get("adherence_min", 0.0)
        red = cfg.get("transmission_percentage", 0.0)
        current_ord = self.start_date_ordinal + t

        start_ord = cfg.get("start_date_ordinal")
        if start_ord is None:
            return 1.0

        end_ord = cfg.get("end_date_ordinal")
        if end_ord is not None:
            in_window = np.logical_and(
                current_ord >= start_ord, current_ord <= end_ord
            )
        else:
            in_window = current_ord >= start_ord

        return np.where(in_window, 1.0 - adh * red, 1.0)

    # ------------------------------------------------------------------
    # ODE derivative
    # ------------------------------------------------------------------

    def derivative(self, y, t, p):
        C = self.COMPARTMENTS
        params = self._unpack_params(p)
        states = {c: y[i] for i, c in enumerate(C)}

        # --- Unpack rates ---
        beta_c = params["beta"]
        mu = params["clearance"]       # natural clearance rate (1/days)
        alpha = params["infection_rate"]  # colonised → infected (1/days)
        rho = params["recovery_rate"]  # infected → recovered (1/days)

        f_e = self.fitness_esbl
        f_r = self.fitness_crk
        kappa = self.super_col
        xi = self.treat_suscept
        mu_drug = self.treat_clear_mult * mu  # treatment-enhanced clearance
        eps = self.plasmid_loss * mu          # plasmid loss rate
        imp_e = self.import_esbl
        imp_r = self.import_crk

        h_adm = self.admission
        h_dis = self.discharge

        tau_a_c = self.tau_a_c
        tau_b_c = self.tau_b_c
        tau_a_h = self.tau_a_h
        tau_b_h = self.tau_b_h
        tau_stop = self.tau_stop

        # --- Apply interventions ---
        ic_mult = self._intervention_multiplier(
            self.intervention_dict.get("infection_control"), t
        )
        steward_mult = self._intervention_multiplier(
            self.intervention_dict.get("antibiotic_stewardship"), t
        )

        beta_h = beta_c * self.hospital_mult * ic_mult
        tau_a_h = tau_a_h * steward_mult
        tau_b_h = tau_b_h * steward_mult

        # --- Setting-level aggregates ---
        G = self._GROUPS

        def _sum3(key):
            u, a, b = G[key]
            return states[u] + states[a] + states[b]

        S_c = _sum3(("S", "c"))
        CW_c = _sum3(("CW", "c"))
        CE_c = _sum3(("CE", "c"))
        CR_c = _sum3(("CR", "c"))
        N_c = S_c + CW_c + CE_c + CR_c

        S_h = _sum3(("S", "h"))
        CW_h = _sum3(("CW", "h"))
        CE_h = _sum3(("CE", "h"))
        CR_h = _sum3(("CR", "h"))
        I_h = states[C.IW_h] + states[C.IE_h] + states[C.IR_h]
        N_h = S_h + CW_h + CE_h + CR_h + I_h

        # --- Force of colonization ---
        # Transmission FOI (within-population contact)
        foi_wt_c = beta_c * CW_c / (N_c + 1e-10)
        foi_esbl_trans_c = beta_c * (1.0 - f_e) * CE_c / (N_c + 1e-10)
        foi_crk_trans_c = beta_c * (1.0 - f_r) * CR_c / (N_c + 1e-10)

        foi_wt_h = beta_h * CW_h / (N_h + 1e-10)
        foi_esbl_trans_h = beta_h * (1.0 - f_e) * CE_h / (N_h + 1e-10)
        foi_crk_trans_h = beta_h * (1.0 - f_r) * CR_h / (N_h + 1e-10)

        # Import FOI (external reservoir, added to both settings)
        foi_esbl_import = beta_c * imp_e
        foi_crk_import = beta_c * imp_r

        # Total primary colonization FOI
        foi_esbl_c = foi_esbl_trans_c + foi_esbl_import
        foi_crk_c = foi_crk_trans_c + foi_crk_import
        foi_esbl_h = foi_esbl_trans_h + foi_esbl_import
        foi_crk_h = foi_crk_trans_h + foi_crk_import

        # --- Initialise derivatives ---
        zero = np.zeros_like(states[C.Sc_u])
        derivs = {c: zero for c in C}

        # ==============================================================
        # Process both settings
        # ==============================================================
        settings = [
            # (setting, groups, tau_a, tau_b, beta_s,
            #  foi_wt, foi_esbl, foi_crk, foi_esbl_trans, foi_crk_trans)
            ("c", G, tau_a_c, tau_b_c, beta_c,
             foi_wt_c, foi_esbl_c, foi_crk_c,
             foi_esbl_trans_c, foi_crk_trans_c),
            ("h", G, tau_a_h, tau_b_h, beta_h,
             foi_wt_h, foi_esbl_h, foi_crk_h,
             foi_esbl_trans_h, foi_crk_trans_h),
        ]

        for (setting, grps, tau_a, tau_b, beta_s,
             foi_wt, foi_esbl, foi_crk,
             foi_esbl_tr, foi_crk_tr) in settings:

            s_u, s_a, s_b = grps[("S", setting)]
            cw_u, cw_a, cw_b = grps[("CW", setting)]
            ce_u, ce_a, ce_b = grps[("CE", setting)]
            cr_u, cr_a, cr_b = grps[("CR", setting)]

            # All (strain, treatment) compartment triples in this setting
            strain_comps = [
                (s_u, s_a, s_b),
                (cw_u, cw_a, cw_b),
                (ce_u, ce_a, ce_b),
                (cr_u, cr_a, cr_b),
            ]

            # ---- A. Treatment transitions ----
            for (u, a, b) in strain_comps:
                # Start Drug A
                f = tau_a * states[u]
                derivs[u] = derivs[u] - f
                derivs[a] = derivs[a] + f
                # Start Drug B
                f = tau_b * states[u]
                derivs[u] = derivs[u] - f
                derivs[b] = derivs[b] + f
                # Stop Drug A
                f = tau_stop * states[a]
                derivs[a] = derivs[a] - f
                derivs[u] = derivs[u] + f
                # Stop Drug B
                f = tau_stop * states[b]
                derivs[b] = derivs[b] - f
                derivs[u] = derivs[u] + f

            # ---- B. Colonization (S → C*) ----
            for ti, (s, cw, ce, cr) in enumerate(
                zip(
                    (s_u, s_a, s_b),
                    (cw_u, cw_a, cw_b),
                    (ce_u, ce_a, ce_b),
                    (cr_u, cr_a, cr_b),
                )
            ):
                # Treatment increases susceptibility
                smult = 1.0 + xi * (1.0 if ti > 0 else 0.0)
                pop_s = states[s]

                # S → CW
                f = smult * foi_wt * pop_s
                derivs[s] = derivs[s] - f
                derivs[cw] = derivs[cw] + f

                # S → CE
                f = smult * foi_esbl * pop_s
                derivs[s] = derivs[s] - f
                derivs[ce] = derivs[ce] + f

                # S → CR
                f = smult * foi_crk * pop_s
                derivs[s] = derivs[s] - f
                derivs[cr] = derivs[cr] + f

            # ---- C. Super-colonization + HGT ----
            for ti, (cw, ce, cr) in enumerate(
                zip(
                    (cw_u, cw_a, cw_b),
                    (ce_u, ce_a, ce_b),
                    (cr_u, cr_a, cr_b),
                )
            ):
                # CW → CE (acquire ESBL via HGT)
                f = kappa * foi_esbl_tr * states[cw]
                derivs[cw] = derivs[cw] - f
                derivs[ce] = derivs[ce] + f

                # CW → CR (acquire CRK via HGT)
                f = kappa * foi_crk_tr * states[cw]
                derivs[cw] = derivs[cw] - f
                derivs[cr] = derivs[cr] + f

                # CE → CR (acquire CRK via HGT)
                f = kappa * foi_crk_tr * states[ce]
                derivs[ce] = derivs[ce] - f
                derivs[cr] = derivs[cr] + f

            # ---- D. Decolonization (natural + treatment-enhanced) ----
            for ti, (s, cw, ce, cr) in enumerate(
                zip(
                    (s_u, s_a, s_b),
                    (cw_u, cw_a, cw_b),
                    (ce_u, ce_a, ce_b),
                    (cr_u, cr_a, cr_b),
                )
            ):
                # Natural clearance — all strains, all treatment states
                for x in (cw, ce, cr):
                    f = mu * states[x]
                    derivs[x] = derivs[x] - f
                    derivs[s] = derivs[s] + f

                # Treatment-enhanced clearance (selective pressure):
                # Drug A (cephalosporins) clears WT only (ESBL/CRK resist)
                # Drug B (carbapenems) clears WT and ESBL (CRK resists)
                if ti == 1:  # Drug A
                    f = mu_drug * states[cw]
                    derivs[cw] = derivs[cw] - f
                    derivs[s] = derivs[s] + f
                elif ti == 2:  # Drug B
                    # Clears WT
                    f = mu_drug * states[cw]
                    derivs[cw] = derivs[cw] - f
                    derivs[s] = derivs[s] + f
                    # Clears ESBL
                    f = mu_drug * states[ce]
                    derivs[ce] = derivs[ce] - f
                    derivs[s] = derivs[s] + f

            # ---- E. Plasmid loss ----
            for (cw, ce, cr) in zip(
                (cw_u, cw_a, cw_b),
                (ce_u, ce_a, ce_b),
                (cr_u, cr_a, cr_b),
            ):
                # CE → CW (lose ESBL plasmid)
                f = eps * states[ce]
                derivs[ce] = derivs[ce] - f
                derivs[cw] = derivs[cw] + f

                # CR → CE (lose carbapenem-R plasmid, retain ESBL)
                f = eps * states[cr]
                derivs[cr] = derivs[cr] - f
                derivs[ce] = derivs[ce] + f

        # ==============================================================
        # Hospital–Community flows (admission / discharge)
        # ==============================================================
        for strain in ("S", "CW", "CE", "CR"):
            for ti in range(3):
                c_comp = G[(strain, "c")][ti]
                h_comp = G[(strain, "h")][ti]

                # Admission
                f = h_adm * states[c_comp]
                derivs[c_comp] = derivs[c_comp] - f
                derivs[h_comp] = derivs[h_comp] + f

                # Discharge
                f = h_dis * states[h_comp]
                derivs[h_comp] = derivs[h_comp] - f
                derivs[c_comp] = derivs[c_comp] + f

        # ==============================================================
        # Infection development (hospital colonized → infected)
        # ==============================================================
        infected_map = {
            "CW": C.IW_h,
            "CE": C.IE_h,
            "CR": C.IR_h,
        }
        for strain, inf_comp in infected_map.items():
            for ti in range(3):
                h_comp = G[(strain, "h")][ti]
                f = alpha * states[h_comp]
                derivs[h_comp] = derivs[h_comp] - f
                derivs[inf_comp] = derivs[inf_comp] + f

                # Accumulate cumulative tracking
                if inf_comp == C.IE_h:
                    derivs[C.IE_total] = derivs[C.IE_total] + f
                elif inf_comp == C.IR_h:
                    derivs[C.IR_total] = derivs[C.IR_total] + f

        # ==============================================================
        # Recovery (infected → susceptible hospital, untreated)
        # ==============================================================
        for inf_comp in (C.IW_h, C.IE_h, C.IR_h):
            f = rho * states[inf_comp]
            derivs[inf_comp] = derivs[inf_comp] - f
            derivs[C.Sh_u] = derivs[C.Sh_u] + f

        return np.stack([derivs[c] for c in C])
