"""
ESBL / CR Klebsiella pneumoniae transmission model.

Based on: Identifying the drivers of multidrug-resistant Klebsiella pneumoniae
at a European level. PLOS Comp Bio (2021). doi:10.1371/journal.pcbi.1008446

Model structure (27 compartments per geographic region):
  Community setting, by treatment status (U/A/B):
    S_C  – susceptible
    CW_C – colonized with wild-type (WT) strain
    CE_C – colonized with ESBL strain
    CC_C – colonized with CR (carbapenem-resistant) strain
  Hospital setting, by treatment status (U/A/B):
    S_H, CW_H, CE_H, CC_H  (same four groups, 3 treatment arms each = 12)
  Infected (all in hospital, strain-specific, single treatment arm T):
    IWT_H, IET_H, ICT_H

Treatment arms:
  U – untreated
  A – treated with 3rd/4th generation cephalosporins
  B – treated with carbapenems

Resistance coefficients (section 2.3.3 of supplementary):
  r_WT_A = 0,  r_WT_B = 0   (WT susceptible to both drugs)
  r_E_A  = 1,  r_E_B  = 0   (ESBL resistant to cephalosporins, susceptible to carbapenems)
  r_CR_A = 1,  r_CR_B = 1   (CR resistant to both)

COMPARTMENT_DELTA_GROUPING aggregates 27 internal compartments into 7 output groups.
"""

import jax.numpy as np
import numpy as onp
import logging
from compartment.helpers import setup_logging
from compartment.model import Model

setup_logging()
logger = logging.getLogger(__name__)


class ESBLJaxModel(Model):
    """27-compartment ESBL/CR Klebsiella pneumoniae transmission model."""

    # fmt: off
    COMPARTMENT_LIST = [
        # Community – susceptible
        "SU_C", "SA_C", "SB_C",
        # Community – WT colonized
        "CWU_C", "CWA_C", "CWB_C",
        # Community – ESBL colonized
        "CEU_C", "CEA_C", "CEB_C",
        # Community – CR colonized
        "CCU_C", "CCA_C", "CCB_C",
        # Hospital – susceptible
        "SU_H", "SA_H", "SB_H",
        # Hospital – WT colonized
        "CWU_H", "CWA_H", "CWB_H",
        # Hospital – ESBL colonized
        "CEU_H", "CEA_H", "CEB_H",
        # Hospital – CR colonized
        "CCU_H", "CCA_H", "CCB_H",
        # Infected (hospital only, one arm each)
        "IWT_H", "IET_H", "ICT_H",
    ]
    # fmt: on

    # Aggregate community + hospital counts by strain for reporting
    COMPARTMENT_DELTA_GROUPING = {
        "S":   ["SU_C",  "SA_C",  "SB_C",  "SU_H",  "SA_H",  "SB_H"],
        "CW":  ["CWU_C", "CWA_C", "CWB_C", "CWU_H", "CWA_H", "CWB_H"],
        "CE":  ["CEU_C", "CEA_C", "CEB_C", "CEU_H", "CEA_H", "CEB_H"],
        "CC":  ["CCU_C", "CCA_C", "CCB_C", "CCU_H", "CCA_H", "CCB_H"],
        "IWT": ["IWT_H"],
        "IET": ["IET_H"],
        "ICT": ["ICT_H"],
    }

    def __init__(self, config):
        # ---- Population ----
        # initial_population shape: (n_zones, 27)
        self.population_matrix = onp.array(config["initial_population"])
        self.compartment_list = self.COMPARTMENT_LIST
        self.start_date   = config["start_date"]
        self.n_timesteps  = config["time_steps"]
        self.admin_units  = config["admin_units"]

        # ---- Disease parameters from Disease config dict ----
        d = config["Disease"]

        self.beta    = float(d.get("beta",   0.01027))
        self.R_HC    = float(d.get("R_HC",   10.0))
        self.s_ESBL  = float(d.get("s_ESBL", 0.1))
        self.s_CR    = float(d.get("s_CR",   0.2))
        self.cr      = float(d.get("cr",     3.0 / 365))
        self.dis     = float(d.get("dis",    1.5 / 365))
        self.nu      = float(d.get("nu",     0.5))
        self.mu      = float(d.get("mu",     1.0))
        self.hr      = float(d.get("hr",     1.0 / 1400))
        self.dr      = float(d.get("dr",     1.0 / 7))
        self.tau_t   = float(d.get("tau_t",  7.0))
        self.tau_r   = float(d.get("tau_r",  10.0))
        self.tau_dC  = float(d.get("tau_dC", 912500.0))
        self.tau_dH  = float(d.get("tau_dH", 912500.0))
        self.IMP_ESBL = float(d.get("IMP_ESBL", 1.0))
        self.IMP_CR   = float(d.get("IMP_CR",   0.1))

        # Antibiotic consumption (DDD / 1000 person-days)
        self.C_A_C = float(d.get("C_A_C", 1.5))
        self.C_B_C = float(d.get("C_B_C", 0.01))
        self.C_A_H = float(d.get("C_A_H", 20.0))
        self.C_B_H = float(d.get("C_B_H", 10.0))

        # Pre-computed fitness-cost ratios (paper section 1.6)
        self.ratio_CR_ESBL       = self.s_CR / self.s_ESBL          # sCR / sESBL
        self.ratio_CR_minus_ESBL = (self.s_CR - self.s_ESBL) / self.s_ESBL  # (sCR-sESBL)/sESBL

        # ---- Treatment rates (per-region arrays, shape (n_zones,)) ----
        # Derived from antibiotic consumption at quasi-equilibrium (paper section 1.3).
        pop = self.population_matrix  # (n_zones, 27)
        T_C0 = pop[:, 0:12].sum(axis=1)   # initial community population per zone
        T_H0 = pop[:, 12:27].sum(axis=1)  # initial hospital population per zone
        T0   = T_C0 + T_H0               # total population per zone

        tau = self.tau_t  # same duration for all treatment arms
        hr  = self.hr
        C_A_C, C_B_C = self.C_A_C, self.C_B_C
        C_A_H, C_B_H = self.C_A_H, self.C_B_H

        eps = 1e-10  # guard against division by zero in degenerate configs

        denom_H = 1000.0 * tau * (1000.0 * T_H0 - T0 * (C_A_H + C_B_H) + eps)
        denom_C = 1000.0 * tau * (1000.0 * T_C0 - T0 * (C_A_C + C_B_C) + eps)

        self.t_A_H = T0 * (C_A_H - hr * tau * C_A_C) / denom_H   # shape (n_zones,)
        self.t_B_H = T0 * (C_B_H - hr * tau * C_B_C) / denom_H
        self.t_A_C = T0 * C_A_C * (1.0 + hr * tau) / denom_C
        self.t_B_C = T0 * C_B_C * (1.0 + hr * tau) / denom_C

        # Clip to non-negative (can go slightly negative in extreme consumption configs)
        self.t_A_H = onp.clip(self.t_A_H, 0.0, None)
        self.t_B_H = onp.clip(self.t_B_H, 0.0, None)
        self.t_A_C = onp.clip(self.t_A_C, 0.0, None)
        self.t_B_C = onp.clip(self.t_B_C, 0.0, None)

        # ---- Interventions (not yet wired – placeholder) ----
        self.intervention_dict = config.get("intervention_dict", {})

        self.payload = config

    # ------------------------------------------------------------------
    # Class-level initial population builder
    # ------------------------------------------------------------------

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """
        Build a (n_zones, 27) initial population matrix.

        Per zone:
          - Hospital fraction  = hr / (hr + dr)
          - Community fraction = 1 - hospital_fraction
          - Community colonization split into WT, ESBL, CR per config prevalences.
          - Hospital colonization = hospital_colonization % (all WT initially).
          - All colonized individuals start in the untreated (U) arm.
          - Infected compartments start at zero.
        """
        # These parameters come from the Disease config via kwargs if provided by
        # a custom processor; otherwise fall back to defaults matching ESBLDiseaseConfig.
        hr  = kwargs.get("hr",   1.0 / 1400)
        dr  = kwargs.get("dr",   1.0 / 7)
        wt_comm_col    = kwargs.get("wt_community_colonization", 20.0) / 100.0
        hosp_col       = kwargs.get("hospital_colonization",     40.0) / 100.0

        hosp_frac = hr / (hr + dr)
        comm_frac = 1.0 - hosp_frac

        idx = {c: i for i, c in enumerate(compartment_list)}
        n_zones = len(admin_zones)
        pop = onp.zeros((n_zones, len(compartment_list)))

        for i, zone in enumerate(admin_zones):
            N = zone["population"]
            # ESBL prevalence uses infected_population field (% of total zone pop)
            esbl_prev = zone.get("infected_population", 0.0) / 100.0
            # CR prevalence (passed through seroprevalence field for convenience)
            cr_prev   = zone.get("seroprevalence", 0.0) / 100.0

            N_C = N * comm_frac
            N_H = N * hosp_frac

            # Community colonization fractions
            esbl_C  = N * esbl_prev           # ESBL-colonized in community
            cr_C    = N * cr_prev             # CR-colonized in community
            wt_C    = N_C * wt_comm_col - esbl_C - cr_C  # remainder is WT
            wt_C    = max(wt_C, 0.0)
            susc_C  = N_C - wt_C - esbl_C - cr_C
            susc_C  = max(susc_C, 0.0)

            # Hospital colonization (all WT initially)
            wt_H    = N_H * hosp_col
            susc_H  = N_H - wt_H

            # Assign to compartments (all in untreated arm)
            pop[i, idx["SU_C"]]  = susc_C
            pop[i, idx["CWU_C"]] = wt_C
            pop[i, idx["CEU_C"]] = esbl_C
            pop[i, idx["CCU_C"]] = cr_C
            pop[i, idx["SU_H"]]  = susc_H
            pop[i, idx["CWU_H"]] = wt_H

        return pop

    # ------------------------------------------------------------------

    @property
    def disease_type(self):
        return "ESBL"

    def prepare_initial_state(self):
        # population_matrix is (n_zones, 27); ODE solver expects (C, R) = (27, n_zones)
        return np.array(self.population_matrix).T, self.compartment_list

    def get_params(self):
        return (
            self.beta, self.R_HC, self.s_ESBL, self.s_CR,
            self.cr, self.dis, self.nu, self.mu,
            self.hr, self.dr, self.tau_t, self.tau_r, self.tau_dC, self.tau_dH,
            np.array(self.t_A_C), np.array(self.t_B_C),
            np.array(self.t_A_H), np.array(self.t_B_H),
            self.IMP_ESBL, self.IMP_CR,
            self.ratio_CR_ESBL, self.ratio_CR_minus_ESBL,
        )

    # ------------------------------------------------------------------
    # ODE derivative (equations 1–27 from supplementary S1)
    # ------------------------------------------------------------------

    def derivative(self, y, t, p):
        """
        Right-hand side of the 27-compartment ODE system.

        y shape: (27, R) — compartments × geographic regions.
        All operations vectorise over R automatically.
        """
        (beta, R_HC, s_ESBL, s_CR,
         cr, dis, nu, mu,
         hr, dr, tau_t, tau_r, tau_dC, tau_dH,
         t_A_C, t_B_C, t_A_H, t_B_H,
         IMP_ESBL, IMP_CR,
         ratio_CR_ESBL, ratio_CR_minus_ESBL) = p

        # Inverse rates (used throughout)
        inv_tau_t  = 1.0 / tau_t
        inv_tau_r  = 1.0 / tau_r
        inv_dC     = 1.0 / tau_dC
        inv_dH     = 1.0 / tau_dH
        inv_tau_tC = 1.0 / tau_t   # same duration for both settings
        inv_tau_tH = 1.0 / tau_t

        # Fixed resistance coefficients (section 2.3.3)
        # rWT_A=0, rWT_B=0; rE_A=1, rE_B=0; rCR_A=1, rCR_B=1
        # These collapse many terms to 0 — annotated inline below.

        eps = 1e-10

        # ---- Unpack compartments (each shape (R,)) ----
        SU_C  = y[0];  SA_C  = y[1];  SB_C  = y[2]
        CWU_C = y[3];  CWA_C = y[4];  CWB_C = y[5]
        CEU_C = y[6];  CEA_C = y[7];  CEB_C = y[8]
        CCU_C = y[9];  CCA_C = y[10]; CCB_C = y[11]
        SU_H  = y[12]; SA_H  = y[13]; SB_H  = y[14]
        CWU_H = y[15]; CWA_H = y[16]; CWB_H = y[17]
        CEU_H = y[18]; CEA_H = y[19]; CEB_H = y[20]
        CCU_H = y[21]; CCA_H = y[22]; CCB_H = y[23]
        IWT_H = y[24]; IET_H = y[25]; ICT_H = y[26]

        # ---- Total populations (section 1.1) ----
        T_C = (SU_C + SA_C + SB_C
               + CWU_C + CWA_C + CWB_C
               + CEU_C + CEA_C + CEB_C
               + CCU_C + CCA_C + CCB_C)

        T_H = (SU_H + SA_H + SB_H
               + CWU_H + CWA_H + CWB_H
               + CEU_H + CEA_H + CEB_H
               + CCU_H + CCA_H + CCB_H
               + IWT_H + IET_H + ICT_H)

        # ---- Forces of colonization (section 1.4) ----
        lam_WT_C = beta * (CWU_C + CWA_C + CWB_C) / (T_C + eps)
        lam_E_C  = beta * (1.0 - s_ESBL) * (CEU_C + CEA_C + CEB_C + IMP_ESBL) / (T_C + eps)
        lam_CR_C = beta * (1.0 - s_CR)   * (CCU_C + CCA_C + CCB_C + IMP_CR)   / (T_C + eps)

        lam_WT_H = R_HC * beta * (CWU_H + CWA_H + CWB_H + IWT_H) / (T_H + eps)
        lam_E_H  = R_HC * beta * (1.0 - s_ESBL) * (CEU_H + CEA_H + CEB_H + IET_H) / (T_H + eps)
        lam_CR_H = R_HC * beta * (1.0 - s_CR)   * (CCU_H + CCA_H + CCB_H + ICT_H) / (T_H + eps)

        # ================================================================
        # COMMUNITY COMPARTMENTS
        # ================================================================

        # ---- Eq (1): SU_C ----
        dSU_C = (
            # Hospitalisation / discharge
            - SU_C * hr + SU_H * dr
            # Treatment initiation / termination
            - SU_C * t_A_C + SA_C * inv_tau_tC
            - SU_C * t_B_C + SB_C * inv_tau_tC
            # Colonisation (all three strains)
            - (lam_WT_C + lam_E_C + lam_CR_C) * SU_C
            # Natural decolonisation of U-arm colonised → susceptible
            + (CWU_C + CEU_C + CCU_C) * cr
            # Recovery from infection
            + (IWT_H + IET_H + ICT_H) * inv_tau_r
        )

        # ---- Eq (2): SA_C ----
        # rWT_A=0 → colonisation term = 0; (1-rWT_A)=1 → full WT clearance by drug A
        # rE_A=1  → colonisation term lam_E_C*SA_C*1; (1-rE_A)=0 → no ESBL clearance by A
        # rCR_A=1 → colonisation term lam_CR_C*SA_C*1; (1-rCR_A)=0 → no CR clearance by A
        dSA_C = (
            - SA_C * hr
            + SU_C * t_A_C - SA_C * inv_tau_tC
            # Colonisation: rWT_A=0 so WT term vanishes; ESBL and CR use r=1
            - lam_E_C * SA_C - lam_CR_C * SA_C
            # Natural decolonisation: A-arm colonised → susceptible-A
            + (CWA_C + CEA_C + CCA_C) * cr
            # Decolonisation by treatment: only WT cleared (rWT_A=0 → (1-0)=1)
            + CWA_C * inv_tau_t
            # ESBL/CR not cleared by cephalosporins: (1-1)=0
        )

        # ---- Eq (3): SB_C ----
        # rWT_B=0 → colonisation = 0; (1-rWT_B)=1 → WT cleared by carbapenems
        # rE_B=0  → colonisation = 0; (1-rE_B)=1  → ESBL cleared by carbapenems
        # rCR_B=1 → colonisation lam_CR_C*SB_C; (1-rCR_B)=0 → CR not cleared by B
        dSB_C = (
            - SB_C * hr
            + SU_C * t_B_C - SB_C * inv_tau_tC
            # Colonisation: only CR (rCR_B=1); WT and ESBL have r=0
            - lam_CR_C * SB_C
            + (CWB_C + CEB_C + CCB_C) * cr
            # Decolonisation by treatment: WT and ESBL cleared (both r=0 → 1-r=1)
            + (CWB_C + CEB_C) * inv_tau_t
        )

        # ---- Eq (4): CWU_C ----
        dCWU_C = (
            - CWU_C * hr + CWU_H * dr
            - CWU_C * t_A_C + CWA_C * inv_tau_tC
            - CWU_C * t_B_C + CWB_C * inv_tau_tC
            # Primary colonisation from susceptible-U
            + lam_WT_C * SU_C
            # Loss of resistance: ESBL→WT and CR→WT
            + dis * CEU_C + dis * ratio_CR_ESBL * CCU_C
            # HGT to ESBL or CR (super-colonisation of WT by resistant strain)
            - nu * lam_E_C * CWU_C - nu * lam_CR_C * CWU_C
            # Natural decolonisation
            - CWU_C * cr
            # Disease progression to infection
            - CWU_C * inv_dC
        )

        # ---- Eq (5): CWA_C ----
        # rWT_A=0 → primary colonisation lam_WT_C*SA_C*0 = 0
        # HGT: -(mu+nu) * lam_E_C * CWA_C * rE_A  = -(mu+nu)*lam_E_C*CWA_C  (rE_A=1)
        #       -(mu+nu) * lam_CR_C * CWA_C * rCR_A = -(mu+nu)*lam_CR_C*CWA_C (rCR_A=1)
        # Decolonisation by A: -CWA_C/tau_t * (1-rWT_A) = -CWA_C/tau_t * 1
        dCWA_C = (
            - CWA_C * hr
            + CWU_C * t_A_C - CWA_C * inv_tau_tC
            # Primary colonisation = 0 (rWT_A=0)
            # HGT to resistant strains (both terms active)
            - (mu + nu) * (lam_E_C + lam_CR_C) * CWA_C
            - CWA_C * cr
            - CWA_C * inv_tau_t   # (1 - rWT_A) = 1
            - CWA_C * inv_dC
        )

        # ---- Eq (6): CWB_C ----
        # rWT_B=0 → primary colonisation = 0
        # HGT: -(mu+nu)*lam_E_C*CWB_C*rE_B = 0 (rE_B=0)
        #       -(mu+nu)*lam_CR_C*CWB_C*rCR_B = -(mu+nu)*lam_CR_C*CWB_C (rCR_B=1)
        # Decolonisation by B: -CWB_C/tau_t * (1-rWT_B) = -CWB_C/tau_t
        dCWB_C = (
            - CWB_C * hr
            + CWU_C * t_B_C - CWB_C * inv_tau_tC
            # Primary colonisation = 0 (rWT_B=0)
            # HGT to CR only (ESBL term: rE_B=0)
            - (mu + nu) * lam_CR_C * CWB_C
            - CWB_C * cr
            - CWB_C * inv_tau_t   # (1 - rWT_B) = 1
            - CWB_C * inv_dC
        )

        # ---- Eq (7): CEU_C ----
        dCEU_C = (
            - CEU_C * hr + CEU_H * dr
            - CEU_C * t_A_C + CEA_C * inv_tau_tC
            - CEU_C * t_B_C + CEB_C * inv_tau_tC
            + lam_E_C * SU_C
            # Loss of resistance: ESBL→WT; CR→ESBL
            - dis * CEU_C + dis * ratio_CR_minus_ESBL * CCU_C
            # HGT: gain from WT super-colonised; lose to CR super-colonisation
            + nu * lam_E_C * CWU_C - nu * lam_CR_C * CEU_C
            - CEU_C * cr
            - CEU_C * inv_dC
        )

        # ---- Eq (8): CEA_C ----
        # Primary colonisation: lam_E_C * SA_C * rE_A = lam_E_C * SA_C (rE_A=1)
        # Loss of resistance from CR: dis * ratio_CR_minus_ESBL * CCA_C
        # HGT gain: (mu+nu)*lam_E_C*CWA_C*rE_A = (mu+nu)*lam_E_C*CWA_C
        # HGT loss: -nu * lam_CR_C * CEA_C * rCR_A = -nu*lam_CR_C*CEA_C (rCR_A=1)
        # Decolonisation by A: -CEA_C/tau_t*(1-rE_A) = 0  (rE_A=1 → ESBL not cleared by A)
        dCEA_C = (
            - CEA_C * hr
            + CEU_C * t_A_C - CEA_C * inv_tau_tC
            + lam_E_C * SA_C                              # primary colonisation (rE_A=1)
            + dis * ratio_CR_minus_ESBL * CCA_C           # loss of CR resistance
            + (mu + nu) * lam_E_C * CWA_C                 # HGT gain
            - nu * lam_CR_C * CEA_C                       # HGT loss to CR
            - CEA_C * cr
            # Decolonisation by A = 0 (rE_A=1)
            - CEA_C * inv_dC
        )

        # ---- Eq (9): CEB_C ----
        # Primary colonisation: lam_E_C * SB_C * rE_B = 0  (rE_B=0)
        # HGT gain: (mu+nu)*lam_E_C*CWB_C*rE_B = 0
        # HGT loss: -(mu+nu)*lam_CR_C*CEB_C*rCR_B = -(mu+nu)*lam_CR_C*CEB_C (rCR_B=1)
        # Decolonisation by B: -CEB_C/tau_t*(1-rE_B) = -CEB_C/tau_t (rE_B=0 → ESBL cleared by B)
        dCEB_C = (
            - CEB_C * hr
            + CEU_C * t_B_C - CEB_C * inv_tau_tC
            # Primary colonisation = 0 (rE_B=0)
            # HGT gain = 0; HGT loss to CR:
            - (mu + nu) * lam_CR_C * CEB_C
            - CEB_C * cr
            - CEB_C * inv_tau_t    # (1 - rE_B) = 1
            - CEB_C * inv_dC
        )

        # ---- Eq (10): CCU_C ----
        # Total loss of resistance:
        #   -(sCR-sESBL)/sESBL * dis * CCU_C  → ESBL level
        #   - sCR/sESBL * dis * CCU_C          → WT level
        # HGT gain: nu * lam_CR_C * (CWU_C + CEU_C)
        dCCU_C = (
            - CCU_C * hr + CCU_H * dr
            - CCU_C * t_A_C + CCA_C * inv_tau_tC
            - CCU_C * t_B_C + CCB_C * inv_tau_tC
            + lam_CR_C * SU_C
            - dis * (ratio_CR_minus_ESBL + ratio_CR_ESBL) * CCU_C
            + nu * lam_CR_C * (CWU_C + CEU_C)
            - CCU_C * cr
            - CCU_C * inv_dC
        )

        # ---- Eq (11): CCA_C ----
        # Primary colonisation: lam_CR_C * SA_C * rCR_A = lam_CR_C*SA_C (rCR_A=1)
        # Loss of resistance: -dis * ratio_CR_minus_ESBL * CCA_C (to ESBL level)
        # HGT: (mu+nu)*lam_CR_C*CWA_C*rCR_A = (mu+nu)*lam_CR_C*CWA_C
        # Decolonisation by A: -CCA_C/tau_t*(1-rCR_A) = 0 (rCR_A=1)
        dCCA_C = (
            - CCA_C * hr
            + CCU_C * t_A_C - CCA_C * inv_tau_tC
            + lam_CR_C * SA_C
            - dis * ratio_CR_minus_ESBL * CCA_C
            + (mu + nu) * lam_CR_C * CWA_C
            - CCA_C * cr
            # Decolonisation by A = 0 (rCR_A=1)
            - CCA_C * inv_dC
        )

        # ---- Eq (12): CCB_C ----
        # Primary colonisation: lam_CR_C * SB_C * rCR_B = lam_CR_C*SB_C (rCR_B=1)
        # HGT: (mu+nu)*lam_CR_C*(CWB_C+CEB_C)*rCR_B
        # Decolonisation by B: -CCB_C/tau_t*(1-rCR_B) = 0 (rCR_B=1)
        dCCB_C = (
            - CCB_C * hr
            + CCU_C * t_B_C - CCB_C * inv_tau_tC
            + lam_CR_C * SB_C
            + (mu + nu) * lam_CR_C * (CWB_C + CEB_C)
            - CCB_C * cr
            # Decolonisation by B = 0 (rCR_B=1)
            - CCB_C * inv_dC
        )

        # ================================================================
        # HOSPITAL COMPARTMENTS
        # ================================================================

        # ---- Eq (13): SU_H ----
        dSU_H = (
            SU_C * hr - SU_H * dr
            - SU_H * t_A_H + SA_H * inv_tau_tH
            - SU_H * t_B_H + SB_H * inv_tau_tH
            - (lam_WT_H + lam_E_H + lam_CR_H) * SU_H
            + (CWU_H + CEU_H + CCU_H) * cr
        )

        # ---- Eq (14): SA_H ----
        # rWT_A=0 → colonisation terms for WT = 0; ESBL and CR use r=1
        # Decolonisation by A: WT cleared (rWT_A=0 → 1-r=1), ESBL/CR not (r=1 → 1-r=0)
        dSA_H = (
            SA_C * hr
            + SU_H * t_A_H - SA_H * inv_tau_tH
            - (lam_E_H + lam_CR_H) * SA_H
            + (CWA_H + CEA_H + CCA_H) * cr
            + CWA_H * inv_tau_t
        )

        # ---- Eq (15): SB_H ----
        # rWT_B=0, rE_B=0 → only CR colonisation active; WT and ESBL cleared by B
        dSB_H = (
            SB_C * hr
            + SU_H * t_B_H - SB_H * inv_tau_tH
            - lam_CR_H * SB_H
            + (CWB_H + CEB_H + CCB_H) * cr
            + (CWB_H + CEB_H) * inv_tau_t
        )

        # ---- Eq (16): CWU_H ----
        dCWU_H = (
            CWU_C * hr - CWU_H * dr
            - CWU_H * t_A_H + CWA_H * inv_tau_tH
            - CWU_H * t_B_H + CWB_H * inv_tau_tH
            + lam_WT_H * SU_H
            + dis * CEU_H + dis * ratio_CR_ESBL * CCU_H
            - nu * (lam_E_H + lam_CR_H) * CWU_H
            - CWU_H * cr
            - CWU_H * inv_dH
        )

        # ---- Eq (17): CWA_H ----
        dCWA_H = (
            CWA_C * hr
            + CWU_H * t_A_H - CWA_H * inv_tau_tH
            # Primary colonisation: rWT_A=0 → 0
            - (mu + nu) * (lam_E_H + lam_CR_H) * CWA_H
            - CWA_H * cr
            - CWA_H * inv_tau_t
            - CWA_H * inv_dH
        )

        # ---- Eq (18): CWB_H ----
        dCWB_H = (
            CWB_C * hr
            + CWU_H * t_B_H - CWB_H * inv_tau_tH
            # Primary colonisation: rWT_B=0 → 0
            # HGT to ESBL: rE_B=0 → 0; HGT to CR: rCR_B=1
            - (mu + nu) * lam_CR_H * CWB_H
            - CWB_H * cr
            - CWB_H * inv_tau_t
            - CWB_H * inv_dH
        )

        # ---- Eq (19): CEU_H ----
        dCEU_H = (
            CEU_C * hr - CEU_H * dr
            - CEU_H * t_A_H + CEA_H * inv_tau_tH
            - CEU_H * t_B_H + CEB_H * inv_tau_tH
            + lam_E_H * SU_H
            - dis * CEU_H + dis * ratio_CR_minus_ESBL * CCU_H
            + nu * lam_E_H * CWU_H - nu * lam_CR_H * CEU_H
            - CEU_H * cr
            - CEU_H * inv_dH
        )

        # ---- Eq (20): CEA_H ----
        # Primary colonisation: lam_E_H * SA_H * rE_A = lam_E_H * SA_H
        # Decolonisation: (1-rE_A)=0 → not cleared by A
        dCEA_H = (
            CEA_C * hr
            + CEU_H * t_A_H - CEA_H * inv_tau_tH
            + lam_E_H * SA_H
            + dis * ratio_CR_minus_ESBL * CCA_H
            + (mu + nu) * lam_E_H * CWA_H - nu * lam_CR_H * CEA_H
            - CEA_H * cr
            # (1 - rE_A) = 0
            - CEA_H * inv_dH
        )

        # ---- Eq (21): CEB_H ----
        # rE_B=0 → primary colonisation = 0; HGT gain = 0; full clearance by B
        dCEB_H = (
            CEB_C * hr
            + CEU_H * t_B_H - CEB_H * inv_tau_tH
            # Primary colonisation: rE_B=0 → 0
            # HGT gain: rE_B=0 → 0; HGT loss to CR: rCR_B=1
            - (mu + nu) * lam_CR_H * CEB_H
            - CEB_H * cr
            - CEB_H * inv_tau_t   # (1-rE_B)=1
            - CEB_H * inv_dH
        )

        # ---- Eq (22): CCU_H ----
        dCCU_H = (
            CCU_C * hr - CCU_H * dr
            - CCU_H * t_A_H + CCA_H * inv_tau_tH
            - CCU_H * t_B_H + CCB_H * inv_tau_tH
            + lam_CR_H * SU_H
            - dis * (ratio_CR_minus_ESBL + ratio_CR_ESBL) * CCU_H
            + nu * lam_CR_H * (CWU_H + CEU_H)
            - CCU_H * cr
            - CCU_H * inv_dH
        )

        # ---- Eq (23): CCA_H ----
        # Primary: lam_CR_H*SA_H*rCR_A = lam_CR_H*SA_H
        # HGT: nu*lam_CR_H*CEA_H*rCR_A + (mu+nu)*lam_CR_H*CWA_H*rCR_A
        # Decolonisation: (1-rCR_A)=0
        dCCA_H = (
            CCA_C * hr
            + CCU_H * t_A_H - CCA_H * inv_tau_tH
            + lam_CR_H * SA_H
            - dis * ratio_CR_minus_ESBL * CCA_H
            + nu * lam_CR_H * CEA_H + (mu + nu) * lam_CR_H * CWA_H
            - CCA_H * cr
            # (1 - rCR_A) = 0
            - CCA_H * inv_dH
        )

        # ---- Eq (24): CCB_H ----
        # Primary: lam_CR_H*SB_H*rCR_B = lam_CR_H*SB_H
        # HGT: (mu+nu)*lam_CR_H*(CWB_H+CEB_H)*rCR_B
        # Decolonisation: (1-rCR_B)=0
        dCCB_H = (
            CCB_C * hr
            + CCU_H * t_B_H - CCB_H * inv_tau_tH
            + lam_CR_H * SB_H
            + (mu + nu) * lam_CR_H * (CWB_H + CEB_H)
            - CCB_H * cr
            # (1 - rCR_B) = 0
            - CCB_H * inv_dH
        )

        # ================================================================
        # INFECTED COMPARTMENTS (hospital only, Eqs 25–27)
        # Development = sum of all colonised progressing to infection
        # Recovery    = -I / tau_r
        # ================================================================

        # ---- Eq (25): IWT_H ----
        dIWT_H = (
            (CWU_C + CWA_C + CWB_C) * inv_dC
            + (CWU_H + CWA_H + CWB_H) * inv_dH
            - IWT_H * inv_tau_r
        )

        # ---- Eq (26): IET_H ----
        dIET_H = (
            (CEU_C + CEA_C + CEB_C) * inv_dC
            + (CEU_H + CEA_H + CEB_H) * inv_dH
            - IET_H * inv_tau_r
        )

        # ---- Eq (27): ICT_H ----
        dICT_H = (
            (CCU_C + CCA_C + CCB_C) * inv_dC
            + (CCU_H + CCA_H + CCB_H) * inv_dH
            - ICT_H * inv_tau_r
        )

        return np.stack([
            dSU_C,  dSA_C,  dSB_C,
            dCWU_C, dCWA_C, dCWB_C,
            dCEU_C, dCEA_C, dCEB_C,
            dCCU_C, dCCA_C, dCCB_C,
            dSU_H,  dSA_H,  dSB_H,
            dCWU_H, dCWA_H, dCWB_H,
            dCEU_H, dCEA_H, dCEB_H,
            dCCU_H, dCCA_H, dCCB_H,
            dIWT_H, dIET_H, dICT_H,
        ])
