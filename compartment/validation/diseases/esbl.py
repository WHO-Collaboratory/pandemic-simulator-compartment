from __future__ import annotations
from typing import Literal, Optional
from pydantic import Field

from compartment.validation.disease_config import BaseDiseaseConfig


class ESBLDiseaseConfig(BaseDiseaseConfig):
    """
    Disease configuration for the ESBL/CR Klebsiella pneumoniae transmission model.

    Based on: Identifying the drivers of multidrug-resistant Klebsiella pneumoniae
    at a European level (PLOS Comp Bio, doi:10.1371/journal.pcbi.1008446).

    The model tracks 27 compartments stratified by:
      - Setting: community (C) vs. hospital (H)
      - Colonization strain: wild-type (WT), ESBL, carbapenem-resistant (CR)
      - Treatment status: untreated (U), cephalosporins (A), carbapenems (B)
    Plus 3 hospitalized infected compartments (WT, ESBL, CR).

    Antibiotic consumption rates (C_*) are in DDD per 1000 person-days.
    Treatment rates (t_*) are derived automatically from consumption data at model init.
    """

    disease_type: Literal["ESBL"] = "ESBL"

    # --- Transmission ---
    beta: float = Field(
        default=0.01027,
        gt=0,
        description="Community colonization transmission rate (per day). "
                    "Default derived from 20% community colonization prevalence and "
                    "cr=3/365: beta = cr / (1 - colonization_fraction).",
    )
    R_HC: float = Field(
        default=10.0,
        gt=0,
        description="Hospital-to-community transmission ratio. "
                    "Hospital transmission rate = R_HC * beta.",
    )

    # --- Fitness costs (0–0.5 range from paper) ---
    s_ESBL: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Population-level fitness cost of ESBL strain (0=no cost, 0.5=high cost).",
    )
    s_CR: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Population-level fitness cost of CR strain. Should be >= s_ESBL.",
    )

    # --- Decolonization ---
    cr: float = Field(
        default=3.0 / 365,
        gt=0,
        description="Natural clearance rate (per day). Default 3/365 from carriage duration literature.",
    )
    dis: float = Field(
        default=1.5 / 365,
        gt=0,
        description="Displacement/plasmid-loss rate (per day). Must satisfy: "
                    "dis <= cr and beta * nu * 0.2 <= dis.",
    )

    # --- HGT (horizontal gene transfer / super-colonization) ---
    nu: float = Field(
        default=0.5,
        ge=0,
        description="Super-colonization HGT rate multiplier (nu). "
                    "Constraint: beta * nu * 0.2 <= dis.",
    )
    mu: float = Field(
        default=1.0,
        ge=0,
        description="Additional HGT probability under antibiotic treatment (mu). "
                    "Treated individuals have HGT rate (mu + nu) instead of nu.",
    )

    # --- Hospital flow ---
    hr: float = Field(
        default=1.0 / 1400,
        gt=0,
        description="Hospitalization rate (per day). Default 1/1400 ≈ 500 beds per 100k "
                    "with 7-day mean length of stay.",
    )
    dr: float = Field(
        default=1.0 / 7,
        gt=0,
        description="Discharge rate (per day) = 1 / mean_length_of_stay.",
    )

    # --- Treatment timing ---
    tau_t: float = Field(
        default=7.0,
        gt=0,
        description="Mean time to bacterial clearance under appropriate treatment (days).",
    )
    tau_r: float = Field(
        default=10.0,
        gt=0,
        description="Mean recovery time from invasive infection (days).",
    )

    # --- Disease progression (colonized → infected) ---
    tau_dC: float = Field(
        default=912500.0,
        gt=0,
        description="Mean time from colonization to invasive infection in community (days). "
                    "Default derived from 8 per 100k per year incidence with 20% colonization.",
    )
    tau_dH: float = Field(
        default=912500.0,
        gt=0,
        description="Mean time from colonization to invasive infection in hospital (days). "
                    "Paper assumes same rate as community.",
    )

    # --- Antibiotic consumption (DDD / 1000 person-days) ---
    C_A_C: float = Field(
        default=1.5,
        ge=0,
        description="3rd/4th generation cephalosporin consumption in community.",
    )
    C_B_C: float = Field(
        default=0.01,
        ge=0,
        description="Carbapenem consumption in community.",
    )
    C_A_H: float = Field(
        default=20.0,
        ge=0,
        description="3rd/4th generation cephalosporin consumption in hospital.",
    )
    C_B_H: float = Field(
        default=10.0,
        ge=0,
        description="Carbapenem consumption in hospital.",
    )

    # --- Import (external colonization pressure) ---
    IMP_ESBL: float = Field(
        default=1.0,
        ge=0,
        description="Effective number of ESBL-colonized importees added to the community "
                    "force-of-colonization numerator per day.",
    )
    IMP_CR: float = Field(
        default=0.1,
        ge=0,
        description="Effective number of CR-colonized importees added to the community "
                    "force-of-colonization numerator per day.",
    )

    # --- Initial conditions ---
    esbl_community_prevalence: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Initial ESBL colonization prevalence in the community (% of total zone population). "
                    "Overrides infected_population from case_file for ESBL seeding.",
    )
    cr_community_prevalence: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Initial CR colonization prevalence in the community (% of total zone population).",
    )
    wt_community_colonization: float = Field(
        default=20.0,
        ge=0,
        le=100,
        description="Total initial colonization prevalence in the community including all strains (%).",
    )
    hospital_colonization: float = Field(
        default=40.0,
        ge=0,
        le=100,
        description="Total initial colonization prevalence in the hospital (%).",
    )
