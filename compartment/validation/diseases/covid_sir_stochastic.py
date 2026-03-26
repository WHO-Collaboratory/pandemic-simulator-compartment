"""
COVID Stochastic SIR disease validation config -- auto-generated from model parameters.
"""

from __future__ import annotations

from compartment.models.covid_sir_stochastic.model import CovidSirStochasticModel
from compartment.schema_generator import generate_disease_config

CovidSirStochasticDiseaseConfig = generate_disease_config(
    CovidSirStochasticModel._build_parameter_schema())

__all__ = ["CovidSirStochasticDiseaseConfig"]
