"""
Disease-specific configuration modules.

This directory contains disease validation configs that users create
when implementing a new disease model.

Each disease module should define a single DiseaseConfig class that:
1. Inherits from BaseDiseaseConfig
2. Defines the disease_type
3. Contains only disease-specific validation fields

Example:
    from compartment.validation.diseases.covid import CovidDiseaseConfig
    from compartment.validation import SimulationConfig
    
    config = SimulationConfig[CovidDiseaseConfig].model_validate(data)
"""
from .covid import CovidDiseaseConfig
from .dengue import DengueDiseaseConfig
from .dengue_2strain import Dengue2StrainDiseaseConfig
from .mpox import MpoxDiseaseConfig

__all__ = [
    "CovidDiseaseConfig",
    "DengueDiseaseConfig",
    "Dengue2StrainDiseaseConfig",
    "MpoxDiseaseConfig",
]
