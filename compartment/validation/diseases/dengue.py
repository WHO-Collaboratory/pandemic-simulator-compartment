from __future__ import annotations
from typing import Literal
from pydantic import Field

from compartment.validation.disease_config import BaseDiseaseConfig, DiseaseCapabilities


class DengueDiseaseConfig(BaseDiseaseConfig):
    disease_type: Literal["VECTOR_BORNE"] = "VECTOR_BORNE"
    immunity_period: int = Field(ge=0)
    
    # Dengue capabilities
    capabilities: DiseaseCapabilities = DiseaseCapabilities(
        supports_travel=True,
        supported_interventions={"physical", "chemical"},
        supports_demographics=True,
        supports_transmission_edges=False,
        supports_temperature=True,
        supports_uncertainty=True
    )