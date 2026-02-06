from __future__ import annotations
from abc import ABC
from typing import Optional, List, Set
from pydantic import BaseModel, ConfigDict, Field


class DiseaseCapabilities(BaseModel):
    """Defines what features/interventions a disease model supports."""
    model_config = ConfigDict(frozen=True)  # Immutable after creation
    
    supports_travel: bool = Field(
        default=False,
        description="Supports inter-admin unit travel/mobility"
    )
    supported_interventions: Set[str] = Field(
        default_factory=set,
        description="Set of intervention IDs supported by this disease model"
    )
    supports_demographics: bool = Field(
        default=False,
        description="Supports age-based demographics"
    )
    supports_transmission_edges: bool = Field(
        default=False,
        description="Uses transmission edge graphs for disease dynamics"
    )
    supports_temperature: bool = Field(
        default=False,
        description="Temperature-dependent transmission (e.g., vector-borne diseases)"
    )
    supports_uncertainty: bool = Field(
        default=False,
        description="Supports stochastic/uncertainty simulation mode"
    )


class BaseDiseaseConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="ignore")
    
    # Common fields that all diseases should have
    id: Optional[str] = None
    disease_name: Optional[str] = None
    disease_type: str  # e.g., "RESPIRATORY", "VECTOR_BORNE", "MONKEYPOX"
    
    # Class attribute defining capabilities - override in subclasses
    capabilities: DiseaseCapabilities = DiseaseCapabilities()
    
    def get_compartments(self) -> List[str]:
        """
        Override this method if the disease has a custom way to derive compartments.
        
        By default, returns None (compartments will be derived by post-processor).
        """
        return None
