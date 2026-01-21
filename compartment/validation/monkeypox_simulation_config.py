from __future__ import annotations

from typing import Optional, List, Any
from pydantic import field_validator, ConfigDict, Field

from compartment.validation.base_simulation import BaseSimulationShared
from compartment.validation.monkeypox_disease import MonkeypoxDiseaseConfig
from compartment.validation.interventions import Intervention

from compartment.helpers import (
    create_initial_population_matrix,
    create_compartment_list,
    create_transmission_dict,
    extract_admin_units,
    get_gravity_model_travel_matrix,
    get_hemisphere,
    get_temperature
)

class MonkeypoxSimulationConfig(BaseSimulationShared):
    """
    Full validated configuration for a Monkeypox SIR simulation.

    Includes:
    - Shared fields from BaseSimulationShared
    - Simple SIR model (no age stratification, no interventions)
    - Travel matrix for multi-region simulations
    """
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    
    Disease: MonkeypoxDiseaseConfig
    # No interventions for simple model
    interventions: Optional[List[Intervention]] = None

    # Computed fields, not part of the input
    compartment_list: List[str] = Field(default_factory=list, exclude=False)
    initial_population: Any = Field(default=None, exclude=False)
    transmission_dict: dict = Field(default_factory=dict, exclude=False)
    admin_units: list = Field(default_factory=list, exclude=False)
    intervention_dict: dict = Field(default_factory=dict, exclude=False)
    travel_matrix: Any = Field(default=None, exclude=False)
    hemisphere: str = Field(default="", exclude=False)
    temperature: dict = Field(default_factory=dict, exclude=False)

    @field_validator("Disease")
    @classmethod
    def ensure_monkeypox(cls, v: MonkeypoxDiseaseConfig):
        """
        Guarantees that only MONKEYPOX disease type is passed to Monkeypox config.
        """
        if v.disease_type != "MONKEYPOX":
            raise ValueError(
                f"MonkeypoxSimulationConfig expects MONKEYPOX disease type, got {v.disease_type}"
            )
        return v

    def model_post_init(self, __context):
        # ---- Build the dict views to work with helpers ----
        disease_dict = self.Disease.model_dump()
        admin_zones_dicts = [z.model_dump() for z in self.case_file.admin_zones]
        admin0_dict = self.AdminUnit0.model_dump() if self.AdminUnit0 else None
        admin1_dict = self.AdminUnit1.model_dump() if self.AdminUnit1 else None
        admin2_dict = self.AdminUnit2.model_dump() if self.AdminUnit2 else None
        travel_volume_dict = self.travel_volume.model_dump()

        # For Monkeypox, always use simple SIR compartments
        self.compartment_list = disease_dict.get('compartment_list', ["S", "I", "R"])
        
        # Build initial population matrix from admin zones
        self.initial_population = create_initial_population_matrix(admin_zones_dicts, self.compartment_list)
        
        # Extract transmission rates from Disease.transmission_edges
        self.transmission_dict = create_transmission_dict(disease_dict.get('transmission_edges', []))
        
        self.admin_units = extract_admin_units(admin_zones_dicts)
        
        # No interventions for simple SIR model
        self.intervention_dict = {}

        # Build travel matrix using gravity model
        self.travel_matrix = get_gravity_model_travel_matrix(admin_zones_dicts, travel_volume_dict)
        self.hemisphere = get_hemisphere(admin2_dict, admin1_dict, admin0_dict)
        self.temperature = get_temperature(admin_zones_dicts)
