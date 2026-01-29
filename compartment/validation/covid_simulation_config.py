from __future__ import annotations

from typing import Optional, List, Any
from pydantic import field_validator, ConfigDict, Field

from compartment.validation.base_simulation import BaseSimulationShared
from compartment.validation.covid_disease import CovidDiseaseConfig
from compartment.validation.interventions import Intervention

from compartment.helpers import create_initial_population_matrix, create_compartment_list, create_transmission_dict, extract_admin_units, create_intervention_dict, get_gravity_model_travel_matrix, get_hemisphere, get_temperature
class CovidSimulationConfig(BaseSimulationShared):
    """
    Full validated configuration for a COVID simulation.

    Includes:
    - Shared fields from BaseSimulationShared
    - Disease-specific COVID configuration
    - Optional COVID-specific interventions
    """
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True) # "forbid" would raise an error for unknown fields
    
    Disease: CovidDiseaseConfig
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
    def ensure_respiratory(cls, v: CovidDiseaseConfig):
        """
        Guarantees that only respiratory diseases are passed to COVID config.
        """
        if v.disease_type != "RESPIRATORY":
            raise ValueError(
                f"CovidSimulationConfig expects RESPIRATORY disease type, got {v.disease_type}"
            )
        return v

    def model_post_init(self, __context):
        # ---- Build the dict views to work with helpers ----
        disease_dict = self.Disease.model_dump()
        admin_zones_dicts = [z.model_dump() for z in self.case_file.admin_zones]
        interventions_list = (
            [i.model_dump() for i in self.interventions] if self.interventions else []
        )
        admin0_dict = self.AdminUnit0.model_dump() if self.AdminUnit0 else None
        admin1_dict = self.AdminUnit1.model_dump() if self.AdminUnit1 else None
        admin2_dict = self.AdminUnit2.model_dump() if self.AdminUnit2 else None
        travel_volume_dict = self.travel_volume.model_dump()

        # Use compartment_list if provided, otherwise derive from disease_nodes
        if disease_dict.get('compartment_list'):
            self.compartment_list = disease_dict['compartment_list']
        elif disease_dict.get('disease_nodes'):
            self.compartment_list = create_compartment_list(disease_dict['disease_nodes'])
        else:
            raise ValueError("Either disease_nodes or compartment_list must be provided in Disease config")

        self.initial_population = create_initial_population_matrix(admin_zones_dicts, self.compartment_list)
        self.transmission_dict = create_transmission_dict(disease_dict['transmission_edges'])
        self.admin_units = extract_admin_units(admin_zones_dicts)
        self.intervention_dict = create_intervention_dict(interventions_list, self.start_date)

        # Build travel matrix - will return identity if travel_volume is None
        self.travel_matrix = get_gravity_model_travel_matrix(
            admin_zones_dicts, 
            travel_volume_dict if self.travel_volume else None
        )
        self.hemisphere = get_hemisphere(admin2_dict, admin1_dict, admin0_dict)
        self.temperature = get_temperature(admin_zones_dicts)