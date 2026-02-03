from __future__ import annotations

from typing import List, Any
from pydantic import field_validator, ConfigDict, Field

from compartment.validation.base_simulation import BaseSimulationShared
from compartment.validation.mpox_disease import MpoxDiseaseConfig

from compartment.helpers import create_initial_population_matrix, create_compartment_list, create_transmission_dict, extract_admin_units

class MpoxSimulationConfig(BaseSimulationShared):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True) # "forbid" would raise an error for unknown fields
    
    Disease: MpoxDiseaseConfig

    # Computed fields, not part of the input
    compartment_list: List[str] = Field(default_factory=list, exclude=False)
    initial_population: Any = Field(default=None, exclude=False)
    transmission_dict: dict = Field(default_factory=dict, exclude=False)
    admin_units: list = Field(default_factory=list, exclude=False)

    @field_validator("Disease")
    @classmethod
    def ensure_monkeypox(cls, v: MpoxDiseaseConfig):
        """
        Guarantees that only monkeypox diseases are passed to Mpox config.
        """
        if v.disease_type != "MONKEYPOX":
            raise ValueError(
                f"MpoxSimulationConfig expects MONKEYPOX disease type, got {v.disease_type}"
            )
        return v

    def model_post_init(self, __context):
        # ---- Build the dict views to work with helpers ----
        disease_dict = self.Disease.model_dump()
        admin_zones_dicts = [z.model_dump() for z in self.case_file.admin_zones]

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