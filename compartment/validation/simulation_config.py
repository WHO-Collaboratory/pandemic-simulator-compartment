from __future__ import annotations
from typing import Optional, List, TypeVar, Generic
from pydantic import ConfigDict, model_validator

from compartment.validation.base_simulation import BaseSimulationShared
from compartment.validation.disease_config import BaseDiseaseConfig
from compartment.validation.interventions import Intervention


# Type variable for the disease config
T = TypeVar('T', bound=BaseDiseaseConfig)


class SimulationConfig(BaseSimulationShared, Generic[T]):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    
    # Disease configuration (generic - works with any BaseDiseaseConfig subclass)
    Disease: T
    
    # Optional interventions (same for all diseases)
    interventions: Optional[List[Intervention]] = None
    
    @model_validator(mode="after")
    def validate_disease_capabilities(self):
        """Validate that config only uses features supported by this disease model."""
        capabilities = self.Disease.__class__.capabilities

        if hasattr(self, 'run_mode') and self.run_mode == "UNCERTAINTY":
            if not capabilities.supports_uncertainty:
                disease_name = self.Disease.disease_type
                raise ValueError(
                    f"Uncertainty mode is not supported for {disease_name} disease model. "
                )
        
        # Check if travel is used but not supported
        if hasattr(self, 'travel_volume') and self.travel_volume is not None:
            if not capabilities.supports_travel:
                disease_name = self.Disease.disease_type
                raise ValueError(
                    f"Travel is not supported for {disease_name} disease model. "
                )
        
        # Check if transmission_edges are used but not supported
        if hasattr(self.Disease, 'transmission_edges') and self.Disease.transmission_edges:
            if not capabilities.supports_transmission_edges:
                disease_name = self.Disease.disease_type
                raise ValueError(
                    f"Transmission edges are not supported for {disease_name} disease model. "
                )
        
        # Check if interventions use unsupported types
        if self.interventions:
            for intervention in self.interventions:
                if intervention.id not in capabilities.supported_interventions:
                    disease_name = self.Disease.disease_type
                    supported_list = ', '.join(sorted(capabilities.supported_interventions)) if capabilities.supported_interventions else 'none'
                    raise ValueError(
                        f"Intervention '{intervention.id}' is not supported for {disease_name} disease model. "
                        f"Supported interventions: {supported_list}"
                    )
        
        # Check if demographics are used but not supported
        if hasattr(self, 'case_file') and self.case_file is not None:
            if hasattr(self.case_file, 'demographics') and self.case_file.demographics is not None:
                if not capabilities.supports_demographics:
                    disease_name = self.Disease.disease_type
                    raise ValueError(
                        f"Demographics are not supported for {disease_name} disease model. "
                    )
            
            # Check if temperature data is used but not supported
            if hasattr(self.case_file, 'admin_zones') and self.case_file.admin_zones:
                has_temp = any(
                    (hasattr(zone, 'temp_min') and zone.temp_min is not None) or
                    (hasattr(zone, 'temp_max') and zone.temp_max is not None) or
                    (hasattr(zone, 'temp_mean') and zone.temp_mean is not None)
                    for zone in self.case_file.admin_zones
                )
                if has_temp and not capabilities.supports_temperature:
                    disease_name = self.Disease.disease_type
                    raise ValueError(
                        f"Temperature data is not supported for {disease_name} disease model. "
                        f"Remove 'temp_min', 'temp_max', or 'temp_mean' from your admin_zones."
                    )
        
        return self
