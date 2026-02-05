from __future__ import annotations
from typing import Optional, List, TypeVar, Generic
from pydantic import ConfigDict

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
    
    # Note: No model_post_init here - processing happens externally
    # This keeps validation separate from transformation/computation
