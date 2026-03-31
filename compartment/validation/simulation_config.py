from __future__ import annotations
from typing import Optional, TypeVar, Generic
from pydantic import ConfigDict

from compartment.validation.base_simulation import BaseSimulationShared
from compartment.validation.disease_config import BaseDiseaseConfig
from compartment.validation.interventions import (
    NormalizedInterventions,
    NormalizedTransmissionEdges,
)


# Type variable for the disease config
T = TypeVar('T', bound=BaseDiseaseConfig)


class SimulationConfig(BaseSimulationShared, Generic[T]):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # Disease configuration (generic - works with any BaseDiseaseConfig subclass)
    Disease: T

    # Normalized interventions from join table
    Interventions: Optional[NormalizedInterventions] = None

    # Normalized transmission edges from join table
    TransmissionEdges: Optional[NormalizedTransmissionEdges] = None
