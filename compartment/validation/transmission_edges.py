from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel

from compartment.validation.field_configs import FieldConfigItems


class TransmissionEdgeLookup(BaseModel):
    """The transmission_edge reference record."""
    id: Optional[str] = None
    value_type: Optional[str] = None
    disease_type: Optional[str] = None
    description: Optional[str] = None
    source: str
    target: str


class NormalizedTransmissionEdge(BaseModel):
    """A single item from TransmissionEdges.items[] in the new schema."""
    id: Optional[str] = None
    simulation_job_id: Optional[str] = None
    transmission_edge_id: Optional[str] = None
    transmission_edge: TransmissionEdgeLookup
    value: float
    FieldConfigs: Optional[FieldConfigItems] = None


class NormalizedTransmissionEdges(BaseModel):
    """Container for TransmissionEdges.items[] on the SimulationJob."""
    items: List[NormalizedTransmissionEdge] = []
