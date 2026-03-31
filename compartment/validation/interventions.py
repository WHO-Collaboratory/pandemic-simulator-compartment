from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import date


class FieldConfig(BaseModel):
    """Shared FieldConfig model for both interventions and transmission edges."""
    id: Optional[str] = None
    field_key: Optional[str] = None
    has_variance: bool = False
    distribution_type: Literal["UNIFORM", "NORMAL"] = "UNIFORM"
    disease_param: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    parent_id: Optional[str] = None


class FieldConfigItems(BaseModel):
    items: List[FieldConfig] = []


# ---------------------------------------------------------------------------
# Normalized Intervention models (from Interventions join table)
# ---------------------------------------------------------------------------


class InterventionLookup(BaseModel):
    """The Intervention reference record from the lookup table."""
    id: str
    name: str
    display_name: Optional[str] = None


class NormalizedIntervention(BaseModel):
    """A single item from Interventions.items[] in the new schema."""
    id: Optional[str] = None
    intervention_id: Optional[str] = None
    Intervention: InterventionLookup
    adherence_min: Optional[float] = None
    adherence_max: Optional[float] = None
    transmission_percentage: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    start_threshold: Optional[float] = None
    end_threshold: Optional[float] = None
    start_threshold_node_id: Optional[str] = None
    end_threshold_node_id: Optional[str] = None
    hour_reduction: Optional[float] = None
    FieldConfigs: Optional[FieldConfigItems] = None

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_optional_iso_date(cls, v):
        if v is None:
            return v
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("Dates must be ISO format YYYY-MM-DD")
        return v


class NormalizedInterventions(BaseModel):
    """Container for Interventions.items[] on the SimulationJob."""
    items: List[NormalizedIntervention] = []


# ---------------------------------------------------------------------------
# Normalized TransmissionEdge models (from TransmissionEdges join table)
# ---------------------------------------------------------------------------


class TransmissionEdgeLookup(BaseModel):
    """The transmittion_edge reference record (note: typo in API schema)."""
    id: str
    value_type: Optional[str] = None
    default_value: Optional[float] = None
    disease_type: Optional[str] = None
    description: Optional[str] = None
    source: str
    target: str
    order: Optional[int] = None


class NormalizedTransmissionEdge(BaseModel):
    """A single item from TransmissionEdges.items[] in the new schema."""
    id: Optional[str] = None
    simulation_job_id: Optional[str] = None
    transmission_edge_id: Optional[str] = None
    transmittion_edge: TransmissionEdgeLookup
    value: float
    FieldConfigs: Optional[FieldConfigItems] = None


class NormalizedTransmissionEdges(BaseModel):
    """Container for TransmissionEdges.items[] on the SimulationJob."""
    items: List[NormalizedTransmissionEdge] = []
