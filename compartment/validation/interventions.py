from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, field_validator
from datetime import date

from compartment.validation.field_configs import FieldConfigItems


class InterventionLookup(BaseModel):
    """The Intervention reference record from the lookup table."""
    id: Optional[str] = None
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
