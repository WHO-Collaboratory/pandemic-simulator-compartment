from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import date


class InterventionVarianceParams(BaseModel):
    has_variance: bool
    distribution_type: Literal["UNIFORM", "NORMAL"]
    field_name: str
    min: float
    max: float


class Intervention(BaseModel):
    """
    Matches exactly the intervention objects from the uploaded config.
    """
    id: Literal["social_isolation", "vaccination", "mask_wearing", "lock_down", "chemical", "physical"]
    type: Literal["INTERVENTION"] = "INTERVENTION"
    label: Optional[str] = None

    start_date: Optional[str] = None
    end_date: Optional[str] = None

    adherence_min: Optional[float] = None
    adherence_max: Optional[float] = None

    transmission_percentage: Optional[float] = None
    hour_reduction: Optional[float] = None

    start_threshold: Optional[float] = None
    end_threshold: Optional[float] = None
    start_threshold_node_id: Optional[str] = None
    end_threshold_node_id: Optional[str] = None

    variance_params: Optional[List[InterventionVarianceParams]] = None

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_optional_iso_date(cls, v):
        if v is None:
            return v  # allowed
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("Dates must be ISO format YYYY-MM-DD")
        return v
