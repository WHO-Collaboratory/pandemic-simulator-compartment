from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel


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
