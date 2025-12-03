from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

class DengueDiseaseConfig(BaseModel):
    """
    Disease-specific config for Dengue.
    """
    model_config = ConfigDict(extra="ignore") # "forbid" would raise an error for unknown fields

    id: Optional[str] = None
    disease_name: Optional[str] = None
    disease_type: Literal["VECTOR_BORNE"] = "VECTOR_BORNE"
    immunity_period: int = Field(ge=0)
