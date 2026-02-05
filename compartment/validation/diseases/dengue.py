from __future__ import annotations
from typing import Literal
from pydantic import Field

from compartment.validation.disease_config import BaseDiseaseConfig


class DengueDiseaseConfig(BaseDiseaseConfig):
    disease_type: Literal["VECTOR_BORNE"] = "VECTOR_BORNE"
    immunity_period: int = Field(ge=0)
