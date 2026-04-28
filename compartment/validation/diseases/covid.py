from __future__ import annotations
from typing import Literal, List, Optional
from pydantic import BaseModel, Field, model_validator

from compartment.validation.disease_config import BaseDiseaseConfig


class CovidVarianceParams(BaseModel):
    has_variance: bool
    distribution_type: Literal["UNIFORM", "NORMAL"]
    field_name: Optional[str]
    min: float
    max: float


class CovidTransmissionEdgeData(BaseModel):
    transmission_rate: float = Field(gt=0)
    variance_params: Optional[CovidVarianceParams] = None


class CovidTransmissionEdge(BaseModel):
    source: str
    target: str
    type: Optional[str] = None
    data: CovidTransmissionEdgeData
    id: str = ""

    @model_validator(mode="before")
    @classmethod
    def build_id(cls, values):
        if "id" not in values or not values["id"]:
            source = values.get("source", "")
            target = values.get("target", "")
            values["id"] = f"{source}->{target}"
        return values


class CovidDiseaseConfig(BaseDiseaseConfig):
    disease_type: Literal["RESPIRATORY"] = "RESPIRATORY"
    transmission_edges: Optional[List[CovidTransmissionEdge]] = None
    compartment_list: Optional[List[str]] = None
