from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

from compartment.validation.disease_config import BaseDiseaseConfig


class TransmissionEdgeData(BaseModel):
    transmission_rate: float = Field(gt=0)


class TransmissionEdge(BaseModel):
    source: str
    target: str
    data: TransmissionEdgeData
    id: str = ""

    @model_validator(mode="before")
    @classmethod
    def build_id(cls, values):
        if "id" not in values or not values["id"]:
            source = values.get("source", "")
            target = values.get("target", "")
            values["id"] = f"{source}->{target}"
        return values


class DiseaseNodeData(BaseModel):
    alias: Optional[str]
    label: str


class DiseaseNode(BaseModel):
    type: Literal["DISEASE_STATE_NODE"]
    data: DiseaseNodeData
    id: str


class MpoxDiseaseConfig(BaseDiseaseConfig):
    disease_type: Literal["MONKEYPOX"] = "MONKEYPOX"
    transmission_edges: List[TransmissionEdge]
    
    # MPOX has a hardcoded compartment list in MpoxJaxModel.COMPARTMENT_LIST
    # No need to require it in config
