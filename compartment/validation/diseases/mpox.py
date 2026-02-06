from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

from compartment.validation.disease_config import BaseDiseaseConfig, DiseaseCapabilities


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
    
    # Mpox capabilities
    capabilities: DiseaseCapabilities = DiseaseCapabilities(
        supports_travel=False,
        supported_interventions=set(),
        supports_demographics=False,
        supports_transmission_edges=True,
        supports_temperature=False,
        supports_uncertainty=False
    )

    disease_nodes: Optional[List[DiseaseNode]] = None
    compartment_list: Optional[List[str]] = None

    @model_validator(mode='after')
    def check_compartment_source(self):
        """
        Ensures that either disease_nodes or compartment_list is provided.
        """
        if self.disease_nodes is None and (self.compartment_list is None or len(self.compartment_list) == 0):
            raise ValueError(
                "Either 'disease_nodes' or 'compartment_list' must be provided. "
                "If 'disease_nodes' is not provided, 'compartment_list' is required."
            )
        return self
