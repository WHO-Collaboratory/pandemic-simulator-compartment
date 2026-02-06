from __future__ import annotations
from typing import Literal, List, Optional, ClassVar
from pydantic import BaseModel, Field, model_validator

from compartment.validation.disease_config import BaseDiseaseConfig, DiseaseCapabilities


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


class CovidDiseaseNodeData(BaseModel):
    alias: Optional[str]
    label: str


class CovidDiseaseNode(BaseModel):
    type: Literal["DISEASE_STATE_NODE"]
    data: CovidDiseaseNodeData
    id: str


class CovidDiseaseConfig(BaseDiseaseConfig):
    disease_type: Literal["RESPIRATORY"] = "RESPIRATORY"
    
    # COVID capabilities
    capabilities: ClassVar[DiseaseCapabilities] = DiseaseCapabilities(
        supports_travel=True,
        supported_interventions={"social_isolation", "vaccination", "mask_wearing", "lock_down"},
        supports_demographics=True,
        supports_transmission_edges=True,
        supports_temperature=False,
        supports_uncertainty=True
    )
    
    transmission_edges: List[CovidTransmissionEdge]
    
    disease_nodes: Optional[List[CovidDiseaseNode]] = None
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