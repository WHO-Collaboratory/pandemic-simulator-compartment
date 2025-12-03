from __future__ import annotations
from typing import Literal, List, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator

class CovidVarianceParams(BaseModel):
    has_variance: bool
    distribution_type: Literal["UNIFORM", "NORMAL"]
    field_name: Optional[str]
    min: float
    max: float


class CovidTransmissionEdgeData(BaseModel):
    transmission_rate: float = Field(gt=0)
    variance_params: CovidVarianceParams


class CovidTransmissionEdge(BaseModel):
    id: str
    source: str
    target: str
    type: Optional[str] = None
    data: CovidTransmissionEdgeData


class CovidDiseaseNodeData(BaseModel):
    alias: Optional[str]
    label: str


class CovidDiseaseNode(BaseModel):
    type: Literal["DISEASE_STATE_NODE"]
    data: CovidDiseaseNodeData
    id: str
class CovidCompartmentList(BaseModel):
    compartment_list: List[str]

class CovidDiseaseConfig(BaseModel):
    """
    Disease-specific config for COVID.
    """
    model_config = ConfigDict(extra="ignore") # "forbid" would raise an error for unknown fields

    id: Optional[str] = None
    disease_name: Optional[str] = None
    disease_type: Literal["RESPIRATORY"]
    transmission_edges: List[CovidTransmissionEdge]
    
    disease_nodes: Optional[List[CovidDiseaseNode]] = None
    compartment_list: Optional[List[str]] = None

    @model_validator(mode='after')
    def check_compartment_source(self):
        """
        Ensures that either disease_nodes or compartment_list is provided.
        If disease_nodes is not provided, compartment_list must be provided.
        """
        if self.disease_nodes is None and (self.compartment_list is None or len(self.compartment_list) == 0):
            raise ValueError(
                "Either 'disease_nodes' or 'compartment_list' must be provided. "
                "If 'disease_nodes' is not provided, 'compartment_list' is required."
            )
        return self