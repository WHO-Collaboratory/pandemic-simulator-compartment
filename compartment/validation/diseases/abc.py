from __future__ import annotations
from typing import Literal, List, Optional
from pydantic import BaseModel, Field, model_validator

from compartment.validation.disease_config import BaseDiseaseConfig


class ABCVarianceParams(BaseModel):
    """Parameters for variance in transmission rates"""
    has_variance: bool
    distribution_type: Literal["UNIFORM", "NORMAL"]
    field_name: Optional[str]
    min: float
    max: float


class ABCTransmissionEdgeData(BaseModel):
    """Data for transmission edges between compartments"""
    transmission_rate: float = Field(gt=0, description="Rate of transmission between compartments")
    variance_params: Optional[ABCVarianceParams] = None


class ABCTransmissionEdge(BaseModel):
    """Transmission edge defining flow between compartments"""
    source: str = Field(description="Source compartment (e.g., 'a_compartment')")
    target: str = Field(description="Target compartment (e.g., 'b_compartment')")
    type: Optional[str] = None
    data: ABCTransmissionEdgeData
    id: str = ""

    @model_validator(mode="before")
    @classmethod
    def build_id(cls, values):
        """Auto-generate edge ID from source and target if not provided"""
        if "id" not in values or not values["id"]:
            source = values.get("source", "")
            target = values.get("target", "")
            values["id"] = f"{source}->{target}"
        return values


class ABCDiseaseNodeData(BaseModel):
    """Data for disease state nodes"""
    alias: Optional[str]
    label: str


class ABCDiseaseNode(BaseModel):
    """Disease state node in the ABC model"""
    type: Literal["DISEASE_STATE_NODE"]
    data: ABCDiseaseNodeData
    id: str


class ABCDiseaseConfig(BaseDiseaseConfig):
    """
    Configuration for the ABC disease model.
    
    The ABC model is a simple three-compartment model where:
    - A: First compartment (analogous to Susceptible)
    - B: Second compartment (analogous to Infected)
    - C: Third compartment (analogous to Recovered)
    
    Transmission edges define the rates of flow between compartments.
    """
    disease_type: Literal["ABC"] = "ABC"
    transmission_edges: List[ABCTransmissionEdge]
    
    disease_nodes: Optional[List[ABCDiseaseNode]] = None
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
    
    @model_validator(mode='after')
    def validate_compartments(self):
        """Validate that compartment list contains A, B, and C"""
        if self.compartment_list:
            required_compartments = {'A', 'B', 'C'}
            provided_compartments = set(self.compartment_list)
            if not required_compartments.issubset(provided_compartments):
                missing = required_compartments - provided_compartments
                raise ValueError(
                    f"ABC model requires compartments A, B, and C. Missing: {missing}"
                )
        return self
