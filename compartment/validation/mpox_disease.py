from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from pydantic import model_validator
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
class MpoxDiseaseConfig(BaseModel):
    disease_type: Literal["MONKEYPOX"]
    compartment_list: List[str]
    transmission_edges: List[TransmissionEdge]