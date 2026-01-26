from typing import List, Literal
from pydantic import BaseModel, ConfigDict, Field
from pydantic import model_validator
from uuid import uuid4
from datetime import date

class TransmissionEdgeData(BaseModel):
    transmission_rate: float = Field(gt=0)

class TransmissionEdge(BaseModel):
    source: str
    target: str
    data: TransmissionEdgeData

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

class CaseFileAdminZone(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    center_lat: float = Field(ge=-90, le=90)
    center_lon: float = Field(ge=-180, le=180)
    name: str
    population: int = Field(ge=0)
    infected_population: float = Field(default=0.05, ge=0, le=100)


class CaseFile(BaseModel):
    admin_zones: List[CaseFileAdminZone]

class AdminUnit(BaseModel):
    id: str
    center_lat: float = Field(ge=-90, le=90)

class MpoxConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    disease: MpoxDiseaseConfig
    case_file: CaseFile
    admin_unit_0_id: str
    start_date: date
    end_date: date
    time_steps: int = Field(gt=0)
    run_mode: Literal["DETERMINISTIC", "UNCERTAINTY"]
    simulation_type: Literal["COMPARTMENTAL"]
    AdminUnit0: AdminUnit