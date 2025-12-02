from __future__ import annotations

from datetime import date
from typing import Literal, Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator


class TravelVolume(BaseModel):
    leaving: float = Field(default=0.2, ge=0, le=1)
    returning: float = Field(default=0.1, ge=0, le=1)

    @field_validator("leaving", "returning", mode="before")
    @classmethod
    def normalize_percentage(cls, v):
        if v is None:
            return v
        v = float(v)
        if v > 1:
            v = v / 100.0
        return v

class AdminUnit(BaseModel):
    id: str
    center_lat: float = Field(ge=-90, le=90)

class CaseFileDemographics(BaseModel):
    age_0_17: float = Field(ge=0, le=100)
    age_18_55: float = Field(ge=0, le=100)
    age_56_plus: float = Field(ge=0, le=100)


class CaseFileAdminZone(BaseModel):
    id: Optional[str] = None
    admin_code: Optional[str] = None
    admin_iso_code: Optional[str] = None
    admin_level: int = Field(ge=0)
    center_lat: float = Field(ge=-90, le=90)
    center_lon: float = Field(ge=-180, le=180)

    viz_name: Optional[str] = None
    name: Optional[str] = None

    population: int = Field(ge=0)
    osm_id: Optional[str] = None

    infected_population: float = Field(ge=0, le=100) # Covid: infected population, Dengue: first infection
    seroprevalence: Optional[float] = Field(default=None, ge=0, le=100) # Dengue: susceptible to second infection
    temp_min: Optional[float] = Field(default=15)
    temp_max: Optional[float] = Field(default=30)
    temp_mean: Optional[float] = Field(default=25)

class CaseFile(BaseModel):
    admin_zones: List[CaseFileAdminZone]
    demographics: CaseFileDemographics

class BaseSimulationShared(BaseModel):
    """
    Shared fields for all disease simulations (COVID, Dengue, etc.).
    """
    model_config = ConfigDict(extra="ignore") # "forbid" would raise an error for unknown fields

    id: Optional[str] = None
    simulation_name: str

    # Admin units
    admin_unit_0_id: str
    admin_unit_1_id: Optional[str] = None
    admin_unit_2_id: Optional[str] = None
    AdminUnit0: AdminUnit
    AdminUnit1: Optional[AdminUnit] = None
    AdminUnit2: Optional[AdminUnit] = None

    # Owner & meta
    owner: Optional[str] = None
    simulation_type: Literal["COMPARTMENTAL", "AGENT_BASED"]
    run_mode: Literal["UNCERTAINTY", "DETERMINISTIC"]

    # Time
    start_date: str
    end_date: str
    time_steps: int = Field(gt=0)

    # Population / mobility
    selected_infected_population: float = Field(ge=0)
    selected_population: int = Field(ge=0)
    travel_volume: TravelVolume

    case_file: CaseFile

    @field_validator("start_date", "end_date")
    @classmethod
    def must_be_iso_date(cls, v: str) -> str:
        try:
            date.fromisoformat(v)  # ensure valid YYYY-MM-DD
        except ValueError:
            raise ValueError("Date must be an ISO string formatted as YYYY-MM-DD")
        return v

    @field_validator("end_date")
    @classmethod
    def end_not_before_start(cls, v: str, info):
        start = info.data.get("start_date")
        if start:
            if date.fromisoformat(v) < date.fromisoformat(start):
                raise ValueError("end_date must be on or after start_date")
        return v
