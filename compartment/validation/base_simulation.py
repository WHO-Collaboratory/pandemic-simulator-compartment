from __future__ import annotations

from datetime import date
from typing import Literal, Optional, List
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class TravelVolume(BaseModel):
    leaving: float = Field(default=0.2, ge=0, le=1)
    returning: Optional[float] = Field(default=None, ge=0, le=1)

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
    age_0_17: float = Field(default=25.0, ge=0, le=100)
    age_18_55: float = Field(default=50.0, ge=0, le=100)
    age_56_plus: float = Field(default=25.0, ge=0, le=100)


class CaseFileAdminZone(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    admin_code: Optional[str] = None
    admin_iso_code: Optional[str] = None
    admin_level: Optional[int] = None
    center_lat: float = Field(ge=-90, le=90)
    center_lon: float = Field(ge=-180, le=180)

    viz_name: Optional[str] = None
    name: str

    population: int = Field(ge=0)
    osm_id: Optional[str] = None

    infected_population: float = Field(default=0.05, ge=0, le=100) # Covid: infected population, Dengue: first infection
    seroprevalence: Optional[float] = Field(default=10.0, ge=0, le=100) # Dengue: susceptible to second infection
    temp_min: Optional[float] = Field(default=15)
    temp_max: Optional[float] = Field(default=30)
    temp_mean: Optional[float] = Field(default=25)

class CaseFile(BaseModel):
    admin_zones: List[CaseFileAdminZone]
    demographics: CaseFileDemographics = CaseFileDemographics()

class BaseSimulationShared(BaseModel):
    """
    Shared fields for all disease simulations (COVID, Dengue, etc.).
    """
    model_config = ConfigDict(extra="ignore") # "forbid" would raise an error for unknown fields

    id: Optional[str] = None
    simulation_name: str = ""

    # Admin units
    admin_unit_0_id: str
    admin_unit_1_id: str = ""
    admin_unit_2_id: str = ""
    AdminUnit0: AdminUnit
    AdminUnit1: Optional[AdminUnit] = None
    AdminUnit2: Optional[AdminUnit] = None

    # Owner & meta
    owner: Optional[str] = None
    simulation_type: Literal["COMPARTMENTAL", "AGENT_BASED"]
    run_mode: Literal["UNCERTAINTY", "DETERMINISTIC"]

    # Time
    start_date: date
    end_date: date
    # time_steps is now optional and will be derived from start/end dates if not provided
    time_steps: Optional[int] = Field(default=None, gt=0)

    # Population / mobility
    # Make travel_volume optional - models can declare if they need it
    travel_volume: Optional[TravelVolume] = TravelVolume()

    case_file: CaseFile

    @field_validator("end_date")
    @classmethod
    def end_not_before_start(cls, v: date, info):
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError("end_date must be on or after start_date")
        return v

    @model_validator(mode="after")
    def derive_time_steps(self):
        if self.time_steps is None:
            delta_days = (self.end_date - self.start_date).days
            # Maintain positive steps for same-day ranges
            self.time_steps = max(delta_days, 1)
        return self
