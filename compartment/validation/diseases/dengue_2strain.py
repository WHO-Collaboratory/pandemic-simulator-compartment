from __future__ import annotations
from typing import Literal
from pydantic import Field
from compartment.validation.disease_config import BaseDiseaseConfig


class Dengue2StrainDiseaseConfig(BaseDiseaseConfig):
    disease_type: Literal["VECTOR_BORNE_2STRAIN"] = "VECTOR_BORNE_2STRAIN"