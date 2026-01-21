from __future__ import annotations
from typing import Literal, List, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator

# Reuse the same transmission edge structure as COVID
from compartment.validation.covid_disease import CovidTransmissionEdge

class MonkeypoxDiseaseConfig(BaseModel):
    """
    Disease-specific config for Monkeypox SIR model.
    Simple structure - just transmission edges for S->I and I->R.
    """
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    disease_name: Optional[str] = None
    disease_type: Literal["MONKEYPOX"]
    transmission_edges: List[CovidTransmissionEdge]
    compartment_list: Optional[List[str]] = None

    @model_validator(mode='after')
    def ensure_sir_compartments(self):
        """
        Ensures compartment_list is SIR for Monkeypox.
        """
        if self.compartment_list is None:
            self.compartment_list = ["S", "I", "R"]
        elif set(self.compartment_list) != {"S", "I", "R"}:
            raise ValueError(
                "Monkeypox model requires exactly S, I, R compartments. "
                f"Got: {self.compartment_list}"
            )
        return self

