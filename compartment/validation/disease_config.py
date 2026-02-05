from __future__ import annotations
from abc import ABC
from typing import Optional, List
from pydantic import BaseModel, ConfigDict


class BaseDiseaseConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="ignore")
    
    # Common fields that all diseases should have
    id: Optional[str] = None
    disease_name: Optional[str] = None
    disease_type: str  # e.g., "RESPIRATORY", "VECTOR_BORNE", "MONKEYPOX"
    
    def get_compartments(self) -> List[str]:
        """
        Override this method if the disease has a custom way to derive compartments.
        
        By default, returns None (compartments will be derived by post-processor).
        """
        return None
