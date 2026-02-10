# New simplified validation structure (default)
from .disease_config import BaseDiseaseConfig
from .simulation_config import SimulationConfig
from .post_processor import ValidationPostProcessor, ProcessedSimulation
from .diseases import (
    CovidDiseaseConfig,
    DengueDiseaseConfig,
    Dengue2StrainDiseaseConfig,
    MpoxDiseaseConfig,
)

# Shared models still used
from .base_simulation import BaseSimulationShared, TravelVolume
from .interventions import (
    Intervention,
    InterventionVarianceParams,
)

import logging
import sys
from pydantic import ValidationError

__all__ = [
    # Core validation classes
    "BaseDiseaseConfig",
    "SimulationConfig",
    "ValidationPostProcessor",
    "ProcessedSimulation",
    # Disease configs
    "CovidDiseaseConfig",
    "DengueDiseaseConfig",
    "Dengue2StrainDiseaseConfig",
    "MpoxDiseaseConfig",
    # Shared models
    "BaseSimulationShared",
    "TravelVolume",
    "Intervention",
    "InterventionVarianceParams",
    # Utility functions
    "log_pydantic_errors",
    "load_simulation_config",
]

logger = logging.getLogger("compartment.validation")

def log_pydantic_errors(err: ValidationError, context: str | None = None) -> None:
    """
    Log Pydantic errors in a compact, readable form.

    Example line:
    [CovidSimulationConfig] Disease.r0 -> Field required [type=value_error.missing]
    """
    prefix = f"[{context}] " if context else ""

    errors = err.errors()
    logger.error("%sValidation failed with %d error(s)", prefix, len(errors))

    for e in errors:
        loc = ".".join(str(part) for part in e.get("loc", ()))
        msg = e.get("msg", "")
        typ = e.get("type", "")

        logger.error("%s%s -> %s [type=%s]", prefix, loc or "<root>", msg, typ)

def load_simulation_config(config: dict, disease_type: str):
    """
    Centralized validation entrypoint using the new structure.

    Args:
        config: Configuration dict (usually from GraphQL/JSON)
        disease_type: Type of disease ("RESPIRATORY", "VECTOR_BORNE", "MONKEYPOX")

    Returns:
        ProcessedSimulation with all computed fields ready for use by Model classes.
        
    Example:
        config = load_simulation_config(data, "RESPIRATORY")
        model = CovidJaxModel(config)
    """
    # Map disease type to disease config class
    if disease_type == "VECTOR_BORNE":
        disease_cls = DengueDiseaseConfig
    elif disease_type == "VECTOR_BORNE_2STRAIN":
        disease_cls = Dengue2StrainDiseaseConfig
    elif disease_type == "RESPIRATORY":
        disease_cls = CovidDiseaseConfig
    elif disease_type == "MONKEYPOX":
        disease_cls = MpoxDiseaseConfig
    else:
        raise ValueError(f"Invalid disease type: {disease_type}")
    
    context = f"SimulationConfig[{disease_cls.__name__}]"
    
    try:
        # Step 1: Validate
        validated_config = SimulationConfig[disease_cls](**config["data"]["getSimulationJob"])
        # Step 2: Post-process
        processed = ValidationPostProcessor.process(validated_config)
        return processed
    except ValidationError as e:
        log_pydantic_errors(e, context=context)
        logger.error("Simulation config validation failed; aborting.")
        sys.exit(2)
