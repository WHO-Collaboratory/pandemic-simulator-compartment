from .base_simulation import BaseSimulationShared, TravelVolume
from .covid_disease import (
    CovidDiseaseConfig,
    CovidDiseaseNode,
    CovidDiseaseNodeData,
    CovidTransmissionEdge,
    CovidTransmissionEdgeData,
    CovidVarianceParams,
)
from .dengue_disease import DengueDiseaseConfig
from .interventions import (
    Intervention,
    InterventionVarianceParams,
)

from .covid_simulation_config import CovidSimulationConfig
from .dengue_simulation_config import DengueSimulationConfig

import logging
import sys
from pydantic import ValidationError

__all__ = [
    "BaseSimulationShared",
    "TravelVolume",
    "CovidDiseaseConfig",
    "CovidDiseaseNode",
    "CovidDiseaseNodeData",
    "CovidTransmissionEdge",
    "CovidTransmissionEdgeData",
    "CovidVarianceParams",
    "Intervention",
    "InterventionVarianceParams",
    "CovidSimulationConfig",
    "DengueDiseaseConfig",
    "DengueSimulationConfig",
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
    Centralized validation entrypoint.

    - Picks the right SimulationConfig model
    - Validates config["data"]["getSimulationJob"]
    - Logs nice error messages on ValidationError
    """
    model_cls = DengueSimulationConfig if disease_type == "VECTOR_BORNE" else CovidSimulationConfig
    context = model_cls.__name__

    try:
        return model_cls(**config["data"]["getSimulationJob"])
    except ValidationError as e:
        log_pydantic_errors(e, context=context)
        logger.error("Simulation config validation failed; aborting.")
        # exit with non-zero code, NO traceback
        sys.exit(2)
