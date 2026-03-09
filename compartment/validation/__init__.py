# New simplified validation structure (default)
from .disease_config import BaseDiseaseConfig
from .simulation_config import SimulationConfig
from .post_processor import ValidationPostProcessor, ProcessedSimulation

# Hand-written disease configs (non-migrated models)
from .diseases import (
    CovidDiseaseConfig,
    DengueDiseaseConfig,
    Dengue2StrainDiseaseConfig,
)

# Auto-generated disease config (migrated to declarative parameters)
from .diseases import MpoxDiseaseConfig

# Shared models still used
from .base_simulation import BaseSimulationShared, TravelVolume
from .interventions import (
    Intervention,
    InterventionVarianceParams,
)

import logging
import sys
from pydantic import ValidationError

from compartment.schema_generator import generate_disease_config, has_parameter_schema

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

# ---------------------------------------------------------------------------
# Model registry: disease_type -> model class
# Lazy import to avoid circular dependencies at module level.
# ---------------------------------------------------------------------------


def _get_model_registry() -> dict:
    from compartment.models.covid_jax_model.model import CovidJaxModel
    from compartment.models.dengue_jax_model.model import DengueJaxModel
    from compartment.models.dengue_2strain.model import Dengue2StrainModel
    from compartment.models.mpox_jax_model.model import MpoxJaxModel

    return {
        "RESPIRATORY": CovidJaxModel,
        "VECTOR_BORNE": DengueJaxModel,
        "VECTOR_BORNE_2STRAIN": Dengue2StrainModel,
        "MONKEYPOX": MpoxJaxModel,
    }


# ---------------------------------------------------------------------------
# Fallback mapping for models that haven't migrated to define_parameters()
# ---------------------------------------------------------------------------

_FALLBACK_DISEASE_CONFIG = {
    "RESPIRATORY": CovidDiseaseConfig,
    "VECTOR_BORNE": DengueDiseaseConfig,
    "VECTOR_BORNE_2STRAIN": Dengue2StrainDiseaseConfig,
}


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


def _resolve_disease_config(disease_type: str):
    """
    Resolve the Pydantic disease config class for a given disease type.

    For models that have implemented ``define_parameters()`` the config is
    auto-generated from the parameter schema.  Otherwise falls back to the
    hand-written config class.
    """
    registry = _get_model_registry()
    model_class = registry.get(disease_type)

    if model_class and has_parameter_schema(model_class):
        schema = model_class._build_parameter_schema()
        return generate_disease_config(schema)

    # Fallback to hand-written configs
    if disease_type in _FALLBACK_DISEASE_CONFIG:
        return _FALLBACK_DISEASE_CONFIG[disease_type]

    raise ValueError(f"Invalid disease type: {disease_type}")


def load_simulation_config(config: dict, disease_type: str):
    """
    Centralized validation entrypoint.

    For models that have migrated to declarative parameter definitions the
    Pydantic disease config is **auto-generated** from ``define_parameters()``.
    Non-migrated models (COVID, Dengue) still use hand-written configs.

    Args:
        config: Configuration dict (usually from GraphQL/JSON)
        disease_type: Type of disease ("RESPIRATORY", "VECTOR_BORNE", "MONKEYPOX")

    Returns:
        ProcessedSimulation with all computed fields ready for use by Model classes.

    Example:
        config = load_simulation_config(data, "RESPIRATORY")
        model = CovidJaxModel(config)
    """
    disease_cls = _resolve_disease_config(disease_type)
    context = f"SimulationConfig[{disease_cls.__name__}]"

    try:
        # Step 1: Validate
        validated_config = SimulationConfig[disease_cls](
            **config["data"]["getSimulationJob"]
        )
        # Step 2: Post-process
        processed = ValidationPostProcessor.process(validated_config)
        return processed
    except ValidationError as e:
        log_pydantic_errors(e, context=context)
        logger.error("Simulation config validation failed; aborting.")
        sys.exit(2)
