# New simplified validation structure (default)
from .disease_config import BaseDiseaseConfig
from .simulation_config import SimulationConfig
from .post_processor import ValidationPostProcessor, ProcessedSimulation

# Hand-written disease configs (non-migrated models)
from .diseases import (
    CovidDiseaseConfig,
    DengueDiseaseConfig,
)

# Auto-generated disease configs (migrated to declarative parameters)
from .diseases import MpoxDiseaseConfig
from .diseases import KlebsiellaAmrDiseaseConfig
from .diseases import CovidSirStochasticDiseaseConfig

# Shared models still used
from .base_simulation import BaseSimulationShared
from .field_configs import FieldConfig, FieldConfigItems
from .interventions import (
    NormalizedIntervention,
    NormalizedInterventions,
    InterventionLookup,
)
from .transmission_edges import (
    NormalizedTransmissionEdge,
    NormalizedTransmissionEdges,
    TransmissionEdgeLookup,
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
    "MpoxDiseaseConfig",
    "KlebsiellaAmrDiseaseConfig",
    "CovidSirStochasticDiseaseConfig",
    # Shared models
    "BaseSimulationShared",
    # Normalized models
    "FieldConfig",
    "FieldConfigItems",
    "NormalizedIntervention",
    "NormalizedInterventions",
    "InterventionLookup",
    "NormalizedTransmissionEdge",
    "NormalizedTransmissionEdges",
    "TransmissionEdgeLookup",
    # Utility functions
    "log_pydantic_errors",
    "load_simulation_config",
]

logger = logging.getLogger("compartment.validation")

# ---------------------------------------------------------------------------
# Model registry: model_key plus unambiguous disease_type aliases -> model class
# Lazy import to avoid circular dependencies at module level.
# ---------------------------------------------------------------------------


def _get_model_registry() -> dict:
    from compartment.registry import (
        DISEASE_TYPE_ALIASES,
        DISEASE_TYPE_REGISTRY,
        MODEL_KEY_REGISTRY,
        MODEL_REGISTRY,
    )
    return {
        **MODEL_REGISTRY,
        **MODEL_KEY_REGISTRY,
        **DISEASE_TYPE_REGISTRY,
        **{
            alias: DISEASE_TYPE_REGISTRY[canonical]
            for alias, canonical in DISEASE_TYPE_ALIASES.items()
            if canonical in DISEASE_TYPE_REGISTRY
        },
    }


# ---------------------------------------------------------------------------
# Fallback mapping for models that haven't migrated to define_parameters()
# ---------------------------------------------------------------------------

_FALLBACK_DISEASE_CONFIG = {
    "VECTOR_BORNE": DengueDiseaseConfig,
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


def _resolve_disease_config(disease_type: str, registry: dict | None = None):
    """
    Resolve the Pydantic disease config class for a given disease type.

    For models that have implemented ``define_parameters()`` the config is
    auto-generated from the parameter schema.  Otherwise falls back to the
    hand-written config class.
    """
    registry = registry if registry is not None else _get_model_registry()
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
        disease_type: Unique model_key, or an unambiguous disease type.

    Returns:
        ProcessedSimulation with all computed fields ready for use by Model classes.

    Example:
        config = load_simulation_config(data, "RESPIRATORY")
        model = CovidJaxModel(config)
    """
    # ``disease_type`` is the legacy parameter name. Modern callers pass a
    # unique model_key so multiple model implementations may intentionally
    # share the same Disease.disease_type value.
    registry = _get_model_registry()
    model_class = registry.get(disease_type)
    disease_cls = _resolve_disease_config(disease_type, registry=registry)
    context = f"SimulationConfig[{disease_cls.__name__}]"

    try:
        # Step 1: Validate
        validated_config = SimulationConfig[disease_cls](
            **config["data"]["getSimulationJob"]
        )
        # Step 2: Post-process
        # Preserve the exact class selected by model_key. Re-resolving from
        # validated_config.Disease.disease_type would be ambiguous when two
        # models cover the same disease.
        processed = ValidationPostProcessor.process(
            validated_config, model_class=model_class
        )
        return processed
    except ValidationError as e:
        log_pydantic_errors(e, context=context)
        logger.error("Simulation config validation failed; aborting.")
        sys.exit(2)
