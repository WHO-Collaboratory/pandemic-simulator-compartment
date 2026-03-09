"""
Auto-generate Pydantic validation models from ``ModelParameterSchema``.

This module bridges the declarative parameter definitions on model classes
with the runtime Pydantic validation layer.  Given a ``ModelParameterSchema``
it produces:

- A ``BaseDiseaseConfig`` subclass with the correct ``disease_type`` Literal,
  ``transmission_edges`` list, and any disease-specific fields.
- Supporting models (``TransmissionEdgeData``, ``TransmissionEdge``) with
  validators derived from parameter metadata.

The generated models are fully compatible with ``SimulationConfig[T]`` and
the existing ``ValidationPostProcessor`` pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, create_model, model_validator

from compartment.parameters import ModelParameterSchema, ValueType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared supporting models (common to all generated disease configs)
# ---------------------------------------------------------------------------


class VarianceParams(BaseModel):
    """Variance configuration for uncertainty analysis."""

    has_variance: bool
    distribution_type: Literal["UNIFORM", "NORMAL"] = "UNIFORM"
    field_name: Optional[str] = None
    min: float = 0
    max: float = 0


class GeneratedTransmissionEdgeData(BaseModel):
    """
    Transmission edge data model (auto-generated compatible).

    Contains the transmission_rate plus optional variance_params
    for uncertainty analysis.
    """

    transmission_rate: float = Field(gt=0)
    variance_params: Optional[VarianceParams] = None


class GeneratedTransmissionEdge(BaseModel):
    """
    Transmission edge model with auto-generated ``id`` from source/target.

    Mirrors the structure of hand-written edge models (e.g.
    ``CovidTransmissionEdge``, ``TransmissionEdge`` in mpox.py).
    """

    source: str
    target: str
    data: GeneratedTransmissionEdgeData
    id: str = ""

    @model_validator(mode="before")
    @classmethod
    def build_id(cls, values):
        if isinstance(values, dict):
            if "id" not in values or not values["id"]:
                source = values.get("source", "")
                target = values.get("target", "")
                values["id"] = f"{source}->{target}"
        return values


# ---------------------------------------------------------------------------
# Python type mapping
# ---------------------------------------------------------------------------

_VALUE_TYPE_TO_PYTHON: dict[ValueType, type] = {
    ValueType.RATE: float,
    ValueType.DAYS: int,
    ValueType.PERCENTAGE: float,
    ValueType.COUNT: int,
    ValueType.DATE: str,
    ValueType.BOOLEAN: bool,
    ValueType.TEXT: str,
    ValueType.SELECT: str,
    ValueType.FLOAT: float,
    ValueType.INTEGER: int,
    ValueType.COORDINATE: float,
}


def _python_type_for(vtype: ValueType) -> type:
    return _VALUE_TYPE_TO_PYTHON.get(vtype, Any)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Cache: disease_type -> generated Pydantic class
_generated_cache: dict[str, type] = {}


def generate_disease_config(schema: ModelParameterSchema) -> type:
    """
    Generate a Pydantic ``BaseDiseaseConfig`` subclass from *schema*.

    The returned class:
    - Has ``disease_type`` as a ``Literal`` matching ``schema.disease_type``
    - Has ``transmission_edges: List[GeneratedTransmissionEdge]``
    - Has any extra disease-specific parameters from ``schema.disease_parameters``
    - Is fully compatible with ``SimulationConfig[T]``

    Results are cached per ``disease_type`` so repeated calls are cheap.
    """
    if schema.disease_type in _generated_cache:
        return _generated_cache[schema.disease_type]

    from compartment.validation.disease_config import BaseDiseaseConfig

    # -- Build dynamic fields for create_model --
    # Pydantic's create_model expects: field_name=(type, FieldInfo | default)
    fields: dict[str, Any] = {}

    # disease_type as a Literal with a default
    disease_literal = Literal[schema.disease_type]  # type: ignore[valid-type]
    fields["disease_type"] = (
        disease_literal, Field(default=schema.disease_type))

    # transmission_edges (required if the model has any edges defined)
    if schema.transmission_edges:
        fields["transmission_edges"] = (
            List[GeneratedTransmissionEdge],
            ...,  # required
        )

    # Disease-specific parameters
    for param in schema.disease_parameters:
        py_type = _python_type_for(param.value_type)

        # Build Field kwargs from parameter metadata
        field_kwargs: dict[str, Any] = {}
        if param.default is not None:
            field_kwargs["default"] = param.default
        if param.min_value is not None:
            field_kwargs["ge"] = param.min_value
        if param.max_value is not None:
            field_kwargs["le"] = param.max_value
        if param.description:
            field_kwargs["description"] = param.description

        if param.required and param.default is None:
            fields[param.name] = (py_type, Field(**field_kwargs))
        elif not param.required and param.default is None:
            fields[param.name] = (
                Optional[py_type],
                Field(default=None, **field_kwargs),
            )
        else:
            fields[param.name] = (py_type, Field(**field_kwargs))

    # Construct a readable class name: "MONKEYPOX" -> "MonkeypoxDiseaseConfig"
    class_name = (
        schema.disease_type.replace("_", " ").title().replace(
            " ", "") + "DiseaseConfig"
    )

    config_cls = create_model(
        class_name,
        __base__=BaseDiseaseConfig,
        **fields,
    )

    _generated_cache[schema.disease_type] = config_cls
    logger.info("Generated Pydantic config class: %s", class_name)
    return config_cls


def has_parameter_schema(model_class: type) -> bool:
    """
    Return True if *model_class* has implemented ``define_parameters()``.

    This lets the validation layer decide whether to use the auto-generated
    Pydantic model or fall back to a hand-written one.
    """
    try:
        model_class._build_parameter_schema()
        return True
    except NotImplementedError:
        return False


def clear_cache() -> None:
    """Clear the generated-model cache (useful in tests)."""
    _generated_cache.clear()
