"""
Monkeypox disease validation config -- auto-generated from model parameters.

This module no longer contains hand-written Pydantic models.  Instead it
imports the ``MpoxJaxModel`` parameter schema and uses the schema generator
to produce a ``MpoxDiseaseConfig`` class at import time.

The generated class is fully compatible with ``SimulationConfig[T]`` and
the ``ValidationPostProcessor`` pipeline.
"""

from __future__ import annotations

from compartment.models.mpox_jax_model.model import MpoxJaxModel
from compartment.schema_generator import generate_disease_config

# Auto-generate MpoxDiseaseConfig from the model's parameter schema
MpoxDiseaseConfig = generate_disease_config(
    MpoxJaxModel._build_parameter_schema())

# Re-export the supporting edge models so existing imports still work
from compartment.schema_generator import (  # noqa: E402, F401
    GeneratedTransmissionEdge as TransmissionEdge,
    GeneratedTransmissionEdgeData as TransmissionEdgeData,
)

__all__ = [
    "MpoxDiseaseConfig",
    "TransmissionEdge",
    "TransmissionEdgeData",
]
