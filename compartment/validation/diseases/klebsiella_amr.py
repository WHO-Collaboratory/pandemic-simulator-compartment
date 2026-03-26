"""
Klebsiella AMR disease validation config -- auto-generated from model parameters.

This module imports the ``KlebsiellaAmrModel`` parameter schema and uses the
schema generator to produce a ``KlebsiellaAmrDiseaseConfig`` class at import
time.

The generated class is fully compatible with ``SimulationConfig[T]`` and
the ``ValidationPostProcessor`` pipeline.
"""

from __future__ import annotations

from compartment.models.test_klebsiella_amr_model.model import KlebsiellaAmrModel
from compartment.schema_generator import generate_disease_config

# Auto-generate KlebsiellaAmrDiseaseConfig from the model's parameter schema
KlebsiellaAmrDiseaseConfig = generate_disease_config(
    KlebsiellaAmrModel._build_parameter_schema())

# Re-export the supporting edge models so existing imports still work
from compartment.schema_generator import (  # noqa: E402, F401
    GeneratedTransmissionEdge as TransmissionEdge,
    GeneratedTransmissionEdgeData as TransmissionEdgeData,
)

__all__ = [
    "KlebsiellaAmrDiseaseConfig",
    "TransmissionEdge",
    "TransmissionEdgeData",
]
