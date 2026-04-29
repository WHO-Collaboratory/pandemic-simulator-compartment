"""
Central model registry — single source of truth for disease_type → model class.

Discovery is automatic: every Model subclass that declares a DISEASE_TYPE class
attribute is registered.  No manual list to maintain.

Adding a new model:
  1. Create compartment/models/<dir>/model.py with a Model subclass.
  2. Add DISEASE_TYPE = "<YOUR_TYPE>" to the class body.
  3. Done — the registry picks it up on next import.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Legacy aliases: old disease_type strings the backend may still send.
DISEASE_TYPE_ALIASES: dict[str, str] = {
    "RESPIRATORY": "COVID_SEIHDR",
}


def _build_registry() -> dict:
    from compartment.model import Model

    registry: dict[str, type] = {}
    models_dir = Path(__file__).parent / "models"

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        for suffix in ("model", "variants"):
            module_name = f"compartment.models.{model_dir.name}.{suffix}"
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(cls, Model)
                    and cls is not Model
                    and cls.__module__ == module_name
                    and hasattr(cls, "DISEASE_TYPE")
                ):
                    if cls.DISEASE_TYPE in registry:
                        if registry[cls.DISEASE_TYPE] is cls:
                            continue
                        raise RuntimeError(
                            f"Duplicate DISEASE_TYPE '{cls.DISEASE_TYPE}': "
                            f"'{registry[cls.DISEASE_TYPE].__name__}' and '{cls.__name__}' "
                            f"both claim the same disease type."
                        )
                    registry[cls.DISEASE_TYPE] = cls

    return registry


MODEL_REGISTRY: dict[str, type] = _build_registry()


def resolve(disease_type: str) -> type | None:
    """Return the model class for a disease_type, resolving legacy aliases."""
    canonical = DISEASE_TYPE_ALIASES.get(disease_type, disease_type)
    return MODEL_REGISTRY.get(canonical)
