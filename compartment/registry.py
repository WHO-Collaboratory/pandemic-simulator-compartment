"""
Central model registry — unique model_key routing with disease_type aliases.

Discovery is automatic: every Model subclass receives a stable MODEL_KEY and is
registered under it. A disease_type remains a shortcut while it is unambiguous.

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


def _discover_model_classes() -> list[type]:
    from compartment.model import Model

    classes: list[type] = []
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
                    if cls not in classes:
                        classes.append(cls)

    return classes


def _build_registries(model_classes: list[type]) -> tuple[dict[str, type], dict[str, type]]:
    """Build unique model-key routes and unambiguous disease-type aliases."""
    from compartment.model import model_key_for_class

    model_registry: dict[str, type] = {}
    disease_claims: dict[str, list[type]] = {}

    for cls in model_classes:
        model_key = getattr(cls, "MODEL_KEY", None) or model_key_for_class(
            cls, cls.DISEASE_TYPE
        )
        cls.MODEL_KEY = model_key
        if model_key in model_registry and model_registry[model_key] is not cls:
            raise RuntimeError(f"Duplicate MODEL_KEY '{model_key}'")
        model_registry[model_key] = cls
        disease_claims.setdefault(cls.DISEASE_TYPE, []).append(cls)

    disease_registry = {
        disease_type: classes[0]
        for disease_type, classes in disease_claims.items()
        if len(classes) == 1
    }
    return model_registry, disease_registry


MODEL_KEY_REGISTRY, DISEASE_TYPE_REGISTRY = _build_registries(
    _discover_model_classes()
)

# Backward-compatible public registry: unique disease types keep their familiar
# key; only ambiguous disease types expand into one model_key entry per class.
MODEL_REGISTRY: dict[str, type] = dict(DISEASE_TYPE_REGISTRY)
for model_key, model_class in MODEL_KEY_REGISTRY.items():
    if model_class.DISEASE_TYPE not in DISEASE_TYPE_REGISTRY:
        MODEL_REGISTRY[model_key] = model_class


def resolve(identifier: str) -> type | None:
    """Resolve a unique model_key or an unambiguous legacy disease_type."""
    if identifier in MODEL_KEY_REGISTRY:
        return MODEL_KEY_REGISTRY[identifier]
    canonical = DISEASE_TYPE_ALIASES.get(identifier, identifier)
    return DISEASE_TYPE_REGISTRY.get(canonical)
