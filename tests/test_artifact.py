"""Artifact discovery and model registry tests.

Verifies that _discover_models_from_dir finds the correct classes and that
MODEL_REGISTRY is consistent with what's defined across all model files.

Run:
    python3 -m pytest tests/test_artifact.py -v
"""

import pytest


class TestArtifactDiscovery:
    """Tests for generate_artifact._discover_models_from_dir."""

    def test_discovers_all_covid_variants(self):
        from compartment.generate_artifact import _discover_models_from_dir
        classes = _discover_models_from_dir("compartment/models/covid_jax_model")
        disease_types = {cls.DISEASE_TYPE for cls in classes}
        expected = {
            "COVID_SEIHDR", "COVID_SIR", "COVID_SEIR",
            "COVID_SIHR", "COVID_SIDR", "COVID_SEIHR",
            "COVID_SEIDR", "COVID_SIHDR",
        }
        assert disease_types == expected, (
            f"Expected {sorted(expected)}, discovered {sorted(disease_types)}"
        )

    def test_discovers_dengue(self):
        from compartment.generate_artifact import _discover_models_from_dir
        from compartment.models.dengue_jax_model.model import DengueJaxModel
        classes = _discover_models_from_dir("compartment/models/dengue_jax_model")
        assert any(issubclass(c, DengueJaxModel) for c in classes), (
            "Expected DengueJaxModel to be discovered"
        )

    def test_discovers_mpox(self):
        from compartment.generate_artifact import _discover_models_from_dir
        from compartment.models.mpox_jax_model.model import MpoxJaxModel
        classes = _discover_models_from_dir("compartment/models/mpox_jax_model")
        assert any(issubclass(c, MpoxJaxModel) for c in classes), (
            "Expected MpoxJaxModel to be discovered"
        )

    def test_covid_class_count(self):
        """Exactly 8 COVID variant classes should be discoverable."""
        from compartment.generate_artifact import _discover_models_from_dir
        classes = _discover_models_from_dir("compartment/models/covid_jax_model")
        assert len(classes) == 8, (
            f"Expected 8 COVID model classes (SIR through SEIHDR), found {len(classes)}: "
            f"{[c.__name__ for c in classes]}"
        )

    def test_registry_covers_all_model_files(self):
        """MODEL_REGISTRY must contain one entry per Model subclass with DISEASE_TYPE
        found across all models/*/model.py and models/*/variants.py files."""
        import importlib
        import inspect
        from pathlib import Path
        from compartment.model import Model
        from compartment.registry import MODEL_REGISTRY

        models_dir = Path("compartment/models")
        discovered: set[str] = set()
        for model_dir in models_dir.iterdir():
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
                        discovered.add(cls.DISEASE_TYPE)

        missing = discovered - set(MODEL_REGISTRY.keys())
        assert not missing, f"Classes with DISEASE_TYPE not in MODEL_REGISTRY: {missing}"
        assert len(MODEL_REGISTRY) == len(discovered), (
            f"Registry has {len(MODEL_REGISTRY)} entries but {len(discovered)} "
            f"model classes with DISEASE_TYPE were found"
        )
