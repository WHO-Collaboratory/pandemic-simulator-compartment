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
        discovered: set[type] = set()
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
                        discovered.add(cls)

        missing = discovered - set(MODEL_REGISTRY.values())
        assert not missing, f"Model classes missing from MODEL_REGISTRY: {missing}"
        assert len(MODEL_REGISTRY) == len(discovered), (
            f"Registry has {len(MODEL_REGISTRY)} entries but {len(discovered)} "
            f"model classes with DISEASE_TYPE were found"
        )

    def test_duplicate_disease_types_get_distinct_internal_routes(self):
        from compartment.registry import _build_registries

        class FirstModel:
            DISEASE_TYPE = "SHARED_DISEASE"
            MODEL_KEY = "SHARED_DISEASE_11111111-1111-5111-8111-111111111111"

        class SecondModel:
            DISEASE_TYPE = "SHARED_DISEASE"
            MODEL_KEY = "SHARED_DISEASE_22222222-2222-5222-8222-222222222222"

        models, disease_types = _build_registries([FirstModel, SecondModel])

        assert models[FirstModel.MODEL_KEY] is FirstModel
        assert models[SecondModel.MODEL_KEY] is SecondModel
        assert "SHARED_DISEASE" not in disease_types

    def test_model_key_is_stable_and_included_in_artifact(self):
        from compartment.models.mpox_jax_model.model import MpoxJaxModel
        from compartment.registry import resolve

        first = MpoxJaxModel._build_parameter_schema()
        second = MpoxJaxModel._build_parameter_schema()

        assert first.model_key == second.model_key == MpoxJaxModel.MODEL_KEY
        assert first.model_key.startswith(f"{first.disease_type}_")
        assert first.to_artifact_dict()["model_key"] == first.model_key
        assert resolve(first.model_key) is MpoxJaxModel
        assert resolve(first.disease_type) is MpoxJaxModel

    def test_validation_cache_isolated_for_duplicate_disease_types(self):
        from compartment.model import Model
        from compartment.parameters import ValueType
        from compartment.schema_generator import clear_cache, generate_disease_config

        class FirstModel(Model):
            @classmethod
            def define_parameters(cls, schema):
                schema.set_model_info("SHARED", "First", "First model")
                schema.add_compartment("S", "Susceptible", "Susceptible")
                schema.add_disease_parameter(
                    "first_only", "First only", "First parameter", ValueType.FLOAT, 1.0
                )

        class SecondModel(Model):
            @classmethod
            def define_parameters(cls, schema):
                schema.set_model_info("SHARED", "Second", "Second model")
                schema.add_compartment("S", "Susceptible", "Susceptible")
                schema.add_disease_parameter(
                    "second_only", "Second only", "Second parameter", ValueType.FLOAT, 1.0
                )

        clear_cache()
        first_config = generate_disease_config(FirstModel._build_parameter_schema())
        second_config = generate_disease_config(SecondModel._build_parameter_schema())

        assert first_config is not second_config
        assert "first_only" in first_config.model_fields
        assert "second_only" in second_config.model_fields
