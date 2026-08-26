"""Artifact discovery and model registry tests.

Verifies that _discover_models_from_dir finds the correct classes and that
MODEL_REGISTRY is consistent with what's defined across all model files.

Run:
    python3 -m pytest tests/test_artifact.py -v
"""

import json

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

    def test_validation_keeps_model_selected_by_unique_key(self, monkeypatch):
        """Post-processing must not re-resolve an ambiguous disease_type."""
        import compartment.registry as registry_module
        import compartment.validation as validation_module
        from compartment.helpers import load_config_from_json
        from compartment.models.example_stochastic_model.model import (
            ExampleStochasticModel,
        )

        raw = load_config_from_json(
            "compartment/models/example_stochastic_model/example-config.json"
        )
        model_key = ExampleStochasticModel.MODEL_KEY

        # Simulate two models claiming the same disease type: the unique key is
        # resolvable, but the disease-type shortcut is deliberately unavailable.
        monkeypatch.setattr(
            validation_module,
            "_get_model_registry",
            lambda: {model_key: ExampleStochasticModel},
        )
        monkeypatch.setattr(registry_module, "resolve", lambda _identifier: None)

        processed = validation_module.load_simulation_config(raw, model_key)

        assert processed.compartment_list == ExampleStochasticModel.COMPARTMENT_LIST


class TestExampleConfigGeneration:
    """Generated examples use the same normalized shape as the runtime."""

    def test_emits_normalized_runtime_sections(self):
        from compartment.models.mpox_jax_model.model import MpoxJaxModel
        from compartment.models.example_stochastic_model.model import (
            ExampleStochasticModel,
        )

        example = MpoxJaxModel.generate_example_config()

        assert "transmission_edges" not in example["Disease"]
        assert "interventions" not in example
        assert example["run_mode"] == "DETERMINISTIC"
        assert example["simulation_type"] == "COMPARTMENTAL"
        assert (
            ExampleStochasticModel.generate_example_config()["run_mode"]
            == "STOCHASTIC"
        )

        edge_items = example["TransmissionEdges"]["items"]
        assert [item["value"] for item in edge_items] == [0.3, 10.0, 60.0]
        assert [
            item["transmission_edge"]["value_type"] for item in edge_items
        ] == ["RATE", "DAYS", "DAYS"]
        assert [
            item["FieldConfigs"]["items"][0]["disease_param"]
            for item in edge_items
        ] == ["BETA", "GAMMA", "OMEGA"]
        assert all(
            item["FieldConfigs"]["items"][0]["has_variance"] is False
            for item in edge_items
        )

        intervention = example["Interventions"]["items"][0]
        assert intervention["Intervention"] == {
            "name": "ring_vaccination",
            "display_name": "Ring Vaccination",
        }
        assert intervention["adherence_min"] == 70.0
        assert intervention["transmission_percentage"] == 75.0

    def test_uncertainty_mode_uses_schema_declared_bounds(self):
        from compartment.registry import resolve

        covid_model = resolve("COVID_SEIHDR")
        example = covid_model.generate_example_config(uncertainty=True)

        assert example["run_mode"] == "UNCERTAINTY"
        assert example["n_simulations"] == 30
        assert "variance_params" not in example["Disease"]

        edge_configs = {
            item["FieldConfigs"]["items"][0]["disease_param"]: item[
                "FieldConfigs"
            ]["items"][0]
            for item in example["TransmissionEdges"]["items"]
        }
        assert edge_configs["BETA"] == {
            "field_key": "value",
            "has_variance": True,
            "distribution_type": "UNIFORM",
            "disease_param": "BETA",
            "min": 0.2,
            "max": 0.3,
        }
        assert edge_configs["THETA"]["min"] == 2.0
        assert edge_configs["THETA"]["max"] == 14.0
        assert all(config["has_variance"] for config in edge_configs.values())

        # Disease-specific parameters use the local config's inline variance
        # representation rather than TransmissionEdges FieldConfigs.
        klebsiella_model = resolve("KLEBSIELLA_AMR")
        disease_example = klebsiella_model.generate_example_config(
            uncertainty=True
        )
        disease_ranges = {
            item["param"]: item
            for item in disease_example["Disease"]["variance_params"]
        }
        assert disease_ranges["hospital_transmission_mult"] == {
            "param": "hospital_transmission_mult",
            "dist": "uniform",
            "min": 5.0,
            "max": 30.0,
        }

    def test_cli_uncertainty_flag(self, monkeypatch, capsys):
        from compartment.generate_artifact import main

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_artifact",
                "COVID_SEIHDR",
                "--example-config",
                "--uncertainty",
            ],
        )

        main()

        example = json.loads(capsys.readouterr().out)
        beta_config = example["TransmissionEdges"]["items"][0][
            "FieldConfigs"
        ]["items"][0]
        assert beta_config["has_variance"] is True
        assert beta_config["min"] == 0.2
        assert beta_config["max"] == 0.3

    def test_generated_config_round_trips_to_model_parameters(self, tmp_path):
        from compartment.helpers import load_config_from_json
        from compartment.models.mpox_jax_model.model import MpoxJaxModel
        from compartment.validation import load_simulation_config

        config_path = tmp_path / "example-config.json"
        config_path.write_text(
            json.dumps(MpoxJaxModel.generate_example_config()),
            encoding="utf-8",
        )

        raw = load_config_from_json(str(config_path))
        processed = load_simulation_config(raw, MpoxJaxModel.MODEL_KEY)

        assert processed.transmission_dict == {
            "beta": 0.3,
            "gamma": 10.0,
            "omega": 60.0,
        }
        assert processed.intervention_dict["ring_vaccination"][
            "adherence_min"
        ] == pytest.approx(0.7)
        assert processed.intervention_dict["ring_vaccination"][
            "transmission_percentage"
        ] == pytest.approx(0.75)

        # Exercise the model's schema-driven conversion without invoking its
        # dataset-backed constructor.
        model = MpoxJaxModel.__new__(MpoxJaxModel)
        model._load_transmission_params(processed.transmission_dict)
        assert model.beta == pytest.approx(0.3)
        assert model.gamma == pytest.approx(0.1)
        assert model.omega == pytest.approx(1 / 60)

    def test_loader_upgrades_legacy_generated_sections(self, tmp_path):
        from compartment.helpers import load_config_from_json
        from compartment.models.test_klebsiella_amr_model.model import (
            KlebsiellaAmrModel,
        )

        schema = KlebsiellaAmrModel._build_parameter_schema()
        legacy = schema.to_example_config()

        # Recreate the format emitted by the old generator. Klebsiella uses
        # custom source/target pairs, so this also verifies schema-aware edge
        # recovery rather than the small hardcoded fallback mapping.
        legacy_edges = [
            {
                "source": edge.source,
                "target": edge.target,
                "data": {"transmission_rate": edge.parameter.default},
            }
            for edge in schema.transmission_edges
        ]
        legacy_edges[0]["data"]["variance_params"] = {
            "has_variance": True,
            "distribution_type": "UNIFORM",
            "field_name": None,
            "min": 0.005,
            "max": 0.015,
        }
        legacy["Disease"]["transmission_edges"] = legacy_edges
        legacy.pop("TransmissionEdges")
        legacy_interventions = [
            {
                "id": intervention.id,
                **{
                    parameter.name: parameter.default
                    for parameter in intervention.parameters
                },
            }
            for intervention in schema.interventions
        ]
        legacy_interventions[0]["variance_params"] = [
            {
                "has_variance": True,
                "distribution_type": "UNIFORM",
                "field_name": "adherence_min",
                "min": 70.0,
                "max": 90.0,
            }
        ]
        legacy["interventions"] = legacy_interventions
        legacy.pop("Interventions")

        config_path = tmp_path / "legacy-example-config.json"
        config_path.write_text(json.dumps(legacy), encoding="utf-8")
        job = load_config_from_json(str(config_path))["data"]["getSimulationJob"]

        assert "transmission_edges" not in job["Disease"]
        assert "interventions" not in job
        assert [
            item["FieldConfigs"]["items"][0]["disease_param"]
            for item in job["TransmissionEdges"]["items"]
        ] == [edge.variable_name.upper() for edge in schema.transmission_edges]
        assert [
            item["transmission_edge"]["value_type"]
            for item in job["TransmissionEdges"]["items"]
        ] == [edge.to_dict()["value_type"] for edge in schema.transmission_edges]
        edge_variance = job["TransmissionEdges"]["items"][0]["FieldConfigs"][
            "items"
        ][0]
        assert edge_variance["has_variance"] is True
        assert edge_variance["min"] == 0.005
        assert edge_variance["max"] == 0.015
        assert [
            item["Intervention"]["name"]
            for item in job["Interventions"]["items"]
        ] == [intervention.id for intervention in schema.interventions]
        intervention_variance = job["Interventions"]["items"][0]["FieldConfigs"][
            "items"
        ][0]
        assert intervention_variance == {
            "field_key": "adherence_min",
            "has_variance": True,
            "distribution_type": "UNIFORM",
            "min": 70.0,
            "max": 90.0,
        }
