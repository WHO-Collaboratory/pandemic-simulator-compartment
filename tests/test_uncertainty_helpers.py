"""Fast unit tests for the variance-collection helpers in compartment.helpers.

These exercise the run_mode promotion logic and multi-source uncertainty
collection without running a full simulation.

Run:
    python -m pytest tests/test_uncertainty_helpers.py -v
"""

from types import SimpleNamespace

from compartment.helpers import (
    collect_uncertainty_params,
    extract_disease_variance_params,
    resolve_run_mode,
)


class TestExtractDiseaseVarianceParams:
    def test_empty_section_returns_empty(self):
        assert extract_disease_variance_params({}) == []
        assert extract_disease_variance_params(None) == []

    def test_no_variance_params_key(self):
        assert extract_disease_variance_params({"immunity_period": 240}) == []

    def test_normalizes_entries(self):
        section = {
            "variance_params": [
                {"param": "immunity_period", "dist": "normal", "min": 90, "max": 365},
            ]
        }
        result = extract_disease_variance_params(section)
        assert result == [
            {"param": "immunity_period", "dist": "normal", "min": 90, "max": 365}
        ]

    def test_dist_defaults_to_uniform(self):
        section = {"variance_params": [{"param": "beta", "min": 0.1, "max": 0.4}]}
        result = extract_disease_variance_params(section)
        assert result[0]["dist"] == "uniform"


class TestResolveRunMode:
    def test_promotes_deterministic_when_params_present(self):
        params = [{"param": "beta", "dist": "uniform", "min": 0.1, "max": 0.4}]
        assert resolve_run_mode("DETERMINISTIC", params) == "UNCERTAINTY"

    def test_no_promotion_without_params(self):
        assert resolve_run_mode("DETERMINISTIC", []) == "DETERMINISTIC"

    def test_leaves_uncertainty_untouched(self):
        params = [{"param": "beta", "dist": "uniform", "min": 0.1, "max": 0.4}]
        assert resolve_run_mode("UNCERTAINTY", params) == "UNCERTAINTY"
        assert resolve_run_mode("UNCERTAINTY", []) == "UNCERTAINTY"


class TestCollectUncertaintyParams:
    def test_disease_params_only(self):
        cfg = SimpleNamespace(TransmissionEdges=None, Interventions=None)
        disease = [{"param": "immunity_period", "dist": "uniform", "min": 90, "max": 365}]
        result = collect_uncertainty_params(cfg, disease)
        assert result == disease

    def test_none_disease_configs(self):
        cfg = SimpleNamespace(TransmissionEdges=None, Interventions=None)
        assert collect_uncertainty_params(cfg, None) == []

    def test_collects_transmission_edge_variance(self):
        cfg = SimpleNamespace(
            TransmissionEdges={
                "items": [
                    {
                        "transmission_edge": {"source": "S", "target": "E"},
                        "FieldConfigs": {
                            "items": [
                                {
                                    "disease_param": "beta",
                                    "has_variance": True,
                                    "distribution_type": "uniform",
                                    "min": 0.1,
                                    "max": 0.4,
                                }
                            ]
                        },
                    }
                ]
            },
            Interventions=None,
        )
        result = collect_uncertainty_params(cfg, None)
        assert {"param": "beta", "dist": "uniform", "min": 0.1, "max": 0.4} in result

    def test_merges_all_sources(self):
        cfg = SimpleNamespace(
            TransmissionEdges={
                "items": [
                    {
                        "transmission_edge": {"source": "S", "target": "E"},
                        "FieldConfigs": {
                            "items": [
                                {
                                    "disease_param": "beta",
                                    "has_variance": True,
                                    "distribution_type": "uniform",
                                    "min": 0.1,
                                    "max": 0.4,
                                }
                            ]
                        },
                    }
                ]
            },
            Interventions={
                "items": [
                    {
                        "Intervention": {"name": "PHYSICAL"},
                        "FieldConfigs": {
                            "items": [
                                {
                                    "field_key": "adherence_min",
                                    "has_variance": True,
                                    "distribution_type": "uniform",
                                    "min": 20.0,
                                    "max": 80.0,
                                }
                            ]
                        },
                    }
                ]
            },
        )
        disease = [{"param": "immunity_period", "dist": "uniform", "min": 90, "max": 365}]
        result = collect_uncertainty_params(cfg, disease)
        params = {p["param"] for p in result}
        assert "beta" in params
        assert "intervention.physical.adherence_min" in params
        assert "immunity_period" in params

    def test_field_config_without_variance_ignored(self):
        cfg = SimpleNamespace(
            TransmissionEdges={
                "items": [
                    {
                        "transmission_edge": {"source": "S", "target": "E"},
                        "FieldConfigs": {
                            "items": [
                                {
                                    "disease_param": "beta",
                                    "has_variance": False,
                                    "min": 0.1,
                                    "max": 0.4,
                                }
                            ]
                        },
                    }
                ]
            },
            Interventions=None,
        )
        assert collect_uncertainty_params(cfg, None) == []
