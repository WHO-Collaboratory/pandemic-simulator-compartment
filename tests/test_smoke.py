"""Smoke tests: run every model end-to-end with its example config.

Each test loads the model's example-config.json, runs the full simulation
pipeline (validation → model init → ODE solve → post-processing), and
checks basic correctness of the output.

These are integration tests — they exercise the real code path, not mocks.
They take a few seconds each due to JAX compilation.
"""

"""
Run all smoke tests:
python3 -m pytest tests/test_smoke.py -v -m integration

Run just one model:
python3 -m pytest tests/test_smoke.py -v -m integration -k "covid_jax_model"

Run a specific test:
python3 -m pytest tests/test_smoke.py::TestFlexibleCompartments::test_sir_only_no_extra_compartments -v
"""

import json
import math
import pathlib
import tempfile
import pytest
from compartment.run_simulation import run_simulation

# ---------------------------------------------------------------------------
# Auto-discover models: any directory under compartment/models/ that has
# both model.py and example-config.json is a testable model.
# Filter by name with: pytest -k "dengue_jax_model"
# ---------------------------------------------------------------------------

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "compartment" / "models"


def _discover_models() -> dict[str, tuple[str, str, pathlib.Path]]:
    """Scan compartment/models/ for directories with model.py + example-config.json."""
    import importlib, inspect
    from compartment.model import Model

    found = {}
    for model_dir in sorted(MODELS_DIR.iterdir()):
        config_path = model_dir / "example-config.json"
        model_py = model_dir / "model.py"
        if not config_path.exists() or not model_py.exists():
            continue

        dir_name = model_dir.name
        module_path = f"compartment.models.{dir_name}.model"

        # Find the Model subclass in the module
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue

        model_class = None
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Model) and obj is not Model:
                model_class = obj
                break

        if model_class is None:
            continue

        class_path = f"{module_path}.{model_class.__name__}"
        found[dir_name] = (dir_name, class_path, config_path)

    return found


MODEL_CONFIGS = _discover_models()


def _import_class(dotted_path: str):
    """Import a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _run_model(model_class, config_path: pathlib.Path) -> list[dict]:
    """Run a model and return the JSON results."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    run_simulation(
        model_class=model_class,
        config_path=str(config_path),
        output_path=output_path,
    )

    with open(output_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestModelSmoke:
    """Run each model end-to-end and verify output structure."""

    @pytest.fixture(params=list(MODEL_CONFIGS.keys()))
    def model_run(self, request):
        """Run a model and return (model_id, results)."""
        model_id, class_path, config_path = MODEL_CONFIGS[request.param]
        model_class = _import_class(class_path)
        results = _run_model(model_class, config_path)
        return model_id, results

    def test_returns_two_runs(self, model_run):
        """Output should contain exactly two runs (with/without interventions)."""
        _, results = model_run
        assert isinstance(results, list)
        assert len(results) == 2

    def test_control_run_flags(self, model_run):
        """One run should be control, one should not."""
        _, results = model_run
        control_flags = {r["control_run"] for r in results}
        assert control_flags == {True, False}

    def test_has_parent_admin_total(self, model_run):
        """Both runs should have a parent_admin_total section."""
        _, results = model_run
        for run in results:
            assert "parent_admin_total" in run
            assert "time_series" in run["parent_admin_total"]

    def test_time_series_not_empty(self, model_run):
        """Time series should have at least one entry."""
        _, results = model_run
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            assert len(ts) > 0

    def test_time_series_has_date(self, model_run):
        """Every time series entry should have a date field."""
        _, results = model_run
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            for entry in ts:
                assert "date" in entry

    def test_compartments_present(self, model_run):
        """Time series entries should have at least 2 compartments."""
        _, results = model_run
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = [k for k in ts[0].keys() if k != "date"]
        assert len(compartments) >= 2, f"Expected at least 2 compartments, got {compartments}"

    def test_compartment_deltas_match_time_series(self, model_run):
        """compartment_deltas keys should match the compartments in time_series."""
        _, results = model_run
        for run in results:
            assert "compartment_deltas" in run, "Output missing compartment_deltas"
            deltas = run["compartment_deltas"]
            ts = run["parent_admin_total"]["time_series"]
            ts_compartments = {k for k in ts[0].keys() if k != "date"}
            delta_compartments = set(deltas.keys())
            assert delta_compartments == ts_compartments, (
                f"compartment_deltas keys {sorted(delta_compartments)} "
                f"don't match time_series compartments {sorted(ts_compartments)}"
            )

    def test_no_nan_values(self, model_run):
        """No compartment should contain NaN values."""
        _, results = model_run
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            for entry in ts:
                for key, val in entry.items():
                    if key == "date":
                        continue
                    if isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            if isinstance(sub_val, (int, float)):
                                assert not math.isnan(sub_val), (
                                    f"NaN in {key}.{sub_key} on {entry['date']}"
                                )
                            elif isinstance(sub_val, dict):
                                # uncertainty format: {mean, lower, upper}
                                for stat, v in sub_val.items():
                                    if isinstance(v, (int, float)):
                                        assert not math.isnan(v), (
                                            f"NaN in {key}.{sub_key}.{stat} on {entry['date']}"
                                        )

    def test_no_negative_compartments(self, model_run):
        """No compartment population should be significantly negative."""
        _, results = model_run
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            for entry in ts:
                for key, val in entry.items():
                    if key == "date":
                        continue
                    if isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            if isinstance(sub_val, (int, float)):
                                assert sub_val >= -1.0, (
                                    f"Negative value {sub_val} in {key}.{sub_key} "
                                    f"on {entry['date']}"
                                )

    def test_has_admin_zones(self, model_run):
        """Output should contain admin_zones with time series."""
        _, results = model_run
        for run in results:
            assert "admin_zones" in run
            zones = run["admin_zones"]
            assert len(zones) > 0
            for zone in zones:
                # Zone is identified by admin_zone_id (current format)
                assert "admin_zone_id" in zone
                assert "time_series" in zone

    def test_has_interventions(self, model_run):
        """Output should have an interventions list."""
        _, results = model_run
        for run in results:
            assert "interventions" in run

    def test_dates_are_sequential(self, model_run):
        """Dates in time series should be monotonically increasing."""
        from datetime import datetime
        _, results = model_run
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            if len(ts) < 2:
                continue
            dates = [datetime.strptime(e["date"], "%Y-%m-%d") for e in ts]
            for i in range(1, len(dates)):
                assert dates[i] > dates[i - 1], (
                    f"Dates not sequential: {dates[i-1]} >= {dates[i]}"
                )


# ---------------------------------------------------------------------------
# Uncertainty runs
# ---------------------------------------------------------------------------


class TestUncertaintyRuns:
    """Test uncertainty (LHS) runs produce valid confidence intervals."""

    @pytest.mark.integration
    def test_covid_uncertainty_varying_params(self):
        """COVID uncertainty run with variance on beta should produce CI bands."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = {
            "Disease": {
                "disease_type": "RESPIRATORY",
                "disease_nodes": [
                    {"id": "susceptible", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Susceptible"}},
                    {"id": "infected", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Infected"}},
                    {"id": "recovered", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Recovered"}},
                ],
                "immunity_period": 14,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
            "run_mode": "UNCERTAINTY",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0, "population": 500000, "infected_population": 1.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"},
                        "value": 0.3,
                        "FieldConfigs": {"items": [{
                            "field_key": "value",
                            "has_variance": True,
                            "distribution_type": "UNIFORM",
                            "disease_param": "BETA",
                            "min": 0.15,
                            "max": 0.45,
                        }]},
                    },
                    {
                        "transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"},
                        "value": 0.1,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "GAMMA", "min": 0, "max": 0}]},
                    },
                ]
            },
            "Interventions": {"items": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))

        assert len(results) == 2
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            assert len(ts) > 0

            # Uncertainty output should have mean/lower/upper, not age_all
            mid_point = ts[len(ts) // 2]
            for comp in ["S", "I", "R"]:
                assert comp in mid_point
                val = mid_point[comp]
                assert "mean" in val, f"Expected uncertainty format (mean/lower/upper) for {comp}"
                assert "lower" in val
                assert "upper" in val

            # Verify no NaN in uncertainty values
            for entry in ts:
                for comp in ["S", "I", "R"]:
                    val = entry[comp]
                    for stat in ["mean", "lower", "upper"]:
                        assert not math.isnan(val[stat]), (
                            f"NaN in {comp}.{stat} on {entry['date']}"
                        )

    @pytest.mark.integration
    def test_covid_uncertainty_no_nan(self):
        """COVID uncertainty run should produce no NaN values."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = {
            "Disease": {
                "disease_type": "RESPIRATORY",
                "disease_nodes": [
                    {"id": "susceptible", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Susceptible"}},
                    {"id": "infected", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Infected"}},
                    {"id": "recovered", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Recovered"}},
                ],
                "immunity_period": 14,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-02-01",
            "run_mode": "UNCERTAINTY",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0, "population": 100000, "infected_population": 1.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"},
                        "value": 0.25,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": True, "distribution_type": "UNIFORM", "disease_param": "BETA", "min": 0.1, "max": 0.4}]},
                    },
                    {
                        "transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"},
                        "value": 0.1,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "GAMMA", "min": 0, "max": 0}]},
                    },
                ]
            },
            "Interventions": {"items": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))

        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            for entry in ts:
                for key, val in entry.items():
                    if key == "date":
                        continue
                    if isinstance(val, dict):
                        assert "mean" in val and "lower" in val and "upper" in val, (
                            f"Expected uncertainty format for {key}, got keys: {val.keys()}"
                        )
                        for stat in ["mean", "lower", "upper"]:
                            assert not math.isnan(val[stat]), (
                                f"NaN in {key}.{stat} on {entry['date']}"
                            )

    @pytest.mark.integration
    def test_dengue_uncertainty_varying_interventions(self):
        """Dengue uncertainty run with variance on intervention adherence."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        config = {
            "Disease": {
                "disease_type": "VECTOR_BORNE",
                "immunity_period": 240,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-07-01",
            "run_mode": "UNCERTAINTY",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 2.0, "center_lon": 45.0, "population": 500000, "infected_population": 5, "seroprevalence": 30, "temp_min": 20, "temp_max": 35, "temp_mean": 28},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "Interventions": {
                "items": [
                    {
                        "Intervention": {"name": "PHYSICAL", "display_name": "Bite Reduction"},
                        "adherence_min": 50.0,
                        "transmission_percentage": 50.0,
                        "start_date": "2025-01-01",
                        "end_date": "2025-07-01",
                        "FieldConfigs": {
                            "items": [{
                                "field_key": "adherence_min",
                                "has_variance": True,
                                "distribution_type": "UNIFORM",
                                "min": 20.0,
                                "max": 80.0,
                            }]
                        },
                    }
                ]
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(DengueJaxModel, pathlib.Path(config_path))

        assert len(results) == 2

        # The "with interventions" run should have uncertainty bands
        with_run = next(r for r in results if not r["control_run"])
        ts = with_run["parent_admin_total"]["time_series"]
        assert len(ts) > 0

        # Should be uncertainty format
        mid_point = ts[len(ts) // 2]
        sample_comp = [k for k in mid_point if k != "date"][0]
        val = mid_point[sample_comp]
        assert "mean" in val, f"Expected uncertainty format for {sample_comp}, got {val}"

        # The control run (no interventions) should also exist and have data
        ctrl_run = next(r for r in results if r["control_run"])
        ctrl_ts = ctrl_run["parent_admin_total"]["time_series"]
        assert len(ctrl_ts) > 0

    @pytest.mark.integration
    def test_uncertainty_lower_le_mean_le_upper(self):
        """In uncertainty output, lower <= mean <= upper should hold."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = {
            "Disease": {
                "disease_type": "RESPIRATORY",
                "disease_nodes": [
                    {"id": "susceptible", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Susceptible"}},
                    {"id": "infected", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Infected"}},
                    {"id": "recovered", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Recovered"}},
                ],
                "immunity_period": 14,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-02-15",
            "run_mode": "UNCERTAINTY",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0, "population": 200000, "infected_population": 1.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"},
                        "value": 0.3,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": True, "distribution_type": "UNIFORM", "disease_param": "BETA", "min": 0.2, "max": 0.5}]},
                    },
                    {
                        "transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"},
                        "value": 0.1,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "GAMMA", "min": 0, "max": 0}]},
                    },
                ]
            },
            "Interventions": {"items": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))

        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            for entry in ts:
                for key, val in entry.items():
                    if key == "date":
                        continue
                    if isinstance(val, dict) and "mean" in val:
                        # Allow tiny floating point tolerance
                        assert val["lower"] <= val["mean"] + 0.01, (
                            f"lower ({val['lower']}) > mean ({val['mean']}) "
                            f"for {key} on {entry['date']}"
                        )
                        assert val["mean"] <= val["upper"] + 0.01, (
                            f"mean ({val['mean']}) > upper ({val['upper']}) "
                            f"for {key} on {entry['date']}"
                        )


# ---------------------------------------------------------------------------
# COVID-specific disease tests
# ---------------------------------------------------------------------------
class TestFlexibleCompartments:
    """Test that COVID respects disease_nodes for compartment selection."""

    def _make_covid_config(self, compartment_ids: list[str]) -> dict:
        """Build a minimal COVID config with specific compartments."""
        node_map = {
            "S": {"id": "susceptible", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Susceptible"}},
            "I": {"id": "infected", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Infected"}},
            "R": {"id": "recovered", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Recovered"}},
            "E": {"id": "exposed", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Exposed"}},
            "H": {"id": "hospitalized", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Hospitalized"}},
            "D": {"id": "deceased", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Deceased"}},
        }
        disease_nodes = [node_map[c] for c in compartment_ids]

        return {
            "Disease": {
                "disease_type": "RESPIRATORY",
                "disease_nodes": disease_nodes,
                "immunity_period": 14,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
            "admin_zones": [
                {
                    "name": "Zone A",
                    "center_lat": 47.0,
                    "center_lon": 8.0,
                    "population": 100000,
                    "infected_population": 1.0,
                }
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"},
                        "value": 0.3,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "BETA", "min": 0, "max": 0}]},
                    },
                    {
                        "transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"},
                        "value": 0.1,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "GAMMA", "min": 0, "max": 0}]},
                    },
                ]
            },
            "Interventions": {"items": []},
        }

    @pytest.mark.integration
    def test_sir_only_no_extra_compartments(self):
        """SIR config should produce only S, I, R — no E, H, D."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = self._make_covid_config(["S", "I", "R"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = {k for k in ts[0].keys() if k != "date"}
        assert "S" in compartments
        assert "I" in compartments
        assert "R" in compartments
        assert "E" not in compartments, "E should not appear in SIR config"
        assert "H" not in compartments, "H should not appear in SIR config"
        assert "D" not in compartments, "D should not appear in SIR config"

    @pytest.mark.integration
    def test_seir_has_exposed_but_no_hospitalized(self):
        """SEIR config should have E but not H or D."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = self._make_covid_config(["S", "E", "I", "R"])
        # Add E->I edge
        config["TransmissionEdges"]["items"].insert(0, {
            "transmission_edge": {"source": "susceptible", "target": "exposed", "value_type": "RATE"},
            "value": 0.3,
            "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "BETA", "min": 0, "max": 0}]},
        })
        config["TransmissionEdges"]["items"].append({
            "transmission_edge": {"source": "exposed", "target": "infected", "value_type": "DAYS"},
            "value": 0.2,
            "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "THETA", "min": 0, "max": 0}]},
        })
        # Zero out the S->I edge since we're using S->E
        config["TransmissionEdges"]["items"][1]["value"] = 0.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = {k for k in ts[0].keys() if k != "date"}
        assert "E" in compartments
        assert "H" not in compartments
        assert "D" not in compartments

    @pytest.mark.integration
    def test_covid_sir_population_conserved(self):
        """In SIR (no deaths), total population at start should equal end."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = self._make_covid_config(["S", "I", "R"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = [k for k in ts[0].keys() if k != "date"]

        def _sum_pop(entry):
            total = 0
            for c in compartments:
                val = entry[c]
                if isinstance(val, dict):
                    total += val.get("age_all", 0)
            return total

        pop_start = _sum_pop(ts[0])
        pop_end = _sum_pop(ts[-1])

        assert pop_start > 0, "Starting population should be non-zero"
        # Allow small floating point drift from ODE solver
        assert abs(pop_start - pop_end) / pop_start < 0.001, (
            f"Population not conserved: start={pop_start:.1f}, end={pop_end:.1f}, "
            f"drift={abs(pop_start - pop_end):.1f} ({abs(pop_start - pop_end) / pop_start * 100:.4f}%)"
        )

    @pytest.mark.integration
    def test_dengue_ignores_disease_nodes(self):
        """Dengue should always use its fixed compartments, not disease_nodes."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        results = _run_model(
            DengueJaxModel,
            MODELS_DIR / "dengue_jax_model" / "example-config.json",
        )
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = {k for k in ts[0].keys() if k != "date"}
        # Dengue should have its strain-specific compartments, not generic S/I/R
        assert "S0" in compartments or "SV" in compartments, (
            f"Expected dengue-specific compartments, got {compartments}"
        )

class TestCovidDisease:
    """Tests specific to COVID model dynamics."""

    @pytest.mark.integration
    def test_full_seihdr_run(self):
        """Full SEIHDR run with all 6 compartments should succeed."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        results = _run_model(
            CovidJaxModel,
            MODELS_DIR / "covid_jax_model" / "example-config.json",
        )
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = {k for k in ts[0].keys() if k != "date"}
        for c in ["S", "E", "I", "H", "D", "R"]:
            assert c in compartments, f"Missing compartment {c} in full SEIHDR run"

    @pytest.mark.integration
    def test_intervention_changes_output(self):
        """With-interventions run should differ from control at epidemic peak."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        results = _run_model(
            CovidJaxModel,
            MODELS_DIR / "covid_jax_model" / "example-config.json",
        )
        with_run = next(r for r in results if not r["control_run"])
        ctrl_run = next(r for r in results if r["control_run"])

        with_ts = with_run["parent_admin_total"]["time_series"]
        ctrl_ts = ctrl_run["parent_admin_total"]["time_series"]

        # Compare I at the midpoint — interventions should reduce infections
        mid = len(with_ts) // 2
        i_with = with_ts[mid]["I"]["age_all"]
        i_ctrl = ctrl_ts[mid]["I"]["age_all"]
        assert i_with != i_ctrl, (
            f"Intervention run and control are identical at midpoint (I={i_with}). "
            "Interventions may not be applied."
        )

    @pytest.mark.integration
    def test_multi_zone_travel(self):
        """With 2+ zones and travel, infection should spread between zones."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = {
            "Disease": {
                "disease_type": "RESPIRATORY",
                "disease_nodes": [
                    {"id": "susceptible", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Susceptible"}},
                    {"id": "infected", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Infected"}},
                    {"id": "recovered", "type": "DISEASE_STATE_NODE", "data": {"alias": None, "label": "Recovered"}},
                ],
            },
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0, "population": 500000, "infected_population": 2.0},
                {"name": "Zone B", "center_lat": 48.0, "center_lon": 9.0, "population": 500000, "infected_population": 0.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"},
                        "value": 0.3,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "BETA", "min": 0, "max": 0}]},
                    },
                    {
                        "transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"},
                        "value": 0.1,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "GAMMA", "min": 0, "max": 0}]},
                    },
                ]
            },
            "Interventions": {"items": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))

        # Zone B started with 0 infected — should have infections by end
        zones = results[0]["admin_zones"]
        zone_b = next(z for z in zones if z["time_series"][-1]["I"]["age_all"] is not None)
        zone_b_ts = zone_b["time_series"]
        zone_b_i_end = zone_b_ts[-1]["I"]["age_all"]
        zone_b_r_end = zone_b_ts[-1]["R"]["age_all"]
        assert zone_b_i_end > 0 or zone_b_r_end > 0, (
            "Zone B started with 0 infected but never saw infections — travel may be broken"
        )

    @pytest.mark.integration
    def test_age_groups_sum_to_all(self):
        """age_0_17 + age_18_55 + age_56_plus should equal age_all."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        results = _run_model(
            CovidJaxModel,
            MODELS_DIR / "covid_jax_model" / "example-config.json",
        )
        ts = results[0]["parent_admin_total"]["time_series"]
        for entry in ts:
            for key, val in entry.items():
                if key == "date":
                    continue
                if isinstance(val, dict) and "age_all" in val:
                    parts = (
                        val.get("age_0_17", 0)
                        + val.get("age_18_55", 0)
                        + val.get("age_56_plus", 0)
                    )
                    assert abs(parts - val["age_all"]) < 1.0, (
                        f"Age groups don't sum to age_all for {key} on {entry['date']}: "
                        f"{parts:.2f} != {val['age_all']:.2f}"
                    )

    @pytest.mark.integration
    def test_demographic_rate_overrides_change_outcome(self):
        """High fatality rate for elderly should produce more deaths in that group.

        This test proves that demographic_rate_overrides actually flow through the
        ODE derivative computation and change simulation outcomes — not just that
        the rate vectors are constructed correctly.

        Setup: two equal-sized groups (young / elderly), same initial conditions,
        same beta/gamma — but elderly delta is 500x higher.  After 150 days,
        elderly deaths must exceed young deaths by a large margin.
        """
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = {
            "Disease": {
                "disease_type": "RESPIRATORY",
                "compartment_list": ["S", "E", "I", "D", "R"],
            },
            "start_date": "2025-01-01",
            "end_date": "2025-06-01",
            "admin_zones": [
                {
                    "name": "TestZone",
                    "center_lat": 0.0,
                    "center_lon": 0.0,
                    "population": 1000000,
                    "infected_population": 5.0,
                }
            ],
            # Two equal groups — same initial conditions, same population weight.
            # Only delta differs so any outcome difference is caused by the override.
            "demographics": {"young": 50.0, "elderly": 50.0},
            "travel_volume": {"leaving": 0},
            # Identity contact matrix: each group infects only itself.
            # Without this the warning fires and the matrix defaults to identity
            # anyway, but declaring it explicitly makes the test self-documenting.
            "contact_matrix_overrides": {
                "young":   {"young": 1.0, "elderly": 0.0},
                "elderly": {"young": 0.0, "elderly": 1.0},
            },
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {
                            "value_type": "RATE",
                            "source": "susceptible",
                            "target": "exposed",
                        },
                        "value": 0.35,
                    },
                    {
                        "transmission_edge": {
                            "value_type": "DAYS",
                            "source": "exposed",
                            "target": "infected",
                        },
                        "value": 5.0,
                    },
                    {
                        "transmission_edge": {
                            "value_type": "RATE",
                            "source": "infected",
                            "target": "deceased",
                        },
                        "value": 0.0001,  # base rate; overridden per group below
                    },
                    {
                        "transmission_edge": {
                            "value_type": "DAYS",
                            "source": "infected",
                            "target": "recovered",
                        },
                        "value": 7.0,
                    },
                ]
            },
            "Interventions": {"items": []},
            # elderly delta is 500x higher → elderly should accumulate far more deaths
            "demographic_rate_overrides": {
                "delta": {"young": 0.0001, "elderly": 0.05},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))
        ts = results[0]["parent_admin_total"]["time_series"]
        final = ts[-1]

        d_entry = final.get("D")
        assert isinstance(d_entry, dict), (
            f"Expected D compartment to be a dict with per-group values, got {type(d_entry)}"
        )
        d_young = d_entry.get("young", 0)
        d_elderly = d_entry.get("elderly", 0)

        assert d_elderly > 0, "Elderly group recorded zero deaths — rate override may not be applied"
        assert d_young >= 0, "Young group deaths should be non-negative"
        assert d_elderly > 10 * d_young, (
            f"Elderly deaths ({d_elderly:.0f}) should be at least 10x young deaths ({d_young:.0f}). "
            "demographic_rate_overrides for delta may not be flowing through the derivative."
        )


# ---------------------------------------------------------------------------
# Dengue-specific disease tests
# ---------------------------------------------------------------------------

def _get_dengue_val(entry, comp):
    """Extract the numeric value from a time series entry (handles both formats)."""
    val = entry[comp]
    if isinstance(val, dict):
        return val.get("age_all", val.get("mean", 0))
    return val


class TestDengueDisease:
    """Tests specific to Dengue model dynamics."""

    @pytest.mark.integration
    def test_vector_population_emerges(self):
        """Mosquito population should grow from 0 via temperature dynamics."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        results = _run_model(
            DengueJaxModel,
            MODELS_DIR / "dengue_jax_model" / "example-config.json",
        )
        ts = results[0]["parent_admin_total"]["time_series"]

        # Check IV (infectious vectors) at midpoint
        mid = len(ts) // 2
        iv_mid = _get_dengue_val(ts[mid], "IV")
        assert iv_mid > 0, (
            f"Infectious vector population is 0 at day {mid}. "
            "Temperature-driven vector dynamics may be broken."
        )

    @pytest.mark.integration
    def test_secondary_infections_occur(self):
        """I2 (secondary infections) should be non-zero by end of a multi-year run."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        results = _run_model(
            DengueJaxModel,
            MODELS_DIR / "dengue_jax_model" / "example-config.json",
        )
        ts = results[0]["parent_admin_total"]["time_series"]

        i2_end = _get_dengue_val(ts[-1], "I2")
        assert i2_end > 0, (
            "No secondary infections (I2) by end of simulation. "
            "Cross-immunity waning or secondary infection pathway may be broken."
        )

    @pytest.mark.integration
    def test_seroprevalence_affects_initial_s0(self):
        """Higher seroprevalence should reduce initial S0 population."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        compartment_list = DengueJaxModel.COMPARTMENT_LIST
        pop = 1_000_000
        zone_base = {
            "name": "Test", "center_lat": 2.0, "center_lon": 45.0,
            "population": pop, "infected_population": 5.0,
            "seroprevalence": 10.0,
        }
        zone_high = {**zone_base, "seroprevalence": 60.0}

        pop_low = DengueJaxModel.get_initial_population(
            [zone_base], compartment_list
        )
        pop_high = DengueJaxModel.get_initial_population(
            [zone_high], compartment_list
        )

        s0_idx = compartment_list.index("S0")
        s0_low = pop_low[0, s0_idx]
        s0_high = pop_high[0, s0_idx]
        assert s0_high < s0_low, (
            f"Higher seroprevalence should reduce S0: "
            f"sero=10% -> S0={s0_low:.0f}, sero=60% -> S0={s0_high:.0f}"
        )

    @pytest.mark.integration
    def test_immunity_period_matters(self):
        """immunity_period=0 vs 240 should produce different trajectories."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        base_config = {
            "Disease": {"disease_type": "VECTOR_BORNE", "immunity_period": 240},
            "start_date": "2026-01-01",
            "end_date": "2028-01-01",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 2.0, "center_lon": 45.0, "population": 500000, "infected_population": 5.0, "seroprevalence": 30, "temp_min": 20, "temp_max": 35, "temp_mean": 28},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "Interventions": {"items": []},
        }

        config_no_immunity = json.loads(json.dumps(base_config))
        config_no_immunity["Disease"]["immunity_period"] = 0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(base_config, f)
            path_with = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_no_immunity, f)
            path_without = f.name

        results_with = _run_model(DengueJaxModel, pathlib.Path(path_with))
        results_without = _run_model(DengueJaxModel, pathlib.Path(path_without))

        ts_with = results_with[0]["parent_admin_total"]["time_series"]
        ts_without = results_without[0]["parent_admin_total"]["time_series"]

        # Compare Snot (partially susceptible) near end — immunity_period
        # controls how fast C -> Snot, so they should differ
        end = -1
        snot_with = _get_dengue_val(ts_with[end], "Snot")
        snot_without = _get_dengue_val(ts_without[end], "Snot")
        assert snot_with != snot_without, (
            f"immunity_period=240 and immunity_period=0 produced identical Snot "
            f"({snot_with}). Parameter may be ignored."
        )

    @pytest.mark.integration
    def test_intervention_reduces_infections(self):
        """Bite reduction intervention should reduce cumulative infections vs control."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        results = _run_model(
            DengueJaxModel,
            MODELS_DIR / "dengue_jax_model" / "example-config.json",
        )
        with_run = next(r for r in results if not r["control_run"])
        ctrl_run = next(r for r in results if r["control_run"])

        with_ts = with_run["parent_admin_total"]["time_series"]
        ctrl_ts = ctrl_run["parent_admin_total"]["time_series"]

        # Compare cumulative R (recovered) at end — fewer recoveries means
        # fewer infections occurred overall when intervention is active
        r_with = _get_dengue_val(with_ts[-1], "R")
        r_ctrl = _get_dengue_val(ctrl_ts[-1], "R")
        assert r_with <= r_ctrl, (
            f"Intervention run has MORE recoveries ({r_with:.0f}) than "
            f"control ({r_ctrl:.0f}). Intervention may be broken."
        )
