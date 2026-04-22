"""Unit tests for schema-driven demographic features.

Tests cover:
- _build_contact_matrix: identity default, schema overrides, config overrides
- _build_rate_vectors: base fallback, per-group absolute rates, unit conversion
- _prepare_demographic_state: 2D → 3D expansion, _total rows appended
- _array_module: returns numpy for numpy arrays, jax for jax arrays
- end-to-end COVID model derivative computation with demographic rates
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

def _make_schema(groups=None, overrides=None, edges=None):
    """Build a minimal mock ModelParameterSchema."""
    from compartment.parameters import DemographicGroupDef, ContactOverrideDef

    schema = MagicMock()
    schema.demographic_groups = [
        DemographicGroupDef(id=g[0], label=g[1], default_weight=g[2])
        for g in (groups or [])
    ]
    schema.contact_matrix_overrides = [
        ContactOverrideDef(from_group=o[0], to_group=o[1], value=o[2])
        for o in (overrides or [])
    ]
    schema.transmission_edges = edges or []
    return schema


def _make_model_with_schema(schema, population_matrix=None):
    """Return a Model-like object whose _get_cached_schema returns schema."""
    from compartment.model import Model

    class _TestModel(Model):
        @classmethod
        def define_parameters(cls, s):
            raise NotImplementedError

    obj = object.__new__(_TestModel)
    obj.population_matrix = (
        population_matrix
        if population_matrix is not None
        else np.zeros((3, 2))  # (K, R)
    )
    obj._cached_schema = schema
    obj._TestModel__class_schema = schema

    # Patch _get_cached_schema on the instance's class
    _TestModel._cached_schema = schema

    return obj


# ---------------------------------------------------------------------------
# _array_module
# ---------------------------------------------------------------------------

class TestArrayModule:
    def test_returns_numpy_for_numpy_array(self):
        from compartment.model import Model

        class _M(Model):
            @classmethod
            def define_parameters(cls, s):
                raise NotImplementedError

        obj = object.__new__(_M)
        obj.population_matrix = np.zeros((3, 2))
        assert obj._array_module() is np

    def test_returns_jnp_for_jax_array(self):
        pytest.importorskip("jax")
        import jax.numpy as jnp
        from compartment.model import Model

        class _M(Model):
            @classmethod
            def define_parameters(cls, s):
                raise NotImplementedError

        obj = object.__new__(_M)
        obj.population_matrix = jnp.zeros((3, 2))
        assert obj._array_module() is jnp


# ---------------------------------------------------------------------------
# _build_contact_matrix
# ---------------------------------------------------------------------------

class TestBuildContactMatrix:
    def _model(self, groups, overrides=None, pm=None):
        schema = _make_schema(
            groups=groups,
            overrides=overrides or [],
        )
        return _make_model_with_schema(schema, pm)

    def test_returns_none_when_no_groups(self):
        model = self._model(groups=[])
        result = model._build_contact_matrix(config={})
        assert result is None

    def test_identity_by_default(self):
        groups = [("g0", "G0", 50.0), ("g1", "G1", 50.0)]
        model = self._model(groups=groups)
        result = model._build_contact_matrix(config={})
        expected = np.eye(2)
        np.testing.assert_allclose(np.array(result), expected)

    def test_schema_overrides_applied(self):
        groups = [("g0", "G0", 50.0), ("g1", "G1", 50.0)]
        overrides = [("g0", "g1", 3.5), ("g1", "g0", 1.2)]
        model = self._model(groups=groups, overrides=overrides)
        result = np.array(model._build_contact_matrix(config={}))
        assert result[0, 1] == pytest.approx(3.5)
        assert result[1, 0] == pytest.approx(1.2)

    def test_config_overrides_take_precedence(self):
        groups = [("g0", "G0", 50.0), ("g1", "G1", 50.0)]
        overrides = [("g0", "g1", 3.5)]
        model = self._model(groups=groups, overrides=overrides)
        config = {"contact_matrix_overrides": {"g0": {"g1": 9.9}}}
        result = np.array(model._build_contact_matrix(config=config))
        assert result[0, 1] == pytest.approx(9.9)

    def test_unknown_config_group_ignored(self):
        groups = [("g0", "G0", 50.0)]
        model = self._model(groups=groups)
        config = {"contact_matrix_overrides": {"unknown_group": {"g0": 5.0}}}
        # Should not raise
        result = model._build_contact_matrix(config=config)
        np.testing.assert_allclose(np.array(result), np.eye(1))

    def test_output_is_numpy_for_numpy_model(self):
        groups = [("g0", "G0", 50.0)]
        model = self._model(groups=groups, pm=np.zeros((1, 2)))
        result = model._build_contact_matrix(config={})
        assert type(result).__module__ == "numpy"


# ---------------------------------------------------------------------------
# _build_rate_vectors
# ---------------------------------------------------------------------------

class TestBuildRateVectors:
    def _make_edge(self, var_name, value_type):
        from compartment.parameters import ValueType
        edge = MagicMock()
        edge.variable_name = var_name
        edge.parameter = MagicMock()
        edge.parameter.value_type = value_type
        return edge

    def _model_with_rates(self, groups, edges, rate_attrs, pm=None):
        schema = _make_schema(groups=groups, edges=edges)
        model = _make_model_with_schema(schema, pm)
        for name, val in rate_attrs.items():
            setattr(model, name, val)
        return model

    def test_returns_none_with_no_groups(self):
        schema = _make_schema(groups=[])
        model = _make_model_with_schema(schema)
        result = model._build_rate_vectors(config={})
        assert result is None

    def test_returns_none_with_no_overrides(self):
        from compartment.parameters import ValueType
        groups = [("a", "A", 50.0), ("b", "B", 50.0)]
        edges = [self._make_edge("gamma", ValueType.RATE)]
        model = self._model_with_rates(groups=groups, edges=edges, rate_attrs={"gamma": 0.14})
        result = model._build_rate_vectors(config={})
        assert result is None

    def test_base_rate_fills_unspecified_groups(self):
        from compartment.parameters import ValueType
        groups = [("g0", "G0", 50.0), ("g1", "G1", 50.0)]
        edges = [self._make_edge("delta", ValueType.RATE)]
        model = self._model_with_rates(
            groups=groups, edges=edges,
            rate_attrs={"delta": 0.001},
        )
        config = {"demographic_rate_overrides": {"delta": {"g1": 0.005}}}
        result = model._build_rate_vectors(config=config)
        assert result is not None
        vec = np.array(result["delta"])
        assert vec[0] == pytest.approx(0.001)
        assert vec[1] == pytest.approx(0.005)

    def test_days_conversion(self):
        from compartment.parameters import ValueType
        groups = [("young", "Young", 50.0), ("old", "Old", 50.0)]
        edges = [self._make_edge("gamma", ValueType.DAYS)]
        model = self._model_with_rates(
            groups=groups, edges=edges,
            rate_attrs={"gamma": 1.0 / 7.14},  # base: 7.14 days
        )
        config = {"demographic_rate_overrides": {"gamma": {"old": 14.0}}}
        result = model._build_rate_vectors(config=config)
        vec = np.array(result["gamma"])
        assert vec[0] == pytest.approx(1.0 / 7.14)
        assert vec[1] == pytest.approx(1.0 / 14.0)

    def test_percentage_conversion(self):
        from compartment.parameters import ValueType
        groups = [("young", "Young", 30.0), ("old", "Old", 70.0)]
        edges = [self._make_edge("zeta", ValueType.PERCENTAGE)]
        model = self._model_with_rates(
            groups=groups, edges=edges,
            rate_attrs={"zeta": 0.04},  # base: 4%
        )
        config = {"demographic_rate_overrides": {"zeta": {"old": 40.0}}}
        result = model._build_rate_vectors(config=config)
        vec = np.array(result["zeta"])
        assert vec[0] == pytest.approx(0.04)
        assert vec[1] == pytest.approx(0.40)  # 40% -> 0.40

    def test_unknown_edge_ignored(self):
        from compartment.parameters import ValueType
        groups = [("g0", "G0", 100.0)]
        edges = [self._make_edge("gamma", ValueType.RATE)]
        model = self._model_with_rates(
            groups=groups, edges=edges,
            rate_attrs={"gamma": 0.1},
        )
        config = {"demographic_rate_overrides": {"nonexistent_rate": {"g0": 0.5}}}
        result = model._build_rate_vectors(config=config)
        assert result is None  # nonexistent_rate not in edges → nothing built


# ---------------------------------------------------------------------------
# _prepare_demographic_state
# ---------------------------------------------------------------------------

class TestPrepareDemographicState:
    def _make_edge_mock(self, target_id):
        e = MagicMock()
        e.target_id = target_id
        return e

    def _model(self, groups, edges, pm):
        schema = _make_schema(groups=groups, edges=edges)
        schema.transmission_edges = edges
        model = _make_model_with_schema(schema, pm)
        model.demographics = {g[0]: g[2] for g in groups}
        model.compartment_list = [f"C{i}" for i in range(pm.shape[0])]
        return model

    def test_expands_2d_to_3d(self):
        K, R = 2, 3
        groups = [("young", "Young", 40.0), ("old", "Old", 60.0)]
        pm = np.ones((K, R)) * 100.0
        model = self._model(groups=groups, edges=[], pm=pm)
        model._prepare_demographic_state()
        assert model.population_matrix.ndim == 3
        assert model.population_matrix.shape == (K, 2, R)

    def test_weights_sum_to_original_population(self):
        K, R = 2, 3
        groups = [("g0", "G0", 30.0), ("g1", "G1", 70.0)]
        pm = np.full((K, R), 1000.0)
        model = self._model(groups=groups, edges=[], pm=pm)
        model._prepare_demographic_state()
        # Each compartment's age slices should sum to the original value
        np.testing.assert_allclose(
            model.population_matrix.sum(axis=1),
            pm,
            rtol=1e-6,
        )

    def test_appends_total_rows(self):
        K, R = 2, 3
        groups = [("g0", "G0", 100.0)]
        edges = [self._make_edge_mock("C1")]
        pm = np.ones((K, R))
        model = self._model(groups=groups, edges=edges, pm=pm)
        model._prepare_demographic_state()
        # C1_total should be appended
        assert "C1_total" in model.compartment_list
        assert model.population_matrix.shape[0] == K + 1

    def test_no_duplicate_total_rows(self):
        """If _total already in compartment_list, don't add it again."""
        K, R = 2, 3
        groups = [("g0", "G0", 100.0)]
        edges = [self._make_edge_mock("C1"), self._make_edge_mock("C1")]  # duplicate
        pm = np.ones((K, R))
        model = self._model(groups=groups, edges=edges, pm=pm)
        model._prepare_demographic_state()
        assert model.compartment_list.count("C1_total") == 1

    def test_no_groups_uses_single_weight(self):
        K, R = 3, 2
        pm = np.full((K, R), 500.0)
        schema = _make_schema(groups=[], edges=[])
        model = _make_model_with_schema(schema, pm)
        model.demographics = {}
        model.compartment_list = ["S", "I", "R"]
        model._prepare_demographic_state()
        # Falls back to single group → shape (K, 1, R)
        assert model.population_matrix.shape == (K, 1, R)
        np.testing.assert_allclose(model.population_matrix[:, 0, :], pm)


# ---------------------------------------------------------------------------
# End-to-end: COVID model with demographic rate overrides
# ---------------------------------------------------------------------------

class TestCovidDemographicEndToEnd:
    @pytest.mark.integration
    def test_covid_smoke_with_demographics(self):
        """Run the COVID JAX model end-to-end to verify demographic pipeline."""
        import json
        import pathlib
        import tempfile
        from compartment.run_simulation import run_simulation
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config_path = (
            pathlib.Path(__file__).parent.parent
            / "compartment" / "models" / "covid_jax_model" / "example-config.json"
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        run_simulation(
            model_class=CovidJaxModel,
            config_path=str(config_path),
            output_path=output_path,
        )

        with open(output_path) as f:
            result = json.load(f)

        # Output is a list of simulation result dicts (one per run)
        assert isinstance(result, list) and len(result) > 0
        run = result[0]
        assert "admin_zones" in run
        assert len(run["admin_zones"]) > 0
        # Each time-series record should have compartment values as dicts
        ts = run["admin_zones"][0]["time_series"]
        assert len(ts) > 0
        first_record = ts[0]
        comp_entries = {k: v for k, v in first_record.items() if k != "date"}
        assert comp_entries, "No compartment entries found"
        for comp, val in comp_entries.items():
            assert isinstance(val, dict), f"Expected dict for {comp}, got {type(val)}"
            assert "age_all" in val, f"'age_all' missing from {comp} entry"
