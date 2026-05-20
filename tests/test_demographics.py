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
    """Build a minimal mock ModelParameterSchema.

    Each entry in ``groups`` is a tuple of (id, label, default_weight) or
    (id, label, default_weight, age_range). The age_range slot is optional
    and defaults to None.
    """
    from compartment.parameters import DemographicGroupDef, ContactOverrideDef

    schema = MagicMock()
    schema.demographic_groups = [
        DemographicGroupDef(
            id=g[0],
            label=g[1],
            default_weight=g[2],
            age_range=g[3] if len(g) > 3 else None,
        )
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
# _build_contact_matrix — Prem 2021 country-aware auto-load
# ---------------------------------------------------------------------------

class TestBuildContactMatrixPrem:
    """When every demographic group declares age_range and no overrides exist,
    the framework auto-loads a country-specific Prem matrix and aggregates it."""

    def _aged_model(self, overrides=None, pm=None):
        groups = [
            ("age_0_17",    "Children", 33.3, (0, 17)),
            ("age_18_55",   "Adults",   44.4, (18, 55)),
            ("age_56_plus", "Elderly",  22.3, (56, 120)),
        ]
        schema = _make_schema(groups=groups, overrides=overrides or [])
        return _make_model_with_schema(schema, pm)

    def test_known_country_loads_and_aggregates_prem(self):
        from compartment.contact_matrices import (
            aggregate_to_bands, load_country_matrix,
        )
        model = self._aged_model()
        result = np.array(model._build_contact_matrix(
            config={"admin_unit_id": "USA.6.1_1"}
        ))
        expected = aggregate_to_bands(
            load_country_matrix("USA"),
            [(0, 17), (18, 55), (56, 120)],
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_unknown_country_falls_back_to_default_matrix(self):
        from compartment.contact_matrices import (
            aggregate_to_bands, default_matrix,
        )
        model = self._aged_model()
        result = np.array(model._build_contact_matrix(
            config={"admin_unit_id": "XYZ.1.1"}
        ))
        expected = aggregate_to_bands(
            default_matrix(),
            [(0, 17), (18, 55), (56, 120)],
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_local_admin_unit_id_falls_back_to_default(self):
        from compartment.contact_matrices import (
            aggregate_to_bands, default_matrix,
        )
        model = self._aged_model()
        result = np.array(model._build_contact_matrix(
            config={"admin_unit_id": "LOCAL"}
        ))
        expected = aggregate_to_bands(
            default_matrix(),
            [(0, 17), (18, 55), (56, 120)],
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_missing_admin_unit_id_falls_back_to_default(self):
        from compartment.contact_matrices import (
            aggregate_to_bands, default_matrix,
        )
        model = self._aged_model()
        result = np.array(model._build_contact_matrix(config={}))
        expected = aggregate_to_bands(
            default_matrix(),
            [(0, 17), (18, 55), (56, 120)],
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_schema_override_beats_prem(self):
        """Existing schema-level overrides still win — preserves covid behavior."""
        overrides = [
            ("age_0_17",    "age_0_17",    5.46),
            ("age_0_17",    "age_18_55",   5.18),
            ("age_0_17",    "age_56_plus", 0.93),
            ("age_18_55",   "age_0_17",    1.70),
            ("age_18_55",   "age_18_55",   9.18),
            ("age_18_55",   "age_56_plus", 1.68),
            ("age_56_plus", "age_0_17",    0.83),
            ("age_56_plus", "age_18_55",   5.90),
            ("age_56_plus", "age_56_plus", 3.80),
        ]
        model = self._aged_model(overrides=overrides)
        result = np.array(model._build_contact_matrix(
            config={"admin_unit_id": "USA.6.1_1"}
        ))
        expected = np.array([
            [5.46, 5.18, 0.93],
            [1.70, 9.18, 1.68],
            [0.83, 5.90, 3.80],
        ])
        np.testing.assert_allclose(result, expected)

    def test_config_override_beats_prem(self):
        model = self._aged_model()
        result = np.array(model._build_contact_matrix(
            config={
                "admin_unit_id": "USA.6.1_1",
                "contact_matrix_overrides": {
                    "age_0_17": {"age_18_55": 9.9},
                },
            }
        ))
        assert result[0, 1] == pytest.approx(9.9)

    def test_age_range_missing_falls_back_to_identity_with_warning(self, caplog):
        # No age_range on any group -> Prem path not taken, identity warning fires
        groups = [("g0", "G0", 50.0), ("g1", "G1", 50.0)]
        schema = _make_schema(groups=groups)
        model = _make_model_with_schema(schema)
        with caplog.at_level("WARNING", logger="compartment.model"):
            result = np.array(model._build_contact_matrix(
                config={"admin_unit_id": "USA.6.1_1",
                        "case_file": {"demographics": {"g0": 50, "g1": 50}}}
            ))
        np.testing.assert_allclose(result, np.eye(2))
        assert any("defaults to identity" in r.message for r in caplog.records)


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


# ---------------------------------------------------------------------------
# add_demographic_group age_range validation
# ---------------------------------------------------------------------------

class TestAgeRangeValidation:
    def _builder(self):
        from compartment.parameters import ParameterSchemaBuilder
        b = ParameterSchemaBuilder()
        b.set_model_info(disease_type="TEST", label="Test", description="t")
        b.add_compartment("S", "Susceptible", "")
        return b

    def test_valid_age_range_accepted(self):
        b = self._builder()
        b.add_demographic_group("a", "A", default_weight=50.0, age_range=(0, 17))
        b.add_demographic_group("b", "B", default_weight=50.0, age_range=(18, 120))
        schema = b.build()
        assert schema.demographic_groups[0].age_range == (0, 17)
        assert schema.demographic_groups[1].age_range == (18, 120)

    def test_age_range_none_is_default(self):
        b = self._builder()
        b.add_demographic_group("a", "A", default_weight=100.0)
        schema = b.build()
        assert schema.demographic_groups[0].age_range is None

    def test_overlap_rejected_at_build(self):
        b = self._builder()
        b.add_demographic_group("a", "A", default_weight=50.0, age_range=(0, 17))
        b.add_demographic_group("b", "B", default_weight=50.0, age_range=(15, 30))
        with pytest.raises(ValueError, match="overlapping age_ranges"):
            b.build()

    def test_invalid_tuple_rejected(self):
        b = self._builder()
        with pytest.raises(ValueError, match="must be a 2-tuple of ints"):
            b.add_demographic_group("a", "A", default_weight=50.0, age_range=(0,))
        with pytest.raises(ValueError, match="must be a 2-tuple of ints"):
            b.add_demographic_group("a", "A", default_weight=50.0, age_range=(0.5, 17))

    def test_inverted_range_rejected(self):
        b = self._builder()
        with pytest.raises(ValueError, match="0 <= low <= high <= 120"):
            b.add_demographic_group("a", "A", default_weight=50.0, age_range=(17, 0))

    def test_out_of_bounds_rejected(self):
        b = self._builder()
        with pytest.raises(ValueError, match="0 <= low <= high <= 120"):
            b.add_demographic_group("a", "A", default_weight=50.0, age_range=(-1, 17))
        with pytest.raises(ValueError, match="0 <= low <= high <= 120"):
            b.add_demographic_group("a", "A", default_weight=50.0, age_range=(0, 200))
