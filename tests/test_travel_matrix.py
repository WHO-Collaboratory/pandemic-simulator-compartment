"""Travel matrix tests.

Mobility is model-owned: a model that travels declares ``travel_sigma`` (plus
any kernel-specific extras) via ``schema.add_disease_parameter()`` and defines
how those become a matrix in ``build_travel_matrix()``. The framework calls
that hook via ``Model._ensure_travel_matrix()`` before ``prepare_initial_state()``.

These tests cover:
  - the invariants every model's matrix must satisfy (rows sum to 1,
    diagonal = 1 - sigma, finite, non-negative)
  - the framework default (identity) for models with no declared travel
  - each model's own mobility kernel, including degenerate inputs
  - the shared geopy gravity helper
  - the lockdown intervention that collapses travel to the identity

Run:
    python3 -m pytest tests/test_travel_matrix.py -v
"""

import numpy as onp
import pytest

from compartment.helpers import (
    create_travel_matrix,
    get_admin_zone_df,
    get_gravity_model_travel_matrix,
    load_config_from_json,
)
from compartment.model import Model
from compartment.parameters import ValueType
from compartment.runtime import Intervention
from compartment.validation import load_simulation_config

from helpers import MODEL_CONFIGS, MODELS_DIR, import_class, model_datasets_are_available


# Three well-separated zones with distinct populations. ``id`` is required —
# the geopy gravity helper pivots on ``id_origin`` / ``id_destination``, and
# validated CaseFileAdminZones always carry one.
ZONES = [
    {"id": "z0", "name": "A", "center_lat": 47.0, "center_lon": 8.0, "population": 500_000},
    {"id": "z1", "name": "B", "center_lat": 48.0, "center_lon": 9.0, "population": 250_000},
    {"id": "z2", "name": "C", "center_lat": 52.0, "center_lon": 13.0, "population": 1_000_000},
]

MODEL_DIRS = sorted(
    model_dir
    for model_dir in MODEL_CONFIGS
    if model_datasets_are_available(MODELS_DIR / model_dir)
)


def _model_class(model_dir):
    return import_class(MODEL_CONFIGS[model_dir][1])


def _declares_travel(model_class):
    """True if the model declared a travel_sigma custom field."""
    schema = model_class._get_cached_schema()
    if schema is None:
        return False
    return any(p.name == "travel_sigma" for p in schema.disease_parameters)


TRAVEL_MODEL_DIRS = [d for d in MODEL_DIRS if _declares_travel(_model_class(d))]
STATIC_MODEL_DIRS = [d for d in MODEL_DIRS if not _declares_travel(_model_class(d))]


def _build_model(model_dir):
    """Construct a model from its example config without running the solver."""
    model_class = _model_class(model_dir)
    config_path = MODEL_CONFIGS[model_dir][2]

    raw = load_config_from_json(str(config_path))
    disease_type = model_class.DISEASE_TYPE
    raw["data"]["getSimulationJob"]["Disease"]["disease_type"] = disease_type
    processed = load_simulation_config(raw, disease_type)
    return model_class(processed)


def _sigma_of(model):
    """The model's outbound travel rate as a fraction."""
    return model._to_rate(model.travel_sigma, ValueType.PERCENTAGE)


def _zones_of(model):
    """The model's admin zones. Models that skip super().__init__() use payload."""
    config = getattr(model, "config", None)
    if not hasattr(config, "get"):
        config = model.payload
    return config["case_file"]["admin_zones"]


def assert_valid_travel_matrix(T, n):
    """Every travel matrix must satisfy these, whatever kernel produced it."""
    T = onp.asarray(T, dtype=float)
    assert T.shape == (n, n), f"expected {(n, n)}, got {T.shape}"
    assert onp.isfinite(T).all(), f"non-finite entries:\n{T}"
    assert (T >= 0).all(), f"negative entries:\n{T}"
    onp.testing.assert_allclose(
        T.sum(axis=1),
        onp.ones(n),
        atol=1e-9,
        err_msg=f"rows must sum to 1 (population is conserved):\n{T}",
    )


# ---------------------------------------------------------------------------
# Framework hook
# ---------------------------------------------------------------------------


class TestFrameworkHook:
    def test_default_build_is_identity(self):
        """A model that declares no mobility gets the identity matrix."""
        T = Model.build_travel_matrix(None, ZONES)
        onp.testing.assert_allclose(T, onp.eye(len(ZONES)))

    @pytest.mark.parametrize("model_dir", MODEL_DIRS)
    def test_ensure_travel_matrix_populates_from_example_config(self, model_dir):
        """_ensure_travel_matrix() must reach the model's own builder.

        Regression guard: the runtime config is a ProcessedSimulation, not a
        dict. An isinstance(config, dict) check here silently falls through to
        the identity fallback, which looks like "travel is off" rather than a
        bug — every travel model would quietly stop mixing.
        """
        model = _build_model(model_dir)
        model._ensure_travel_matrix()
        assert_valid_travel_matrix(model.travel_matrix, len(_zones_of(model)))

    @pytest.mark.parametrize("model_dir", TRAVEL_MODEL_DIRS)
    def test_diagonal_is_one_minus_sigma(self, model_dir):
        model = _build_model(model_dir)
        model._ensure_travel_matrix()

        T = onp.asarray(model.travel_matrix)
        sigma = _sigma_of(model)
        onp.testing.assert_allclose(
            onp.diag(T),
            1.0 - sigma,
            atol=1e-9,
            err_msg=f"{model_dir}: diagonal should be the stay-home fraction 1 - {sigma}",
        )

    @pytest.mark.parametrize("model_dir", TRAVEL_MODEL_DIRS)
    def test_example_config_actually_exercises_travel(self, model_dir):
        """A travel model's example config should have a non-zero sigma."""
        model = _build_model(model_dir)
        assert _sigma_of(model) > 0, (
            f"{model_dir}'s example config sets travel_sigma to 0, so its "
            "smoke test never exercises spatial mixing"
        )

    @pytest.mark.parametrize("model_dir", STATIC_MODEL_DIRS)
    def test_models_without_travel_get_identity(self, model_dir):
        model = _build_model(model_dir)
        model._ensure_travel_matrix()

        T = onp.asarray(model.travel_matrix)
        onp.testing.assert_allclose(T, onp.eye(T.shape[0]), atol=1e-12)


# ---------------------------------------------------------------------------
# Uncertainty sampling
# ---------------------------------------------------------------------------


class TestUncertaintyRebuild:
    """travel_sigma must survive LHS sampling and re-derive the matrix.

    It isn't an edge variable_name or an intervention id, so
    build_overridden_config() routes it to the Disease dict. The rebuilt model
    then has to actually rebuild its matrix from the sampled value — if the
    matrix were cached from the original config, uncertainty runs would all
    share one mobility pattern and the CI bands would be wrong.
    """

    @pytest.mark.parametrize("model_dir", TRAVEL_MODEL_DIRS)
    def test_sampled_sigma_rebuilds_the_matrix(self, model_dir):
        model = _build_model(model_dir)
        model_class = type(model)

        overridden = model_class(model.build_overridden_config({"travel_sigma": 40.0}))
        overridden._ensure_travel_matrix()

        T = onp.asarray(overridden.travel_matrix)
        assert_valid_travel_matrix(T, len(_zones_of(overridden)))
        onp.testing.assert_allclose(
            onp.diag(T),
            0.6,
            atol=1e-9,
            err_msg=f"{model_dir}: sampled travel_sigma=40% should give a 0.6 diagonal",
        )


# ---------------------------------------------------------------------------
# Per-model kernels
# ---------------------------------------------------------------------------


class TestModelKernels:
    @pytest.mark.parametrize("model_dir", TRAVEL_MODEL_DIRS)
    def test_kernel_invariants(self, model_dir):
        model = _build_model(model_dir)
        T = model.build_travel_matrix(ZONES)

        assert_valid_travel_matrix(T, len(ZONES))
        onp.testing.assert_allclose(
            onp.diag(onp.asarray(T)), 1.0 - _sigma_of(model), atol=1e-9
        )

    @pytest.mark.parametrize("model_dir", TRAVEL_MODEL_DIRS)
    def test_zero_sigma_disables_travel(self, model_dir):
        model = _build_model(model_dir)
        model.travel_sigma = 0.0

        T = onp.asarray(model.build_travel_matrix(ZONES))
        onp.testing.assert_allclose(T, onp.eye(len(ZONES)), atol=1e-12)

    @pytest.mark.parametrize("model_dir", TRAVEL_MODEL_DIRS)
    def test_single_zone_is_self_contained(self, model_dir):
        model = _build_model(model_dir)

        T = onp.asarray(model.build_travel_matrix(ZONES[:1]))
        assert_valid_travel_matrix(T, 1)
        onp.testing.assert_allclose(T, [[1.0]], atol=1e-12)

    @pytest.mark.parametrize("model_dir", TRAVEL_MODEL_DIRS)
    def test_coincident_zones_do_not_blow_up(self, model_dir):
        """Two zones at the same coordinates make distance 0 — no NaN/inf."""
        model = _build_model(model_dir)
        coincident = [
            dict(ZONES[0]),
            {**ZONES[1], "center_lat": ZONES[0]["center_lat"],
             "center_lon": ZONES[0]["center_lon"]},
        ]

        T = onp.asarray(model.build_travel_matrix(coincident))
        assert_valid_travel_matrix(T, 2)

    @pytest.mark.parametrize("model_dir", TRAVEL_MODEL_DIRS)
    def test_closer_and_larger_zones_attract_more_travel(self, model_dir):
        """Every kernel here is population-weighted with distance decay."""
        model = _build_model(model_dir)
        zones = [
            {"id": "home", "name": "home", "center_lat": 47.0, "center_lon": 8.0,
             "population": 100_000},
            {"id": "near", "name": "near", "center_lat": 47.2, "center_lon": 8.0,
             "population": 500_000},
            {"id": "far", "name": "far", "center_lat": 60.0, "center_lon": 8.0,
             "population": 500_000},
        ]

        T = onp.asarray(model.build_travel_matrix(zones))
        assert T[0, 1] > T[0, 2], (
            f"{model_dir}: travel to the near zone ({T[0, 1]:.6f}) should exceed "
            f"travel to the equally-populous far zone ({T[0, 2]:.6f})"
        )


# ---------------------------------------------------------------------------
# Shared geopy gravity helper
# ---------------------------------------------------------------------------


class TestGravityHelper:
    def test_invariants(self):
        T = get_gravity_model_travel_matrix(ZONES, 0.2)

        assert_valid_travel_matrix(T, len(ZONES))
        onp.testing.assert_allclose(onp.diag(T), 0.8, atol=1e-9)

    @pytest.mark.parametrize("sigma", [0.0, None])
    def test_no_sigma_is_identity(self, sigma):
        T = get_gravity_model_travel_matrix(ZONES, sigma)
        onp.testing.assert_allclose(T, onp.eye(len(ZONES)))

    def test_single_zone(self):
        onp.testing.assert_allclose(
            get_gravity_model_travel_matrix(ZONES[:1], 0.2), [[1.0]]
        )

    def test_sigma_scales_off_diagonal_mass(self):
        """Total off-diagonal mass per row is exactly sigma."""
        for sigma in (0.05, 0.2, 0.5):
            T = get_gravity_model_travel_matrix(ZONES, sigma)
            off_diagonal = T.sum(axis=1) - onp.diag(T)
            onp.testing.assert_allclose(off_diagonal, sigma, atol=1e-9)

    def test_bigger_destination_attracts_proportionally_more_travel(self):
        """At equal distance, trip share scales with destination population."""
        zones = [
            {"id": "origin", "name": "origin", "center_lat": 47.0,
             "center_lon": 8.0, "population": 100_000},
            {"id": "small", "name": "small", "center_lat": 48.0,
             "center_lon": 8.0, "population": 250_000},
            {"id": "big", "name": "big", "center_lat": 46.0,
             "center_lon": 8.0, "population": 1_000_000},
        ]

        T = get_gravity_model_travel_matrix(zones, 0.2)
        assert T[0, 2] > T[0, 1]
        # 4x the population => ~4x the trip share. Not exact: geodesic
        # distances for one degree of latitude differ slightly with latitude
        # on the WGS84 ellipsoid.
        onp.testing.assert_allclose(T[0, 2] / T[0, 1], 4.0, rtol=1e-2)

    def test_matrix_follows_input_zone_order(self):
        """Reversing the zone list must permute the matrix, not reshuffle it.

        Regression: pivot_table sorts its labels, so the matrix used to come
        back ordered by zone id while the population matrix stayed in
        admin_zones order — silently routing infections into the wrong zones.
        """
        T = get_gravity_model_travel_matrix(ZONES, 0.2)
        T_reversed = get_gravity_model_travel_matrix(list(reversed(ZONES)), 0.2)

        perm = [2, 1, 0]
        onp.testing.assert_allclose(T_reversed, T[onp.ix_(perm, perm)], atol=1e-12)

    def test_zone_ids_do_not_affect_ordering(self):
        """Ids that sort differently from the input order change nothing."""
        relabelled = [
            {**zone, "id": new_id} for zone, new_id in zip(ZONES, ["zz", "aa", "mm"])
        ]
        onp.testing.assert_allclose(
            get_gravity_model_travel_matrix(relabelled, 0.2),
            get_gravity_model_travel_matrix(ZONES, 0.2),
            atol=1e-12,
        )

    def test_create_travel_matrix_directly(self):
        df = get_admin_zone_df(ZONES)
        T = create_travel_matrix(df, 0.2, zone_order=[z["id"] for z in ZONES])

        assert_valid_travel_matrix(T, len(ZONES))
        onp.testing.assert_allclose(onp.diag(onp.asarray(T)), 0.8, atol=1e-9)

    def test_coincident_zones_keep_population_at_home(self):
        """Zero reachable gravity must not silently delete sigma of the population."""
        coincident = [
            {"id": "a", "name": "a", "center_lat": 47.0, "center_lon": 8.0,
             "population": 500_000},
            {"id": "b", "name": "b", "center_lat": 47.0, "center_lon": 8.0,
             "population": 250_000},
        ]

        T = get_gravity_model_travel_matrix(coincident, 0.2)
        assert_valid_travel_matrix(T, 2)
        onp.testing.assert_allclose(T, onp.eye(2), atol=1e-12)


# ---------------------------------------------------------------------------
# Lockdown intervention
# ---------------------------------------------------------------------------


class TestApplyToTravel:
    @staticmethod
    def _intervention(modifies_travel):
        return Intervention(
            id="lock_down",
            target_rates=["beta"],
            modifies_travel=modifies_travel,
            adherence=0.8,
            transmission_reduction=0.7,
            start_date_ordinal=None,
            end_date_ordinal=None,
            start_threshold=None,
            end_threshold=None,
        )

    def test_active_lockdown_collapses_to_identity(self):
        T = get_gravity_model_travel_matrix(ZONES, 0.2)
        out = self._intervention(True).apply_to_travel(T, True)

        onp.testing.assert_allclose(out, onp.eye(len(ZONES)))
        assert_valid_travel_matrix(out, len(ZONES))

    def test_inactive_lockdown_leaves_matrix_alone(self):
        T = get_gravity_model_travel_matrix(ZONES, 0.2)
        out = self._intervention(True).apply_to_travel(T, False)

        onp.testing.assert_allclose(out, T)

    def test_non_travel_intervention_is_a_noop(self):
        T = get_gravity_model_travel_matrix(ZONES, 0.2)
        out = self._intervention(False).apply_to_travel(T, True)

        onp.testing.assert_allclose(out, T)
