"""Abstract base class for all models"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

import numpy as np
import jax.numpy as jnp

from compartment.parameters import (
    AdminZoneFieldRegistry,
    CompartmentDef,
    CompartmentRegistry,
    DiseaseParamRegistry,
    DiseaseParamValues,
    ModelParameterSchema,
    ParameterDef,
    ParameterSchemaBuilder,
    ValueType,
)
from compartment.runtime import Intervention, TransmissionEdge

# Re-export so model authors can write:
#   from compartment.model import Model, ValueType
# without needing to import from compartment.parameters directly.
__all__ = ["Model", "ParameterSchemaBuilder", "ValueType"]


class Model(ABC):
    """
    Base class for all compartmental disease models.

    Subclasses **must** implement:
    - ``define_parameters(schema)``  -- fill the builder with parameter declarations

    For models that have migrated to declarative parameters, the following
    are **derived automatically** from the schema:

    - ``disease_type`` property
    - ``COMPARTMENT_LIST`` class attribute
    - ``get_params()`` — returns transmission params in schema edge order
    - Transmission param attributes (e.g. ``self.beta``, ``self.gamma``)

    Subclasses **should** override:
    - ``prepare_initial_state()``
    - ``derivative()``

    Non-migrated models can still override ``disease_type``, define
    ``COMPARTMENT_LIST`` manually, and implement ``get_params()``.
    """

    # Schema is cached per-subclass by __init_subclass__.
    # None means the model hasn't migrated to define_parameters() yet.
    _cached_schema: ClassVar[ModelParameterSchema | None] = None

    def __init_subclass__(cls, **kwargs):
        """
        Auto-populate schema-derived class attributes when a subclass is
        defined.  For migrated models this sets:

        - ``COMPARTMENTS`` — :class:`CompartmentRegistry` with attribute
          access (e.g. ``cls.COMPARTMENTS.S``).
        - ``COMPARTMENT_LIST`` — plain ``list[str]`` for backward
          compatibility with the post-processor and other legacy code.

        After building the schema the framework auto-generates
        ``_total`` cumulative tracking compartments for every
        compartment that is the target of at least one transmission
        edge.  These are appended to the schema's compartment list
        before the :class:`CompartmentRegistry` is created, so they
        are accessible via ``cls.COMPARTMENTS`` and included in
        ``cls.COMPARTMENT_LIST``.

        Non-migrated models that raise ``NotImplementedError`` from
        ``define_parameters()`` are silently skipped.
        """
        super().__init_subclass__(**kwargs)
        try:
            schema = cls._build_parameter_schema()
            cls._cached_schema = schema

            # Derive the display order for the results sidebar from
            # COMPARTMENT_DELTA_GROUPING (if the model defines one) or
            # fall back to the raw compartment IDs.  Must run before
            # _add_total_compartments so that _total entries are excluded.
            grouping = getattr(cls, "COMPARTMENT_DELTA_GROUPING", None)
            if grouping:
                schema.compartment_display_order = list(grouping.keys())
            else:
                schema.compartment_display_order = [c.id for c in schema.compartments]

            # Auto-generate _total compartments for each edge target.
            # These are a framework concern (derivative accumulation,
            # post-processing) — model authors never declare them.
            cls._add_total_compartments(schema)

            cls.COMPARTMENTS = CompartmentRegistry(schema.compartments)
            # Keep COMPARTMENT_LIST as a plain list for backward compat
            # with the post-processor and other downstream code.
            cls.COMPARTMENT_LIST = list(cls.COMPARTMENTS)
            # Auto-set DISEASE_TYPE from define_parameters() so models
            # only declare disease_type once (in set_model_info).
            if "DISEASE_TYPE" not in cls.__dict__ and schema.disease_type:
                cls.DISEASE_TYPE = schema.disease_type

            # Custom field registries: attribute-style access to
            # disease parameter and admin zone field names.
            cls.DISEASE_PARAMS = DiseaseParamRegistry(schema.disease_parameters)
            cls.ADMIN_ZONE_FIELDS = AdminZoneFieldRegistry(schema.admin_zone_fields)
        except (NotImplementedError, Exception):
            # Non-migrated model — leave everything as-is.
            pass

    @classmethod
    def _add_total_compartments(cls, schema: ModelParameterSchema) -> None:
        """
        Append ``_total`` cumulative tracking compartments to *schema*.

        For every compartment that is the target of at least one
        transmission edge a corresponding ``<id>_total`` compartment is
        created (unless the model author already declared one).  Their
        derivatives are computed automatically by
        :meth:`_compute_derivatives`.

        Called from :meth:`__init_subclass__` — model authors never
        need to call this directly.
        """
        existing_ids = {c.id for c in schema.compartments}
        seen_targets: set[str] = set()
        for edge in schema.transmission_edges:
            tid = edge.target_id
            total_id = f"{tid}_total"
            if tid not in seen_targets and total_id not in existing_ids:
                original = next(c for c in schema.compartments if c.id == tid)
                schema.compartments.append(
                    CompartmentDef(
                        id=total_id,
                        label=f"{original.label} Total",
                        description=f"Cumulative total entering {original.label}",
                    )
                )
                seen_targets.add(tid)

    def __init__(self, config):
        """
        Initialize common model state from config.

        For **migrated models** (those with a cached parameter schema),
        calling ``super().__init__(config)`` extracts the common fields
        every model needs:

        - ``population_matrix``, ``compartment_list``
        - ``start_date``, ``start_date_ordinal``, ``n_timesteps``
        - ``admin_units``, ``payload``
        - Transmission params (``self.beta``, ``self.gamma``, etc.)
        - ``disease_params`` — :class:`DiseaseParamValues` with
          attribute access (e.g. ``self.disease_params.immunity_period``)
        - Top-level aliases for each disease parameter
          (e.g. ``self.immunity_period``) for uncertainty compatibility
        - ``intervention_dict`` (kept for post-processing compat)
        - ``interventions`` — list of :class:`Intervention` runtime objects
        - ``intervention_statuses`` — auto-generated from schema

        **Non-migrated models** that don't call ``super().__init__()``
        are unaffected — they continue to extract fields manually.

        Subclasses override ``__init__`` and call ``super().__init__(config)``
        first, then add model-specific fields (travel matrix, demographics,
        etc.).
        """
        self.config = config
        self.contact_matrix = None
        self._rate_vectors = None

        schema = type(self)._get_cached_schema()
        if not schema:
            # Non-migrated model — subclass handles everything.
            return

        # -- Common fields (identical across all migrated models) --
        self.population_matrix = jnp.array(config["initial_population"]).T
        self.compartment_list = list(self.COMPARTMENTS)
        self.start_date = config["start_date"]
        self.start_date_ordinal = self.start_date.toordinal()
        self.n_timesteps = config["time_steps"]
        self.admin_units = config["admin_units"]
        self.payload = config

        # Transmission params: self.beta, self.gamma, etc.
        self._load_transmission_params(config.get("transmission_dict", {}))

        # -- Disease parameters --
        # Auto-unpack from the Disease dict using schema declarations.
        # Values are accessible via self.disease_params.<name> and also
        # as top-level aliases (self.<name>) for uncertainty compatibility
        # (BatchSimulationManager._single_run uses setattr(model, key, val)).
        disease_dict = config.get("Disease", {}) or {}
        self.disease_params = DiseaseParamValues(
            schema.disease_parameters, disease_dict
        )
        for p in schema.disease_parameters:
            setattr(self, p.name, getattr(self.disease_params, p.name))

        # -- Interventions --
        self.intervention_dict = config.get("intervention_dict", {})

        # Build runtime Intervention objects from schema defs + config
        self.interventions: list[Intervention] = []
        for intv_def in schema.interventions:
            if intv_def.id in self.intervention_dict:
                self.interventions.append(
                    Intervention.from_config(
                        intv_def, self.intervention_dict[intv_def.id]
                    )
                )

        # Auto-generate statuses from schema intervention declarations
        self.intervention_statuses = {
            intv_def.id: False for intv_def in schema.interventions
        }

        # Demographics: prefer config-supplied weights; otherwise fall back to
        # the schema's declared default_weight values for each demographic group.
        # Downstream consumers (post-processor, output formatter) read
        # ``self.demographics`` to derive age-group labels and weights, so it
        # must always be populated when the schema declares groups.
        config_demo = {}
        case_file = config.get("case_file")
        if isinstance(case_file, dict):
            config_demo = case_file.get("demographics") or {}
        if config_demo:
            self.demographics = dict(config_demo)
        elif schema.demographic_groups:
            self.demographics = {
                g.id: g.default_weight for g in schema.demographic_groups
            }
        else:
            self.demographics = {}

        self.contact_matrix = self._build_contact_matrix(config)
        self._rate_vectors = self._build_rate_vectors(config)

    # ------------------------------------------------------------------
    # Schema-derived properties
    # ------------------------------------------------------------------

    @classmethod
    def _get_cached_schema(cls) -> ModelParameterSchema | None:
        """Return the cached schema, or ``None`` for non-migrated models."""
        return cls._cached_schema

    @property
    def disease_type(self):
        """
        Disease type identifier (e.g. ``"MONKEYPOX"``).

        Migrated models get this from the schema automatically. Non-migrated models should override this property.
        """
        schema = type(self)._get_cached_schema()
        if schema:
            return schema.disease_type
        raise NotImplementedError(
            f"{type(self).__name__} must either implement define_parameters() "
            "or override the disease_type property."
        )

    # ------------------------------------------------------------------
    # Declarative parameter schema
    # ------------------------------------------------------------------

    @classmethod
    def define_parameters(cls, schema: ParameterSchemaBuilder) -> None:
        """
        Declare all parameters for this disease model.

        Called by the framework with a ``ParameterSchemaBuilder`` that has
        been pre-populated with shared simulation parameters.  Subclasses
        call ``schema.add_*()`` / ``schema.set_*()`` to register
        compartments, edges, interventions, custom fields, etc.

        Args:
            schema: Builder instance -- call its methods to declare parameters.

        Raises:
            NotImplementedError: For non-migrated models that haven't
                adopted declarative parameter definitions yet.
        """
        raise NotImplementedError(
            f"{cls.__name__} has not implemented define_parameters(). "
            "Implement it to enable declarative parameter definitions."
        )

    @classmethod
    def _build_parameter_schema(cls) -> ModelParameterSchema:
        """
        Internal: create a builder, pre-populate shared params, call
        ``define_parameters()``, and return the built schema.

        This is the single entry point used by ``generate_artifact()``,
        ``generate_example_config()``, the validation layer, and the CLI.
        """
        schema = ParameterSchemaBuilder()

        # Pre-populate shared simulation parameters so subclasses don't
        # have to remember to include them.
        for p in cls._shared_simulation_parameters():
            schema._simulation_parameters.append(p)

        # Let the subclass fill the rest
        cls.define_parameters(schema)

        built = schema.build()

        # Expose the UI compartment display order. When the model groups raw
        # compartments (COMPARTMENT_DELTA_GROUPING), the grouped keys are what
        # appear in time series + deltas — surface those (in order) so the
        # results view orders bespoke compartments correctly. Ungrouped models
        # fall back to declared-compartment order in to_artifact_dict.
        grouping = getattr(cls, "COMPARTMENT_DELTA_GROUPING", None)
        if grouping:
            built.compartment_display_order = list(grouping.keys())

        return built

    # ------------------------------------------------------------------
    # Shared parameter helpers (inherited by all subclasses)
    # ------------------------------------------------------------------

    @classmethod
    def _shared_simulation_parameters(cls) -> list[ParameterDef]:
        """
        Simulation-level parameters common to every model.

        Subclasses may extend this list but should rarely need to.
        """
        return [
            ParameterDef(
                name="start_date",
                label="Start Date",
                description="Simulation start date",
                value_type=ValueType.DATE,
                required=True,
            ),
            ParameterDef(
                name="end_date",
                label="End Date",
                description="Simulation end date",
                value_type=ValueType.DATE,
                required=True,
            ),
            ParameterDef(
                name="run_mode",
                label="Run Mode",
                description="Run a single deterministic simulation or uncertainty analysis with multiple runs",
                value_type=ValueType.SELECT,
                default="DETERMINISTIC",
                options=["DETERMINISTIC", "UNCERTAINTY"],
            ),
            ParameterDef(
                name="simulation_type",
                label="Simulation Type",
                description="Type of simulation engine",
                value_type=ValueType.SELECT,
                default="COMPARTMENTAL",
                options=["COMPARTMENTAL", "AGENT_BASED"],
            ),
        ]

    @classmethod
    def _shared_admin_zone_fields(cls) -> list[ParameterDef]:
        """
        Admin-zone fields common to every model.

        Disease models that need additional zone-level fields (e.g.
        seroprevalence for Dengue, temperature for vector-borne) should
        call ``schema.add_admin_zone_field()`` in ``define_parameters()``.
        """
        return [
            ParameterDef(
                name="name",
                label="Zone Name",
                description="Name of the administrative zone",
                value_type=ValueType.TEXT,
                required=True,
            ),
            ParameterDef(
                name="population",
                label="Population",
                description="Total population of the administrative zone",
                value_type=ValueType.COUNT,
                min_value=0,
                required=True,
            ),
            ParameterDef(
                name="infected_population",
                label="Initial Infected",
                description="Percentage of population initially infected",
                value_type=ValueType.PERCENTAGE,
                default=0.05,
                min_value=0,
                max_value=100,
                unit="%",
            ),
            ParameterDef(
                name="center_lat",
                label="Latitude",
                description="Center latitude of the zone",
                value_type=ValueType.COORDINATE,
                min_value=-90,
                max_value=90,
                required=True,
            ),
            ParameterDef(
                name="center_lon",
                label="Longitude",
                description="Center longitude of the zone",
                value_type=ValueType.COORDINATE,
                min_value=-180,
                max_value=180,
                required=True,
            ),
        ]

    # ------------------------------------------------------------------
    # Convenience class-methods for artifact / config generation
    # ------------------------------------------------------------------

    @classmethod
    def generate_artifact(cls) -> dict[str, Any]:
        """
        Generate the model artifact JSON dict.

        This is the schema document consumed by the UI layer, DB seeding,
        and downstream Zod schema generation.
        """
        return cls._build_parameter_schema().to_artifact_dict()

    @classmethod
    def generate_example_config(cls) -> dict[str, Any]:
        """
        Generate an example simulation config from parameter defaults.

        The output is in the "short-form" JSON format accepted by
        ``load_config_from_json()``.
        """
        return cls._build_parameter_schema().to_example_config()

    # ------------------------------------------------------------------
    # Runtime helpers for schema-driven models
    # ------------------------------------------------------------------

    @staticmethod
    def _to_rate(value: float, value_type) -> float:
        """Convert a native-unit value to a per-day rate for ODE computation."""
        from compartment.parameters import ValueType

        if value_type == ValueType.DAYS:
            return 1.0 / value if value > 0 else 0.0
        elif value_type == ValueType.PERCENTAGE:
            return value / 100.0
        return value

    def _load_transmission_params(self, transmission_dict: dict) -> None:
        """
        Set transmission-rate attributes (e.g. ``self.beta``, ``self.gamma``)
        from the config's ``transmission_dict``, using variable names
        declared in the parameter schema.

        Values are stored in **native units** (days, percentages, or raw
        rates) and converted to per-day rates here based on each edge's
        ``value_type``.

        Call this from ``__init__`` instead of manually extracting each
        parameter by name.

        Args:
            transmission_dict: ``{"beta": 0.3, "gamma": 0.1, ...}``

        Raises:
            RuntimeError: If the model has no cached parameter schema.
        """
        schema = type(self)._get_cached_schema()
        if not schema:
            raise RuntimeError(
                f"{type(self).__name__} has no parameter schema. "
                "Implement define_parameters() or load params manually."
            )
        for edge in schema.transmission_edges:
            raw = transmission_dict.get(edge.variable_name)
            if raw is not None:
                rate = self._to_rate(raw, edge.parameter.value_type)
            else:
                rate = None
            setattr(self, edge.variable_name, rate)

    def _array_module(self):
        """
        Return the array module (``jax.numpy`` or ``numpy``) that matches the
        type of ``self.population_matrix``.

        This allows demographic helpers to produce arrays that are compatible
        with the model's compute backend without hardcoding ``jnp``.  JAX
        models get JAX arrays; plain-numpy models get numpy arrays.
        """
        pm = self.population_matrix
        if type(pm).__module__.startswith("jax"):
            return jnp
        return np

    def _build_contact_matrix(self, config: dict = None):
        """
        Build the demographic contact matrix from schema declarations and
        optional config overrides.

        Starts from identity, applies schema-declared overrides (model
        defaults from ``set_contact_override()``), then applies any
        per-run config overrides from ``config["contact_matrix_overrides"]``::

            "contact_matrix_overrides": {
                "age_0_17":    { "age_18_55": 3.0 },
                "age_56_plus": { "age_0_17": 0.5, "age_56_plus": 2.5 }
            }

        Config overrides take precedence over schema defaults, so a run can
        use a different contact structure without modifying the model.

        Returns ``None`` if the schema declares no demographic groups.
        """
        schema = type(self)._get_cached_schema()
        if not schema or not schema.demographic_groups:
            return None

        xp = self._array_module()

        # Resolve the effective group IDs and matrix size.
        # Config demographics take full precedence: a config can declare any
        # number of groups with any names, regardless of what the schema
        # declares as defaults.  The schema group IDs are only used when the
        # config provides no demographics.
        config_demo = {}
        if config:
            cf = config.get("case_file", {})
            if isinstance(cf, dict):
                config_demo = cf.get("demographics", {})

        if config_demo:
            effective_ids = list(config_demo.keys())
        else:
            effective_ids = [g.id for g in schema.demographic_groups]
        A = len(effective_ids)

        config_contact_overrides = (
            (config.get("contact_matrix_overrides") or {}) if config else {}
        )
        schema_ids = [g.id for g in schema.demographic_groups]
        schema_has_overrides = (
            bool(schema.contact_matrix_overrides) and effective_ids == schema_ids
        )

        matrix = np.eye(A)

        # Auto-load a country-aware Prem 2021 contact matrix when:
        #   - the model's declared groups are in use (effective == schema), so we
        #     know each group's age_range from the schema;
        #   - every group has an age_range declared;
        #   - no schema- or config-level overrides are present.
        # This is the recommended path — model authors only need to declare
        # age_range on each demographic group and the framework supplies a
        # sensible country-specific matrix.  Explicit overrides still win.
        import logging

        log = logging.getLogger(__name__)
        schema_group_by_id = {g.id: g for g in schema.demographic_groups}
        # Use set comparison so the Prem auto-load is not blocked by ordering
        # differences between the schema and the cloud API (which may return
        # demographic groups in a different order than they were declared).
        prem_applied = False
        schema_group_by_id = {g.id: g for g in schema.demographic_groups}
        # Use set comparison so the Prem auto-load is not blocked by ordering
        # differences between the schema and the cloud API (which may return
        # demographic groups in a different order than they were declared).
        if (
            set(effective_ids) == set(schema_ids)
            and not schema_has_overrides
            and not config_contact_overrides
            and all(g.age_range is not None for g in schema.demographic_groups)
        ):
            from compartment.contact_matrices import (
                load_country_matrix,
                default_matrix,
                aggregate_to_bands,
            )

            iso3 = self._resolve_iso3_for_run(config)
            source = load_country_matrix(iso3)
            if source is None:
                source = default_matrix()
                if iso3:
                    log.info(
                        "Country '%s' not in Prem bundle; using global-average "
                        "contact matrix aggregated to %d bands.",
                        iso3,
                        A,
                    )
                else:
                    log.info(
                        "No admin_unit_id resolvable; using global-average "
                        "contact matrix aggregated to %d bands.",
                        A,
                    )
            else:
                log.info(
                    "Loaded Prem contact matrix for '%s', aggregated to %d bands.",
                    iso3,
                    A,
                )
            # Build age_ranges in effective_ids order so the aggregated matrix
            # rows/cols align with the state array's demographic dimension.
            age_ranges = [schema_group_by_id[gid].age_range for gid in effective_ids]
            matrix = aggregate_to_bands(source, age_ranges)
            prem_applied = True

        # Warn when demographics are provided but the matrix is still identity
        # (no Prem aggregation, no schema overrides, no config overrides).
        # Without anything, each group only contacts itself — almost always a bug.
        if (
            config_demo
            and not config_contact_overrides
            and not schema_has_overrides
            and not prem_applied
        ):
            log.warning(
                "Demographics provided (%s) but no contact_matrix_overrides found "
                "and age_range is not declared on every group. The contact matrix "
                "defaults to identity, meaning demographic groups do not mix. "
                "Declare age_range on every demographic group to auto-load Prem 2021 "
                "country matrices, or supply contact_matrix_overrides.",
                ", ".join(effective_ids),
            )

        # Schema-level contact overrides (only applied when schema IDs match)
        if effective_ids == schema_ids and schema.contact_matrix_overrides:
            applied = []
            for override in schema.contact_matrix_overrides:
                if (
                    override.from_group in effective_ids
                    and override.to_group in effective_ids
                ):
                    i = effective_ids.index(override.from_group)
                    j = effective_ids.index(override.to_group)
                    matrix[i, j] = override.value
                    applied.append(f"{override.from_group}->{override.to_group}")
            if applied:
                log.info(
                    "Applied %d schema contact override(s) (Prem suppressed): %s.",
                    len(applied),
                    ", ".join(applied),
                )

        # Config-level overrides always applied (keys match whatever the config declared)
        if config:
            config_applied = []
            for from_group, targets in (
                config.get("contact_matrix_overrides") or {}
            ).items():
                if from_group not in effective_ids:
                    continue
                i = effective_ids.index(from_group)
                for to_group, value in targets.items():
                    if to_group in effective_ids:
                        matrix[i, effective_ids.index(to_group)] = value
                        config_applied.append(f"{from_group}->{to_group}")
            if config_applied:
                log.info(
                    "Applied %d config contact_matrix_override(s): %s.",
                    len(config_applied),
                    ", ".join(config_applied),
                )

        return xp.array(matrix)

    def _resolve_iso3_for_run(self, config: dict | None) -> str | None:
        """Extract ISO3 country code from the job's ``admin_unit_id``.

        The simulation config carries a single top-level ``admin_unit_id``
        like ``"MDG.1.3_1"`` or just ``"MDG"``.  The ISO3 is the first
        period-separated segment, uppercased.  Returns ``None`` when no
        admin_unit_id is set or the value is the placeholder ``"LOCAL"``,
        which triggers the global-average fallback in the caller.
        """
        if not config:
            return None
        admin_unit_id = config.get("admin_unit_id")
        if not admin_unit_id or admin_unit_id == "LOCAL":
            return None
        return admin_unit_id.split(".", 1)[0].upper()

    def _build_rate_vectors(self, config: dict) -> dict | None:
        """
        Build per-demographic absolute rate vectors from the config.

        Reads ``config["demographic_rate_overrides"]``, which maps
        transmission-edge variable names to per-group absolute rates in the
        same native units as the base rate (days, percentage, or raw rate)::

            "demographic_rate_overrides": {
                "zeta": {"age_56_plus": 15},          # 15% for elderly
                "gamma": {"age_0_17": 5, "age_56_plus": 10}  # days to recover
            }

        For each declared edge, starts from the base scalar rate (already
        converted to a per-day rate by :meth:`_load_transmission_params`)
        and fills the ``(A,)`` vector.  Groups not listed get the base rate.
        Specified groups get their absolute value converted by the edge's
        ``value_type`` (so DAYS and PERCENTAGE are handled correctly).

        These vectors are used directly in :meth:`_compute_derivatives`,
        bypassing the intervention-scaled scalar.  This means:

        - Uncertainty runs vary only the base scalar — demographic rates
          are fixed epidemiological values, not sampled.
        - Interventions do not scale edges that have demographic rate vectors.
          This is intentional: demographic rates apply to edges like
          ``zeta``/``delta``/``gamma`` that interventions do not target.

        Returns ``None`` if no demographic groups are declared or no
        overrides are provided.
        """
        schema = type(self)._get_cached_schema()
        if not schema or not schema.demographic_groups:
            return None

        overrides = config.get("demographic_rate_overrides") or {}
        if not overrides:
            return None

        xp = self._array_module()

        # Resolve effective group IDs from config or schema (same logic as
        # _build_contact_matrix — config demographics always win)
        config_demo = config.get("case_file", {})
        if isinstance(config_demo, dict):
            config_demo = config_demo.get("demographics", {})
        else:
            config_demo = {}
        group_ids = (
            list(config_demo.keys())
            if config_demo
            else [g.id for g in schema.demographic_groups]
        )
        A = len(group_ids)
        edge_value_types = {
            e.variable_name: e.parameter.value_type for e in schema.transmission_edges
        }
        result = {}

        for variable_name, group_rates in overrides.items():
            value_type = edge_value_types.get(variable_name)
            if value_type is None:
                continue
            base_rate = getattr(self, variable_name, None)
            if base_rate is None:
                continue
            vec = np.full(A, base_rate, dtype=float)
            for group_id, raw_value in group_rates.items():
                if group_id in group_ids:
                    vec[group_ids.index(group_id)] = self._to_rate(
                        raw_value, value_type
                    )
            result[variable_name] = xp.array(vec)

        return result or None

    def get_params(self):
        """
        Return transmission parameters as a tuple in schema edge order.

        For migrated models this is derived automatically from the
        parameter schema — no need to override.  The tuple order matches
        the ``add_transmission_edge()`` call order in
        ``define_parameters()``, which must match the positional unpack
        in ``derivative()``.

        Non-migrated models should override this method.
        """
        schema = type(self)._get_cached_schema()
        if not schema:
            return None
        return tuple(
            getattr(self, edge.variable_name) for edge in schema.transmission_edges
        )

    def _unpack_params(self, p: tuple) -> dict[str, Any]:
        """
        Unpack a params tuple into a dict keyed by variable name.

        Optional convenience for ``derivative()`` implementations that
        prefer named access over positional unpacking::

            params = self._unpack_params(p)
            beta = params["beta"]

        The tuple order must match schema edge order (same as
        ``get_params()``).
        """
        schema = type(self)._get_cached_schema()
        if not schema:
            raise RuntimeError("No parameter schema available")
        return {
            edge.variable_name: p[i] for i, edge in enumerate(schema.transmission_edges)
        }

    def _compute_derivatives(
        self,
        states: dict[str, Any],
        rates: dict[str, Any],
        skip_edges: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compute per-compartment derivatives from the declared transmission
        edges and compartment flags.  No callables or intermediates needed.

        For each edge the flow formula is selected by the edge's
        ``frequency_dependent`` flag:

        - **Standard** (default): ``rate * source_population``
        - **Frequency-dependent FOI**: ``source * rate * sum(infective) / N_total``

        The sum of all compartments marked ``infective=True`` and the
        total population ``N_total`` (sum of all non-``_total``
        compartments) are pre-computed once per call.

        This is JAX-safe: it iterates over a fixed-size Python list of
        edges and uses only array arithmetic.

        **Escape hatch**: pass ``skip_edges`` to exclude specific edges
        by variable name.  The model handles those edges manually,
        typically via :meth:`_apply_flow`.  Edges whose source or target
        compartment is absent from ``states`` are also silently skipped,
        supporting models with optional compartments.

        Args:
            states: Mapping of compartment IDs to their current state
                arrays (e.g. ``{"S": jax_array, ...}``).  Only
                compartments present in this dict are included in the
                output.  This allows models with optional compartments
                to pass a subset of the schema's compartments.
            rates: Mapping of edge variable names to their current
                (post-intervention) rate values
                (e.g. ``{"beta": ..., "gamma": ...}``).  Only rates
                for non-skipped, active edges need to be present.
            skip_edges: Variable names of edges to exclude from
                automatic computation (e.g. ``{"beta"}``).  The model
                is responsible for computing those flows manually and
                applying them to the returned ``derivs`` dict via
                :meth:`_apply_flow` or direct mutation.

        Returns:
            Dict mapping every compartment ID present in ``states``
            to its derivative array.  Models that skipped edges should
            add their manual flows to this dict before stacking.
        """
        schema = type(self)._get_cached_schema()
        assert schema is not None

        C = self.COMPARTMENTS
        zero = jnp.zeros_like(next(iter(states.values())))
        derivs: dict[str, Any] = {c: zero for c in states}

        # Pre-compute shared values for FOI edges
        non_total = [c for c in states if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)
        infective_in_states = [c for c in C.infective_ids if c in states]
        infective_sum = (
            sum(states[c] for c in infective_in_states) if infective_in_states else zero
        )

        for edge in schema.transmission_edges:
            # Skip edges the model handles manually
            if skip_edges and edge.variable_name in skip_edges:
                continue
            # Skip edges whose compartments aren't active at runtime
            if edge.source_id not in states or edge.target_id not in states:
                continue

            rate = rates.get(edge.variable_name)
            # Skip edges with no rate (compartment not in this config variant)
            if rate is None:
                continue

            # Use absolute demographic rate vector if declared, bypassing the
            # intervention-scaled scalar.  Demographic rates are fixed
            # epidemiological values and do not participate in uncertainty
            # sampling or intervention scaling.
            rate_vectors = getattr(self, "_rate_vectors", None)
            if rate_vectors and edge.variable_name in rate_vectors:
                rate = rate_vectors[edge.variable_name]

            # Reshape (A,) rate vector to (A, 1) so it broadcasts correctly
            # against (A, R) state arrays without touching scalar rates.
            if hasattr(rate, "ndim") and rate.ndim == 1:
                rate = rate[:, None]

            source = states[edge.source_id]

            if edge.frequency_dependent:
                # Force of infection: source * rate * infective / N
                flow = source * rate * infective_sum / (N_total + 1e-10)
            else:
                # Standard: rate * source_population
                flow = rate * source

            derivs[edge.source_id] = derivs[edge.source_id] - flow
            derivs[edge.target_id] = derivs[edge.target_id] + flow

            # Auto-accumulate into _total compartments
            total_key = f"{edge.target_id}_total"
            if total_key in derivs:
                derivs[total_key] = derivs[total_key] + flow

        return derivs

    def _apply_flow(
        self,
        derivs: dict[str, Any],
        source_id: str,
        target_id: str,
        flow: Any,
    ) -> None:
        """
        Apply a manually-computed flow to the derivatives dict.

        Subtracts *flow* from the source, adds it to the target, and
        auto-accumulates into the target's ``_total`` compartment if
        present.  This is the companion to :meth:`_compute_derivatives`
        for edges excluded via ``skip_edges``.

        Modifies *derivs* in place (returns ``None``).

        Args:
            derivs: The derivatives dict returned by
                :meth:`_compute_derivatives`.
            source_id: Compartment ID the flow leaves (e.g. ``"S"``).
            target_id: Compartment ID the flow enters (e.g. ``"E"``).
            flow: The flow array (same shape as the state arrays).

        Example::

            derivs = self._compute_derivatives(states, rates, skip_edges={"beta"})

            # Manual FOI with age-stratified contact matrix
            omega = age_trans @ ((rates["beta"] * contact) @ I_frac.T).T
            self._apply_flow(derivs, "S", "E", S * omega)
        """
        derivs[source_id] = derivs[source_id] - flow
        derivs[target_id] = derivs[target_id] + flow
        total_key = f"{target_id}_total"
        if total_key in derivs:
            derivs[total_key] = derivs[total_key] + flow

    def _apply_interventions(
        self,
        t: Any,
        rates: dict[str, Any],
        prop_infective: Any,
    ) -> tuple[dict[str, Any], Any]:
        """
        Apply all declared interventions to rates and travel matrix.

        Replicates the original two-pass structure from
        ``interventions.py``: first date-based activation + rate
        reduction, then threshold-based activation + rate reduction.
        A single intervention can reduce a rate **twice** — once via
        date window and once via threshold — matching the original
        behavior where ``jax_timestep_intervention`` modifies rates
        and then ``jax_prop_intervention`` takes the already-modified
        rates and potentially reduces them further.

        Requires ``self.start_date_ordinal``, ``self.intervention_statuses``,
        and ``self.travel_matrix`` to be set (the first two are set by
        ``super().__init__()``; travel_matrix is set by the subclass or
        ``prepare_initial_state()``).

        Args:
            t: Current time step (passed to derivative by the solver).
            rates: Dict of rate variable names → current values.
                Only rates targeted by interventions are modified.
            prop_infective: Proportion of infective population (scalar).

        Returns:
            Tuple of (modified rates dict, modified travel_matrix).
        """
        current_ordinal_day = self.start_date_ordinal + t
        travel_matrix = self.travel_matrix

        # Only apply interventions that are present in intervention_dict.
        # The runner creates a "model_without" by setting
        # intervention_dict = {}, which disables all interventions.
        active_interventions = [
            intv for intv in self.interventions if intv.id in self.intervention_dict
        ]

        # Pass 1: Date-based activation + rate reduction
        # (replicates jax_timestep_intervention)
        # In the original code, _update_date applies rate reduction
        # based on `in_window` (not the persisted status), so the
        # reduction only applies during the date window.
        for intv in active_interventions:
            status = self.intervention_statuses.get(intv.id, False)
            apply_flag, new_status = intv.check_date_activation(
                current_ordinal_day, status
            )
            rates = intv.apply_to_rates(rates, apply_flag)
            travel_matrix = intv.apply_to_travel(travel_matrix, apply_flag)
            self.intervention_statuses = {
                **self.intervention_statuses,
                intv.id: new_status,
            }

        # Pass 2: Threshold-based activation + rate reduction
        # (replicates jax_prop_intervention)
        # In the original code, _update_one applies rate reduction
        # based on `new_status` (the hysteresis-updated status), so
        # the reduction persists as long as the status is active.
        for intv in active_interventions:
            status = self.intervention_statuses.get(intv.id, False)
            new_status = intv.check_threshold_activation(prop_infective, status)
            rates = intv.apply_to_rates(rates, new_status)
            travel_matrix = intv.apply_to_travel(travel_matrix, new_status)
            self.intervention_statuses = {
                **self.intervention_statuses,
                intv.id: new_status,
            }

        self.travel_matrix = travel_matrix
        return rates, travel_matrix

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """
        Compute initial population distribution for this disease model.

        Default implementation for simple S-I models.
        Override in subclasses for disease-specific initialization logic.

        Args:
            admin_zones: List of admin zone dicts with population data
            compartment_list: List of compartment names
            **kwargs: Model-specific parameters (run_mode, etc.)

        Returns:
            numpy array of shape (n_zones, n_compartments)
        """
        column_mapping = {value: index for index, value in enumerate(compartment_list)}
        initial_population = np.zeros((len(admin_zones), len(compartment_list)))

        for i, zone in enumerate(admin_zones):
            infected = round(zone["infected_population"] / 100 * zone["population"], 2)
            susceptible = zone["population"] - infected
            initial_population[i, column_mapping["S"]] = susceptible
            initial_population[i, column_mapping["I"]] = infected

        return initial_population

    def _prepare_demographic_state(self) -> None:
        """
        Expand ``self.population_matrix`` from *(K, R)* to *(K, A, R)* using
        schema-declared demographic groups.

        Each compartment slice is split across age groups by distributing the
        population according to the fractional weights declared in
        ``self.demographics`` (a ``{group_id: percentage}`` dict).  Groups not
        present in the dict receive a weight of zero.

        After stratification, zero-valued rows are appended for every
        ``_total`` cumulative tracking compartment that is active in the
        current run but not yet in the population matrix.

        This method is intended for models that declare demographic groups via
        ``schema.add_demographic_group()`` in ``define_parameters()``.  Models
        that do not use demographics can ignore it entirely.

        Modifies ``self.population_matrix`` and ``self.compartment_list`` in
        place; returns ``None``.
        """
        import numpy as onp

        schema = type(self)._get_cached_schema()
        demographics = getattr(self, "demographics", None)

        if demographics:
            # Use the config's demographics values directly.  The keys may
            # differ from the schema's declared group IDs (e.g. "low_risk"
            # instead of "age_0_17") — using values() preserves the correct
            # population split regardless of naming.
            weights = onp.array(list(demographics.values()), dtype=float) / 100.0
        elif schema and schema.demographic_groups:
            # Fall back to schema default weights when no config demographics.
            raw = onp.array(
                [g.default_weight for g in schema.demographic_groups], dtype=float
            )
            weights = raw / raw.sum()
        else:
            weights = onp.array([1.0])

        # population_matrix is (K, R) — expand to (K, A, R)
        pm = onp.array(self.population_matrix)
        self.population_matrix = pm[:, None, :] * weights[None, :, None]

        # Append zero rows for _total compartments not already present
        if schema:
            active_set = set(self.compartment_list)
            seen: set[str] = set()
            for edge in schema.transmission_edges:
                tid = edge.target_id
                total_id = f"{tid}_total"
                if tid in active_set and total_id not in active_set and tid not in seen:
                    zero_row = onp.zeros((1, *self.population_matrix.shape[1:]))
                    self.population_matrix = onp.vstack(
                        (self.population_matrix, zero_row)
                    )
                    self.compartment_list = self.compartment_list + [total_id]
                    active_set.add(total_id)
                    seen.add(tid)

    def prepare_initial_state(self):
        pass

    def derivative(self, y, t, p):
        pass
