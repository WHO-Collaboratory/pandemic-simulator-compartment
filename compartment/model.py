"""Abstract base class for all models"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

import numpy as np
import jax.numpy as jnp

from compartment.parameters import (
    CompartmentDef,
    CompartmentRegistry,
    ModelParameterSchema,
    ParameterDef,
    ParameterSchemaBuilder,
    ValueType,
)

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

            # Auto-generate _total compartments for each edge target.
            # These are a framework concern (derivative accumulation,
            # post-processing) — model authors never declare them.
            cls._add_total_compartments(schema)

            cls.COMPARTMENTS = CompartmentRegistry(schema.compartments)
            # Keep COMPARTMENT_LIST as a plain list for backward compat
            # with the post-processor and other downstream code.
            cls.COMPARTMENT_LIST = list(cls.COMPARTMENTS)
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

    def __init__(self, config: dict):
        self.config = config

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

        return schema.build()

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

    def _load_transmission_params(self, transmission_dict: dict) -> None:
        """
        Set transmission-rate attributes (e.g. ``self.beta``, ``self.gamma``)
        from the config's ``transmission_dict``, using variable names
        declared in the parameter schema.

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
            setattr(
                self,
                edge.variable_name,
                transmission_dict.get(edge.variable_name),
            )

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

            rate = rates[edge.variable_name]
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

    def prepare_initial_state(self):
        pass

    def derivative(self, y, t, p):
        pass
