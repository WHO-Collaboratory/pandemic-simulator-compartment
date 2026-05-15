"""
Declarative parameter definition framework for disease models.

This module provides the building blocks for defining model parameters
with rich metadata. From a single ``ModelParameterSchema``, we can generate:

- Artifact JSON for UI form population, DB tables, and Zod schema generation
- Pydantic validation models for runtime config validation
- Example config JSON files with sensible defaults

Model authors interact with ``ParameterSchemaBuilder`` (passed into
``define_parameters()``) via type-safe ``add_*`` / ``set_*`` methods.
The internal dataclasses (``ParameterDef``, ``CompartmentDef``, etc.) are
implementation details -- model authors never need to import or construct them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ValueType(str, Enum):
    """Type of value -- drives UI input rendering and serialization."""

    RATE = "rate"  # e.g., transmission_rate: 0.3 per day
    DAYS = "days"  # e.g., incubation period: 5 days
    PERCENTAGE = "percentage"  # e.g., adherence: 80%
    COUNT = "count"  # e.g., population: 1_000_000
    DATE = "date"  # e.g., start_date: 2025-01-01
    BOOLEAN = "boolean"  # e.g., has_variance: true
    TEXT = "text"  # e.g., name: "New York"
    SELECT = "select"  # e.g., run_mode: one of [...]
    FLOAT = "float"  # generic float  (mapped → NUMBER in artifact)
    INTEGER = "integer"  # generic integer  (mapped → NUMBER in artifact)
    COORDINATE = "coordinate"  # lat/lon


# ---------------------------------------------------------------------------
# Core parameter definition
# ---------------------------------------------------------------------------


@dataclass
class ParameterDef:
    """
    Metadata for a single model parameter.

    This is the atomic unit -- every configurable value in a disease model
    should have a corresponding ParameterDef.
    """

    name: str  # machine key: "transmission_rate"
    label: str  # human label: "Transmission Rate (S->I)"
    description: str  # tooltip / help text
    value_type: ValueType  # drives input widget type

    default: Any = None  # default value
    min_value: Optional[float] = None  # hard minimum (validation)
    max_value: Optional[float] = None  # hard maximum (validation)

    # Variance / uncertainty defaults
    default_min: Optional[float] = None  # default lower bound for variance
    default_max: Optional[float] = None  # default upper bound for variance

    required: bool = True
    unit: Optional[str] = None  # display unit: "per day", "%"
    options: Optional[list[str]] = None  # for SELECT type

    def to_dict(self) -> dict:
        """Serialize to a plain dict (None values omitted).

        Field names are mapped to match the GraphQL ``FieldMetadata`` type
        used by the pandemic-simulator DB schema:

        - ``default`` → ``default_value``
        - ``min_value`` → ``min``
        - ``max_value`` → ``max``
        - ``options`` → ``enum``
        - ``value_type`` is emitted UPPERCASE to match the GraphQL
          ``ValueType`` enum (``RATE``, ``DAYS``, ``PERCENTAGE``, …).
        - ``FLOAT`` and ``INTEGER`` are mapped to ``NUMBER`` because the
          GraphQL ``ValueType`` enum does not have those variants; the
          ``min`` / ``max`` validation already distinguishes them.

        Extra runtime fields (``name``, ``required``, ``unit``) are
        preserved for UI/runtime consumers but are not part of
        ``FieldMetadata``.
        """
        # Map modeler-friendly types to the GraphQL ValueType enum values
        _ARTIFACT_VALUE_TYPE_MAP = {
            "FLOAT": "NUMBER",
            "INTEGER": "NUMBER",
        }
        raw_vt = self.value_type.value.upper()
        mapped_vt = _ARTIFACT_VALUE_TYPE_MAP.get(raw_vt, raw_vt)

        d = {
            # FieldMetadata fields
            "label": self.label,
            "description": self.description,
            "value_type": mapped_vt,
            "default_value": self.default,
            "default_min": self.default_min,
            "default_max": self.default_max,
            "min": self.min_value,
            "max": self.max_value,
            "enum": self.options,
            # Extra runtime fields (not in FieldMetadata)
            "name": self.name,
            "required": self.required,
            "unit": self.unit,
        }
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Compartment definition
# ---------------------------------------------------------------------------


@dataclass
class CompartmentDef:
    """
    Describes a single compartment in the model (e.g. S, I, R).

    Set ``infective=True`` on compartments whose population contributes
    to the force of infection (e.g. the *I* compartment in an SIR model).
    When a transmission edge is marked ``frequency_dependent=True``,
    :meth:`Model._compute_derivatives` uses the sum of all infective
    compartments to compute the flow:
    ``source * rate * sum(infective) / N_total``.
    """

    id: str  # short key used in matrices: "S"
    label: str  # human-readable: "Susceptible"
    description: str  # explanation for UI
    infective: bool = False  # contributes to force of infection

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "infective": self.infective,
        }


class CompartmentRegistry:
    """
    Attribute-style access to compartment IDs, auto-populated from the
    parameter schema.

    Set as ``cls.COMPARTMENTS`` on migrated model classes by
    ``Model.__init_subclass__``.

    Usage::

        # Named access (returns the string ID)
        cls.COMPARTMENTS.S   # → "S"
        cls.COMPARTMENTS.I   # → "I"

        # Iterable (drop-in replacement for COMPARTMENT_LIST)
        list(cls.COMPARTMENTS)      # → ["S", "I", "R"]
        len(cls.COMPARTMENTS)       # → 3
        "S" in cls.COMPARTMENTS     # → True

        # In derivative():
        states = {c: y[i] for i, c in enumerate(self.COMPARTMENTS)}
        S = states[self.COMPARTMENTS.S]

        # Stack results in schema order:
        return np.stack([derivs[c] for c in self.COMPARTMENTS])
    """

    def __init__(self, compartment_defs: list[CompartmentDef]) -> None:
        self._defs = compartment_defs
        self._ids = [c.id for c in compartment_defs]
        self._def_map: dict[str, CompartmentDef] = {c.id: c for c in compartment_defs}
        self._infective_ids: list[str] = [c.id for c in compartment_defs if c.infective]
        for c in compartment_defs:
            setattr(self, c.id, c.id)

    def get_def(self, compartment_id: str) -> CompartmentDef:
        """Look up the full :class:`CompartmentDef` by its short ID."""
        return self._def_map[compartment_id]

    @property
    def infective_ids(self) -> list[str]:
        """Compartment IDs marked ``infective=True``."""
        return self._infective_ids

    def __iter__(self):
        return iter(self._ids)

    def __len__(self):
        return len(self._ids)

    def __contains__(self, item):
        return item in self._ids

    def __getattr__(self, name: str) -> str:
        # Provide a clear error for typos / wrong compartment names
        raise AttributeError(f"No compartment '{name}'. Available: {self._ids}")

    def __repr__(self):
        return f"CompartmentRegistry({self._ids})"


# ---------------------------------------------------------------------------
# Transmission edge definition
# ---------------------------------------------------------------------------


@dataclass
class TransmissionEdgeDef:
    """
    Defines a directed edge in the compartment graph with parameter metadata.

    Example: susceptible -> infected  (variable_name="beta")

    When ``frequency_dependent`` is ``True``, the framework computes the
    flow as ``source * rate * sum(infective) / N_total`` instead of the
    default ``rate * source``.  This is the standard frequency-dependent
    force of infection used in most compartmental models.
    """

    source: str  # human-readable label: "susceptible"
    target: str  # human-readable label: "infected"
    source_id: str  # resolved compartment ID: "S"
    target_id: str  # resolved compartment ID: "I"
    variable_name: str  # model attribute name: "beta"
    parameter: ParameterDef  # metadata for the rate on this edge
    frequency_dependent: bool = False  # use FOI formula instead of simple rate

    def to_dict(self, order: int | None = None) -> dict:
        """Serialize to a dict matching the ``TransmissionEdge`` DB schema.

        The nested ``parameter`` is emitted as ``metadata`` with
        ``FieldMetadata``-compatible keys.  ``value_type`` is also
        hoisted to the edge level (UPPERCASE) so consumers can read it
        without digging into ``metadata``.  The optional *order*
        argument is injected into ``metadata.order``.
        """
        _ARTIFACT_VALUE_TYPE_MAP = {"FLOAT": "NUMBER", "INTEGER": "NUMBER"}
        raw_vt = self.parameter.value_type.value.upper()
        mapped_vt = _ARTIFACT_VALUE_TYPE_MAP.get(raw_vt, raw_vt)

        metadata = self.parameter.to_dict()
        if order is not None:
            metadata["order"] = order
        return {
            "variable_name": self.variable_name,
            "value_type": mapped_vt,
            "description": self.parameter.description,
            "source": self.source,
            "target": self.target,
            "frequency_dependent": self.frequency_dependent,
            "metadata": metadata,
        }


# ---------------------------------------------------------------------------
# Intervention definition
# ---------------------------------------------------------------------------

# Shared parameter templates for common intervention fields.
# Models reference these when building their InterventionDef.parameters list.


def _intervention_shared_parameters() -> list[ParameterDef]:
    """Common parameters that every intervention type may expose."""
    return [
        ParameterDef(
            name="adherence_min",
            label="Adherence",
            description="Minimum population adherence to the intervention",
            value_type=ValueType.PERCENTAGE,
            default=50.0,
            min_value=0,
            max_value=100,
            unit="%",
            required=False,
        ),
        ParameterDef(
            name="transmission_percentage",
            label="Transmission Reduction",
            description="Percentage reduction in transmission while intervention is active",
            value_type=ValueType.PERCENTAGE,
            default=5.0,
            min_value=0,
            max_value=100,
            unit="%",
            required=False,
        ),
        ParameterDef(
            name="start_date",
            label="Start Date",
            description="Date the intervention begins (leave blank for threshold-based)",
            value_type=ValueType.DATE,
            required=False,
        ),
        ParameterDef(
            name="end_date",
            label="End Date",
            description="Date the intervention ends (leave blank for threshold-based)",
            value_type=ValueType.DATE,
            required=False,
        ),
        ParameterDef(
            name="start_threshold",
            label="Start Threshold",
            description="Proportion of infected population that triggers the intervention",
            value_type=ValueType.PERCENTAGE,
            min_value=0,
            max_value=100,
            unit="%",
            required=False,
        ),
        ParameterDef(
            name="end_threshold",
            label="End Threshold",
            description="Proportion of infected population below which the intervention stops",
            value_type=ValueType.PERCENTAGE,
            min_value=0,
            max_value=100,
            unit="%",
            required=False,
        ),
    ]


@dataclass
class InterventionDef:
    """
    Defines a supported intervention type and its configurable parameters.

    Example: social_isolation with adherence, transmission_percentage, etc.

    ``target_rates`` declares which transmission-edge variable names this
    intervention modifies (e.g. ``["beta"]``).  At runtime the
    ``Intervention`` class reads this list and applies the reduction
    formula only to those rates — no per-disease if/else branching.

    If ``modifies_travel`` is ``True`` the intervention replaces the
    travel matrix with an identity matrix when active (lockdown behavior).
    """

    id: str  # "social_isolation", "vaccination", ...
    label: str  # "Social Isolation"
    description: str
    target_rates: list[str] = field(default_factory=list)  # e.g. ["beta"]
    modifies_travel: bool = False  # lockdown: replace travel matrix with I
    parameters: list[ParameterDef] = field(
        default_factory=_intervention_shared_parameters
    )

    def to_dict(self) -> dict:
        """Serialize to a dict matching the ``Intervention`` DB schema.

        Key mappings:
        - ``id`` → ``name``  (machine key, replaces the legacy enum)
        - ``label`` → ``display_name``  (human-readable name)
        - A top-level ``metadata`` object is added for the intervention
          itself (used by the ``FieldMetadata`` column on ``Intervention``).
        - Each entry in ``parameters`` is wrapped as
          ``{"name": …, "metadata": {…}}`` with auto-assigned ``order``.
        """
        params = []
        for idx, p in enumerate(self.parameters, start=1):
            meta = p.to_dict()
            meta["order"] = idx
            params.append(
                {
                    "name": p.name,
                    "metadata": meta,
                }
            )

        return {
            "name": self.id,
            "display_name": self.label,
            "description": self.description,
            "target_rates": self.target_rates,
            "modifies_travel": self.modifies_travel,
            "metadata": {
                "label": self.label,
                "description": self.description,
            },
            "parameters": params,
        }


# ---------------------------------------------------------------------------
# Demographic group and contact matrix definitions
# ---------------------------------------------------------------------------


@dataclass
class DemographicGroupDef:
    """
    Defines a single demographic group (e.g. an age band).

    ``default_weight`` is the percentage of the total population in this
    group and is used to split the initial population tensor when no
    per-zone overrides are provided.

    ``age_range`` is optional: when every group in a schema declares one,
    the framework will auto-load a country-specific contact matrix (Prem
    2021 synthetic matrices) and aggregate it to the declared bands.  When
    any group is missing ``age_range``, the contact matrix falls back to
    identity + explicit ``set_contact_override`` calls.
    """

    id: str  # "age_0_17"
    label: str  # "Children (0-17)"
    default_weight: float  # percentage of total population (0-100)
    age_range: Optional[tuple[int, int]] = None  # inclusive (low, high), e.g. (0, 17)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "default_weight": self.default_weight,
            "age_range": list(self.age_range) if self.age_range else None,
        }


@dataclass
class ContactOverrideDef:
    """
    A single (from_group, to_group) entry in the contact matrix.

    The contact matrix defaults to identity (each group only contacts
    itself).  Add overrides to model cross-demographic exposure.
    """

    from_group: str  # "age_0_17"
    to_group: str  # "age_18_55"
    value: float  # contact rate (replaces the identity default)

    def to_dict(self) -> dict:
        return {
            "from_group": self.from_group,
            "to_group": self.to_group,
            "value": self.value,
        }


# ---------------------------------------------------------------------------
# Full model parameter schema
# ---------------------------------------------------------------------------


@dataclass
class ModelParameterSchema:
    """
    Complete parameter schema for a disease model.

    This is the single source of truth.  From it we generate:
    - artifact JSON  (for UI / tables / Zod)
    - Pydantic validation models  (runtime)
    - example config JSON  (for testing / docs)
    """

    disease_type: str  # "MONKEYPOX"
    disease_label: str  # "Monkeypox"
    description: str  # model description

    compartments: list[CompartmentDef]
    transmission_edges: list[TransmissionEdgeDef]

    interventions: list[InterventionDef] = field(default_factory=list)
    travel_volume: Optional[dict[str, ParameterDef]] = (
        None  # {"leaving": ParameterDef, ...}
    )

    # Fields that appear on each admin zone (beyond the shared defaults)
    admin_zone_fields: list[ParameterDef] = field(default_factory=list)

    # Disease-specific top-level params (e.g. immunity_period for Dengue)
    disease_parameters: list[ParameterDef] = field(default_factory=list)

    # Shared simulation-level params (start_date, end_date, run_mode, ...)
    simulation_parameters: list[ParameterDef] = field(default_factory=list)

    # Demographic groups and contact matrix (optional — models without
    # demographics leave these empty and the framework is unaffected)
    demographic_groups: list[DemographicGroupDef] = field(default_factory=list)
    contact_matrix_overrides: list[ContactOverrideDef] = field(default_factory=list)

    # ---------------------------------------------------------------
    # Serialization helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _wrap_parameter(param: ParameterDef, order: int) -> dict:
        """Wrap a ``ParameterDef`` as ``{"name": …, "metadata": {…}}``."""
        meta = param.to_dict()
        meta["order"] = order
        return {"name": param.name, "metadata": meta}

    def to_artifact_dict(self) -> dict:
        """
        Generate the artifact JSON structure consumed by the UI layer,
        database seeding, and downstream Zod schema generation.

        The output is shaped so that each section maps directly to the
        corresponding DynamoDB / GraphQL table in pandemic-simulator:

        - Top-level ``name`` / ``definition`` → ``ModelArtifact`` table.
        - ``transmission_edges[].metadata`` → ``TransmissionEdge.metadata``
          (``FieldMetadata``).
        - ``interventions[].metadata`` → ``Intervention.metadata``.
        - ``custom_fields`` → ``CustomField`` records with a ``category``
          tag (``admin_zone`` or ``disease_parameter``).
        - ``simulation_parameters`` → shared simulation-level fields
          (start_date, end_date, run_mode, travel volume, …).
        """

        # -- Transmission edges with auto-assigned order --------------------
        edges = [
            e.to_dict(order=idx)
            for idx, e in enumerate(self.transmission_edges, start=1)
        ]

        # -- Custom fields (model-specific bespoke inputs) ------------------
        # admin_zone and disease_parameter fields are merged into a single
        # ``custom_fields`` array with a ``category`` tag.  Consumers
        # filter by category when they need a specific subset.
        admin_zone = [
            self._wrap_parameter(f, idx)
            for idx, f in enumerate(self.admin_zone_fields, start=1)
        ]
        disease_params = [
            self._wrap_parameter(p, idx)
            for idx, p in enumerate(self.disease_parameters, start=1)
        ]

        custom_fields: list[dict[str, Any]] = []
        for entry in admin_zone:
            custom_fields.append({**entry, "category": "admin_zone"})
        for entry in disease_params:
            custom_fields.append({**entry, "category": "disease_parameter"})

        # -- Simulation parameters (shared across all models) ---------------
        # These are non-negotiable baseline fields every simulation gets.
        # Travel volume parameters are appended here as well.
        sim_params = [
            self._wrap_parameter(p, idx)
            for idx, p in enumerate(self.simulation_parameters, start=1)
        ]
        if self.travel_volume:
            for _key, param in self.travel_volume.items():
                sim_params.append(self._wrap_parameter(param, len(sim_params) + 1))

        # -- Assemble -------------------------------------------------------
        result: dict[str, Any] = {
            # ModelArtifact identity
            "name": self.disease_label,
            "definition": self.description,
            # Compartment graph
            "compartments": [c.to_dict() for c in self.compartments],
            "transmission_edges": edges,
            # Interventions
            "interventions": [i.to_dict() for i in self.interventions],
            # Model-specific custom fields (admin_zone + disease_parameter)
            "custom_fields": custom_fields,
            # Shared simulation parameters (+ travel volume)
            "simulation_parameters": sim_params,
            # Demographics (optional)
            "demographic_groups": [g.to_dict() for g in self.demographic_groups],
            "contact_matrix_overrides": [
                o.to_dict() for o in self.contact_matrix_overrides
            ],
        }
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize the artifact dict to a JSON string."""
        return json.dumps(self.to_artifact_dict(), indent=indent)

    def to_example_config(self) -> dict:
        """
        Generate an example simulation config JSON from parameter defaults.

        The output matches the format expected by ``load_config_from_json()``
        (the "short-form" that gets wrapped in the GraphQL envelope).

        For required date fields that have no default, sensible placeholder
        values are generated automatically so the config is immediately
        runnable.
        """
        from datetime import date, timedelta

        # --- Disease section ---
        disease: dict[str, Any] = {
            "disease_type": self.disease_type,
        }

        # Transmission edges
        if self.transmission_edges:
            edges = []
            for edge_def in self.transmission_edges:
                edges.append(
                    {
                        "source": edge_def.source,
                        "target": edge_def.target,
                        "data": {
                            "transmission_rate": edge_def.parameter.default,
                        },
                    }
                )
            disease["transmission_edges"] = edges

        # Disease-specific params
        for param in self.disease_parameters:
            disease[param.name] = param.default

        # --- Top-level simulation fields ---
        config: dict[str, Any] = {}

        # Emit all simulation parameters with defaults or sensible fallbacks
        sim_params = {p.name: p for p in self.simulation_parameters}

        # Dates: use provided defaults or generate 90-day window from today
        today = date.today()
        if "start_date" in sim_params:
            val = sim_params["start_date"].default or today.isoformat()
            config["start_date"] = val
        if "end_date" in sim_params:
            val = (
                sim_params["end_date"].default
                or (today + timedelta(days=90)).isoformat()
            )
            config["end_date"] = val

        config["Disease"] = disease

        # --- Admin unit stub ---
        admin0_id = "USA"
        config["admin_unit_id"] = admin0_id
        config["AdminUnit"] = {"id": admin0_id, "center_lat": 37.0902}

        # --- Travel volume ---
        if self.travel_volume:
            tv: dict[str, Any] = {}
            for key, param in self.travel_volume.items():
                tv[key] = param.default
            config["travel_volume"] = tv

        # --- Interventions stub ---
        if self.interventions:
            interventions_list = []
            for intv_def in self.interventions:
                intv: dict[str, Any] = {"id": intv_def.id}
                for p in intv_def.parameters:
                    intv[p.name] = p.default
                interventions_list.append(intv)
            config["interventions"] = interventions_list

        # --- Case file with example zones ---
        base_zone_fields: dict[str, Any] = {
            "center_lat": 40.7128,
            "center_lon": -74.0060,
            "name": "Example Zone",
            "population": 1_000_000,
            "infected_population": 1,
        }
        # Overlay any model-specific admin zone fields
        for field_def in self.admin_zone_fields:
            base_zone_fields[field_def.name] = field_def.default

        config["case_file"] = {"admin_zones": [base_zone_fields]}

        return config


# ---------------------------------------------------------------------------
# Builder — the public API for model authors
# ---------------------------------------------------------------------------


class ParameterSchemaBuilder:
    """
    Builder for constructing a ``ModelParameterSchema`` via type-safe method calls.

    An instance is passed into ``Model.define_parameters(schema)`` by the
    framework.  Model authors call ``add_*`` / ``set_*`` methods to declare
    parameters.  The framework then calls ``build()`` to produce the final
    ``ModelParameterSchema``.

    Example::

        @classmethod
        def define_parameters(cls, schema):
            schema.set_model_info("MONKEYPOX", "Monkeypox", "SIR model for Monkeypox")

            schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
            schema.add_compartment("I", "Infected", "Currently infected population")
            schema.add_compartment("R", "Recovered", "Recovered and immune population")

            schema.add_transmission_edge(
                source="susceptible", target="infected", variable_name="beta",
                label="Transmission Rate (S->I)",
                description="Rate of new infections from contact",
                default=0.3, min_value=0.01, max_value=2.0,
            )
    """

    def __init__(self) -> None:
        self._disease_type: str | None = None
        self._disease_label: str | None = None
        self._description: str | None = None
        self._compartments: list[CompartmentDef] = []
        self._transmission_edges: list[TransmissionEdgeDef] = []
        self._interventions: list[InterventionDef] = []
        self._travel_volume: dict[str, ParameterDef] | None = None
        self._admin_zone_fields: list[ParameterDef] = []
        self._disease_parameters: list[ParameterDef] = []
        self._simulation_parameters: list[ParameterDef] = []
        self._demographic_groups: list[DemographicGroupDef] = []
        self._contact_overrides: list[ContactOverrideDef] = []

    # ----- Identity --------------------------------------------------------

    def set_model_info(
        self,
        disease_type: str,
        label: str,
        description: str,
    ) -> None:
        """
        Set the disease model identity.  Must be called exactly once.

        Args:
            disease_type: Machine identifier (e.g. ``"MONKEYPOX"``).
            label: Human-readable name (e.g. ``"Monkeypox"``).
            description: Short description shown in the UI.
        """
        if self._disease_type is not None:
            raise ValueError("set_model_info() has already been called")
        self._disease_type = disease_type
        self._disease_label = label
        self._description = description

    # ----- Compartments ----------------------------------------------------

    def add_compartment(
        self,
        id: str,
        label: str,
        description: str,
        infective: bool = False,
    ) -> None:
        """
        Add a compartment to the model (e.g. S, I, R).

        Set ``infective=True`` for compartments whose population
        contributes to the force of infection.  When a transmission
        edge is marked ``frequency_dependent=True``, the framework
        uses the sum of all infective compartments to compute the
        flow: ``source * rate * sum(infective) / N_total``.

        For most models only the *Infected* compartment (or its
        equivalents) should be marked infective.

        Args:
            id: Short key used in matrices and edge definitions (e.g. ``"S"``).
            label: Human-readable name (e.g. ``"Susceptible"``).
            description: Explanation shown in the UI.
            infective: Whether this compartment's population contributes
                to the force of infection used by ``frequency_dependent``
                transmission edges.  Defaults to ``False``.

        Raises:
            ValueError: If a compartment with the same *id* already exists.
        """
        existing_ids = {c.id for c in self._compartments}
        if id in existing_ids:
            raise ValueError(
                f"Duplicate compartment id '{id}'. "
                f"Already registered: {sorted(existing_ids)}"
            )
        self._compartments.append(
            CompartmentDef(
                id=id,
                label=label,
                description=description,
                infective=infective,
            )
        )

    def remove_compartment(self, id: str) -> None:
        """Remove a compartment and any edges that reference it by source or target."""
        self._compartments = [c for c in self._compartments if c.id != id]
        self._transmission_edges = [
            e
            for e in self._transmission_edges
            if e.source_id != id and e.target_id != id
        ]

    def remove_transmission_edge(self, variable_name: str) -> None:
        """Remove a transmission edge by its variable name."""
        self._transmission_edges = [
            e for e in self._transmission_edges if e.variable_name != variable_name
        ]

    # ----- Transmission edges ----------------------------------------------

    def _resolve_compartment_id(self, name: str) -> str:
        """
        Resolve a compartment label or ID to its canonical short ID.

        Matches case-insensitively against both ``CompartmentDef.id``
        and ``CompartmentDef.label``.

        Args:
            name: Label or ID to resolve (e.g. ``"susceptible"`` or ``"S"``).

        Returns:
            The canonical compartment ID (e.g. ``"S"``).

        Raises:
            ValueError: If no matching compartment is found.
        """
        for c in self._compartments:
            if c.id.lower() == name.lower() or c.label.lower() == name.lower():
                return c.id
        display = [f"{c.id} ({c.label})" for c in self._compartments]
        raise ValueError(f"Unknown compartment '{name}'. Available: {sorted(display)}")

    def add_transmission_edge(
        self,
        source: str,
        target: str,
        variable_name: str,
        label: str,
        description: str,
        default: float,
        min_value: float | None = None,
        max_value: float | None = None,
        default_min: float | None = None,
        default_max: float | None = None,
        unit: str = "per day",
        frequency_dependent: bool = False,
        value_type: ValueType = ValueType.RATE,
    ) -> None:
        """
        Add a directed transmission edge between two compartments.

        The ``source`` and ``target`` are matched against previously added
        compartment IDs or labels (case-insensitive).  The resolved short
        IDs are stored as ``source_id`` and ``target_id`` on the edge for
        direct use in derivative computations.

        All numeric arguments (``default``, ``min_value``, ``max_value``,
        ``default_min``, ``default_max``) should be in the **native units**
        of the chosen ``value_type``:

        - ``ValueType.RATE``: per-day rate (e.g. ``0.3``).
        - ``ValueType.DAYS``: duration in days (e.g. ``5``).  The framework
          converts to a rate (``1 / days``) at model load time.
        - ``ValueType.PERCENTAGE``: percentage 0-100 (e.g. ``4``).  The
          framework converts to a fraction (``value / 100``) at model load
          time.

        Args:
            source: Source compartment (e.g. ``"susceptible"``).
            target: Target compartment (e.g. ``"infected"``).
            variable_name: Model attribute name (e.g. ``"beta"``).
            label: Human-readable edge label (e.g. ``"Transmission Rate (S->I)"``).
            description: Tooltip text for the UI.
            default: Default value in native units of ``value_type``.
            min_value: Hard minimum for validation (native units).
            max_value: Hard maximum for validation (native units).
            default_min: Default lower bound for variance / uncertainty (native units).
            default_max: Default upper bound for variance / uncertainty (native units).
            unit: Display unit (defaults to ``"per day"``).
            frequency_dependent: If ``True``, the framework computes flow
                as ``source * rate * sum(infective) / N_total`` instead
                of the default ``rate * source``.  Use this for edges
                where transmission depends on the proportion of infective
                individuals in the population (e.g. S→I in SIR).
            value_type: The unit system for this edge's value.  Defaults
                to ``ValueType.RATE``.

        Raises:
            ValueError: If source or target doesn't match a known compartment.
        """
        source_id = self._resolve_compartment_id(source)
        target_id = self._resolve_compartment_id(target)

        self._transmission_edges.append(
            TransmissionEdgeDef(
                source=source,
                target=target,
                source_id=source_id,
                target_id=target_id,
                variable_name=variable_name,
                parameter=ParameterDef(
                    name="transmission_rate",
                    label=label,
                    description=description,
                    value_type=value_type,
                    default=default,
                    min_value=min_value,
                    max_value=max_value,
                    default_min=default_min,
                    default_max=default_max,
                    unit=unit,
                ),
                frequency_dependent=frequency_dependent,
            )
        )

    # ----- Interventions ---------------------------------------------------

    def add_intervention(
        self,
        id: str,
        label: str,
        description: str,
        target_rates: list[str] | None = None,
        modifies_travel: bool = False,
        adherence: float | None = None,
        transmission_reduction: float | None = None,
    ) -> None:
        """
        Add a supported intervention type.

        Uses shared default parameters (adherence, transmission_percentage,
        start/end date, start/end threshold).  Pass ``adherence`` and/or
        ``transmission_reduction`` to override the generic defaults with
        disease-specific values.

        ``target_rates`` declares which transmission-edge variable names
        this intervention modifies at runtime (e.g. ``["beta"]``).  This
        eliminates the hardcoded per-disease if/else chains in
        ``interventions.py``.

        Set ``modifies_travel=True`` for lockdown-style interventions that
        replace the travel matrix with an identity matrix when active.

        Args:
            id: Machine key (e.g. ``"social_isolation"``, ``"vaccination"``).
            label: Human-readable name (e.g. ``"Social Isolation"``).
            description: Explanation shown in the UI.
            target_rates: Variable names of transmission edges this
                intervention modifies (e.g. ``["beta"]``).  Defaults to
                an empty list (intervention has no rate effect, e.g.
                lockdown only modifies travel).
            modifies_travel: If ``True``, replaces the travel matrix with
                identity when the intervention is active (lockdown).
            adherence: Default population adherence percentage (0-100).
                Overrides the generic 50% default on ``adherence_min``.
            transmission_reduction: Default transmission reduction percentage
                (0-100).  Overrides the generic 5% default on
                ``transmission_percentage``.
        """
        # Start from shared parameter templates; apply overrides if given.
        overrides = {}
        if adherence is not None:
            overrides["adherence_min"] = adherence
        if transmission_reduction is not None:
            overrides["transmission_percentage"] = transmission_reduction

        params = _intervention_shared_parameters()
        if overrides:
            for p in params:
                if p.name in overrides:
                    p.default = overrides[p.name]

        self._interventions.append(
            InterventionDef(
                id=id,
                label=label,
                description=description,
                target_rates=target_rates or [],
                modifies_travel=modifies_travel,
                parameters=params,
            )
        )

    # ----- Travel volume ---------------------------------------------------

    def set_travel_volume(
        self,
        leaving_default: float = 0.2,
        leaving_description: str = "Proportion of population traveling out of their zone per day",
        leaving_min: float = 0.0,
        leaving_max: float = 1.0,
        returning_default: float | None = None,
        returning_description: str = "Proportion of traveling population returning per day",
        returning_required: bool = False,
    ) -> None:
        """
        Enable inter-zone travel for this model.

        Args:
            leaving_default: Default fraction of population leaving.
            leaving_description: UI description for the leaving parameter.
            leaving_min: Hard minimum for leaving.
            leaving_max: Hard maximum for leaving.
            returning_default: Default fraction returning (None = optional).
            returning_description: UI description for the returning parameter.
            returning_required: Whether the returning parameter is required.
        """
        tv: dict[str, ParameterDef] = {
            "leaving": ParameterDef(
                name="leaving",
                label="Outbound Travel Rate",
                description=leaving_description,
                value_type=ValueType.PERCENTAGE,
                default=leaving_default,
                min_value=leaving_min,
                max_value=leaving_max,
                unit="%",
            ),
        }
        tv["returning"] = ParameterDef(
            name="returning",
            label="Return Travel Rate",
            description=returning_description,
            value_type=ValueType.PERCENTAGE,
            default=returning_default,
            min_value=0.0,
            max_value=1.0,
            unit="%",
            required=returning_required,
        )
        self._travel_volume = tv

    # ----- Custom fields (enforced contract) --------------------------------

    def add_admin_zone_field(
        self,
        name: str,
        label: str,
        description: str,
        value_type: ValueType,
        default: Any,
        min_value: float | None = None,
        max_value: float | None = None,
        default_min: float | None = None,
        default_max: float | None = None,
        unit: str | None = None,
        required: bool = False,
        options: list[str] | None = None,
    ) -> None:
        """
        Add a per-zone field beyond the shared defaults (name, population, etc.).

        Use this for disease-specific zone data like seroprevalence or temperature.

        Args:
            name: Machine key (e.g. ``"seroprevalence"``).
            label: Human-readable label (e.g. ``"Seroprevalence"``).
            description: Tooltip / help text for the UI.
            value_type: Controls input widget rendering (e.g. ``ValueType.PERCENTAGE``).
            default: Default value -- required so the example config is runnable.
            min_value: Hard minimum for validation.
            max_value: Hard maximum for validation.
            default_min: Default lower bound for variance / uncertainty.
            default_max: Default upper bound for variance / uncertainty.
            unit: Display unit (e.g. ``"%"``, ``"days"``).
            required: Whether this field is required in the config.
            options: Valid choices for ``ValueType.SELECT`` fields.
        """
        self._admin_zone_fields.append(
            ParameterDef(
                name=name,
                label=label,
                description=description,
                value_type=value_type,
                default=default,
                min_value=min_value,
                max_value=max_value,
                default_min=default_min,
                default_max=default_max,
                unit=unit,
                required=required,
                options=options,
            )
        )

    def add_disease_parameter(
        self,
        name: str,
        label: str,
        description: str,
        value_type: ValueType,
        default: Any,
        min_value: float | None = None,
        max_value: float | None = None,
        default_min: float | None = None,
        default_max: float | None = None,
        unit: str | None = None,
        required: bool = True,
        options: list[str] | None = None,
    ) -> None:
        """
        Add a disease-specific top-level parameter.

        Use this for parameters that don't fit into transmission edges,
        interventions, or admin zone fields -- e.g. immunity_period for Dengue.

        Args:
            name: Machine key (e.g. ``"immunity_period"``).
            label: Human-readable label (e.g. ``"Cross-Immunity Period"``).
            description: Tooltip / help text for the UI.
            value_type: Controls input widget rendering (e.g. ``ValueType.DAYS``).
            default: Default value -- required so the example config is runnable.
            min_value: Hard minimum for validation.
            max_value: Hard maximum for validation.
            default_min: Default lower bound for variance / uncertainty.
            default_max: Default upper bound for variance / uncertainty.
            unit: Display unit (e.g. ``"days"``, ``"per day"``).
            required: Whether this parameter is required in the config.
            options: Valid choices for ``ValueType.SELECT`` fields.
        """
        self._disease_parameters.append(
            ParameterDef(
                name=name,
                label=label,
                description=description,
                value_type=value_type,
                default=default,
                min_value=min_value,
                max_value=max_value,
                default_min=default_min,
                default_max=default_max,
                unit=unit,
                required=required,
                options=options,
            )
        )

    # ----- Demographics ----------------------------------------------------

    def add_demographic_group(
        self,
        id: str,
        label: str,
        default_weight: float,
        age_range: Optional[tuple[int, int]] = None,
    ) -> None:
        """
        Declare a demographic group (e.g. an age band).

        Groups are ordered by declaration order — that order defines axis 1
        of the population tensor and the rows/columns of the contact matrix.

        Args:
            id: Machine key (e.g. ``"age_0_17"``).
            label: Human-readable name (e.g. ``"Children (0-17)"``).
            default_weight: Percentage of the total population in this group
                (0-100).  Used to split the initial population tensor when
                no per-zone override is provided.
            age_range: Optional inclusive (low, high) age tuple, e.g. ``(0, 17)``.
                When every group in the schema declares one, the framework
                auto-loads the country's Prem 2021 contact matrix and
                aggregates it to these bands.  Ranges must be non-overlapping
                across groups (enforced at :meth:`build`).

        Raises:
            ValueError: If a group with the same *id* already exists, or if
                ``age_range`` is not a 2-tuple of ints with
                ``0 <= low <= high <= 120``.
        """
        existing_ids = {g.id for g in self._demographic_groups}
        if id in existing_ids:
            raise ValueError(
                f"Duplicate demographic group id '{id}'. "
                f"Already registered: {sorted(existing_ids)}"
            )
        if age_range is not None:
            if (
                not isinstance(age_range, tuple)
                or len(age_range) != 2
                or not all(isinstance(v, int) for v in age_range)
            ):
                raise ValueError(
                    f"age_range for '{id}' must be a 2-tuple of ints, got {age_range!r}"
                )
            low, high = age_range
            if not (0 <= low <= high <= 120):
                raise ValueError(
                    f"age_range for '{id}' must satisfy 0 <= low <= high <= 120, "
                    f"got ({low}, {high})"
                )
        self._demographic_groups.append(
            DemographicGroupDef(
                id=id,
                label=label,
                default_weight=default_weight,
                age_range=age_range,
            )
        )

    def set_contact_override(
        self,
        from_group: str,
        to_group: str,
        value: float,
    ) -> None:
        """
        Set a single entry in the contact matrix.

        The contact matrix defaults to identity (each group only contacts
        itself at rate 1.0).  Call this to override any entry — including
        the diagonal if the self-contact rate differs from 1.0.

        Args:
            from_group: ID of the group being exposed (row index).
            to_group: ID of the group doing the infecting (column index).
            value: Contact rate replacing the identity default.
        """
        self._contact_overrides.append(
            ContactOverrideDef(from_group=from_group, to_group=to_group, value=value)
        )

    # ----- Build -----------------------------------------------------------

    def build(self) -> ModelParameterSchema:
        """
        Finalize and return the ``ModelParameterSchema``.

        The schema contains only the compartments and edges explicitly
        declared by the model author.  Cumulative ``_total`` tracking
        compartments are added later by the framework in
        :meth:`Model.__init_subclass__` so that the schema itself stays
        a clean representation of the model author's declarations.

        Raises:
            ValueError: If ``set_model_info()`` was not called or no
                compartments were added.
        """
        if not self._disease_type:
            raise ValueError("set_model_info() must be called before build()")
        if not self._compartments:
            raise ValueError(
                "At least one compartment must be added via add_compartment()"
            )

        # Validate age_range non-overlap among groups that declare one.
        # Overlapping ranges would double-count contacts during aggregation
        # from a synthetic source matrix (e.g. Prem 2021).
        ranged = [g for g in self._demographic_groups if g.age_range is not None]
        for i in range(len(ranged)):
            for j in range(i + 1, len(ranged)):
                a_lo, a_hi = ranged[i].age_range
                b_lo, b_hi = ranged[j].age_range
                if a_lo <= b_hi and b_lo <= a_hi:
                    raise ValueError(
                        f"Demographic groups '{ranged[i].id}' ({a_lo}-{a_hi}) and "
                        f"'{ranged[j].id}' ({b_lo}-{b_hi}) have overlapping age_ranges. "
                        f"Age ranges must be non-overlapping."
                    )

        return ModelParameterSchema(
            disease_type=self._disease_type,
            disease_label=self._disease_label or "",
            description=self._description or "",
            compartments=self._compartments,
            transmission_edges=self._transmission_edges,
            interventions=self._interventions,
            travel_volume=self._travel_volume,
            admin_zone_fields=self._admin_zone_fields,
            disease_parameters=self._disease_parameters,
            simulation_parameters=self._simulation_parameters,
            demographic_groups=self._demographic_groups,
            contact_matrix_overrides=self._contact_overrides,
        )
