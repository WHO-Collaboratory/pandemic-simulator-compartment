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


MAX_DEMOGRAPHIC_GROUPS = 20
"""Hard upper bound on the number of demographic groups per schema.

Prem 2021 source matrices are 16×16, so beyond ~16 age-ranged bands the
aggregated contact matrix becomes meaningless.  The limit is set slightly
higher (20) to accommodate models that mix ranged and unranged groups, while
still preventing accidental runaway declarations that would silently produce
enormous population tensors.
"""


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

    # Parameter variance / uncertainty defaults
    default_min: Optional[float] = None  # default lower bound for variance
    default_max: Optional[float] = None  # default upper bound for variance

    required: bool = True
    unit: Optional[str] = None  # display unit: "per day", "%"
    options: Optional[list[str]] = None  # for SELECT type
    enable_variance: bool = True  # set False to hide the variance checkbox in the UI

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
            # Only emit when explicitly disabled so the artifact stays clean.
            "enable_variance": None if self.enable_variance else False,
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
    :meth:`Model._compute_equations` uses the sum of all infective
    compartments to compute the flow:
    ``source * rate * sum(infective) / N_total``.
    """

    id: str  # short key used in matrices: "S"
    label: str  # human-readable: "Susceptible"
    description: str  # explanation for UI
    infective: bool = False  # contributes to force of infection

    def to_dict(self, order: int | None = None) -> dict:
        d = {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "infective": self.infective,
        }
        if order is not None:
            d["order"] = order
        return d


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

        # In equation():
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
# Custom field registries (disease parameters & admin zone fields)
# ---------------------------------------------------------------------------


class DiseaseParamRegistry:
    """
    Attribute-style access to disease parameter names, auto-populated
    from the parameter schema.

    Set as ``cls.DISEASE_PARAMS`` on migrated model classes by
    ``Model.__init_subclass__``.

    Usage::

        # Named access (returns the string name)
        cls.DISEASE_PARAMS.immunity_period   # → "immunity_period"
        cls.DISEASE_PARAMS.latent_period     # → "latent_period"

        # Iterable
        list(cls.DISEASE_PARAMS)                    # → ["immunity_period", ...]
        len(cls.DISEASE_PARAMS)                     # → 5
        "immunity_period" in cls.DISEASE_PARAMS     # → True

        # Full definition lookup
        cls.DISEASE_PARAMS.get_def("immunity_period")  # → ParameterDef(...)
    """

    def __init__(self, param_defs: list[ParameterDef]) -> None:
        self._defs = param_defs
        self._names = [p.name for p in param_defs]
        self._def_map: dict[str, ParameterDef] = {p.name: p for p in param_defs}
        for p in param_defs:
            setattr(self, p.name, p.name)

    def get_def(self, name: str) -> ParameterDef:
        """Look up the full :class:`ParameterDef` by its name."""
        return self._def_map[name]

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)

    def __contains__(self, item):
        return item in self._names

    def __getattr__(self, name: str) -> str:
        if name.startswith("_"):
            raise AttributeError(name)
        names = self.__dict__.get("_names", [])
        raise AttributeError(f"No disease parameter '{name}'. Available: {names}")

    def __repr__(self):
        return f"DiseaseParamRegistry({self._names})"


class AdminZoneFieldRegistry:
    """
    Attribute-style access to admin zone field names, auto-populated
    from the parameter schema.

    Set as ``cls.ADMIN_ZONE_FIELDS`` on migrated model classes by
    ``Model.__init_subclass__``.

    Admin zone fields are per-zone values (e.g. seroprevalence,
    vector_population) so this registry provides **name references
    only** — actual values live on each zone dict in the case file.

    Usage::

        # Named access (returns the string name — avoids magic strings)
        cls.ADMIN_ZONE_FIELDS.seroprevalence   # → "seroprevalence"

        # In get_initial_population():
        sero = zone.get(cls.ADMIN_ZONE_FIELDS.seroprevalence, 0)

        # Iterable
        list(cls.ADMIN_ZONE_FIELDS)                       # → ["seroprevalence", ...]
        "seroprevalence" in cls.ADMIN_ZONE_FIELDS         # → True

        # Full definition lookup
        cls.ADMIN_ZONE_FIELDS.get_def("seroprevalence")   # → ParameterDef(...)
    """

    def __init__(self, field_defs: list[ParameterDef]) -> None:
        self._defs = field_defs
        self._names = [f.name for f in field_defs]
        self._def_map: dict[str, ParameterDef] = {f.name: f for f in field_defs}
        for f in field_defs:
            setattr(self, f.name, f.name)

    def get_def(self, name: str) -> ParameterDef:
        """Look up the full :class:`ParameterDef` by its name."""
        return self._def_map[name]

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)

    def __contains__(self, item):
        return item in self._names

    def __getattr__(self, name: str) -> str:
        if name.startswith("_"):
            raise AttributeError(name)
        names = self.__dict__.get("_names", [])
        raise AttributeError(f"No admin zone field '{name}'. Available: {names}")

    def __repr__(self):
        return f"AdminZoneFieldRegistry({self._names})"


class DiseaseParamValues:
    """
    Attribute-style access to disease parameter *values* from config.

    Created at instance time in ``Model.__init__`` from the schema's
    disease parameter definitions combined with the config's Disease dict.

    Provides typed values with schema defaults as fallbacks::

        self.disease_params.immunity_period  # → 240 (int, from config or default)
        self.disease_params.latent_period    # → 5.9 (float)

    Iterable over parameter names::

        for name in self.disease_params:
            print(name, getattr(self.disease_params, name))
    """

    def __init__(self, param_defs: list[ParameterDef], disease_dict: dict) -> None:
        self._names: list[str] = []
        for p in param_defs:
            raw = disease_dict.get(p.name, p.default)
            setattr(self, p.name, raw)
            self._names.append(p.name)

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)

    def __contains__(self, item):
        return item in self._names

    def __getattr__(self, name: str):
        # Guard against recursion during deepcopy / pickle reconstruction:
        # when Python reconstructs the object, _names doesn't exist yet
        # and __getattr__ would recurse trying to access it.
        if name.startswith("_"):
            raise AttributeError(name)
        # Avoid accessing self._names if it hasn't been set yet
        names = self.__dict__.get("_names", [])
        raise AttributeError(f"No disease parameter value '{name}'. Available: {names}")

    def __repr__(self):
        vals = {n: getattr(self, n) for n in self._names}
        return f"DiseaseParamValues({vals})"


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
# Model metadata
# ---------------------------------------------------------------------------


@dataclass
class ModelMetadata:
    """
    Editorial and publication metadata for a disease model.

    All fields are optional — omitted fields are excluded from the artifact JSON.
    Stored under ``metadata`` in the artifact root.
    """

    authors: Optional[list[dict]] = None
    """List of author dicts: ``{"name": str, "email": str, "affiliation": str}``.
    All sub-keys are optional within each dict."""

    license: Optional[str] = None
    """SPDX identifier or free-text license name (e.g. ``"MIT"``, ``"CC BY 4.0"``)."""

    citations: Optional[list[str]] = None
    """DOI URLs, paper URLs, or repo links related to this model."""

    model_type: Optional[str] = None
    """Structural category of the model (e.g. ``"Compartmental"``, ``"Network"``)."""

    diseases: Optional[list[str]] = None
    """ICD-11 codes or free-text disease names this model was designed for."""

    transmission_routes: Optional[list[str]] = None
    """Transmission pathways (e.g. ``["Airborne", "Droplet"]``).
    No strict enum — pass any string; the UI offers a picker with a free-text fallback."""

    questions_answered: Optional[list[str]] = None
    """Research or policy questions this model is designed to answer."""

    key_assumptions: Optional[list[str]] = None
    """Core modelling assumptions authors want to surface to users."""

    applicability: Optional[str] = None
    """Contexts or settings where this model is appropriate."""

    not_for: Optional[str] = None
    """Contexts or use-cases where this model should *not* be applied."""

    constraints: Optional[str] = None
    """Known technical or data constraints."""

    biases: Optional[str] = None
    """Known biases in the model or its parameterisation."""

    validation: Optional[str] = None
    """Summary of validation work done on this model."""

    def to_dict(self) -> dict:
        """Serialize to a plain dict (None values omitted)."""
        d = {
            "authors": self.authors,
            "license": self.license,
            "citations": self.citations,
            "model_type": self.model_type,
            "diseases": self.diseases,
            "transmission_routes": self.transmission_routes,
            "questions_answered": self.questions_answered,
            "key_assumptions": self.key_assumptions,
            "applicability": self.applicability,
            "not_for": self.not_for,
            "constraints": self.constraints,
            "biases": self.biases,
            "validation": self.validation,
        }
        return {k: v for k, v in d.items() if v is not None}


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

    disease_type: str  # "MPOX"
    disease_label: str  # "MPOX"
    description: str  # model description

    compartments: list[CompartmentDef]
    transmission_edges: list[TransmissionEdgeDef]

    interventions: list[InterventionDef] = field(default_factory=list)

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

    # Ordered list of grouped display keys for the results sidebar.
    # Derived automatically from COMPARTMENT_DELTA_GROUPING when
    # available, otherwise defaults to the raw compartment IDs
    # (excluding cumulative _total entries).
    compartment_display_order: list[str] = field(default_factory=list)

    # Editorial / publication metadata (optional — omitted when not set).
    metadata: Optional[ModelMetadata] = None

    # Free-form Markdown documentation sourced from a ``model.md`` file in the
    # model's folder (optional — omitted when not present). Rendered on the
    # results page "You Should Know" accordion via Tailwind Typography.
    model_documentation: Optional[str] = None

    # Model-level run mode: DETERMINISTIC or STOCHASTIC.
    # Derived from the model class's STOCHASTIC attribute.
    run_mode: str = "DETERMINISTIC"

    # Number of stochastic/parameter uncertainty trajectories for this model.
    # Overrides the global default of 30 when set on the model class as NUM_RUNS.
    num_runs: int = 30
    num_runs_min: int = 1
    num_runs_max: int = 100

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
        # Mobility is *not* one of them — models that travel declare their own
        # travel parameters as disease parameters, so they land in custom_fields.
        sim_params = [
            self._wrap_parameter(p, idx)
            for idx, p in enumerate(self.simulation_parameters, start=1)
        ]

        # -- Assemble -------------------------------------------------------
        compartment_dicts = [
            c.to_dict(order=idx) for idx, c in enumerate(self.compartments, start=1)
        ]
        # Display order for the UI. Prefer an explicit order derived from the
        # model's COMPARTMENT_DELTA_GROUPING (the grouped/display compartments
        # that actually appear in time series + deltas — set on the schema in
        # Model.__init_subclass__); otherwise fall back to declared compartment
        # ids, excluding auto-generated cumulative (_total) compartments (which
        # the time-series/delta builders also drop). The frontend reads this
        # from the stored artifact JSON instead of hardcoding per-disease lists.
        display_order = getattr(self, "compartment_display_order", None) or [
            c["id"] for c in compartment_dicts if not c["id"].endswith("_total")
        ]
        result: dict[str, Any] = {
            # ModelArtifact identity
            "disease_type": self.disease_type,
            "name": self.disease_label,
            "definition": self.description,
            "run_mode": self.run_mode,
            # Editorial / publication metadata (omitted when not set)
            **({"metadata": self.metadata.to_dict()} if self.metadata else {}),
            # Free-form Markdown documentation from the model's model.md file
            # (omitted when not present). Rendered on the results page.
            **(
                {"documentation": self.model_documentation}
                if self.model_documentation
                else {}
            ),
            # num_runs is surfaced solely as a "disease_parameter" custom field
            # (with default_value/min/max in its metadata) — declare it via
            # schema.add_parameter(name="num_runs", ...). It is
            # intentionally NOT emitted at the artifact root; the runtime falls
            # back to the model-class NUM_RUNS default when no value is provided.
            # Compartment graph
            "compartments": compartment_dicts,
            "compartment_display_order": display_order,
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
            schema.set_model_info("MPOX", "MPOX", "SIR model for MPOX")

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
        self._admin_zone_fields: list[ParameterDef] = []
        self._disease_parameters: list[ParameterDef] = []
        self._simulation_parameters: list[ParameterDef] = []
        self._demographic_groups: list[DemographicGroupDef] = []
        self._contact_overrides: list[ContactOverrideDef] = []
        self._num_runs: int = 30
        self._num_runs_min: int = 1
        self._num_runs_max: int = 100
        self._metadata: ModelMetadata | None = None

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
            disease_type: Machine identifier (e.g. ``"MPOX"``).
            label: Human-readable name (e.g. ``"MPOX"``).
            description: Short description shown in the UI.
        """
        if self._disease_type is not None:
            raise ValueError("set_model_info() has already been called")
        self._disease_type = disease_type
        self._disease_label = label
        self._description = description

    def set_model_metadata(
        self,
        authors: list[dict] | None = None,
        license: str | None = None,
        citations: list[str] | None = None,
        model_type: str | None = None,
        diseases: list[str] | None = None,
        transmission_routes: list[str] | None = None,
        questions_answered: list[str] | None = None,
        key_assumptions: list[str] | None = None,
        applicability: str | None = None,
        not_for: str | None = None,
        constraints: str | None = None,
        biases: str | None = None,
        validation: str | None = None,
    ) -> None:
        """
        Set editorial and publication metadata for the model.  All fields are
        optional — omit any that don't apply.  Metadata is emitted under a
        top-level ``metadata`` key in the artifact JSON and has no effect on
        simulation runtime.

        Args:
            authors: List of author dicts.  Each dict may contain any subset of
                ``"name"``, ``"email"``, and ``"affiliation"`` keys.
            license: SPDX identifier or free-text license name
                (e.g. ``"MIT"``, ``"CC BY 4.0"``).
            citations: DOI URLs, paper URLs, or repo links related to this model.
            model_type: Structural category (e.g. ``"Compartmental"``, ``"Network"``).
            diseases: ICD-11 codes or free-text disease names this model targets.
            transmission_routes: Transmission pathways
                (e.g. ``["Airborne", "Droplet"]``).  No strict enum.
            questions_answered: Research or policy questions this model addresses.
            key_assumptions: Core modelling assumptions to surface to users.
            applicability: Contexts or settings where this model is appropriate.
            not_for: Use-cases where this model should *not* be applied.
            constraints: Known technical or data constraints.
            biases: Known biases in the model or its parameterisation.
            validation: Summary of validation work done on this model.
        """
        self._metadata = ModelMetadata(
            authors=authors,
            license=license,
            citations=citations,
            model_type=model_type,
            diseases=diseases,
            transmission_routes=transmission_routes,
            questions_answered=questions_answered,
            key_assumptions=key_assumptions,
            applicability=applicability,
            not_for=not_for,
            constraints=constraints,
            biases=biases,
            validation=validation,
        )

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
        direct use in equation computations.

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
            default_min: Default lower bound for parameter variance / uncertainty (native units).
            default_max: Default upper bound for parameter variance / uncertainty (native units).
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
            default_min: Default lower bound for parameter variance / uncertainty.
            default_max: Default upper bound for parameter variance / uncertainty.
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

    def add_parameter(
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
        enable_variance: bool = True,
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
            default_min: Default lower bound for parameter variance / uncertainty.
            default_max: Default upper bound for parameter variance / uncertainty.
            unit: Display unit (e.g. ``"days"``, ``"per day"``).
            required: Whether this parameter is required in the config.
            options: Valid choices for ``ValueType.SELECT`` fields.
            enable_variance: Set ``False`` to hide the variance checkbox for this
                parameter in the UI (e.g. for integer count fields like num_runs).
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
                enable_variance=enable_variance,
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
        if len(self._demographic_groups) >= MAX_DEMOGRAPHIC_GROUPS:
            raise ValueError(
                f"Too many demographic groups: the maximum is {MAX_DEMOGRAPHIC_GROUPS}. "
                f"Attempted to add '{id}' but {MAX_DEMOGRAPHIC_GROUPS} groups are already "
                f"registered: {sorted(existing_ids)}"
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

        # If a disease parameter named "num_runs" was declared via
        # add_parameter(), use its default/min/max to populate the
        # artifact-level fields so ModelArtifact.num_runs stays accurate.
        num_runs_param = next(
            (p for p in self._disease_parameters if p.name == "num_runs"), None
        )
        if num_runs_param is not None:
            self._num_runs = int(num_runs_param.default or self._num_runs)
            if num_runs_param.min_value is not None:
                self._num_runs_min = int(num_runs_param.min_value)
            if num_runs_param.max_value is not None:
                self._num_runs_max = int(num_runs_param.max_value)

        return ModelParameterSchema(
            disease_type=self._disease_type,
            disease_label=self._disease_label or "",
            description=self._description or "",
            compartments=self._compartments,
            transmission_edges=self._transmission_edges,
            interventions=self._interventions,
            admin_zone_fields=self._admin_zone_fields,
            disease_parameters=self._disease_parameters,
            simulation_parameters=self._simulation_parameters,
            demographic_groups=self._demographic_groups,
            contact_matrix_overrides=self._contact_overrides,
            num_runs=self._num_runs,
            num_runs_min=self._num_runs_min,
            num_runs_max=self._num_runs_max,
            metadata=self._metadata,
        )
