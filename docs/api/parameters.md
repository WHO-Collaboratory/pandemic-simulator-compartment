# Parameter Framework Reference

This is the primary reference for modelers. The parameter framework lets you
declaratively define all configurable values in a disease model -- compartments,
transmission edges, interventions, demographic groups, and more -- using the
`ParameterSchemaBuilder` API.

---

## Builder API

The `ParameterSchemaBuilder` is the main interface model authors use inside
`define_parameters()`. It provides type-safe `add_*` / `set_*` methods for
declaring every aspect of a model's parameter schema.

::: compartment.parameters.ParameterSchemaBuilder
    options:
      show_root_heading: true
      members_order: source

---

## Value Types

`ValueType` controls how parameter values are rendered in the UI and how
the framework converts them at load time (e.g. `DAYS` → `1/days` rate).

::: compartment.parameters.ValueType
    options:
      show_root_heading: true
      members: true

---

## Core Dataclasses

### ParameterDef

The atomic unit of parameter metadata. Every configurable value in a disease
model has a corresponding `ParameterDef`.

::: compartment.parameters.ParameterDef
    options:
      show_root_heading: true
      members_order: source

### CompartmentDef

Describes a single compartment in the model (e.g. S, I, R).

::: compartment.parameters.CompartmentDef
    options:
      show_root_heading: true
      members_order: source

### TransmissionEdgeDef

A directed edge in the compartment graph with its associated rate parameter.

::: compartment.parameters.TransmissionEdgeDef
    options:
      show_root_heading: true
      members_order: source

### InterventionDef

Defines a supported intervention type and its configurable parameters.

::: compartment.parameters.InterventionDef
    options:
      show_root_heading: true
      members_order: source

### DemographicGroupDef

Defines a demographic group (e.g. an age band) for age-stratified models.

::: compartment.parameters.DemographicGroupDef
    options:
      show_root_heading: true
      members_order: source

### ContactOverrideDef

A single entry in the contact matrix for cross-demographic exposure.

::: compartment.parameters.ContactOverrideDef
    options:
      show_root_heading: true
      members_order: source

---

## Schema and Registries

### ModelParameterSchema

The complete parameter schema -- the single source of truth for a model.

::: compartment.parameters.ModelParameterSchema
    options:
      show_root_heading: true
      members_order: source

### CompartmentRegistry

Attribute-style access to compartment IDs (set as `cls.COMPARTMENTS` on
model classes).

::: compartment.parameters.CompartmentRegistry
    options:
      show_root_heading: true
      members_order: source

### DiseaseParamRegistry

Attribute-style access to disease parameter names (set as
`cls.DISEASE_PARAMS`).

::: compartment.parameters.DiseaseParamRegistry
    options:
      show_root_heading: true
      members_order: source

### AdminZoneFieldRegistry

Attribute-style access to admin zone field names (set as
`cls.ADMIN_ZONE_FIELDS`).

::: compartment.parameters.AdminZoneFieldRegistry
    options:
      show_root_heading: true
      members_order: source

### DiseaseParamValues

Attribute-style access to disease parameter *values* from runtime config.

::: compartment.parameters.DiseaseParamValues
    options:
      show_root_heading: true
      members_order: source
