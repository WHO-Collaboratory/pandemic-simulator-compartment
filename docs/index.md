# Pandemic Simulator - Compartment Models

Compartmental disease models built with systems of ordinary differential
equations for the [WHO Pandemic Simulator](https://uat.pandemicsimulator.com/).

## Overview

This framework provides a declarative parameter system for defining
compartmental models (SIR, SEIR, vector-borne, etc.) with rich metadata.
From a single `ModelParameterSchema`, the framework generates:

- **Artifact JSON** for UI form population, database tables, and Zod schema generation
- **Pydantic validation models** for runtime config validation
- **Example config JSON** files with sensible defaults

## For Modelers

If you are building or configuring a disease model, start here:

- [**Parameter Framework Reference**](api/parameters.md) -- all available
  parameter types, the `ParameterSchemaBuilder` API, and `ValueType` options
- [**Developing Models Guide**](guides/developing-models.md) -- step-by-step
  walkthrough for creating a new model
- [**Disease Models**](#disease-models) -- reference for each implemented model

## API Reference

| Module | Description |
|--------|-------------|
| [Parameter Framework](api/parameters.md) | `ParameterSchemaBuilder`, `ValueType`, `ParameterDef`, and all schema dataclasses |
| [Base Model](api/model.md) | `Model` ABC -- the base class all disease models extend |
| [Interventions](api/interventions.md) | JAX-compatible intervention functions for modifying transmission rates |

## Disease Models

| Model | Module | Description |
|-------|--------|-------------|
| [Dengue](models/dengue.md) | `dengue_jax_model` | 4-serotype vector-borne model with temperature-driven dynamics |
| [COVID SEIHDR](models/covid.md) | `covid_jax_model` | SEIHDR compartmental model with age-stratified transmission and spatial mobility |
| [Mpox](models/mpox.md) | `mpox_jax_model` | Simple SIRS compartmental model with spatial mobility |
| [Hantavirus (Rodent)](models/hantavirus.md) | `hantavirus_jax_model` | Spatial rodent-to-human hantavirus spillover (urban / rural-pop / rural-empty) |
| [Hantavirus (Human)](models/hantavirus-human.md) | `hantavirus_human_jax_model` | SEIR hantavirus model with person-to-person transmission and risk-perception state |

## Quick Start

```bash
# Install with docs dependencies
uv sync --group docs

# Serve docs locally with live reload
uv run mkdocs serve

# Build static HTML
uv run mkdocs build
```
