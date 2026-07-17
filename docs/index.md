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
| [COVID SEIHDR](models/covid.md) | `covid_jax_model` | SEIHDR compartmental model with age-stratified transmission and spatial mobility |
| [COVID SEIHR](models/covid-seihr.md) | `covid_jax_model.variants` | An SEIHR compartmental model for novel respiratory diseases with age-stratified transmission |
| [COVID SEIDR](models/covid-seidr.md) | `covid_jax_model.variants` | An SEIDR compartmental model for novel respiratory diseases with age-stratified transmission |
| [COVID SEIR](models/covid-seir.md) | `covid_jax_model.variants` | An SEIR compartmental model for novel respiratory diseases with age-stratified transmission |
| [COVID SIHDR](models/covid-sihdr.md) | `covid_jax_model.variants` | An SIHDR compartmental model for novel respiratory diseases with age-stratified transmission |
| [COVID SIHR](models/covid-sihr.md) | `covid_jax_model.variants` | An SIHR compartmental model for novel respiratory diseases with age-stratified transmission |
| [COVID SIDR](models/covid-sidr.md) | `covid_jax_model.variants` | An SIDR compartmental model for novel respiratory diseases with age-stratified transmission |
| [COVID SIR](models/covid-sir.md) | `covid_jax_model.variants` | An SIR compartmental model for novel respiratory diseases with age-stratified transmission |
| [Dengue (4-Serotype)](models/dengue.md) | `dengue_jax_model` | A 4-serotype vector-borne dengue model with temperature-driven mosquito dynamics |
| [Dengue (2-Strain)](models/dengue-2strain.md) | `dengue_2strain_jax_model` | A two-strain dengue model with seasonal transmission, temporary cross-protective immunity, and antibody-dependent enhancement of secondary infections |
| [Ebola](models/ebola.md) | `ebola_jax_model` | Discrete-time stochastic SEIR model with hospitalisation and funeral transmission, Erlang(2) passage times via linear chain (Li et al. 2019, Getz & Dougherty 2018) |
| [Hantavirus (Rodent)](models/hantavirus.md) | `hantavirus_jax_model` | Spatial rodent-to-human hantavirus spillover (urban / rural-pop / rural-empty) following Cornejo-Donoso et al. 2023 |
| [Hantavirus (Human)](models/hantavirus-human.md) | `hantavirus_human_jax_model` | SEIR hantavirus model with person-to-person transmission and endogenous risk-perception state |
| [Mpox](models/mpox.md) | `mpox_jax_model` | A simple SIRS compartmental model for MPOX with spatial mobility |
