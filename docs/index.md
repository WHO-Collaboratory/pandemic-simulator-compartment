# Pandemic Simulator - Compartmental Models

A library of compartmental disease models in the
[WHO Pandemic Simulator](https://uat.pandemicsimulator.com/).

## For Modelers

Describe a model once -- its compartments, the flows between them, and the
parameters it takes -- and the framework builds everything else from that
description:

- the file the web app reads to build its input forms and store model settings
- the checks that catch a bad or incomplete configuration before a run starts
- a ready-to-run example configuration, filled in with sensible defaults

You can use functions from any of the [existing models](#disease-models), or write your
own custom model -- see the
[Model Integration Documentation](guides/model-integration-documentation.md) to
get started.
- [**Parameter Framework Reference**](api/parameters.md) -- all available
  parameter types, the `ParameterSchemaBuilder` API, and `ValueType` options

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
