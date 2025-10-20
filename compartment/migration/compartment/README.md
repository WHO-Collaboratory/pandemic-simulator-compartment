# Compartment Models

This directory contains the core compartment model implementations and simulation management components for epidemiological modeling.

## Purpose

This directory houses the main simulation infrastructure including:
- **Compartment Model Implementations** - Epidemiological models for ODEs implemented in JAX. Currently we operate distinct models for respiratory and vector-borne diseases with some common features such as interventions and travel.
- **Simulation Management** - Simulations can be run in batches, allowing for runs with uncertainty and parameter values across a range.
- **Intervention Systems** - Simulations support interventions which can be triggered by date or based on infections crossing thresholds. Interventions are unique for different diseases, have different effects, and can have variable adherence.
- **Helper Utilities** - Configuration loading, parameter generation, and data processing

## Key Components

### Model Implementations
- **`covid_jax_model.py`** - COVID-19 compartmental model with dynamic travel and intervention mechanisms
- **`dengue_jax_model.py`** - Dengue fever model incorporating temperature seasonality and vector dynamics

### Simulation Infrastructure
- **`main.py`** - Primary entry point for batch simulation execution
- **`simulation_manager.py`** - Core simulation execution engine using JAX ODE solvers
- **`batch_simulation_manager.py`** - Batch processing coordination and parallel execution
- **`simulation_postprocessor.py`** - Post-simulation data processing and output formatting

### Supporting Systems
- **`interventions.py`** - JAX-compatible intervention mechanisms (proportional and timestep-based)
- **`helpers.py`** - Comprehensive utility functions for configuration, parameter generation, and data processing
- **`temperature.py`** - Temperature-dependent modeling components for vector-borne diseases

### Infrastructure
- **`Dockerfile`** - Container configuration for cloud deployment. This is configured for the monorepo, and not currently supported in this repository.
- **`requirements.txt`** - Python dependencies for the simulation environment

## Architecture

The compartment models use [JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html) for high-performance numerical computing, enabling:
- Fast parallel simulation execution
- Automatic differentiation for parameter optimization
- GPU acceleration support
- Efficient batch processing of multiple parameter sets

## Usage

This code represents the current implementation during the migration process and serves as the foundation for the new epidemiological modeling framework being developed.