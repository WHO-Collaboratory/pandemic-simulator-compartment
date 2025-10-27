# Compartmental Models

This directory contains the core compartmental model implementations and simulation management components for epidemiological modeling.

## Purpose

This directory houses the main simulation infrastructure including:
- **Compartment Model Implementations** - Epidemiological models for ordinary differential equations (ODEs) implemented in JAX. Currently we operate distinct models for respiratory and vector-borne diseases with some common features such as interventions and travel.
- **Simulation Management** - Simulations can be run in batches, allowing for runs with uncertainty and a range of parameter values.
- **Intervention Systems** - Interventions can be implemented or removed in two scenarios: at a specific timestep or when the proportion of the infected population reaches a user-defined threshold. Interventions are unique for respiratory and vector-borne diseases and can have variable adherence.
- **Helper Utilities** - Configuration loading, parameter generation, and data processing

## Key Components

### Model Implementations
- **`covid_jax_model.py`** - Contains functions specific to the respiratory compartmental model including travel and intervention mechanisms
- **`dengue_jax_model.py`** - Contains functions specific to the vector-borne compartmental model and incorporates temperature and vector dynamics via thermal response curves.

### Simulation Infrastructure
- **`main.py`** - Primary entry point for batch simulation execution
- **`simulation_manager.py`** - Core simulation execution engine using JAX ODE solvers, houses run_simulation, responsible for running a single simulation of any disease
- **`batch_simulation_manager.py`** - Batch processing coordination and parallel execution, wrapper of simulation_manager, invoked when multi-run is called, creates n simulation objects
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

## Running Locally

To run simulations locally using the `local-run.py` script:

### Prerequisites

- Python 3.8 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create and activate virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
   **On Windows**:
   ```bash
   .venv\Scripts\activate
   ```
### Running Simulations

The `local-run.py` script accepts a configuration file and optionally an output file:

```bash
# Run with default output filename (includes timestamp)
python local-run.py config.json

# Run with custom output filename
python local-run.py config.json my-results.json
```

#### Output

- Results are saved as JSON files containing both simulation metadata and results
- Default output format: `{config-name}-results-{timestamp}.json`
- Example: `pansim-config-results-20251023-133124.json`

#### Configuration File

The script expects a JSON configuration file with the structure used by the simulation system. See the existing `pansim-config.json` for reference.

### Example Usage

```bash
# Basic run with timestamped output
python local-run.py pansim-config.json

# Custom output file
python local-run.py pansim-config.json my-simulation-results.json
```