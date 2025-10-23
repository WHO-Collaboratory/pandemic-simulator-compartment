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

## Usage

This code represents the current implementation during the migration process and serves as the foundation for the new epidemiological modeling framework being developed.