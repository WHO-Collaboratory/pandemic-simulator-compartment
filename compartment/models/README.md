# Pandemic Simulator Compartment Models

This directory contains compartmental models built using the pandemic-simulator-compartment framework. These models demonstrate the package's capabilities and provide starting points for further development.

## Current Models

| Model | Disease Type | Description |
|-------|--------------|-------------|
| `covid_jax_model/` | RESPIRATORY | Age-stratified SEIHDR model for respiratory diseases |
| `dengue_jax_model/` | VECTOR_BORNE | Multi-serotype dengue model with vector dynamics |

## Adding a New Model

To add a new model, create a new directory with the following structure:

```
compartment/models/your_model_name/
├── __init__.py
├── main.py             # Entry point with lambda_handler and CLI
├── model.py            # Model class implementing the Model interface
└── example-config.json # Sample configuration file
```

### Required: Implement the Model Interface

Your model class must inherit from `compartment.model.Model` and implement:

```python
from compartment.model import Model

class YourModel(Model):
    def __init__(self, config: dict):
        super().__init__(config)
        # Initialize model parameters from config
    
    @property
    def disease_type(self):
        # Return "RESPIRATORY" or "VECTOR_BORNE"
        return "RESPIRATORY"
    
    def prepare_initial_state(self):
        # Set up initial population matrices
        pass
    
    def get_params(self):
        # Return tuple of parameters for ODE solver
        pass
    
    def derivative(self, y, t, p):
        # Calculate dy/dt for each compartment
        pass
```

### Required: Create main.py Entry Point

```python
from compartment.driver import drive_simulation
from compartment.models.your_model_name.model import YourModel

def lambda_handler(event, context):
    drive_simulation(
        model_class=YourModel,
        args={"mode": "cloud", "simulation_job_id": event["simulation_job_id"]}
    )
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['local', 'cloud'], default='local')
    parser.add_argument('--config_file', help='Path to config JSON')
    parser.add_argument('--output_file', nargs='?', default=None)
    parser.add_argument('--simulation_job_id', nargs='?', default=None)
    args = parser.parse_args()
    
    drive_simulation(model_class=YourModel, args=vars(args))
```

### Running Your Model

```bash
# Local execution
python -m compartment.models.your_model_name.main \
    --mode local \
    --config_file compartment/models/your_model_name/example-config.json \
    --output_file results/output.json

# Docker execution
docker build --build-arg MODEL_DIR=compartment/models/your_model_name/ . -t your-model
docker run -v $(pwd)/reference:/opt/reference -v $(pwd)/results:/opt/results your-model
```

---

## Modifying Model Components

Models can be customized through configuration files without code changes for most use cases.

### 1. Modifying Compartments

Compartments define the disease states in the model (e.g., Susceptible, Exposed, Infected, Recovered).

**Configuration:**
Specify compartments via `Disease.compartment_list` in your config JSON:

```json
"Disease": {
    "disease_type": "RESPIRATORY",
    "compartment_list": ["S", "E", "I", "H", "D", "R"]
}
```

**Available Compartments (Respiratory):**

| Code | Meaning | Required |
|------|---------|----------|
| S | Susceptible | Yes |
| E | Exposed | No |
| I | Infected | Yes |
| H | Hospitalized | No |
| D | Deceased | No |
| R | Recovered | Yes |

**Flexibility:**
- Minimum: `["S", "I", "R"]` (basic SIR model)
- Maximum: `["S", "E", "I", "H", "D", "R"]` (full SEIHDR model)
- The model dynamically handles which compartments are present

**Adding New Compartment Types (requires code changes):**
1. Update the `derivative()` function in your model
2. Add mappings in `compartment/helpers.py` (`edge_to_variable`, `covid_compartment_grouping`)
3. Add validation schema in `compartment/validation/`

---

### 2. Modifying Parameters (Transmission Rates)

Parameters control the rates of flow between compartments.

**Configuration:**
Specify via `Disease.transmission_edges` in your config JSON:

```json
"transmission_edges": [
    {
        "source": "susceptible",
        "target": "exposed",
        "data": {
            "transmission_rate": 0.25,
            "variance_params": {
                "has_variance": true,
                "distribution_type": "UNIFORM",
                "min": 0.2,
                "max": 0.3
            }
        }
    },
    {
        "source": "exposed",
        "target": "infected",
        "data": { "transmission_rate": 0.2 }
    },
    {
        "source": "infected",
        "target": "recovered",
        "data": { "transmission_rate": 0.14 }
    }
]
```

**Parameter Mapping:**

| Edge (source -> target) | Variable | Description |
|------------------------|----------|-------------|
| susceptible -> infected | beta | Infection rate (SIR) |
| susceptible -> exposed | beta | Infection rate (SEIR) |
| exposed -> infected | theta | Incubation rate |
| infected -> recovered | gamma | Recovery rate |
| infected -> hospitalized | zeta | Hospitalization rate |
| infected -> deceased | delta | Direct mortality rate |
| hospitalized -> recovered | eta | Hospital recovery rate |
| hospitalized -> deceased | epsilon | Hospital mortality rate |

**Uncertainty Quantification:**
Add `variance_params` to any transmission edge to enable stochastic sampling:
- `distribution_type`: "UNIFORM" or "NORMAL"
- `min` / `max`: bounds for the distribution
- When `run_mode` is "UNCERTAINTY", the model uses Latin Hypercube Sampling across parameter ranges

---

### 3. Modifying Interventions

Interventions modify transmission dynamics during the simulation.

**Configuration:**
Specify via `interventions` array in your config JSON:

```json
"interventions": [
    {
        "id": "mask_wearing",
        "start_date": "2025-11-18",
        "end_date": "2025-12-18",
        "adherence_min": 20.0,
        "transmission_percentage": 35.0,
        "variance_params": [
            {
                "has_variance": true,
                "distribution_type": "UNIFORM",
                "field_name": "adherence_min",
                "min": 20.0,
                "max": 70.0
            }
        ]
    },
    {
        "id": "lock_down",
        "start_threshold": 5.0,
        "end_threshold": 1.0,
        "adherence_min": 80.0,
        "transmission_percentage": 90.0
    }
]
```

**Available Intervention Types:**

| Disease Type | Intervention ID | Effect |
|--------------|-----------------|--------|
| RESPIRATORY | `social_isolation` | Reduces transmission rate (beta) |
| RESPIRATORY | `vaccination` | Reduces transmission rate (beta) |
| RESPIRATORY | `mask_wearing` | Reduces transmission rate (beta) |
| RESPIRATORY | `lock_down` | Eliminates inter-zone travel |
| VECTOR_BORNE | `physical` | Reduces vector biting rate |
| VECTOR_BORNE | `chemical` | Reduces vector survival rate |

**Intervention Parameters:**

| Parameter | Description | Units |
|-----------|-------------|-------|
| `start_date` | Date to activate intervention | ISO format (YYYY-MM-DD) |
| `end_date` | Date to deactivate intervention | ISO format (YYYY-MM-DD) |
| `start_threshold` | Infected % to trigger activation | 0-100 |
| `end_threshold` | Infected % to trigger deactivation | 0-100 |
| `adherence_min` | Population compliance rate | 0-100 (%) |
| `transmission_percentage` | Transmission reduction when adhering | 0-100 (%) |

**Trigger Logic:**
- **Date-based:** Activates on `start_date`, deactivates on `end_date`
- **Threshold-based:** Activates when infected proportion exceeds `start_threshold`
- **Combined:** If both date and threshold are provided, activates on whichever condition is met first

**Effect Calculation:**
```
effective_rate = original_rate * (1 - adherence_min/100 * transmission_percentage/100)
```
Example: 20% adherence with 35% transmission reduction = 7% overall rate reduction

---

### 4. Modifying Population and Geography

**Admin Zones:**
Define geographic regions in `case_file.admin_zones`:

```json
"case_file": {
    "admin_zones": [
        {
            "name": "Region A",
            "center_lat": -18.93,
            "center_lon": 46.80,
            "population": 7982937,
            "infected_population": 0.05
        }
    ],
    "demographics": {
        "age_0_17": 25.0,
        "age_18_55": 50.0,
        "age_56_plus": 25.0
    }
}
```

**Travel/Mobility:**
Control inter-zone movement via `travel_volume`:

```json
"travel_volume": {
    "leaving": 20.0
}
```
- `leaving`: Percentage of population that travels between zones (0-100)
- Set to 0 for isolated zones with no mixing

---

## Quick Reference: Configuration vs Code Changes

| What to Modify | Configuration File | Code Changes Required |
|----------------|-------------------|----------------------|
| Compartments (existing types) | `Disease.compartment_list` | No |
| Compartments (new types) | - | Yes |
| Transmission rates | `Disease.transmission_edges` | No |
| Rate uncertainty | Add `variance_params` to edges | No |
| Interventions (existing types) | `interventions` array | No |
| Interventions (new types) | - | Yes |
| Population data | `case_file.admin_zones` | No |
| Demographics | `case_file.demographics` | No |
| Travel/mobility | `travel_volume` | No |
| Simulation dates | `start_date`, `end_date` | No |
| Run mode | `run_mode` ("DETERMINISTIC" or "UNCERTAINTY") | No |

---

## Validation

All configuration files are validated using Pydantic schemas located in `compartment/validation/`. To test your configuration locally:

```python
from compartment.validation import load_simulation_config
import json

with open("your-config.json") as f:
    config = {"data": {"getSimulationJob": json.load(f)}}

disease_type = config["data"]["getSimulationJob"]["Disease"]["disease_type"]
validated = load_simulation_config(config, disease_type)
print(f"Validated as {type(validated).__name__}")
```

Validation errors will provide field-level messages indicating what needs to be fixed.
