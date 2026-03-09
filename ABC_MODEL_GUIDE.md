# ABC Model - Usage Guide

## Overview

I've created a complete ABC compartmental model for your pandemic simulator platform. The model is an SIR-like structure but uses the letters A, B, and C instead, with configurable transmission rates between compartments.

## What Was Created

### 1. Model Files
- **`compartment/models/abc_jax_model/model.py`** - Main model class (`ABCJaxModel`)
- **`compartment/models/abc_jax_model/main.py`** - Entry point for running simulations
- **`compartment/models/abc_jax_model/example-config.json`** - Example configuration file
- **`compartment/models/abc_jax_model/README.md`** - Detailed model documentation
- **`compartment/models/abc_jax_model/__init__.py`** - Package initialization

### 2. Validation Files
- **`compartment/validation/diseases/abc.py`** - Pydantic validation schemas
- Updated **`compartment/validation/diseases/__init__.py`** to register ABC model

### 3. Integration
- Updated **`compartment/helpers.py`** with ABC edge-to-variable mappings

### 4. Test Files
- **`test_abc_model.py`** - Verification script (all tests passing ✓)

## Model Structure

### Compartments
- **A**: First compartment (analogous to Susceptible)
- **B**: Second compartment (analogous to Infected)
- **C**: Third compartment (analogous to Recovered)

### Transmission Edges

You can configure the rates between compartments using transmission edges:

| Edge | Parameter | Description |
|------|-----------|-------------|
| `a_compartment → b_compartment` | `alpha` | Rate of transition from A to B (influenced by contact with B individuals) |
| `b_compartment → c_compartment` | `beta` | Rate of transition from B to C |

## Running the Model

### Quick Start

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run with the example configuration
python -m compartment.models.abc_jax_model.main \
    --mode local \
    --config_file compartment/models/abc_jax_model/example-config.json \
    --output_file results/abc_output.json
```

### Configuration

Create a JSON configuration file with your parameters:

```json
{
    "admin_unit_0_id": "YOUR_REGION",
    "start_date": "2025-11-01",
    "end_date": "2025-12-31",
    "Disease": {
        "disease_type": "ABC",
        "compartment_list": ["A", "B", "C"],
        "transmission_edges": [
            {
                "source": "a_compartment",
                "target": "b_compartment",
                "data": {
                    "transmission_rate": 0.3
                }
            },
            {
                "source": "b_compartment",
                "target": "c_compartment",
                "data": {
                    "transmission_rate": 0.1
                }
            }
        ]
    },
    "case_file": {
        "admin_zones": [
            {
                "name": "Region1",
                "population": 1000000,
                "infected_population": 0.05
            }
        ]
    }
}
```

## Key Features

✅ **Configurable Transmission Rates** - Adjust `alpha` and `beta` via transmission edges  
✅ **Age Stratification** - Supports multiple age groups with interaction matrices  
✅ **Multi-Region Support** - Simulate across multiple administrative units  
✅ **Travel Dynamics** - Population movement between regions  
✅ **Interventions** - Time-based and proportion-based interventions  
✅ **Cumulative Tracking** - Automatic tracking of cumulative B compartment  

## Model Equations

```
dA/dt = -α * A * (B/N)
dB/dt = α * A * (B/N) - β * B
dC/dt = β * B
```

Where:
- `N` = Total population (A + B + C)
- `α` = alpha (transmission rate A→B)
- `β` = beta (transition rate B→C)

## Customization

### Changing Transmission Rates

Edit the transmission edges in your config file:

```json
"transmission_edges": [
    {
        "source": "a_compartment",
        "target": "b_compartment",
        "data": {
            "transmission_rate": 0.5  // Increase A→B transition
        }
    },
    {
        "source": "b_compartment",
        "target": "c_compartment",
        "data": {
            "transmission_rate": 0.2  // Increase B→C transition
        }
    }
]
```

### Adding Variance

You can add uncertainty to transmission rates:

```json
"data": {
    "transmission_rate": 0.3,
    "variance_params": {
        "has_variance": true,
        "distribution_type": "UNIFORM",
        "min": 0.2,
        "max": 0.4
    }
}
```

## Next Steps

1. **Test the model**: Run with the example config to verify it works
2. **Customize parameters**: Adjust transmission rates to match your scenario
3. **Add your data**: Replace admin zones with your actual regions
4. **Add interventions**: Configure lockdowns, isolation, etc. in the config
5. **Run simulations**: Execute and analyze results

## Support

- Full model documentation: `compartment/models/abc_jax_model/README.md`
- Example config: `compartment/models/abc_jax_model/example-config.json`
- Validation schema: `compartment/validation/diseases/abc.py`

## Verification

Run the test script to verify everything is working:

```bash
source .venv/bin/activate
python test_abc_model.py
```

Expected output:
```
Testing ABC Model...

✓ ABC model imported successfully
✓ ABC validation schema works correctly
  - Compartments: ['A', 'B', 'C']
  - Transmission edges: 2
✓ Example config is valid JSON
  - Admin units: 6
  - Compartments: ['A', 'B', 'C']

==================================================
Tests passed: 3/3
==================================================
```

---

**Ready to use!** The ABC model is fully integrated into your pandemic simulator platform.
