# ABC JAX Model

A simple three-compartment epidemiological model with configurable transmission rates.

## Model Overview

The ABC model is a simplified compartmental model where:

- **A**: First compartment (analogous to Susceptible in SIR models)
- **B**: Second compartment (analogous to Infected in SIR models)
- **C**: Third compartment (analogous to Recovered in SIR models)

## Dynamics

The model implements the following transitions:

1. **A → B**: Individuals move from compartment A to B at rate `alpha` (influenced by contact with B individuals)
2. **B → C**: Individuals move from compartment B to C at rate `beta`

## Configuration

### Transmission Edges

The transmission rates between compartments are configured via `transmission_edges` in the Disease configuration:

```json
{
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
    }
}
```

### Parameters

- **alpha** (default: 0.3): Transmission rate from A to B
- **beta** (default: 0.1): Transition rate from B to C

### Supported Features

- ✅ Age-stratified populations
- ✅ Multiple administrative units
- ✅ Travel between regions
- ✅ Time-based interventions
- ✅ Proportion-based interventions
- ✅ Cumulative tracking (B_total compartment)

## Running the Model

### Local Execution

```bash
python -m compartment.models.abc_jax_model.main \
    --mode local \
    --config_file path/to/your-config.json \
    --output_file results/abc_output.json
```

### Cloud Execution

The model supports AWS Lambda execution via the `lambda_handler` function.

## Example Configuration

See [example-config.json](./example-config.json) for a complete configuration example with:
- Multiple administrative regions (Madagascar provinces)
- Age stratification
- Initial population setup
- Transmission edge definitions

## Model Equations

The differential equations governing the ABC model are:

```
dA/dt = -α * A * (B/N)
dB/dt = α * A * (B/N) - β * B
dC/dt = β * B
```

Where:
- `N` = Total population (A + B + C)
- `α` = alpha (transmission rate A→B)
- `β` = beta (transition rate B→C)
- Contact patterns are modulated by age-specific interaction matrices

## Customization

To modify transmission rates:

1. Update the `transmission_edges` in your configuration file
2. The first edge (a_compartment → b_compartment) sets `alpha`
3. The second edge (b_compartment → c_compartment) sets `beta`

Rates can also be modified dynamically through interventions.
