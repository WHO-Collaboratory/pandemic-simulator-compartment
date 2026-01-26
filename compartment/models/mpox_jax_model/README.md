# MPOX JAX Model

A simple SIR (Susceptible-Infected-Recovered) compartmental model for MPOX transmission.

## Model Structure

This is a basic SIR model with three compartments:
- **S**: Susceptible individuals
- **I**: Infected individuals  
- **R**: Recovered individuals

## Parameters

- `beta`: Transmission rate (S → I)
- `gamma`: Recovery rate (I → R)

## Features

- No demographics (single population group)
- No interventions
- No travel between regions
- Simple frequency-dependent transmission

## Usage

Run locally with:
```bash
python -m compartment.models.mpox_jax_model.main --mode local --config_file compartment/models/mpox_jax_model/example-config.json --output_file results/mpox_output.json
```

## Example Configuration

See `example-config.json` for a simple configuration with:
- 3 regions (New York, Los Angeles, Chicago)
- 90 day simulation
- Basic transmission parameters

