# Pandemic Simulator — Compartmental Disease Modeling Framework

A Python framework for building and running compartmental disease models, developed for the World Health Organization Pandemic Hub's [Pandemic Simulator](https://uat.pandemicsimulator.com/).

Its purpose is to provide accessible, research-based modeling tools that help decision makers, epidemiologists, and modelers assess public health intervention strategies. Models written here run locally on your own machine, and once integrated into the Pandemic Simulator, any user can run those models via its interface.

## Documentation

Begin with the **[Model Integration guide](docs/guides/model-integration-documentation.md)** — it covers the full workflow, from environment setup through a local run to pull request submission. The other guides each go deeper on a single topic:

| Guide | Read it when |
| :---- | :---- |
| [Model Integration](docs/guides/model-integration-documentation.md) | **Start here.** Setup, writing a model, config format, running, testing, submitting. |
| [Model Conversion](docs/guides/model-conversion.md) | You are porting a model from a published paper or code. |
| [Interventions](docs/guides/interventions.md) | Your model offers a control measure — masks, distancing, vaccination, a lockdown, etc. |
| [Uncertainty Quantification](docs/guides/uncertainty-quantification.md) | Parameters are known only within a range, or your model is stochastic. |
| [Contact Matrices](docs/guides/contact-matrices.md) | Your model is age-structured and needs realistic mixing between age groups. |
| [Mobility](docs/guides/mobility.md) | People move between administrative zones. |
| [Adding Your Own Data](docs/guides/adding-datasets.md) | Your model reads a data file you supply. |

The published site also carries the **[API reference](https://who-collaboratory.github.io/pandemic-simulator-compartment/)** — generated from source docstrings, so it is the place to look up an exact signature while writing. Each model in the repository gets a generated page there too.

## Repository layout

```
compartment/            # The Python package
├── model.py            # Base Model class every model extends
├── parameters.py       # ParameterSchemaBuilder — the schema API you write against
├── registry.py         # Automatic model discovery
├── schema_generator.py # Builds config validation from your schema
├── driver.py           # Runs a simulation
├── models/             # Every disease model — this is where you work
├── validation/         # Config validation
├── contact_matrices/   # Age-mixing data
├── datasets/           # Modeler-supplied data files
└── cloud_helpers/      # Pandemic Simulator integration; ignore for local work
docs/                   # Documentation source (guides + API reference)
tests/                  # Test suite; smoke tests discover models automatically
reference/              # Example configs for local runs — add your own
results/                # Local simulation output (gitignored)
tools/                  # Local utilities, including the results viewer
```

A model directory holds `model.py` (the model), `main.py` (starts a run), `example-config.json` (example inputs), and `model.md` (the write-up users see with their results). Optionally it also holds `datasets.yaml` for a data file and `artifacts/` for the generated file the web app reads.


### Example models

Three example models exist in the repository. The Model Integration guide follows all three.

| Directory | What it demonstrates |
| :---- | :---- |
| `example_parameter_uncertainty_declarative_model` | The default path — the framework generates the equations from declared transmission edges. Adds age demographics, an intervention, and parameter uncertainty. |
| `example_parameter_uncertainty_custom_model` | The same SIR model with the equations written by hand, plus an intervention that ramps up and down instead of switching on and off. |
| `example_stochastic_model` | A stochastic SIR model (tau-leaping, Euler integration, many trajectories) with split asymptomatic and symptomatic compartments. |

Alongside the examples are models for multiple diseases. To list every registered model:

```bash
python -m compartment.generate_artifact --list
```

## What the framework supports

- Compartmental models only — dividing a population into compartments and defining the rates that move people between them. Agent-based, network-based, and individual-level models do not fit.
- Three run modes: a single deterministic run, parameter uncertainty by Latin Hypercube Sampling (uniform distributions), and stochastic models run as many trajectories. The mode is inferred from your configuration rather than set directly.
- Geography as administrative zones with populations and coordinates, plus age structure, contact matrices, and movement between zones.
- Interventions with two built-in levers: scaling transmission rates and stopping travel between zones. Anything beyond those involves writing a custom intervention.
- Simulation forward from parameter values you supply. The framework does not fit or calibrate parameters to data.
- Every simulation runs twice, once with interventions and once without, so there is always a control to compare against.

## Contributing

Contributions are welcome, both to the core framework and as new models. The [Model Integration guide](docs/guides/model-integration-documentation.md) covers the full workflow, including what a submission needs and how models are reviewed.

Models submitted here are reviewed by community subject matter experts. Public release is subject to approval and cannot be guaranteed. A submitted model must include an example config, which is what the smoke tests run.

## License

Apache License 2.0. Copyright 2025-2026 World Health Organization. See [LICENSE](LICENSE).
