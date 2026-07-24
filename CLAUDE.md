# Pandemic Simulator — Compartmental Models

JAX-based compartmental disease modeling framework for the WHO Pandemic Hub's Pandemic Simulator. Built around a **declarative, schema-driven** model authoring API: authors describe compartments/edges/interventions in `define_parameters()`, and the framework derives Pydantic validation, the artifact JSON, the example config, the registry entry, automatic `*_total` cumulative compartments, and runtime rate-attribute wiring.

Python 3.13+, managed with `uv`. JAX (`odeint`) is the default ODE solver; stochastic models use a fixed-step Euler integrator.

## Reference docs — read these before doing the matching kind of work

- **Authoring or modifying a disease model** → [.claude/MODEL_AUTHORING_REFERENCE.md](.claude/MODEL_AUTHORING_REFERENCE.md). This is the authoritative playbook with patterns, pitfalls, and the closest-analog index. Always read it first before suggesting a model change.
- **User-facing model development guide** → [docs/DEVELOPING_MODELS.md](docs/DEVELOPING_MODELS.md). Same material as the authoring reference but framed for human contributors; use it when explaining concepts back to the user.
- **Config schema & required fields** → [README.md](README.md) covers `admin_unit_id`, `Disease`, `case_file`, intervention shapes, etc.

## Directory map

```
compartment/
├── model.py                 # Base Model class — schema wiring, _compute_equations, _apply_flow, _apply_interventions
├── parameters.py            # ParameterSchemaBuilder + ValueType enum
├── registry.py              # Auto-discovery of Model subclasses with DISEASE_TYPE
├── schema_generator.py      # Generates Pydantic config classes from the schema
├── simulation_manager.py    # odeint / Euler dispatch
├── simulation_postprocessor.py
├── driver.py                # drive_simulation() — main.py is a thin wrapper around this
├── generate_artifact.py     # CLI for artifact + example-config generation
├── validation/              # Pydantic validators; post_processor.py builds the runtime config dict
├── cloud_helpers/           # Pandemic Simulator web app integration (not for local users)
└── models/<disease>_model/          # (existing models use the legacy <disease>_jax_model)
    ├── model.py             # define_parameters() + equation()
    ├── main.py              # CLI wrapper
    ├── example-config.json
    ├── variants.py          # OPTIONAL — fixed-compartment variants
    └── artifacts/           # OPTIONAL — generated artifact JSON

tests/
├── test_smoke.py            # Auto-discovers every model dir with example-config.json
├── test_artifact.py
├── test_generate_artifact_model_dir.py
└── helpers.py

reference/                   # Example local-mode configs (users may add their own)
results/                     # Local simulation outputs (gitignored output area)
tools/                       # Local modeler utilities — view_results.py: parent_admin_total results viewer (with/without interventions, uncertainty bands)
```

## Common commands

```bash
# Environment
uv venv && uv sync && source .venv/bin/activate

# Run a model locally
python -m compartment.models.<name>.main \
    --mode local \
    --config_file compartment/models/<name>/example-config.json \
    --output_file results/<name>-test.json

# View a local results file (parent_admin_total; with vs. without interventions, uncertainty bands)
python tools/view_results.py results/<name>-test.json

# List models registered via auto-discovery
python -m compartment.generate_artifact --list

# Generate artifact + example config for a model
python -m compartment.generate_artifact <DISEASE_TYPE> \
    --output compartment/models/<name>/artifacts/<DISEASE_TYPE>.json \
    --example-config \
    --config-output compartment/models/<name>/example-config.json

# Generate one artifact per variant into a directory
python -m compartment.generate_artifact \
    --model-dir compartment/models/<name> \
    --output-dir compartment/models/<name>/artifacts

# Smoke-test a single model (integration tests are marked, run on demand)
python -m pytest tests/test_smoke.py -v -m integration -k "<name>"

# Full unit suite (fast)
python -m pytest tests/ -x -q

# All integration smoke tests
python -m pytest tests/test_smoke.py -m integration

# Quick syntax check on an edited file
python -m py_compile compartment/path/to/file.py
```

## Conventions

- **Model directory naming**: `compartment/models/<disease>_model/` (this is what `python -m compartment.new_model` generates). Existing models predating this convention keep the legacy `<disease>_jax_model` suffix and their `JaxModel` class names — leave those as-is. The suffix is not load-bearing: the registry discovers models by scanning every directory under `models/` and keying on `DISEASE_TYPE`, and the ECR/Lambda deploy path derives image tags/handlers from the directory name and the Lambda name from the model label, so neither depends on the `jax`/`_jax_model` string. Prefix with `test_` for in-repo example/test models that aren't intended for production use (e.g. `test_covid_sir_stochastic`, `test_klebsiella_amr_model`).
- **No manual registry edits.** The registry scans `compartment/models/*/model.py` and `variants.py` for `Model` subclasses with a `DISEASE_TYPE`. If a model isn't picked up, the cause is almost always an import error in `model.py` or a missing `DISEASE_TYPE`/`set_model_info()` call.
- **No manual `validation/__init__.py` edits.** Pydantic configs are generated from each model's schema; only legacy non-migrated disease types (`COVID_*`, `VECTOR_BORNE`, `VECTOR_BORNE_2STRAIN`) have hand-written config classes.
- **Compartment order is load-bearing.** `equation()` indexes the state array by position. Always stack with `jnp.stack([derivs[c] for c in self.compartment_list])`, never with a hardcoded order.
- **Prefer schema edges over manual flows.** Every compartment-to-compartment movement of the form `rate * source` or frequency-dependent FOI should be a `schema.add_transmission_edge(...)`. Reserve manual `_apply_flow()` calls for legitimate cases (multi-rate FOI, demographic births, density-dependent deaths, spatial coupling). See the authoring reference for the canonical examples.
- **`*_total` compartments are auto-generated** for every edge target. Don't declare them by hand unless the flow is manual or you're overriding `_add_total_compartments()` (dengue does).
- **`value_type=ValueType.DAYS`** means `default=10.0` represents a 10-day mean (auto-converted to a `0.1/day` rate at load). Don't pre-divide.
- **Local-mode config "short form"**: top-level `admin_zones` and `demographics` are auto-wrapped into `case_file` by `load_config_from_json`. Don't double-nest in hand-written configs.

## Workflow notes

- When asked to add or modify a model, read [.claude/MODEL_AUTHORING_REFERENCE.md](.claude/MODEL_AUTHORING_REFERENCE.md) first and pick the closest-analog model from its "Reference implementations by complexity" table before suggesting an approach from scratch.
- After any model change, the verification loop is: regenerate the artifact → regenerate or update `example-config.json` → run the model in `--mode local` → run `pytest tests/test_smoke.py -m integration -k "<name>"`. The smoke suite auto-discovers any directory with both `model.py` and `example-config.json`, so no test-registration step is needed.
- `run_simulation` runs each model **twice in parallel** (with and without interventions) and writes both into the output JSON. `UNCERTAINTY` run mode draws Latin Hypercube samples over edge `variance_params` and emits median + CI bands.
- Cloud mode (`--mode cloud`) is for the Pandemic Simulator web app and isn't supported for local use; `cloud_helpers/` is off-limits unless the user explicitly asks about it.

## What NOT to do

- Don't edit a manual model registry — there isn't one.
- Don't write a hand-crafted `BaseDiseaseConfig` subclass for a new migrated model — `schema_generator` produces it.
- Don't declare `*_total` compartments for typical models.
- Don't override `disease_type` as a property when `set_model_info()` already sets it.
- Don't suggest editing `compartment/validation/post_processor.py` to register a new disease type — its default dispatch already handles any registered model.
