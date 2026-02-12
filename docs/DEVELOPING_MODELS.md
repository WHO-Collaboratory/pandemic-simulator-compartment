# Developing a new disease model

This guide walks you through adding a new compartmental model to the simulator. It is based on the current architecture and working examples in this repo.

## Prerequisites
- Python 3.10+
- jax and numpy familiarity (models use jax.numpy in the ODE `derivative()`)
- Basic Pydantic understanding (configs are validated before reaching your model)

## Key concepts and files
- Base model: [compartment/model.py](compartment/model.py)
- Simulation loop: [compartment/simulation_manager.py](compartment/simulation_manager.py)
- Output formatting: [compartment/simulation_postprocessor.py](compartment/simulation_postprocessor.py) and [compartment/helpers.py](compartment/helpers.py)
- Existing models to copy from:
  - Simple SIR (MPOX): [compartment/models/mpox_jax_model/model.py](compartment/models/mpox_jax_model/model.py)
  - Respiratory with travel and age stratification (COVID): [compartment/models/covid_jax_model/model.py](compartment/models/covid_jax_model/model.py)
  - Vector-borne dengue (4 serotypes): [compartment/models/dengue_jax_model/model.py](compartment/models/dengue_jax_model/model.py)

## Developing your model
- Create a directory under compartment/models/{my_disease_model}
- Within compartment/models/{my_disease_model} create the following:
  - `__init__.py`
  - `main.py`
  - `model.py`
  - `example-config.json`

### What your Model class must provide
- `disease_type` property
  - A short string the platform uses to route configs to your class. It must match `Disease.disease_type` in the validated config (e.g. `VECTOR_BORNE`, `RESPIRATORY`).
- Optional class attributes
  - `COMPARTMENT_LIST`: fixed ordered list of base compartments for your disease (e.g., ['S','E','I','H','D','R']). If omitted, the list is derived from the config (e.g., disease graph).
  - `COMPARTMENT_DELTA_GROUPING`: optional mapping used to aggregate “compartment deltas” for reporting, e.g. {"E": ["E1","E2","E3","E4"], ...}. If omitted, a safe default 1:1 mapping is generated automatically (excluding any cumulative columns ending with `_total`).
- `__init__(self, config)`
  - Accepts a validated config object (see [compartment/validation/post_processor.py](compartment/validation/post_processor.py)). It behaves like a dict for convenience.
  - Typically set:
    - `self.population_matrix` from `config["initial_population"]`
    - `self.compartment_list` (either your fixed list or from `config["compartment_list"]`)
    - `self.start_date`, `self.n_timesteps`, `self.admin_units`
    - Disease parameters (e.g., rates) and `self.intervention_dict`
    - `self.payload = config` (handy for post-processing)
- `@classmethod get_initial_population(cls, admin_zones, compartment_list, **kwargs)`
  - Returns a numpy array shaped `(n_zones, n_compartments)` with the initial distribution. You can rely on `admin_zones` entries having `population`, and optionally `infected_population`, `seroprevalence`, etc.
  - The base implementation in [compartment/model.py](compartment/model.py) does a simple S/I split. Override as needed (see dengue examples).
- `prepare_initial_state(self)` → `(population_matrix, compartment_list)`
  - Return the state array used by the ODE solver. Supported shapes:
    - 2D: `(C, R)` — compartments by regions
    - 3D: `(C, A, R)` — compartments by age by regions
  - If you add cumulative/internal tracking compartments, append them here and name them with a `_total` suffix. Example: `'E_total'`, `'I_total'`.
  - Keep `self.compartment_list` in the same order as the stacked state you return.
- `get_params(self)`
  - Return the tuple/array of parameter values your `derivative()` expects.
- `derivative(self, y, t, p)`
  - Implement your ODE right-hand side using `jax.numpy` operations only.
  - `y` has the same shape you returned in `prepare_initial_state()`; return a `np.stack([...])` with derivatives in the same order as `self.compartment_list`.
  - If you maintain cumulative `_total` compartments, also return their derivatives (e.g., flows into E, I, etc.).

### Shapes and conventions
- Regions-first vs compartments-first: JAX ODEs allow any array shape. The project convention is returning `(C, R)` or `(C, A, R)` from `prepare_initial_state()` so `derivative()` can index with the compartment order.
- Age stratification: Respiratory models often use `(C, A, R)`. See `prepare_covid_initial_state()` in [compartment/helpers.py](compartment/helpers.py).
- Cumulative columns: If you track cumulative counts, always suffix the compartment name with `_total`. These are used internally for accuracy and are automatically excluded from GraphQL outputs and parent totals.

### Register your model
- Map your new `disease_type` to your class in [compartment/validation/post_processor.py](compartment/validation/post_processor.py) within `MODEL_REGISTRY`.
- If your model has a fixed structure, set `COMPARTMENT_LIST` on the class; otherwise ensure your config provides enough information to derive it (e.g., `disease_nodes`).

### Compartment deltas and reporting
- The system computes “compartment deltas” at the end of a run using [get_compartment_delta_grouping() and compute_jax_compartment_deltas()](compartment/helpers.py#L1-L500).
- Provide `COMPARTMENT_DELTA_GROUPING` on your class if you want grouped outputs (see dengue). If omitted, the default uses each base compartment as its own group and ignores any `_total` columns.
- If a cumulative column like `E_total` exists, it will be used internally to compute the delta for group `E`, but the output key remains `E` (no `_total` suffix appears in API results).

Configuration tips (example-config.json)
- Put a minimal working example in your model folder, e.g. `models/your_model/example-config.json` containing:
  - `Disease` with `disease_type` matching your class, and either `disease_nodes` or `compartment_list` (unless your class defines `COMPARTMENT_LIST`).
  - `time_steps`, `start_date` (YYYY-MM-DD), `run_mode` (DETERMINISTIC or UNCERTAINTY)
  - `case_file` with `admin_zones` (population, lat/lon, etc.) and optional `demographics`
  - Optional: `interventions`, `travel_volume`, and `transmission_edges`
- After validation, the engine provides these computed fields to your `__init__` via `config`:
  - `compartment_list`, `initial_population`, `transmission_dict`, `admin_units`, `intervention_dict`, `travel_matrix`, `hemisphere`, `temperature`

## How to run locally
- Testing can be done by running a command like this from the repo root:

```
python3 -m compartment.models.dengue_jax_model.main \
  --mode local \
  --config_file compartment/models/dengue_jax_model/example-config.json \
  --output_file results/test_dengue.json
```

### Interventions and travel
- Interventions: Set and use `self.intervention_dict` in your model; built-in helper hooks handle date- and threshold-based activations during the run.
- Travel: Use the provided `travel_matrix` (already gravity-modeled). If you expect an identity (no travel) for 1 region or missing rates, the validator will supply it.

## Quality checklist before opening a PR
- `prepare_initial_state()` returns the correct shape and appends any `_total` columns you need.
- `derivative()` stacks outputs in the exact order of `self.compartment_list` (including any `_total`).
- `disease_type` string matches your example config and is registered in `MODEL_REGISTRY`.
- If you added `_total` compartments, confirm they do not appear in the API outputs (they are filtered by the formatters).
- If you need aggregated reporting, define `COMPARTMENT_DELTA_GROUPING`. Otherwise rely on the default 1:1 mapping.
- Add a small `example-config.json` in your model folder and verify a local run completes without errors.

## Where to look when in doubt
- Minimal: MPOX model for the simplest SIR example.
- Age-stratified respiratory: COVID model.
- Complex vector-borne with groups and cumulative tracking: Dengue models.

## Support
- If you hit issues with validation or output formats, check the formatters and helpers referenced above, or mirror how existing models supply shapes and `_total` columns.
