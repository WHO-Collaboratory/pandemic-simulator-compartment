# Developing a new disease model

This guide walks you through adding a new compartmental model to the simulator. The framework is **declarative**: you describe your model's compartments, transmission edges, interventions, and parameters once in `define_parameters()`, and the framework derives almost everything else (the `COMPARTMENT_LIST`, the `disease_type`, the Pydantic validation schema, the model artifact JSON, an example config, the runtime registry entry, automatic `_total` cumulative compartments, and rate-attribute wiring on `__init__`).

You only need to write code for the parts that are genuinely model-specific: the ODE right-hand side and any non-trivial initial-state preparation.

## Prerequisites
- Python 3.10+ (managed via `uv`)
- Familiarity with `jax.numpy` for the ODE `derivative()`
- Light Pydantic exposure (you rarely touch validators directly — they're generated from your schema)

## Where things live

| Concern | File | Notes |
|---|---|---|
| Base `Model` class | [compartment/model.py](../compartment/model.py) | Auto-wires the schema, contact matrix, interventions, helpers like `_compute_derivatives()` |
| Schema builder + types | [compartment/parameters.py](../compartment/parameters.py) | `ParameterSchemaBuilder`, `ValueType`, `CompartmentDef`, `TransmissionEdgeDef`, etc. |
| Auto model registry | [compartment/registry.py](../compartment/registry.py) | Discovers every `Model` subclass with a `DISEASE_TYPE` — no manual mapping |
| Pydantic config generator | [compartment/schema_generator.py](../compartment/schema_generator.py) | Builds the `BaseDiseaseConfig` subclass from your schema |
| Validation post-processor | [compartment/validation/post_processor.py](../compartment/validation/post_processor.py) | Computes `compartment_list`, `initial_population`, `transmission_dict`, etc. |
| Simulation loop | [compartment/simulation_manager.py](../compartment/simulation_manager.py) | Defaults to JAX `odeint`; `STOCHASTIC = True` (or `SOLVER = "euler"`) switches to fixed-step Euler |
| Output formatter | [compartment/simulation_postprocessor.py](../compartment/simulation_postprocessor.py) | Aggregates age groups, applies `COMPARTMENT_DELTA_GROUPING`, emits `_total` deltas |
| Generic driver / CLI plumbing | [compartment/driver.py](../compartment/driver.py) | `drive_simulation(model_class, args)` — your `main.py` is just a thin wrapper |
| Artifact generator CLI | [compartment/generate_artifact.py](../compartment/generate_artifact.py) | `python -m compartment.generate_artifact <DISEASE_TYPE> [--example-config]` |

Existing models to copy from:
- Minimal SIR with mobility: [compartment/models/mpox_jax_model/model.py](../compartment/models/mpox_jax_model/model.py)
- Age-stratified SEIHDR with variants: [compartment/models/covid_jax_model/model.py](../compartment/models/covid_jax_model/model.py) and [variants.py](../compartment/models/covid_jax_model/variants.py)
- 4-serotype vector-borne with cumulative groupings: [compartment/models/dengue_jax_model/model.py](../compartment/models/dengue_jax_model/model.py)
- Stochastic SIR (tau-leaping): [compartment/models/test_covid_sir_stochastic/model.py](../compartment/models/test_covid_sir_stochastic/model.py)
- Multi-dimensional AMR model: [compartment/models/test_klebsiella_amr_model/model.py](../compartment/models/test_klebsiella_amr_model/model.py)

## File layout

Create a new directory under `compartment/models/<your_model>/`:

```
compartment/models/your_model/
├── __init__.py            # empty
├── model.py               # Model subclass with define_parameters() + derivative()
├── main.py                # CLI entry point — thin wrapper around drive_simulation()
├── example-config.json    # minimal runnable config (can be auto-generated)
├── variants.py            # OPTIONAL — fixed-compartment variants of the same model
└── artifacts/             # OPTIONAL — generated artifact JSON checked into the repo
```

The auto-discovery registry scans every directory under `compartment/models/`, importing `model.py` and (if present) `variants.py`. Any class that subclasses `Model` and exposes a `DISEASE_TYPE` is registered automatically. **There is no manual registry edit.**

## Writing `model.py`

A migrated model has two responsibilities:

1. **`define_parameters(cls, schema)`** — declare the model.
2. **`derivative(self, y, t, p)`** — implement the ODE right-hand side.

The base class handles `disease_type`, `COMPARTMENT_LIST`, `COMPARTMENTS` (attribute-style compartment registry), `get_params()`, transmission rate attributes, intervention runtime objects, contact matrices, and demographic rate vectors.

### `define_parameters()`

Called once when the class is defined. Use the builder API (full reference in [parameters.py](../compartment/parameters.py)):

```python
from compartment.model import Model
from compartment.parameters import ValueType


class MyModel(Model):

    @classmethod
    def define_parameters(cls, schema):
        # 1. Identity (required, exactly once)
        schema.set_model_info(
            disease_type="MY_DISEASE",         # machine key — used everywhere
            label="My Disease",                # UI / artifact name
            description="A short SIR model for my disease",
        )

        # 2. Compartments — id is the short key used in arrays
        #    Mark `infective=True` on compartments that contribute to FOI.
        schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
        schema.add_compartment("I", "Infected", "Currently infectious population", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered and immune")

        # 3. Transmission edges — variable_name becomes self.<name> on instances.
        #    Defaults / mins / maxes are in NATIVE units (RATE / DAYS / PERCENTAGE).
        #    `frequency_dependent=True` makes the framework compute
        #       flow = source * rate * sum(infective) / N_total
        #    instead of the default mass-action form.
        #
        #    Rule of thumb: declare an edge for every compartment-to-compartment
        #    movement whose flow has the form `rate * source` (mass action) or
        #    `source * rate * sum(infective) / N` (frequency-dependent). The
        #    framework then handles intervention scaling, uncertainty sampling,
        #    and _total accumulation for free. Flows that do NOT fit that shape
        #    (multi-rate FOI, demographic births, density-dependent deaths) are
        #    handled manually — see "Flows without an edge" below.
        schema.add_transmission_edge(
            source="susceptible", target="infected",
            variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (S->I)",
            description="Infection rate",
            default=0.3, default_min=0.1, default_max=0.5,
            min_value=0.01, max_value=2.0,
            unit="per day",
        )
        schema.add_transmission_edge(
            source="infected", target="recovered",
            variable_name="gamma",
            label="Recovery Period (I->R)",
            description="Days to recover",
            default=10.0, min_value=1.0, max_value=100.0,
            value_type=ValueType.DAYS,         # framework converts 1/days → rate at load time
            unit="days",
        )

        # 4. (optional) Interventions — target_rates lists the edges they reduce.
        schema.add_intervention(
            id="social_isolation",
            label="Social Isolation",
            description="Reduces transmission while active",
            target_rates=["beta"],
            adherence=40.0, transmission_reduction=50.0,
        )

        # 5. (optional) Travel — declares the case-file `travel_volume` block
        # See GRAVITY_MODEL.md for comprehensive documentation on spatial mobility
        schema.set_travel_volume(leaving_default=0.2)

        # 6. (optional) Demographic groups + contact matrix — see CovidJaxModel.
        # Declare each group with an inclusive (low, high) age range to opt in
        # to country-aware contact matrices (Prem 2021 synthetic matrices,
        # 177 countries, 5-year bands, aggregated to your declared bands).
        # The country is read from the run's top-level `admin_unit_id`
        # (e.g. "MDG.1.3_1" -> "MDG"); unknown countries fall back to a
        # global average across all 177 source matrices.
        # schema.add_demographic_group("age_0_17",  "Children", default_weight=33.3, age_range=(0, 17))
        # schema.add_demographic_group("age_18_55", "Adults",   default_weight=44.4, age_range=(18, 55))
        #
        # `set_contact_override` still works and takes precedence over the
        # Prem default — useful when a model has bespoke (e.g. POLYMOD-derived)
        # values that should not be replaced by country aggregation:
        # schema.set_contact_override("age_0_17", "age_18_55", 5.18)

        # 7. (optional) Bespoke admin-zone fields — e.g. seroprevalence
        # schema.add_admin_zone_field(name="seroprevalence", ...)

        # 8. (optional) Disease-specific top-level params — e.g. immunity_period
        # schema.add_disease_parameter(name="immunity_period", ...)
```

What the base class does for you after `define_parameters()`:
- `cls.DISEASE_TYPE` — set from `schema.set_model_info()` if not already declared.
- `cls.COMPARTMENT_LIST` — a plain ordered `list[str]` of compartment IDs.
- `cls.COMPARTMENTS` — a `CompartmentRegistry` exposing attribute-style access (`cls.COMPARTMENTS.S` → `"S"`) and an `infective_ids` property.
- **Automatic `_total` compartments**: for every transmission edge target, the framework appends a cumulative `<target>_total` compartment to the schema (e.g. `I_total`, `H_total`). You do **not** need to declare these. They are filtered out of API outputs and used internally to compute compartment deltas. To opt out, override `_add_total_compartments(cls, schema)` as a no-op (see Dengue, which declares its own aggregate totals).

### `__init__(self, config)`

For typical models, **call `super().__init__(config)` first**. It handles:
- `self.population_matrix` (jax array of shape `(K, R)` — compartments × regions)
- `self.compartment_list`, `self.start_date`, `self.start_date_ordinal`, `self.n_timesteps`, `self.admin_units`, `self.payload`
- Transmission rate attributes (`self.beta`, `self.gamma`, …) loaded from `config["transmission_dict"]` and converted to per-day rates based on each edge's `value_type`
- `self.intervention_dict` and `self.interventions` (list of runtime `Intervention` objects)
- `self.intervention_statuses` initialized to `{intv_id: False, ...}`
- `self.contact_matrix` (built from declared demographic groups + config overrides)
- `self._rate_vectors` (per-demographic absolute rate vectors when `demographic_rate_overrides` is present)

Then set whatever else your model needs (travel matrix, demographics, temperature, etc.). Look at [CovidJaxModel.__init__](../compartment/models/covid_jax_model/model.py) for a typical migrated `__init__`.

If your model genuinely cannot use `super().__init__()` (e.g. dengue's hand-rolled disease parameters), you can still extract the same fields manually — but keep `self.compartment_list = list(self.COMPARTMENTS)` so downstream code finds the canonical order.

### `prepare_initial_state()` → `(state, compartment_list)`

Return the state array used by the ODE solver. Supported shapes:
- `(K, R)` — compartments × regions (default)
- `(K, A, R)` — compartments × demographic groups × regions (call `self._prepare_demographic_state()` to expand from `(K, R)` and append zero rows for `_total` compartments)

`compartment_list` must list compartment IDs in the **same order** as the first axis of `state`, including any `_total` rows.

Set `self.travel_matrix` here (or in `__init__`) — built-in helpers like `_apply_interventions()` expect it to exist.

### `derivative(self, y, t, p)`

Implement the ODE right-hand side using `jax.numpy`. The shape of `y` matches what `prepare_initial_state()` returned. Return a `jnp.stack([...])` whose first axis matches `self.compartment_list` exactly (including `_total` columns).

The framework provides three helpers that handle the boilerplate. Use them where you can; reach for manual flow math only when you need spatially-coupled or age-stratified force-of-infection terms.

```python
def derivative(self, y, t, p):
    C = self.COMPARTMENTS
    params = self._unpack_params(p)        # {"beta": ..., "gamma": ...}

    states = {c: y[i] for i, c in enumerate(self.compartment_list)}
    I = states[C.I]

    # Optional: apply schema-driven interventions to rates and travel matrix
    rates = {"beta": params["beta"]}
    prop_infective = I.sum() / sum(states[c] for c in C if not c.endswith("_total")).sum()
    rates, travel_matrix = self._apply_interventions(t, rates, prop_infective)
    rates["gamma"] = params["gamma"]

    # Standard mass-action / FOI flows — handled by the framework.
    # Returns a dict {compartment_id: deriv_array}.
    derivs = self._compute_derivatives(states, rates)

    # If you need a custom flow (e.g. spatial coupling), skip that edge
    # and apply it manually:
    #   derivs = self._compute_derivatives(states, rates, skip_edges={"beta"})
    #   self._apply_flow(derivs, "S", "I", S * lambda_force)

    return jnp.stack([derivs[c] for c in self.compartment_list])
```

`_compute_derivatives()` reads `frequency_dependent` and `infective` flags from your schema, automatically accumulates flow into `_total` compartments, and skips edges whose source/target compartments aren't active in the current run (which is what makes the COVID variants work via `schema.remove_compartment()`).

### Flows without an edge

Some compartment-to-compartment movements can't be expressed as a single schema edge because the flow rate isn't a single tunable scalar. Examples seen in this repo:

- **Multi-rate force of infection.** When several rates contribute to the same S→E (or S→I) flow with different infectious sources, e.g. hantavirus's `Sm → Em` driven by `β_mm·Im + β_f·If`. No single rate exists to put on the edge.
- **Demographic births.** Inflow into a compartment from outside the modelled population (e.g. dengue's birth term `μ·N`, hantavirus's harmonic-mean births).
- **Density-dependent deaths.** Outflow from every compartment with rate `a + c·N` — the rate depends on aggregate state, not just the source compartment.
- **Spatial coupling.** Mpox's S→I uses a travel-matrix-weighted FOI; the rate that the framework would multiply by `S` doesn't capture the coupling.

For these, the pattern is:
1. Declare the underlying constants via `schema.add_disease_parameter()` so they show up in the artifact and the validated config.
2. Skip the corresponding edge in `_compute_derivatives()` (or just don't declare it) and apply the flow manually with `self._apply_flow(derivs, source_id, target_id, flow)`.
3. **Declare the target's `*_total` compartment by hand** with `schema.add_compartment("X_total", ...)` if you want cumulative tracking. The framework only auto-generates `*_total` for compartments that are the target of at least one declared edge — flows applied via `_apply_flow()` will populate the `_total` counter, but only if it exists.

Hantavirus is the canonical example here: `Em` and `Ef` are not edge targets, the three β rates are `disease_parameter`s, and `Em_total` / `Ef_total` are declared manually. See [hantavirus_jax_model/model.py](../compartment/models/hantavirus_jax_model/model.py).

### Cumulative `_total` compartments

The framework appends a `<target>_total` compartment for every edge target the first time it sees one. You don't declare them. They:
- Track the cumulative inflow into the target compartment.
- Are excluded from API outputs by the post-processor.
- Are used to compute correct compartment deltas (the delta key is the bare name, e.g. `I`, even though the underlying counter is `I_total`).

If you want different aggregation (e.g. one `_total` per group of compartments instead of per-edge), override `_add_total_compartments()` as a no-op and declare your own (see [DengueJaxModel](../compartment/models/dengue_jax_model/model.py)).

### Contact matrices

Age-stratified models need an NxN contact matrix where N is the number of demographic groups. The framework supports three ways of supplying one, in order of precedence (later wins over earlier).

> **📚 For a comprehensive guide to contact matrices**, see **[CONTACT_MATRICES.md](./CONTACT_MATRICES.md)** — detailed documentation on how contact matrices are loaded, aggregated, and used in the platform.

1. **Country-aware Prem 2021 default** (recommended). Declare an inclusive `age_range=(low, high)` on every demographic group. At model instantiation the framework reads the run's `admin_unit_id`, takes the ISO3 prefix (split on `.`), and looks up that country's synthetic 16×16 contact matrix from a bundled dataset of 177 countries (Prem 2021). It then aggregates the matrix down to your declared bands using fractional age-year membership: row direction mean-averages (the contactor is a typical person sampled from the band), column direction sums (total contacts with everyone in the band). Aggregating Prem back to its own 16 bands is an exact identity. If the ISO3 isn't in the bundle, the framework falls back to the mean across all 177 country matrices (`default_matrix()`). Code lives in [compartment/contact_matrices/](../compartment/contact_matrices/).

   ```python
   schema.add_demographic_group("age_0_17",    "Children", default_weight=33.3, age_range=(0, 17))
   schema.add_demographic_group("age_18_55",   "Adults",   default_weight=44.4, age_range=(18, 55))
   schema.add_demographic_group("age_56_plus", "Elderly",  default_weight=22.3, age_range=(56, 120))
   # No set_contact_override calls needed.
   ```

   Age ranges must be non-overlapping across groups; `schema.build()` raises `ValueError` if they overlap.

2. **Schema-level overrides** via `schema.set_contact_override(from_group, to_group, value)`. When the schema declares any contact override, the framework uses those values for the cells they cover and **does not** load the Prem matrix at all. Use this when a model has bespoke contact values (e.g. POLYMOD-derived rates baked into the model) that should be used regardless of country.

3. **Per-run config overrides** via `contact_matrix_overrides` in the simulation config. These beat both schema overrides and the Prem default for the cells they specify. Useful for sensitivity analyses or sweeping non-default mixing scenarios without modifying the model.

   ```jsonc
   "contact_matrix_overrides": {
       "age_0_17":  { "age_18_55": 3.0 },
       "age_56_plus": { "age_56_plus": 2.5 }
   }
   ```

If a model declares demographic groups but supplies neither `age_range` nor any `set_contact_override` calls, and the config provides no `contact_matrix_overrides`, the matrix defaults to identity (no cross-group mixing) and the framework emits a warning — that's almost always a bug.

**Data attribution.** The bundled synthetic contact matrices are derived from [kieshaprem/synthetic-contact-matrices](https://github.com/kieshaprem/synthetic-contact-matrices), which accompanies Prem et al. 2021, *"Projecting contact matrices in 177 geographical regions: an update and comparison with empirical data for the COVID-19 era"* (PLOS Computational Biology). The dataset is released under CC-BY. Bands are 5-year (0-4, 5-9, …, 70-74, 75+), ISO3 keyed.

### Spatial mobility and travel matrices

Multi-region models use **gravity models** to generate travel matrices describing population flows between administrative zones. The travel matrix `T[i,j]` represents the fraction of region i's population present in region j.

> **📚 For comprehensive documentation on gravity models**, see **[GRAVITY_MODEL.md](./GRAVITY_MODEL.md)** — detailed explanation of spatial mobility, distance-decay functions, and how travel matrices work in disease models.

**Quick overview:**

The framework provides automatic gravity-model travel matrices when `travel_volume.leaving` is specified in the config:

```jsonc
"travel_volume": {
    "leaving": 0.2  // 20% of each region's population travels
}
```

The default implementation uses the classic inverse-square gravity model: attraction from region i to j is proportional to `(pop_i * pop_j) / distance²`. Models can override this with custom mobility functions (exponential decay, power-law with custom exponent, etc.) by defining their own mobility methods.

**Spatial + demographic mixing:** When your model has both multiple regions and age groups, the force of infection combines spatial travel mixing with demographic contact mixing:

```python
# Step 1: Spatial mixing via travel matrix
BETA = ((beta * travel_matrix) @ I_frac.T).T  # (R, A)

# Step 2: Age mixing via contact matrix  
omega = contact_matrix @ BETA  # (R, A)

# Force of infection
foi = S * omega
```

See [GRAVITY_MODEL.md](./GRAVITY_MODEL.md) for detailed mathematical explanation and usage patterns.

### Compartment delta grouping

Set `COMPARTMENT_DELTA_GROUPING` on the class when you want grouped output (e.g. dengue collapses 4 serotypes into a single `I` group):

```python
COMPARTMENT_DELTA_GROUPING = {
    "I": ["I1", "I2", "I3", "I4"],
    "R": ["R1", "R2", "R3", "R4"],
    ...
}
```

Otherwise the post-processor uses a 1:1 mapping (each compartment groups to itself, excluding `_total` columns).

### Stochastic / Euler-integrated models

By default the simulation manager uses JAX's adaptive `odeint`. If your model is stochastic or otherwise needs fixed-step integration, set:

```python
class MyModel(Model):
    STOCHASTIC = True       # or: SOLVER = "euler"
```

This switches the integrator to a fixed-step Euler loop where `derivative()` returns the **delta per timestep**, not the instantaneous rate. See [test_covid_sir_stochastic/model.py](../compartment/models/test_covid_sir_stochastic/model.py) for a tau-leaping example.

### Variants (multiple disease types from one model class)

When you have several fixed-compartment variants of the same dynamics (e.g. SEIHDR / SEIR / SIHR for respiratory), put a `variants.py` next to `model.py`:

```python
# variants.py
from .model import MyBaseModel

class MySEIRModel(MyBaseModel):
    DISEASE_TYPE = "MY_SEIR"
    DISEASE_LABEL = "My Disease (SEIR)"
    DISEASE_DESCRIPTION = "..."

    @classmethod
    def define_parameters(cls, schema):
        super().define_parameters(schema)
        schema.remove_compartment("H")  # cascades: drops edges that reference H
        schema.remove_compartment("D")
```

Each variant gets its own `DISEASE_TYPE`, its own auto-generated `COMPARTMENT_LIST`, and its own artifact JSON. The shared `derivative()` works as long as it tolerates missing compartments (use `_compute_derivatives()` and check `if "E" in states` before referencing optional compartments — see CovidJaxModel.derivative). The auto-discovery registry picks up every variant in `variants.py` automatically.

## `main.py`

Boilerplate — copy from any existing model. The driver handles arg parsing for both modes:

```python
# main.py
import argparse
from compartment.driver import drive_simulation
from compartment.models.your_model.model import MyModel


def lambda_handler(event, context):
    drive_simulation(
        model_class=MyModel,
        args={"mode": "cloud", "simulation_job_id": event["simulation_job_id"]},
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--config_file")
    parser.add_argument("--output_file", nargs="?", default=None)
    parser.add_argument("--simulation_job_id", nargs="?", default=None)
    args = parser.parse_args()
    drive_simulation(model_class=MyModel, args=vars(args))
```

## `example-config.json`

You can write this by hand or auto-generate it from the schema:

```bash
python -m compartment.generate_artifact MY_DISEASE --example-config \
       --config-output compartment/models/your_model/example-config.json
```

Local-mode configs accept the "short form" — top-level `admin_zones` and `demographics` are wrapped into `case_file` automatically by `load_config_from_json()`. Required keys:

- `Disease.disease_type` — must match `set_model_info(disease_type=…)`.
- `start_date`, `end_date` (YYYY-MM-DD).
- `admin_zones` (or `case_file.admin_zones`) — each zone needs `name`, `population`, `center_lat`, `center_lon`, `infected_population` plus any custom `add_admin_zone_field` you declared.
- `Disease.transmission_edges` (or top-level `TransmissionEdges.items` for the GraphQL form) — one entry per edge you declared.

Optional keys: `interventions`, `travel_volume`, `demographics`, `contact_matrix_overrides`, `demographic_rate_overrides`, plus any `disease_parameters` you declared on the schema.

The validation layer auto-generates a Pydantic model from your schema and validates the config against it before your model ever sees it. Any required field you forget is rejected with a structured error.

## Generating model artifacts

Artifacts are JSON descriptions of the model — consumed by the UI, DB seeding, and downstream Zod schema generation. The CLI lives at [compartment/generate_artifact.py](../compartment/generate_artifact.py):

```bash
# List models that support artifact generation
python -m compartment.generate_artifact --list

# Print artifact JSON to stdout
python -m compartment.generate_artifact MY_DISEASE

# Write to a file
python -m compartment.generate_artifact MY_DISEASE --output artifact.json

# Write artifact + example config
python -m compartment.generate_artifact MY_DISEASE \
       --output artifact.json \
       --example-config --config-output example.json

# Generate one artifact per variant (model + variants.py) into a directory
python -m compartment.generate_artifact \
       --model-dir compartment/models/your_model \
       --output-dir compartment/models/your_model/artifacts
```

## Running locally

```bash
python -m compartment.models.your_model.main \
  --mode local \
  --config_file compartment/models/your_model/example-config.json \
  --output_file results/test_run.json
```

`run_simulation` runs your model **twice in parallel** — once with interventions and once without (the "control run") — and writes both into the output JSON. In `UNCERTAINTY` mode (set `run_mode` in the config) it draws Latin Hypercube samples over edge variances and produces median + CI bands.

## Tests

Smoke tests in [tests/test_smoke.py](../tests/test_smoke.py) automatically discover any model directory that contains both a `model.py` and an `example-config.json`. Adding your model to the test sweep requires nothing beyond those two files — the suite picks it up on the next run:

```bash
python -m pytest tests/test_smoke.py -v -m integration -k "your_model"
```

There's also [tests/test_artifact.py](../tests/test_artifact.py) for artifact generation and [tests/test_generate_artifact_model_dir.py](../tests/test_generate_artifact_model_dir.py) for `--model-dir` discovery.

## Quality checklist before opening a PR

- `define_parameters()` calls `set_model_info()` and adds at least one compartment. Every transmission edge's `source`/`target` matches a declared compartment id or label (case-insensitive).
- Every compartment-to-compartment movement in your dynamics is either a declared edge (preferred) **or** intentionally a manual flow because the rate isn't a single tunable scalar (multi-rate FOI, demographic births, density-dependent deaths). For each manual flow the underlying rates are exposed as `add_disease_parameter()` and the target's `*_total` compartment is declared by hand if you need cumulative tracking.
- The compartments your `derivative()` indexes are spelled the same as `self.COMPARTMENTS.X` — typos raise `AttributeError` with a helpful list.
- `derivative()` returns `jnp.stack([...])` in `self.compartment_list` order, including any `_total` rows.
- `prepare_initial_state()` sets `self.travel_matrix` (use `np.eye(R)` if you don't model travel).
- Custom flows skip their edges via `_compute_derivatives(states, rates, skip_edges={...})` and apply manually with `_apply_flow()`.
- `infective=True` is set on the right compartments — frequency-dependent edges read from `cls.COMPARTMENTS.infective_ids`.
- `value_type` matches the unit of `default` (e.g. `ValueType.DAYS` with `default=10.0` means "10-day average," automatically converted to a `0.1/day` rate at load time).
- A minimal `example-config.json` exists and `python -m compartment.models.your_model.main --mode local --config_file …` completes without errors.
- If you added a variant, it appears alongside the base model in `python -m compartment.generate_artifact --list`.
- `python -m pytest tests/test_smoke.py -v -m integration -k your_model` passes.

## Where to look when in doubt

- Minimal SIR: [mpox_jax_model](../compartment/models/mpox_jax_model/model.py).
- Age-stratified respiratory with variants: [covid_jax_model](../compartment/models/covid_jax_model/model.py) + [variants.py](../compartment/models/covid_jax_model/variants.py).
- Vector-borne with custom totals & disease-specific params: [dengue_jax_model](../compartment/models/dengue_jax_model/model.py).
- Multi-rate FOI / demographic births / density-dependent deaths (flows without an edge): [hantavirus_jax_model](../compartment/models/hantavirus_jax_model/model.py).
- Stochastic / Euler-integrated: [test_covid_sir_stochastic](../compartment/models/test_covid_sir_stochastic/model.py).
- Multi-axis structure (settings × strains × treatment): [test_klebsiella_amr_model](../compartment/models/test_klebsiella_amr_model/model.py).

## Support

If validation rejects your config, check the structured error output — it points at the offending `loc` path. If the runtime fails inside `derivative()`, JAX errors usually trace back to a shape mismatch between `states[X]` and the rate you're multiplying it by. Reach for `_unpack_params(p)` and a fresh `print(states[c].shape for c in C)` block to debug.
