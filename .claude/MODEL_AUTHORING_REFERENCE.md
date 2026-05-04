# Claude reference: authoring a new compartmental model

This file is for me (Claude) when the user asks for help adding or modifying a disease model. The companion user-facing doc is [docs/DEVELOPING_MODELS.md](../docs/DEVELOPING_MODELS.md). Use this file as the *authoring playbook*: concrete patterns, file paths, and pitfalls. Keep this terse and pattern-focused.

## Mental model in three sentences

1. The framework is **schema-driven**. The model author calls `schema.add_*()` / `schema.set_*()` inside `define_parameters()`, and almost every other concern (`COMPARTMENT_LIST`, `disease_type`, the Pydantic config class, the artifact JSON, default rate attributes on `__init__`, automatic `_total` cumulative compartments, registry entry) is *derived* from that schema by the base class.
2. The two pieces of code I always have to write by hand are the schema declaration and the ODE `derivative()`. Initialization usually just calls `super().__init__(config)` and tacks on model-specific fields.
3. The model registry is **auto-discovered** — every `Model` subclass under `compartment/models/*/model.py` (and `variants.py`) with a `DISEASE_TYPE` is picked up. There is no manual list to edit anywhere.

## Authoritative source files (reach for these first)

| Need | File |
|---|---|
| Base class, helpers, `_compute_derivatives`, `_apply_flow`, `_apply_interventions`, `_prepare_demographic_state` | [compartment/model.py](../compartment/model.py) |
| Schema builder API and the `ValueType` enum | [compartment/parameters.py](../compartment/parameters.py) |
| Registry / discovery rules | [compartment/registry.py](../compartment/registry.py) |
| What the validated config looks like at runtime (the dict the model receives) | [compartment/validation/post_processor.py](../compartment/validation/post_processor.py) |
| How Pydantic disease configs are auto-generated | [compartment/schema_generator.py](../compartment/schema_generator.py) |
| ODE solver dispatch (`STOCHASTIC` / `SOLVER`) | [compartment/simulation_manager.py](../compartment/simulation_manager.py) |
| Running config loader (the "short form" for local configs) | [compartment/helpers.py](../compartment/helpers.py) `load_config_from_json` |
| CLI driver wrapper | [compartment/driver.py](../compartment/driver.py) |
| Artifact generation CLI | [compartment/generate_artifact.py](../compartment/generate_artifact.py) |
| Smoke test sweep that auto-discovers new models | [tests/test_smoke.py](../tests/test_smoke.py) + [tests/helpers.py](../tests/helpers.py) |

When in doubt, *read the file* — the framework changes faster than this reference.

## Reference implementations by complexity

- **Bare minimum SIR** (with mobility + custom intervention): [mpox_jax_model/model.py](../compartment/models/mpox_jax_model/model.py)
- **Age-stratified, multi-variant**: [covid_jax_model/model.py](../compartment/models/covid_jax_model/model.py) + [variants.py](../compartment/models/covid_jax_model/variants.py)
- **Stochastic (Euler + Poisson tau-leap)**: [test_covid_sir_stochastic/model.py](../compartment/models/test_covid_sir_stochastic/model.py)
- **Vector-borne, custom `_total` compartments, disease-specific params**: [dengue_jax_model/model.py](../compartment/models/dengue_jax_model/model.py)
- **Multi-axis structure (settings × strains × treatment)**: [test_klebsiella_amr_model/model.py](../compartment/models/test_klebsiella_amr_model/model.py)

Pick the closest analog before suggesting from scratch.

## Authoring recipe (use this as the default plan)

1. **Scaffold the directory** under `compartment/models/<name>/`:
   - `__init__.py` (empty)
   - `model.py`
   - `main.py` (copy from `mpox_jax_model/main.py`, swap the class name)
   - `example-config.json` (generate via CLI after the schema is written)
2. **Write `define_parameters()`** in this order: `set_model_info` → `add_compartment` (mark `infective=True` on FOI sources) → `add_transmission_edge` → `add_intervention` → `set_travel_volume` → demographics / contact matrix → `add_admin_zone_field` → `add_disease_parameter`.
3. **Write `__init__(self, config)`**. Default to `super().__init__(config)` then add what's missing (typically `self.travel_matrix`, demographics, temperature, model-specific scalars).
4. **Write `prepare_initial_state(self)`**. Set `self.travel_matrix` (use `jnp.eye(R)` if no travel). Return `(state, list(self.compartment_list))`.
5. **Write `derivative(self, y, t, p)`**. Lean on `_compute_derivatives()` first; only drop to manual flows for spatially-coupled or age-stratified FOI.
6. **Generate the example config**:
   ```bash
   python -m compartment.generate_artifact <DISEASE_TYPE> --example-config \
          --config-output compartment/models/<name>/example-config.json
   ```
7. **Run end-to-end** locally and check the smoke tests:
   ```bash
   python -m compartment.models.<name>.main --mode local \
          --config_file compartment/models/<name>/example-config.json \
          --output_file results/<name>.json
   python -m pytest tests/test_smoke.py -v -m integration -k <name>
   ```

## Schema builder cheat sheet

`schema` is a `ParameterSchemaBuilder`. Every method is in [parameters.py](../compartment/parameters.py); these are the ones I use almost every time.

```python
schema.set_model_info(disease_type, label, description)              # required, once
schema.add_compartment(id, label, description, infective=False)
schema.remove_compartment(id)                                        # also drops referencing edges
schema.add_transmission_edge(
    source, target, variable_name, label, description,
    default, min_value, max_value, default_min, default_max, unit,
    frequency_dependent=False, value_type=ValueType.RATE,
)
schema.remove_transmission_edge(variable_name)
schema.add_intervention(id, label, description,
                        target_rates=[...], modifies_travel=False,
                        adherence=..., transmission_reduction=...)
schema.set_travel_volume(leaving_default=0.2, ...)
schema.add_demographic_group(id, label, default_weight)
schema.set_contact_override(from_group, to_group, value)
schema.add_admin_zone_field(name, label, description, value_type, default, ...)
schema.add_disease_parameter(name, label, description, value_type, default, ...)
```

`ValueType` choices: `RATE`, `DAYS`, `PERCENTAGE`, `COUNT`, `DATE`, `BOOLEAN`, `TEXT`, `SELECT`, `FLOAT`, `INTEGER`, `COORDINATE`. `DAYS` and `PERCENTAGE` get auto-converted to per-day fractional rates inside `_load_transmission_params()`.

## Patterns I reach for in `derivative()`

```python
def derivative(self, y, t, p):
    C = self.COMPARTMENTS
    params = self._unpack_params(p)
    states = {c: y[i] for i, c in enumerate(self.compartment_list)}

    # population for FOI / proportion calculations
    non_total = [c for c in self.compartment_list if not c.endswith("_total")]
    N_total = sum(states[c] for c in non_total)
    prop_infective = (
        sum(states[c] for c in C.infective_ids if c in states).sum()
        / (N_total.sum() + 1e-10)
    )

    # interventions (for migrated models with schema interventions)
    rates = {name: params[name] for name in params}
    rates, travel_matrix = self._apply_interventions(t, rates, prop_infective)

    # framework computes standard edges and accumulates into _total
    derivs = self._compute_derivatives(states, rates)

    # custom flow example (spatial coupling): skip the edge above and do it manually
    # derivs = self._compute_derivatives(states, rates, skip_edges={"beta"})
    # foi = beta * jnp.einsum("ij,j->i", travel_matrix, I / N_total)
    # self._apply_flow(derivs, "S", "I", S * foi)

    return jnp.stack([derivs[c] for c in self.compartment_list])
```

`_compute_derivatives()` already:
- Skips edges whose source/target compartments aren't in `states` (this is what makes COVID's compartment-removal variants work).
- Auto-accumulates flow into `<target>_total` if that compartment exists.
- Uses `frequency_dependent` / `infective` flags to pick the FOI formula.
- Applies per-demographic absolute rate vectors from `self._rate_vectors` when present.

## Pitfalls I keep tripping on

- **`super().__init__()` must be called when the model is "migrated"** (i.e. uses `define_parameters()`). It's what populates `self.beta`, `self.gamma`, contact matrix, intervention runtime objects. If I see `AttributeError: ... has no attribute 'beta'`, the cause is usually a forgotten `super().__init__()` or a typo in `variable_name`.
- **`compartment_list` order matters**. `derivative()` indexes `y` by position, and `_compute_derivatives()` returns a dict keyed by ID. Always stack with `jnp.stack([derivs[c] for c in self.compartment_list])`, never with a hardcoded order.
- **`_total` compartments are auto-generated** — don't declare `I_total` in `define_parameters()` unless I'm overriding `_add_total_compartments()` (dengue does). Declaring duplicates raises `ValueError`.
- **`infective=True` is critical for `frequency_dependent=True` edges**. Without it, the FOI sum is empty and the model produces zero flow. Mark every compartment that contributes infectious pressure (in dengue that's all primary `Ix` and all secondary `Ixy`).
- **`value_type=ValueType.DAYS`** means `default=10.0` is interpreted as a 10-day mean ⇒ rate `0.1`. Do not pre-divide.
- **Travel matrix must exist before `_apply_interventions()`** — it reads `self.travel_matrix`. Set it in `__init__` or `prepare_initial_state()` before the first `derivative()` call. Use `jnp.eye(R)` when there's no travel model.
- **Variants** use `super().define_parameters(schema)` and then mutate. Removing a compartment cascades and removes any edges that reference it. To re-add an edge that the parent referenced via the removed compartment (e.g. an S→I beta after E is gone), call `schema.add_transmission_edge(**_BETA_SI)` — see covid `variants.py` for the canonical pattern.
- **Stochastic / Euler models** must set `STOCHASTIC = True` (or `SOLVER = "euler"`) and have `derivative()` return the **per-step delta**, not the rate. Multiplying by `dt` happens inside `_euler_integrate`.
- **The "short form" config loader** wraps top-level `admin_zones` and `demographics` into `case_file` automatically. Don't double-nest when writing example configs by hand.
- **`schema.remove_compartment("X")` cascades to edges**, but only by `source_id`/`target_id`. If a manual flow inside `derivative()` still indexes `states["X"]`, that branch must be guarded with `if "X" in states:`.
- **The user does not need to edit `MODEL_REGISTRY` or `validation/__init__.py`**. The registry is built by scanning the filesystem; the validation entry resolves through `_get_model_registry()` and `has_parameter_schema()`. If a model isn't being picked up, it's because either (a) the `model.py` import fails, or (b) the class doesn't subclass `Model` / lacks a `DISEASE_TYPE`.

## Common fix-it flows

- **"My new model isn't found by the registry"** → `python -m compartment.generate_artifact --list`. If absent: check that `model.py` imports cleanly (`python -c "import compartment.models.<name>.model"`) and that the class subclasses `Model` and either declares `DISEASE_TYPE` or calls `set_model_info()`.
- **"Validation rejects my config"** → Look at the structured Pydantic error — `loc` path tells me exactly which field. If the issue is "extra field" complaints, the user is using the GraphQL form when the loader expects short form (or vice versa); check `load_config_from_json` semantics.
- **"NaN / negative compartments"** → Almost always a missing `infective=True`, a `frequency_dependent` edge with an empty FOI, or a manual flow that forgot to subtract from the source. Check the smoke-test invariants.
- **"Output deltas don't match expectation"** → Either (a) `_total` compartment isn't being accumulated because the manual flow path forgot `_apply_flow()`, or (b) `COMPARTMENT_DELTA_GROUPING` is mapping IDs that aren't active in the current run.

## When the user is editing an existing model

- Adding a parameter? Add it to `define_parameters()`; the artifact, validation schema, and example config regenerate themselves.
- Adding an intervention? Use `schema.add_intervention(target_rates=[...])` — `_apply_interventions()` handles it generically. Avoid hand-rolled intervention code unless the model has a non-standard activation pattern (e.g. the mpox ring vaccination intervention is bespoke because beta is applied with spatial coupling).
- Renaming a compartment? Search the codebase for the old ID — variant subclasses (`variants.py`), `COMPARTMENT_DELTA_GROUPING`, and any manual indexing in `derivative()` will all need updating in lockstep.
- Adding a variant? Drop a class into `variants.py` (or create one). The discovery scan picks it up next import.

## What I should NOT do

- Don't add the new `DISEASE_TYPE` to a manual registry — there isn't one to edit.
- Don't write a hand-crafted `BaseDiseaseConfig` subclass for a new migrated model — `schema_generator` produces it from the schema. Hand-written configs only exist for legacy models that haven't migrated (`COVID_*`, `VECTOR_BORNE`, `VECTOR_BORNE_2STRAIN`).
- Don't declare `_total` compartments by hand in `define_parameters()` for typical models.
- Don't override `disease_type` as a property when `set_model_info()` already sets it (the base class wires this up via `__init_subclass__`).
- Don't suggest editing `compartment/validation/post_processor.py` to register a new disease type — it dispatches to a default that already works for any registered model.
