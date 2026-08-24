# Agent reference: authoring a compartmental model

Playbook for adding or modifying a disease model in this repo: concrete patterns, file paths, and pitfalls. The companion user-facing doc is [docs/guides/model-integration-documentation.md](../docs/guides/model-integration-documentation.md). Keep this terse and pattern-focused.

## Mental model in three sentences

1. The framework is **schema-driven**. An author calls `schema.add_*()` / `schema.set_*()` inside `define_parameters()`, and almost everything else (`COMPARTMENT_LIST`, `DISEASE_TYPE`, the Pydantic config class, the artifact JSON, rate attributes set on `__init__`, automatic `*_total` cumulative compartments, the registry entry) is *derived* from that schema by the base class.
2. The two pieces of code that must be written by hand are the schema declaration and the ODE `equation()`. `__init__` usually just calls `super().__init__(config)` and adds model-specific fields.
3. The registry is **auto-discovered** — every `Model` subclass defined in `compartment/models/*/model.py` or `variants.py` that has a `DISEASE_TYPE` is picked up. There is no manual list to edit.

## Authoritative source files (reach for these first)

| Need | File |
|---|---|
| Base class: `_compute_equations`, `_apply_flow`, `_apply_interventions`, `_to_rate`, `_prepare_demographic_state`, contact-matrix resolution | [compartment/model.py](../compartment/model.py) |
| Schema builder API and the `ValueType` enum | [compartment/parameters.py](../compartment/parameters.py) |
| Registry / discovery rules | [compartment/registry.py](../compartment/registry.py) |
| Scaffold CLI for a new model directory | [compartment/new_model.py](../compartment/new_model.py) |
| What the validated config looks like at runtime (the dict the model receives) | [compartment/validation/post_processor.py](../compartment/validation/post_processor.py) |
| How Pydantic disease configs are auto-generated (`generate_disease_config`) | [compartment/schema_generator.py](../compartment/schema_generator.py) |
| Solver dispatch (`STOCHASTIC` / `SOLVER` → Euler or `odeint`) | [compartment/simulation_manager.py](../compartment/simulation_manager.py) |
| Config loader (the "short form" for local configs), `load_config_from_json`, `generate_LHS_samples` | [compartment/helpers.py](../compartment/helpers.py) |
| CLI driver wrapper | [compartment/driver.py](../compartment/driver.py) |
| Artifact generation CLI | [compartment/generate_artifact.py](../compartment/generate_artifact.py) |
| Smoke sweep that auto-discovers new models | [tests/test_smoke.py](../tests/test_smoke.py) + [tests/helpers.py](../tests/helpers.py) |
| **Contact matrices** — loading, aggregation, precedence | [docs/guides/contact-matrices.md](../docs/guides/contact-matrices.md); code in [compartment/contact_matrices/](../compartment/contact_matrices/) |
| **Spatial mobility and travel matrices** | [docs/guides/mobility.md](../docs/guides/mobility.md); `build_travel_matrix()` / `_ensure_travel_matrix()` in [compartment/model.py](../compartment/model.py), shared gravity kernel in [compartment/helpers.py](../compartment/helpers.py) |
| **Interventions** — declaration, configuration, application | [docs/guides/interventions.md](../docs/guides/interventions.md); `Intervention` dataclass in [compartment/runtime.py](../compartment/runtime.py), `_apply_interventions()` in [compartment/model.py](../compartment/model.py) |
| **Uncertainty quantification** — LHS, distributions, CI interpretation | [docs/guides/uncertainty-quantification.md](../docs/guides/uncertainty-quantification.md); orchestration in [compartment/run_simulation.py](../compartment/run_simulation.py) |
| **Porting an external model into this framework** | [docs/guides/model-conversion.md](../docs/guides/model-conversion.md) |
| **Modeler-supplied data files** | [docs/guides/adding-datasets.md](../docs/guides/adding-datasets.md); code in [compartment/datasets/](../compartment/datasets/) |

When in doubt, *read the file* — the framework changes faster than this reference.

## Reference implementations by complexity

Start from the closest analog rather than from scratch. The `example_*` models are the maintained teaching templates.

| Need | Model |
|---|---|
| Simplest declarative SIR — schema edges + `_compute_equations()`, age groups, declared uncertainty | [example_parameter_uncertainty_declarative_model](../compartment/models/example_parameter_uncertainty_declarative_model/model.py) |
| Same, but with a hand-written `equation()` and a bespoke ramped intervention | [example_parameter_uncertainty_custom_model](../compartment/models/example_parameter_uncertainty_custom_model/model.py) |
| Stochastic tau-leap (asymptomatic + symptomatic, custom `*_total`) | [example_stochastic_model](../compartment/models/example_stochastic_model/model.py) |
| Minimal stochastic SIR | [test_covid_sir_stochastic](../compartment/models/test_covid_sir_stochastic/model.py) |
| Live mobility + bespoke spatial intervention + a declared dataset | [mpox_jax_model](../compartment/models/mpox_jax_model/model.py) |
| Age-stratified, spatial, multi-variant (contact matrix × travel matrix) | [covid_jax_model](../compartment/models/covid_jax_model/model.py) + [variants.py](../compartment/models/covid_jax_model/variants.py) |
| Vector-borne, temperature-driven, overrides `_add_total_compartments()` | [dengue_jax_model](../compartment/models/dengue_jax_model/model.py) |
| Two-strain with cross-immunity / ADE | [dengue_2strain_jax_model](../compartment/models/dengue_2strain_jax_model/model.py) |
| Manual flows: multi-rate FOI, harmonic births, density-dependent deaths, hand-declared `*_total` | [hantavirus_jax_model](../compartment/models/hantavirus_jax_model/model.py) |
| Coupled human/rodent SEIR with a local power-law gravity kernel | [hantavirus_human_jax_model](../compartment/models/hantavirus_human_jax_model/model.py) |
| Stochastic Erlang(2) linear chain | [ebola_jax_model](../compartment/models/ebola_jax_model/model.py) |
| SEIHFR with community/hospital/burial routes, heavy `_apply_flow()` use | [ebola_seihfr_burial_legrand_model](../compartment/models/ebola_seihfr_burial_legrand_model/model.py) |
| Multi-axis structure (settings × strains × treatment, 29 compartments) | [test_klebsiella_amr_model](../compartment/models/test_klebsiella_amr_model/model.py) |

## Authoring recipe (default plan)

1. **Scaffold the directory.** Don't hand-copy another model:
   ```bash
   python -m compartment.new_model <name> --label "<Label>" \
          --disease-type <DISEASE_TYPE> --description "<description>"
   ```
   The name gets `_model` appended, so `my_disease` → `compartment/models/my_disease_model/`. It writes `__init__.py`, `model.py` (a working S/I/R template), `main.py`, `model.md`, and `example-config.json`. It does **not** create `artifacts/`. `--disease-type` defaults to the name upper-cased; `--dry-run` prints the plan without writing.
2. **Write `define_parameters()`** in this order: `set_model_info` → `set_model_metadata` → `add_compartment` (mark `infective=True` on FOI sources) → `add_transmission_parameter` → `add_intervention` → mobility params → demographics / contact overrides → `add_admin_zone_field` → `add_parameter`.
3. **Write `__init__(self, config)`.** Default to `super().__init__(config)` then add what's missing. **Don't set `self.travel_matrix` here** — the framework does it.
4. **Write `prepare_initial_state(self)`.** Return the state array (usually `self.population_matrix`). For inter-zone travel, override `build_travel_matrix(admin_zones)` instead (see *Mobility*).
5. **Write `equation(self, y, t, p)`.** Lean on `_compute_equations()` first; drop to manual flows only for spatially-coupled or multi-rate FOI. Stochastic/Euler models return a per-step delta, not a derivative.
6. **Regenerate the artifact and example config**:
   ```bash
   python -m compartment.generate_artifact <DISEASE_TYPE> \
          --output compartment/models/<name>/artifacts/<DISEASE_TYPE>.json \
          --example-config --config-output compartment/models/<name>/example-config.json
   ```
   `--uncertainty` adds schema-declared variance to the example config and *requires* `--example-config`. For a model with variants, use `--model-dir <dir> --output-dir <dir>/artifacts` to emit one artifact per discovered class. `--list` prints every registered disease type.
7. **Run end-to-end, then the smoke tests**:
   ```bash
   python -m compartment.models.<name>.main --mode local \
          --config_file compartment/models/<name>/example-config.json \
          --output_file results/<name>.json
   python -m pytest tests/test_smoke.py -v -m integration -k <name>
   ```

## Schema builder cheat sheet

`schema` is a `ParameterSchemaBuilder` ([parameters.py](../compartment/parameters.py)). Defaults below are the real ones.

```python
schema.set_model_info(disease_type, label, description)               # required, once
schema.set_model_metadata(authors=None, license=None, citations=None, # all optional
                          model_type=None, diseases=None, transmission_routes=None,
                          questions_answered=None, key_assumptions=None,
                          applicability=None, not_for=None, constraints=None,
                          biases=None, validation=None)
schema.add_compartment(id, label, description, infective=False)
schema.remove_compartment(id)                                        # cascades to referencing edges
schema.add_transmission_parameter(
    source, target, variable_name, label, description, default,
    min_value=None, max_value=None, default_min=None, default_max=None,
    unit="per day", frequency_dependent=False, value_type=ValueType.RATE,
)
schema.remove_transmission_parameter(variable_name)
schema.add_intervention(id, label, description, target_rates=None,
                        modifies_travel=False, adherence=None,
                        transmission_reduction=None)
schema.add_demographic_group(id, label, default_weight, age_range=None)
schema.set_contact_override(from_group, to_group, value)              # bespoke values; suppresses Prem
schema.add_admin_zone_field(name, label, description, value_type, default,
                            min_value=None, max_value=None, default_min=None,
                            default_max=None, unit=None, required=False, options=None)
schema.add_parameter(name, label, description, value_type, default,
                     min_value=None, max_value=None, default_min=None,
                     default_max=None, unit=None, required=True,
                     options=None, enable_variance=True)              # alias: add_disease_parameter
```

`ValueType`: `RATE`, `DAYS`, `PERCENTAGE`, `COUNT`, `DATE`, `BOOLEAN`, `TEXT`, `SELECT`, `FLOAT`, `INTEGER`, `COORDINATE`.

`DAYS` and `PERCENTAGE` are auto-converted to per-day fractional rates in `_load_transmission_params()` — **for transmission edges only**. Parameters declared with `add_parameter()` arrive in native units (a `PERCENTAGE` param is `20.0`, not `0.2`); convert at the point of use with `self._to_rate(value, ValueType.PERCENTAGE)`.

## Mobility

There **is** a shared gravity kernel — `get_gravity_model_travel_matrix()` in [compartment/helpers.py](../compartment/helpers.py) — but **nothing applies it by default**. `Model.build_travel_matrix()` returns the identity matrix, so a model with inter-zone travel must opt in explicitly.

1. Declare the parameters as ordinary custom fields so they render in the UI and can be sampled during uncertainty runs. Convention is `travel_sigma` (`PERCENTAGE`, 0–100) plus whatever else the kernel needs:
   ```python
   schema.add_parameter(
       name="travel_sigma", label="Travel Rate (σ)",
       description="Percentage of each zone's population away from home on a given day.",
       value_type=ValueType.PERCENTAGE, default=20.0,
       min_value=0.0, max_value=100.0, unit="%",
   )
   ```
2. Override `build_travel_matrix(self, admin_zones)` to return an `(R, R)` matrix. The framework calls it via `_ensure_travel_matrix()` **before** `prepare_initial_state()` and stores the result on `self.travel_matrix`.
   ```python
   from compartment.helpers import get_gravity_model_travel_matrix

   def build_travel_matrix(self, admin_zones):
       sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
       return get_gravity_model_travel_matrix(admin_zones, sigma)
   ```
3. **Apply it in `equation()`.** The framework builds the matrix but never applies it; `_compute_equations()` ignores it. A model that computes FOI from local prevalence only has mobility in name. Use the matrix returned by `_apply_interventions()` (not `self.travel_matrix`) if any intervention sets `modifies_travel=True`, or the travel restriction is silently ignored.

**Never name the parameter plain `sigma`.** `build_overridden_config()` routes edge `variable_name`s to the transmission dict and everything else to `Disease`, so a mobility param colliding with an edge name (ebola's `sigma` is its E→I rate) misroutes during uncertainty runs.

Invariants, enforced by [tests/test_travel_matrix.py](../tests/test_travel_matrix.py) for every discovered model whose schema declares `travel_sigma`: `T[i, j]` is the fraction of zone *i* present in zone *j*, shape is `(R, R)`, all entries finite and non-negative, **rows sum to 1**, the diagonal is `1 - sigma`, and row/column order matches `admin_zones`. Models without `travel_sigma` are asserted to be identity.

Kernels available, by decay shape:

- **inverse-square gravity (geopy)** — `get_gravity_model_travel_matrix()` in `compartment/helpers.py`; used by covid, dengue, ebola
- **power-law gravity, α configurable (Haversine)** — local `gravity()` in `hantavirus_human_jax_model/model.py`
- **exponential distance decay** — local `mobility()` in `mpox_jax_model/model.py`

Everything else inherits the identity matrix: `dengue_2strain_jax_model`, `ebola_seihfr_burial_legrand_model`, `hantavirus_jax_model`, `test_covid_sir_stochastic`, `test_klebsiella_amr_model`, and all three `example_*` models (which ship the override commented out).

## Patterns for `equation()`

```python
def equation(self, y, t, p):
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

    rates = {name: params[name] for name in params}
    rates, travel_matrix = self._apply_interventions(t, rates, prop_infective)

    derivs = self._compute_equations(states, rates)

    # spatial coupling: exclude the edge above, then do it by hand
    # derivs = self._compute_equations(states, rates, skip_edges={"beta"})
    # foi = rates["beta"] * jnp.einsum("ij,j->i", travel_matrix, I / N_total)
    # self._apply_flow(derivs, "S", "I", states["S"] * foi)

    return jnp.stack([derivs[c] for c in self.compartment_list])
```

`_compute_equations(states, rates, skip_edges=None)` already:

- Skips edges named in `skip_edges`, edges whose rate is `None`, and edges whose source or target is absent from `states` (this is what makes COVID's compartment-removal variants work).
- Accumulates flow into `<target>_total` when that key exists in `derivs`.
- Picks the FOI formula from the **edge's** `frequency_dependent` flag, summing over compartments carrying the **compartment-level** `infective=True` flag.
- Substitutes per-demographic rate vectors from `self._rate_vectors` when present, bypassing the intervention-scaled scalar.

`_apply_flow(derivs, source_id, target_id, flow)` subtracts from source, adds to target, and accumulates the target's `*_total` if it exists.

## Contact matrices

Three precedence layers, later wins:

1. **Prem 2021 auto-load**, which requires *all* of: every demographic group declares `age_range=(low, high)`, no schema overrides, no config overrides, **and** the runtime group IDs match the schema group IDs (a config that renames or replaces groups via `case_file.demographics` disables it). ISO3 is parsed from `config["admin_unit_id"]` (split on `.`, take `[0]`, upper-case; `"LOCAL"` or missing → `None`). Lookup falls back in three tiers: country-specific matrix → average across that country's World Bank income level → global average over all 177 bundled countries. Source matrices are 16×16 in 5-year bands, aggregated to N×N by fractional-year overlap — **mean across rows, sum across columns** (preserves "total contacts per person"; aggregating Prem to its own bands is exact). Code: [compartment/contact_matrices/](../compartment/contact_matrices/) (`loader.py`, `aggregator.py`).
2. **Schema overrides** via `schema.set_contact_override(...)`. Any schema override suppresses the Prem path entirely — the model is asserting bespoke values. Mixing Prem and overrides isn't supported.
3. **Per-run config overrides** via `config["contact_matrix_overrides"]` — narrowest scope, always wins.

`schema.build()` raises `ValueError` on overlapping `age_range`s. With no usable source the matrix is identity (groups don't mix); the framework only *warns* about that when the config supplied `case_file.demographics`, so a schema-only model with missing `age_range`s degrades silently. Bundled data comes from [kieshaprem/synthetic-contact-matrices](https://github.com/kieshaprem/synthetic-contact-matrices) (Prem et al. 2021).

COVID's nine hardcoded POLYMOD `set_contact_override` calls are **commented out**; it declares `age_range` on all three groups and relies on Prem auto-load.

## Modeler-supplied data

A model that needs a data file declares it in a `datasets.yaml` next to `model.py` and reads it with `self.dataset(name)`, which returns a `Path`:

```yaml
# compartment/models/<name>/datasets.yaml
datasets:
  - name: kenya-contact-matrix
    version: "1"                    # quote it — unquoted 1.10 becomes 1.1
    file: data/kenya-contacts.csv   # relative to datasets.yaml
```
```python
contacts = pd.read_csv(self.dataset("kenya-contact-matrix"))
```

`dataset()` is a `classmethod`, so it also works from `get_initial_population()` and other classmethods. It resolves the manifest entry and checks the file exists, raising `ManifestError` — with the exact `datasets pull` command to run — when it doesn't. Resolution in [compartment/datasets/resolve.py](../compartment/datasets/resolve.py); `ManifestError` and the size limit in [compartment/datasets/manifest.py](../compartment/datasets/manifest.py); user guide: [docs/guides/adding-datasets.md](../docs/guides/adding-datasets.md).

The same call works locally and in the cloud. Locally the file arrives via `python -m compartment.datasets pull <name>@<version> --dest data/` (other subcommands: `push`, `list`, `check-status`, `stage`); `pull` writes to `<dest>/<filename>`, so point `--dest` at the directory the manifest's `file:` path implies. `compartment/models/*/data/` is gitignored — only the manifest is tracked. In the cloud, the release pipeline's *Stage datasets* step downloads each entry before `docker build`, so it lands in the image at the same relative path. Limit: **500 MB per dataset** (`MAX_DATASET_BYTES`), checked client-side before upload.

`mpox_jax_model` is the only model with a manifest today. This is distinct from the bundled Prem matrices under `compartment/contact_matrices/`, which are repo data loaded by relative path and are not datasets in this sense.

## Pitfalls

- **`super().__init__()` must be called** for a schema-driven model. It populates `self.beta`, `self.gamma`, the contact matrix, and intervention runtime objects. `AttributeError: ... has no attribute 'beta'` usually means a forgotten `super().__init__()` or a typo in `variable_name`.
- **`compartment_list` order is load-bearing.** `equation()` indexes `y` by position and `_compute_equations()` returns a dict keyed by ID. Always stack with `jnp.stack([derivs[c] for c in self.compartment_list])`, never a hardcoded order.
- **`*_total` compartments are auto-generated** for edge targets. Don't declare `I_total` by hand unless overriding `_add_total_compartments()` (dengue, ebola, hantavirus, `example_stochastic_model`, and klebsiella do). Duplicates raise `ValueError`.
- **Uncommenting the scaffold's `build_travel_matrix()` requires adding the import.** The generated template comments out the override *and* omits `from compartment.helpers import get_gravity_model_travel_matrix`, so enabling it without adding the import raises `NameError` at `_ensure_travel_matrix()` time — before the first integration step, on every run.
- **Every compartment-to-compartment movement should be a schema edge** when the flow is `rate * source` or `source * rate * sum(infective) / N`. Skipping an edge silently is almost always a bug. Legitimate manual flows: multi-rate FOI mixing several β values across infectious sources (hantavirus's `Sm → Em = Sm * (β_mm·Im + β_f·If)`), demographic births (`μ·N`, harmonic-mean), density-dependent deaths (`a + c·N`), and spatial coupling. For those, declare the rate constants with `add_parameter()`, move mass with `_apply_flow()`, and hand-declare the target's `*_total` if the target isn't already an edge target.
- **`infective=True` is critical for `frequency_dependent=True` edges.** Without it the FOI sum is empty and the model produces zero flow. Mark every compartment contributing infectious pressure (in dengue, all primary `Ix` and all secondary `Ixy`).
- **`value_type=ValueType.DAYS`** means `default=10.0` is a 10-day mean ⇒ rate `0.1`. Do not pre-divide.
- **Don't build a data-file path by hand.** `Path(__file__).parent / "data" / "contacts.csv"` appears to work, then silently reads stale data the moment `datasets.yaml` bumps the version — and resolves to nothing in the cloud, since only declared datasets get staged into the image. Use `self.dataset(name)`.
- **Don't set `self.travel_matrix` by hand.** The framework guarantees it exists by calling `_ensure_travel_matrix()` before `prepare_initial_state()`, including the identity default. Assigning it in `__init__` or `prepare_initial_state()` is overwritten. Override `build_travel_matrix()` instead.
- **Variants** call `super().define_parameters(schema)` and then mutate. Removing a compartment cascades to edges referencing it. To re-add an edge the parent routed through the removed compartment (e.g. an S→I beta after E is gone), call `schema.add_transmission_parameter(**_BETA_SI)` — see covid `variants.py`. Only covid has a `variants.py`.
- **Stochastic models** set `STOCHASTIC = True` and return the **per-step delta** from `equation()`, not a rate; `dt` is multiplied inside the Euler integrator. `SOLVER = "euler"` forces Euler independently, but no model currently sets it. Today's stochastic models: `ebola_jax_model`, `example_stochastic_model`, `test_covid_sir_stochastic`.
- **The "short form" config loader** wraps top-level `admin_zones` and `demographics` into `case_file` automatically. Don't double-nest in hand-written configs.
- **`schema.remove_compartment("X")` cascades to edges** by `source_id`/`target_id` only. A manual flow in `equation()` still indexing `states["X"]` must be guarded with `if "X" in states:`.
- **The registry only sees classes defined in the scanned module.** `cls.__module__` must equal `compartment.models.<dir>.model` (or `.variants`), so re-exporting a `Model` subclass imported from elsewhere won't register it. Discovery also swallows `ImportError` silently, so a broken import looks identical to a missing model.
- **`COMPARTMENT_DELTA_GROUPING`** is a real optional class attribute mapping display keys to raw compartment IDs. It drives compartment deltas, time-series grouping, and artifact display order. Set by dengue, ebola, hantavirus, `example_stochastic_model`, and klebsiella.

## Common fix-it flows

- **"My new model isn't found by the registry"** → `python -m compartment.generate_artifact --list`. If absent, check that `model.py` imports cleanly (`python -c "import compartment.models.<name>.model"`), that the class subclasses `Model`, is defined in that module, and either declares `DISEASE_TYPE` or calls `set_model_info()`.
- **"My model isn't in the smoke run"** → discovery needs both `model.py` and `example-config.json` in the directory, and the config needs `admin_zones` (top level or under `case_file`) or the fixture skips. The suite imports only `model.py`, not `variants.py`, and swallows any import exception silently. Tests are behind `-m integration`.
- **"Validation rejects my config"** → read the Pydantic `loc` path. "Extra field" complaints usually mean the config is in the GraphQL form where the loader expects short form, or vice versa; check `load_config_from_json`.
- **"NaN / negative compartments"** → usually a missing `infective=True`, a `frequency_dependent` edge with an empty FOI, or a manual flow that forgot to subtract from the source.
- **"Output deltas don't match expectation"** → either a manual flow bypassed `_apply_flow()` so the `*_total` never accumulated, or `COMPARTMENT_DELTA_GROUPING` maps IDs that aren't active in this run.
- **"Mobility has no effect"** → the matrix is built but `equation()` never reads it, or it reads `self.travel_matrix` instead of the matrix returned by `_apply_interventions()`. Also check the example config has more than one admin zone; with one zone the matrix is `[[1.0]]` regardless.

## When editing an existing model

- **Adding a parameter?** Add it to `define_parameters()`; the artifact, validation schema, and example config regenerate from it.
- **Adding an intervention?** `schema.add_intervention(target_rates=[...])` — `_apply_interventions()` handles it generically. Hand-rolled intervention code is only for non-standard activation (mpox's ring vaccination is bespoke because beta is applied with spatial coupling).
- **Renaming a compartment?** Search for the old ID: `variants.py`, `COMPARTMENT_DELTA_GROUPING`, and manual indexing in `equation()` all need updating in lockstep.
- **Adding a variant?** Drop a class into `variants.py`; discovery picks it up on next import.
- After any change: regenerate the artifact → regenerate or update `example-config.json` → run in `--mode local` → `pytest tests/test_smoke.py -m integration -k "<name>"`.

## What NOT to do

- Don't add a `DISEASE_TYPE` to a manual registry — there isn't one to edit.
- Don't hand-write a `BaseDiseaseConfig` subclass for a new model; `generate_disease_config()` derives it from the schema. The only hand-written configs left are `CovidDiseaseConfig` (`RESPIRATORY`) and `DengueDiseaseConfig` (`VECTOR_BORNE`), and only `VECTOR_BORNE` is wired as a runtime fallback.
- Don't declare `*_total` compartments by hand for typical models.
- Don't override `disease_type` as a property when `set_model_info()` already sets it — the base class wires `DISEASE_TYPE` via `__init_subclass__`. No model in the repo overrides it today.
- Don't edit `compartment/validation/post_processor.py` to register a new disease type; `process()` falls through to `_process_default()`, which works for any registered model.
- Don't edit `compartment/cloud_helpers/` unless the user explicitly asks — it's web-app integration, not local modeling.
