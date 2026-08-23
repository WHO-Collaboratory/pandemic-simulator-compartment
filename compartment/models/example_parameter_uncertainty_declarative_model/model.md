# Example Disease (Declarative Parameter Uncertainty) — Model Notes

A teaching example: a minimal **SIR** model authored the **declarative** way — you describe compartments, edges, interventions, and age groups in `define_parameters()`, and the framework derives the equations, config validation, and artifact from that schema. It also demonstrates **parameter uncertainty**, where the simulator runs many parameter draws and reports a median with an interval band. It is a reference implementation for the declarative workflow, not a calibrated model of any real disease.

## Model overview

Susceptible people become infected through contact with infectious people, then recover and stay immune. Transmission is **frequency-dependent**: the force of infection scales with the *proportion* of the population that is infectious, not the raw count. The population is closed — nobody is born, dies, or arrives from outside — so the compartment totals always add up to the starting population.

The population is split into five age bands that mix according to a contact matrix, so an outbreak can move through age groups at different speeds. One intervention reduces transmission while it is active.

The model is deterministic: it integrates an ODE system with JAX's `odeint`. Uncertainty does not change the equations — it re-runs them across sampled parameter values.

## Compartment and state definitions

| Compartment | Meaning |
|---|---|
| `S` | Susceptible — never infected, can be infected |
| `I` | Infected and infectious. Marked `infective=True`, so it drives the force of infection |
| `R` | Recovered and immune. There is no route back to `S` |
| `I_total` | Cumulative count of everyone who has *ever* entered `I` |
| `R_total` | Cumulative count of everyone who has *ever* entered `R` |

`I` and `R` are **prevalence** — how many people are in that state right now. `I_total` and `R_total` are **cumulative** and only ever increase, which is what you want for "total infections over the outbreak".

The two `_total` compartments are generated automatically for every transmission-edge target. Do not declare them by hand. They are internal accumulators: they do not appear in the output time series, but they are what `compartment_deltas` reports (see [Outputs](#outputs)).

Internally each compartment is stratified by age group and administrative zone, so the state array is *(compartments × age groups × zones)*. `equation()` receives it already shaped that way and indexes it positionally in `compartment_list` order.

## Inputs and parameters

Transmission parameters, set per edge under `TransmissionEdges` in the config:

| Parameter | Edge | Unit | Default | Valid range | Uncertainty range |
|---|---|---|---|---|---|
| `beta` | `S → I` | per day (`RATE`) | `0.3` | `0.01` – `2.0` | `0.1` – `0.5` declared; example config varies `0.2` – `0.4` |
| `gamma` | `I → R` | days (`DAYS`) | `10` | `1` – `100` | `5` – `20` declared; **not** varied in the example config |

`gamma` is an average **recovery period in days** — the mean time someone stays infectious — so a larger value means a longer infectious period and more onward transmission. Because the edge is declared as `ValueType.DAYS`, the framework converts the `10` to a `0.1`/day rate at load time. Enter it in days; do not pre-divide.

Intervention (`my_intervention`), which targets `beta`:

| Field | Unit | Default | Uncertainty range in example config |
|---|---|---|---|
| `adherence_min` | % | `50` | `40` – `60` |
| `transmission_percentage` | % | `50` | `45` – `55` |
| `start_date` / `end_date` | date | — | — |

Per-zone inputs under `admin_zones`: `name`, `population` (count), `infected_population` (**percentage** of that zone infected at day 0, `0` – `100`, default `0.05`), `center_lat`, and `center_lon`.

Demographic weights are percentages of the population per age band and must sum to 100: `age_0_4` 6, `age_5_17` 16, `age_18_49` 42, `age_50_64` 19, `age_65_plus` 17.

Simulation-level inputs: `start_date`, `end_date`, and `n_simulations` (parameter draws for an uncertainty run — `20` in the example config, defaulting to 30 when omitted).

## Initial conditions

The base class's default seeding applies, per zone:

- `infected` = `infected_population` ÷ 100 × `population`
- `S` = `population` − `infected`
- `R`, `I_total`, and `R_total` all start at `0`

The example config uses a population of 1,000,000 with `infected_population: 0.01`, which is **0.01 percent** — 100 infected people and 999,900 susceptible. It is not a fraction: writing `0.01` when you meant 1% seeds a hundredth of the infections you intended.

That per-zone total is then divided across the five age bands by their declared weights, so each band starts with its weighted share of both `S` and `I`. Nobody starts immune.

## Outputs

Every simulation produces **two runs**: one with interventions applied and one control run without them (`"control_run": true`). Both are written to the output JSON as a two-element array, which is what makes the "what did the intervention buy us?" comparison possible.

Each run carries `start_date`, `end_date`, `time_steps`, `interventions`, and:

- `admin_zones[].time_series` — one daily series per zone
- `parent_admin_total.time_series` — the whole-population series, summed across zones
- `compartment_deltas` — one summary figure per compartment for the whole run

All values are **people**, at daily resolution, and only `S`, `I`, and `R` are reported — the `_total` accumulators are not surfaced as series.

The shape of each compartment entry depends on the run mode, and the example config's variance flags put this model in the uncertainty mode:

```json
"I": { "median": 1234.5, "lower": 890.1, "upper": 1602.7 }
```

`median` is the 50th percentile across the parameter draws and the bounds are the 2.5th and 97.5th percentiles — a 95% interval describing how much the declared input ranges move the result. It is not a forecast confidence interval and says nothing about whether the model structure is right.

**Age detail is only available in deterministic runs.** With no variance declared anywhere, each compartment is instead keyed by demographic group (`age_all` plus one key per band, with the bands summing to `age_all`). Multi-run output replaces that nesting with the median/interval summary, so an uncertainty run reports population-wide values only — even though the model is age-stratified internally.

`compartment_deltas` mixes two semantics, which is easy to misread. For a compartment that has a `_total` accumulator it reports the **cumulative** figure, so `I` is every infection that occurred over the run and `R` is every recovery. `S` has no accumulator, so it reports the **final-day** susceptible count instead.

To plot a local result file: `python tools/view_results.py results/<file>.json`.

## Model nuances

- **Run mode is inferred, not configured.** Any field with `"has_variance": true` — on an edge, an intervention, or a disease parameter — switches the whole simulation to an uncertainty run with `n_simulations` Latin Hypercube draws. Remove every variance flag and the same model runs deterministically.
- **The intervention's two percentages multiply.** The effect while active is `beta × (1 − adherence × transmission_reduction)`. The defaults of 50% and 50% give a **25%** reduction in transmission, not 50%.
- **The example config's intervention starts after the peak.** With the default parameters the unmitigated epidemic peaks in mid-February, but the intervention window runs 1 March to 1 June. Both runs therefore share an identical peak, and the intervention only trims the tail — around 917,000 cumulative infections against 939,000 for the control. Move the window earlier to see it bite.
- **The contact matrix is loaded for you.** Because all five groups declare an `age_range` and the model sets no overrides, the framework loads a synthetic contact matrix for the ISO3 code parsed from the config's `admin_unit_id`, falling back to that country's income-group average and then to a global average. The example config sets no `admin_unit_id`, so it gets the global-average matrix. Supply a real `admin_unit_id` to get country-specific mixing.
- **Zones do not interact.** No mobility parameters are declared, so the travel matrix is the identity and each zone evolves independently. The `travel_sigma` parameter and `build_travel_matrix()` override are stubbed out in `model.py` if you want to change that.
- **`_total` compartments are excluded from the population.** `equation()` sums only non-`_total` compartments when computing the infectious proportion. Including them would double-count.
- **Compartment order is load-bearing.** `equation()` indexes the state array positionally, so the derivatives are always stacked with `jnp.stack([derivs[c] for c in self.compartment_list])` rather than a hardcoded order.
- **`set_model_metadata()` is editorial only.** Authors, license, assumptions, and the rest land in the artifact JSON for display and have no effect on the simulation.

## Known edge cases

- **`infected_population: 0`** — nothing is seeded, so every series stays flat. The output is valid but empty of dynamics.
- **Demographic weights that do not sum to 100** — the population is distributed by the weights as given, so the totals will not match the zone population you entered. Groups omitted from the config get a weight of zero.
- **`n_simulations: 1` on an uncertainty run** — the median and both bounds collapse onto the single draw, producing a band of zero width that looks deterministic but is not.
- **An uncertainty range wider than the validation range** — `beta` accepts `0.01`–`2.0`, so a variance `min`/`max` outside that window will draw values the schema considers invalid.
- **A config with no interventions** — both runs are then identical, and the control comparison shows no difference. This is expected, not a bug.
- **A zone with population `0`** — the force-of-infection denominator is guarded with a small epsilon, so you get zeros rather than `NaN`.
- **Very short recovery periods with high `beta`** — the epidemic can peak within a few days, and daily output may under-resolve the peak even though the adaptive solver integrates it correctly.

## Differences from the source model

There is no source publication. This is the textbook Kermack–McKendrick SIR model, included as an authoring reference. Three deliberate choices are worth naming:

- **Frequency-dependent transmission** (scaled by `I/N`) rather than the mass-action `beta × S × I` of the original formulation.
- **Age structure and contact mixing** layered on top of a model that is classically unstructured.
- **Uncertainty by uniform sampling** over declared input ranges, rather than analytic sensitivity analysis.

Its parameter defaults are illustrative round numbers and are not fitted to any disease.

## Related models

| Model | Choose it when |
|---|---|
| `example_parameter_uncertainty_custom_model` | Same SIR structure and uncertainty setup, but `equation()` and a gradually ramped intervention are written out by hand. Start here when your dynamics do not fit the declarative edge patterns. |
| `example_stochastic_model` | Chance matters — small populations, possible outbreak extinction, or run-to-run variability. Uses tau-leaping instead of an ODE, and splits infections into asymptomatic and symptomatic. |
| `mpox_jax_model` | You need real inter-zone mobility, a bespoke intervention, or an example of reading a modeler-supplied data file. |
| `covid_jax_model` | You need age-stratified dynamics with multiple structural variants (SIR, SEIR, SIHR, …) generated from one schema. |
| `dengue_jax_model` | Vector-borne transmission, temperature-driven seasonality, or hand-managed `_total` compartments. |
| `hantavirus_jax_model` | Flows that are not simple `rate × source` edges — multi-rate force of infection, births, or density-dependent mortality. |
