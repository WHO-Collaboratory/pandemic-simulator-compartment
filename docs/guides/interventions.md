# Interventions

An intervention is a control measure that can change the course of a simulated outbreak — distancing, masks, vaccination, case isolation, vector control, a lockdown. This guide covers how a model integrates them and how you configure them in a run.

There are two kinds. A **built-in intervention** is declared in one call and the framework handles the rest. A **custom intervention** is logic you write yourself, for effects the built-in intervention does not express.

## Contents

- [What an intervention can change](#what-an-intervention-can-change)
- [Built-in interventions](#built-in-interventions)
  - [Declaring interventions](#declaring-interventions)
  - [Writing it into `equation()`](#writing-it-into-equation)
  - [Formula](#formula)
  - [Activating by date window](#activating-by-date-window)
  - [Activating by prevalence threshold](#activating-by-prevalence-threshold)
  - [Config field reference](#config-field-reference)
  - [Multiple interventions at once](#multiple-interventions-at-once)
  - [Restricting travel](#restricting-travel)
  - [Uncertainty on intervention settings](#uncertainty-on-intervention-settings)
- [Custom interventions](#custom-interventions)
- [Checking that it worked](#checking-that-it-worked)

## What an intervention can change

A built-in intervention has two levers:

1. **It scales one or more transmission rates.** Whatever rates the model author lists in `target_rates` are scaled down while the intervention is active. This is how masks, distancing, vaccination, and vector control are represented.
2. **It stops movement between zones.** With `modifies_travel=True`, the travel matrix is replaced by the identity matrix while the intervention is active, so nobody moves between administrative zones. This is only effective if the model has mobility — see [Restricting travel](#restricting-travel).

An intervention never adds a compartment or moves people directly. If you need any of those, write a [custom intervention](#custom-interventions).

Every simulation runs **twice**: once with your interventions and once without. The second is the control run, tagged `"control_run": true` in the output.

## Built-in interventions

### Declaring interventions

The model author declares each available intervention in `define_parameters()`. `example_parameter_uncertainty_declarative_model/model.py` shows the minimal case:

```python
schema.add_intervention(
    id="my_intervention",
    label="My Intervention",
    description="Reduces transmission while active",
    target_rates=["beta"],
    adherence=50.0,
    transmission_reduction=50.0,
)
```

| Argument | Required | Default | Meaning |
|---|---|---|---|
| `id` | yes | — | Machine name. Config entries are matched to it, case-insensitively. |
| `label` | yes | — | Name shown to the user. |
| `description` | yes | — | One line explaining what the measure does. |
| `target_rates` | no | `[]` | Transmission variable names to scale down, e.g. `["beta"]`. |
| `modifies_travel` | no | `False` | Also switch off inter-zone travel while active. |
| `adherence` | no | `50.0` | Starting value for the adherence control, in percent. |
| `transmission_reduction` | no | `5.0` | Starting value for the reduction control, in percent. |

`adherence` and `transmission_reduction` set only the **starting values of the two controls**, not the values the model runs with. Running locally, the numbers in your config file take precedence; in the Pandemic Simulator UI they are the initial position of the sliders on the Simulation Configuration page, and whatever the user selects there takes precedence instead.

So against the declaration above, this config gives a 30% reduction in `beta` rather than the declared 25%:

```json
"adherence_min": 60.0,
"transmission_percentage": 50.0
```

One catch: an omitted field does **not** fall back to the declared default. Leave `adherence_min` out and adherence is read as zero, which leaves the intervention active but completely ineffective, so the run comes out identical to the control.

Each name in `target_rates` must match the `variable_name` of a transmission parameter *and* be passed into `_apply_interventions()` by the model. A name that matches nothing is skipped silently, so a typo gives you an intervention that does nothing without any error. Declaring an intervention with an empty `target_rates` is only sensible when it is travel-only.

An intervention can target several rates at once. `example_stochastic_model` scales both the asymptomatic and symptomatic transmission rates with `target_rates=["beta", "beta_sym"]`, and `ebola_seihfr_burial_legrand_model` declares three separate interventions — community, hospital, and funeral — each aimed at its own route.

### Writing it into `equation()`

Interventions only take effect if the model calls `_apply_interventions()`, passing the rates that may be modified:

```python
rates, self.travel_matrix = self._apply_interventions(
    t, {"beta": params["beta"]}, prop_infective
)
```

It returns the scaled rates and the updated travel matrix. The third argument is the current proportion of the population that is infectious, needed for threshold activation. When no interventions are configured — as in the control run — it keeps the rates and travel matrix exactly as it received them, leaving the run unchanged. Models without mobility can discard the travel matrix, as `ebola_jax_model` does with `rates, _ = self._apply_interventions(...)`.

### Formula

While an intervention is active, each targeted rate becomes:

```
new_rate = rate * (1 - adherence * transmission_reduction)
```

Both percentages are divided by 100 to convert them to a decimal, then **multiplied together**. For example, adherence at 50% with a 50% reduction is a **25%** drop in transmission. Interpret it as "half the population complies, and those who do cut their transmission in half."

So an original `beta` of 0.3 under that intervention becomes `0.3 × (1 − 0.5 × 0.5) = 0.225`.

### Activating by date window

The intervention runs between two calendar dates:

```json
"Interventions": {
    "items": [
        {
            "Intervention": { "name": "MY_INTERVENTION", "display_name": "My intervention" },
            "adherence_min": 50.0,
            "transmission_percentage": 50.0,
            "start_date": "2026-03-01",
            "end_date": "2026-06-01"
        }
    ]
}
```

The rate is reduced from `start_date` through `end_date` and snaps back to baseline outside that window. Omit `end_date` and the intervention runs to the end of the simulation. `covid_jax_model/example-config.json` uses this form for mask wearing and social isolation.


### Activating by prevalence threshold

The alternative is to let the outbreak trigger the response. Replace the dates with thresholds:

```json
{
    "Intervention": { "name": "MY_INTERVENTION", "display_name": "My intervention" },
    "adherence_min": 50.0,
    "transmission_percentage": 50.0,
    "start_threshold": 2.0,
    "end_threshold": 1.0
}
```

**Thresholds are percentages of the population currently infectious, not fractions.** `2.0` means 2% and `0.02` means 0.02%.

The intervention switches on once prevalence reaches `start_threshold`. On stochastic models it then stays on until prevalence falls to `end_threshold`, which stops it flickering on and off around the trigger point. Under the default solver for deterministic models, activation status is not carried between solver steps, so the intervention is active whenever prevalence is at or above `start_threshold`.

`example_parameter_uncertainty_declarative_model/example-config.json` is configured this way, with a 2% trigger and a 1% release. Running it shows the effect plainly: peak prevalence falls from about 25% to 19% and arrives ten days later, and cumulative infections drop from roughly 939,000 to 865,000.

### Config field reference

These are the fields you write by hand into a model's `example-config.json` for a local run. Running through the Pandemic Simulator UI, the same values come from the intervention's on-screen controls and you do not edit any of this yourself.

Here is one entry with every field set:

```json
{
    "Intervention": { "name": "MY_INTERVENTION", "display_name": "My intervention" },
    "adherence_min": 50.0,
    "transmission_percentage": 50.0,
    "start_date": "2026-03-01",
    "end_date": "2026-06-01",
    "start_threshold": 2.0,
    "end_threshold": 1.0
}
```

| Field | Unit | Notes |
|---|---|---|
| `Intervention.name` | — | Must match the declared `id` (case-insensitive). |
| `Intervention.display_name` | — | Label for the run; cosmetic. |
| `adherence_min` | percent, 0–100 | Share of the population complying. |
| `transmission_percentage` | percent, 0–100 | Reduction among those complying. Defaults to `5.0` if omitted. |
| `start_date`, `end_date` | `YYYY-MM-DD` | Date window. |
| `start_threshold`, `end_threshold` | percent, 0–100 | Prevalence trigger and release. |

If you supply **none** of the four activation fields, `start_date` falls back to the simulation start date, so the intervention is on for the entire run.

You can set both a date window and thresholds on one intervention. At each step the date window is checked first and reduces the rate if the current day falls inside it. The threshold is checked second and reduces the rate only on the steps where the date window did not. So the threshold covers the stretches outside the window, and the reduction is never applied twice in the same step.

> Some older models, such as `ebola_jax_model` and `hantavirus_human_jax_model`, have configs using a flat `"interventions": [{ "id": ..., ... }]` list instead. The loader still accepts it and converts it to the form above, but write new configs the way shown here.

### Multiple interventions at once

Interventions are applied in the order the model declares them, and each one scales the rate the previous one produced, so the reductions **compound rather than add**.

`covid_jax_model/example-config.json` is the worked example: mask wearing at 20% adherence and a 35% reduction, then social isolation at 40% and 50%, both aimed at `beta` over the same two-month window. Against that model's `beta` of 0.25:

```
new_rate = rate * (1 - adherence * transmission_reduction)

mask wearing      0.25   × (1 - 0.20 × 0.35) = 0.25   × 0.93 = 0.2325
social isolation  0.2325 × (1 - 0.40 × 0.50) = 0.2325 × 0.80 = 0.186
```

`beta` therefore runs at 0.186 while both are active. Since `0.186 ÷ 0.25 = 0.744`, 74.4% of the original transmission rate survives and 25.6% has been removed.


### Restricting travel

For a measure that confines people to their own zone, add `modifies_travel=True` to the `schema.add_intervention()` call in your model's `model.py` — the same declaration shown in [Declaring interventions](#declaring-interventions), which `example_parameter_uncertainty_declarative_model/model.py` demonstrates. This is a modeling decision rather than a config field, so it cannot be switched on for a model that does not declare it.

While the intervention is active the travel matrix becomes the identity matrix, so people stop traveling outside of their zone and each one runs its own local epidemic. `covid_jax_model` uses both levers in one intervention, cutting `beta` through `target_rates` and halting travel through `modifies_travel`:

```python
schema.add_intervention(
            id="lock_down",
            label="Lockdown",
            description="Severe movement restrictions: halts inter-regional travel and reduces transmission through reduced social contact",
            target_rates=["beta"],
            modifies_travel=True,
            adherence=80.0,
            transmission_reduction=70.0,
)
```

This only matters for models that **have** mobility, meaning they override `build_travel_matrix()` — see [mobility.md](./mobility.md) for how to build one. If your model already uses the identity matrix (there is no travel), `modifies_travel` changes nothing there. Note also that having mobility is not the same as restricting it: `mpox_jax_model` and `ebola_jax_model` both model movement between zones, but neither has an intervention that touches it.

### Uncertainty on intervention settings

Adherence is rarely known in advance, so you can give intervention fields a range instead of a point value and let the simulator sample across it. Add a `FieldConfigs` block to the intervention; [uncertainty-quantification.md](./uncertainty-quantification.md#on-interventions) covers what the sampling does and how to read the resulting bands. This is the complete `Interventions` block from `example_parameter_uncertainty_declarative_model/example-config.json`, which varies both adherence and the reduction:

```json
"Interventions": {
    "items": [
        {
            "Intervention": { "name": "MY_INTERVENTION", "display_name": "My intervention" },
            "adherence_min": 50.0,
            "transmission_percentage": 50.0,
            "start_threshold": 2.0,
            "end_threshold": 1.0,
            "FieldConfigs": {
                "items": [
                    {
                        "field_key": "adherence_min",
                        "has_variance": true,
                        "distribution_type": "UNIFORM",
                        "default": 50.0,
                        "min": 40.0,
                        "max": 60.0
                    },
                    {
                        "field_key": "transmission_percentage",
                        "has_variance": true,
                        "distribution_type": "UNIFORM",
                        "default": 50.0,
                        "min": 45.0,
                        "max": 55.0
                    }
                ]
            }
        }
    ]
}
```

`field_key` names the intervention field to vary and `min`/`max` are in the same percent units as the field itself, so adherence here is drawn between 40% and 60%. The `adherence_min` and `transmission_percentage` values above still matter: they are what the run uses if you turn the variance off. Any `"has_variance": true` anywhere in the config switches the whole run to uncertainty mode, drawing `n_simulations` parameter sets by Latin Hypercube sampling and reporting a median with a 95% simulation-based interval instead of a single trajectory.

## Custom interventions

Write one when the built-in behaviour cannot express what you need — for example a measure that phases in gradually rather than switching on instantly, coverage that depends on the current state of the outbreak, or an effect on something other than a transmission rate.

Still declare the intervention with `schema.add_intervention()`, since that is what creates the user-facing controls and lets the config validate. Then skip `_apply_interventions()` and apply your own logic in `equation()`.

`example_parameter_uncertainty_custom_model` is the reference example. Its `custom_intervention()` ramps the effect up over `ramp_up_days`, holds it through the window, then releases it over `ramp_down_days`:

```python
def custom_intervention(self, t, beta):
    intv = next(
        (
            i
            for i in self.interventions
            if i.id == "my_intervention" and i.id in self.intervention_dict
        ),
        None,
    )
    if intv is None or intv.start_date_ordinal is None:
        return beta  # intervention not configured — leave beta unchanged

    current_ordinal_day = self.start_date_ordinal + t

    ramp_in = jnp.clip(
        (current_ordinal_day - intv.start_date_ordinal) / self.ramp_up_days, 0.0, 1.0
    )
    ...
    full_reduction = intv.adherence * intv.transmission_reduction
    return beta * (1.0 - ramp * full_reduction)
```

and calls it in place of the helper:

```python
beta = self.custom_intervention(t, params["beta"])
```

Three rules to follow:

- **Gate on `self.intervention_dict`.** The runner builds the control run by emptying that dictionary, so this check is what makes the "without interventions" run skip yours. In the example above it is the `and i.id in self.intervention_dict` clause, paired with the `if intv is None` early return. `mpox_jax_model` does the same thing more directly, reading the config entry and returning early when it is absent: `cfg = self.intervention_dict.get("ring_vaccination")` then `if cfg is None: return beta, self.intervention_statuses`. Miss this and both runs come out identical, hiding the effect you were trying to measure.
- **No Python `if` on values that change during the run.** `t`, compartment values, and anything derived from them are traced by JAX, so choose between two values with `jnp.where`, or hold a value inside a range with `jnp.clip`, instead. Plain `if` is fine for config values that are fixed before the run starts, such as the `None` checks above.
- **`t` is days since the simulation start, not a date.** Convert with `self.start_date_ordinal + t` before comparing against `intv.start_date_ordinal`.

Two more examples worth reading. `mpox_jax_model.ring_vaccination_intervention()` implements date-window and threshold activation by hand for a measure targeted at the contacts of confirmed cases. `test_klebsiella_amr_model._intervention_multiplier()` applies the same reduction formula to parameters that are not transmission rates, which is how its stewardship intervention acts on antibiotic consumption rates.

## Checking that it worked

After running your model locally (see [Run the model locally](./model-integration-documentation.md#run-the-model-locally)), plot the run and compare the two trajectories by replacing `<output>` with the name of the JSON results file you generated:

```bash
python tools/view_results.py results/<output>.json
```


If the two runs look identical, check in this order: the config `name` matches the declared `id`; the activation window or threshold is actually reached during the simulated period; `target_rates` names match rates the model passes to `_apply_interventions()`; and `adherence_min × transmission_percentage` is large enough to matter.

---

**Last Updated:** August 24, 2026
