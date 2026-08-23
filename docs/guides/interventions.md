# Interventions

An intervention is a control measure that can change the course of a simulated outbreak — distancing, masks, vaccination, case isolation, vector control, a lockdown. This guide covers how a model integrates them and how you configure them in a run.

There are two kinds. A **built-in intervention** is declared in one call and the framework handles the rest. A **custom intervention** is logic you write yourself, for effects the built-in intervention does not express.

## Contents

- [What an intervention can change](#what-an-intervention-can-change)
- [Built-in interventions](#built-in-interventions)
  - [Declaring interventions](#declaring-interventions)
  - [Writing it into `equation()`](#writing-it-into-equation)
  - [How much transmission drops](#how-much-transmission-drops)
  - [Switching it on: date window](#switching-it-on-date-window)
  - [Switching it on: prevalence threshold](#switching-it-on-prevalence-threshold)
  - [Config field reference](#config-field-reference)
  - [Several interventions at once](#several-interventions-at-once)
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

### How much transmission drops

While an intervention is active, each targeted rate becomes:

```
new_rate = rate * (1 - adherence * transmission_reduction)
```

Both percentages are divided by 100 at load, then **multiplied together**. This trips people up: adherence 50% with a 50% reduction is a **25%** drop in transmission, not 50%. Read it as "half the population complies, and those who do cut their transmission in half."

So a `beta` of 0.3 under that intervention becomes `0.3 × (1 − 0.5 × 0.5) = 0.225`.

### Switching it on: date window

The common case — the measure runs between two calendar dates:

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

Choose the dates against the outbreak you are actually simulating. A window that opens after the unmitigated peak will barely change the result, because most infections have already happened.

### Switching it on: prevalence threshold

The alternative is to let the outbreak trigger the response — closer to how measures are introduced in practice. Replace the dates with thresholds:

```json
{
    "Intervention": { "name": "MY_INTERVENTION", "display_name": "My intervention" },
    "adherence_min": 50.0,
    "transmission_percentage": 50.0,
    "start_threshold": 2.0,
    "end_threshold": 1.0
}
```

**Thresholds are percentages of the population currently infectious, not fractions.** `2.0` means 2%. Writing `0.02` means 0.02%, which will fire almost immediately in most outbreaks — a common and easily missed mistake.

The intervention switches on once prevalence reaches `start_threshold`. On stochastic models it then stays on until prevalence falls to `end_threshold`, which stops it flickering on and off around the trigger point. Under `odeint`, the default solver for deterministic models, activation status is not carried between solver steps, so the intervention is simply active whenever prevalence is at or above `start_threshold` and `end_threshold` has no effect.

`example_parameter_uncertainty_declarative_model/example-config.json` is configured this way, with a 2% trigger and a 1% release. Running it shows the effect plainly: peak prevalence falls from about 25% to 19% and arrives ten days later, and cumulative infections drop from roughly 939,000 to 865,000.

### Config field reference

| Field | Unit | Notes |
|---|---|---|
| `Intervention.name` | — | Must match the declared `id` (case-insensitive). |
| `Intervention.display_name` | — | Label for the run; cosmetic. |
| `adherence_min` | percent, 0–100 | Share of the population complying. |
| `transmission_percentage` | percent, 0–100 | Reduction among those complying. Defaults to `5.0` if omitted. |
| `start_date`, `end_date` | `YYYY-MM-DD` | Date window. |
| `start_threshold`, `end_threshold` | percent, 0–100 | Prevalence trigger and release. |

If you supply **none** of the four activation fields, `start_date` falls back to the simulation start date, so the intervention is on for the entire run.

You can set both a date window and thresholds on one intervention. The date window wins while it is open, and the threshold rule can only fire outside it — the reduction is never applied twice in the same step.

> Some older models, such as `ebola_jax_model` and `hantavirus_human_jax_model`, ship configs using a flat `"interventions": [{ "id": ..., ... }]` list instead. The loader still accepts it and converts it to the form above, but write new configs the way shown here.

### Several interventions at once

List as many as you like. They apply in the order the model declares them, and each one scales the rate the previous one produced, so the reductions **compound rather than add**:

```
beta               = 0.300
after 50% × 50%    = 0.300 × 0.75 = 0.225
after 60% × 30%    = 0.225 × 0.82 = 0.185
```

That is a 38% total reduction, not the 25% + 18% = 43% you would get by adding them.

### Restricting travel

Set `modifies_travel=True` for a measure that confines people to their own zone. While it is active the travel matrix becomes the identity matrix, so zones stop seeding each other and each one runs its own local epidemic. `covid_jax_model` declares lockdown as both effects at once:

```python
schema.add_intervention(
    id="lock_down",
    label="Lockdown",
    ...
    target_rates=["beta"],
    modifies_travel=True,
    adherence=80.0,
    transmission_reduction=70.0,
)
```

This only matters for models that **have** mobility, meaning they override `build_travel_matrix()`. Most models leave the travel matrix as the identity already, so `modifies_travel` changes nothing there. Note also that having mobility is not the same as restricting it: `mpox_jax_model` and `ebola_jax_model` both model movement between zones, but neither has an intervention that touches it.

### Uncertainty on intervention settings

Adherence is rarely known in advance, so you can give intervention fields a range instead of a point value and let the simulator sample across it. Add a `FieldConfigs` block to the intervention:

```json
"FieldConfigs": {
    "items": [
        {
            "field_key": "adherence_min",
            "has_variance": true,
            "distribution_type": "UNIFORM",
            "default": 50.0,
            "min": 40.0,
            "max": 60.0
        }
    ]
}
```

`field_key` names the intervention field to vary and `min`/`max` are in the same percent units as the field itself. Any `"has_variance": true` anywhere in the config switches the whole run to uncertainty mode, drawing `n_simulations` parameter sets by Latin Hypercube sampling and reporting a median with a 95% interval instead of a single trajectory. `example_parameter_uncertainty_declarative_model/example-config.json` varies both adherence and the reduction this way.

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

- **Check `id in self.intervention_dict`.** This is how the control run skips the intervention. Miss it and your with- and without-intervention runs come out identical, hiding the effect you were trying to measure.
- **No Python `if` on values that change during the run.** `t`, compartment values, and anything derived from them are traced by JAX, so branch with `jnp.where` or clamp with `jnp.clip` instead. Plain `if` is fine for config values that are fixed before the run starts, such as the `None` checks above.
- **`t` is days since the simulation start, not a date.** Convert with `self.start_date_ordinal + t` before comparing against `intv.start_date_ordinal`.

Two more examples worth reading. `mpox_jax_model.ring_vaccination_intervention()` implements date-window and threshold activation by hand for a measure targeted at the contacts of confirmed cases. `test_klebsiella_amr_model._intervention_multiplier()` applies the same reduction formula to parameters that are not transmission rates, which is how its stewardship intervention acts on antibiotic consumption rates.

## Checking that it worked

Plot the run and compare the two trajectories:

```bash
python tools/view_results.py results/<output>.json
```

The viewer draws the with- and without-intervention runs together and marks each intervention's start and stop dates, so a measure that fired too late is obvious. Threshold-triggered interventions have no fixed date and are noted on the panel instead. See [tools/README.md](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tools/README.md).

For a single number, compare `compartment_deltas` between the two runs — for a compartment with a cumulative accumulator it reports the total over the whole run, so the infected entry gives you total infections averted.

If the two runs look identical, check in this order: the config `name` matches the declared `id`; the activation window or threshold is actually reached during the simulated period; `target_rates` names match rates the model passes to `_apply_interventions()`; and `adherence_min × transmission_percentage` is large enough to matter.
