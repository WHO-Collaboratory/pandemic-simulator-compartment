# Example Disease (Parameter Uncertainty) — Model Notes

This folder contains a teaching example: a minimal **SIR** model plus one **variant**
that demonstrates how to hand-write your own equations and interventions. Together they
show the two ends of the authoring spectrum — fully declarative vs. fully custom.

## Models in this folder


| Class                                                  | `disease_type`                          | What it demonstrates                                                                     |
| ------------------------------------------------------ | --------------------------------------- | ---------------------------------------------------------------------------------------- |
| `ExampleDiseaseParameterUncertaintyModel` (`model.py`) | `example_disease_parameter_uncertainty` | The standard declarative path: define the schema, let the framework build the equations. |
| `ExampleDiseaseChangeEquation` (`variants.py`)         | `example_disease_ramped_intervention`   | Overriding `equation()` and writing a custom, gradually-ramping intervention by hand.    |


> **Which one runs?** The `disease_type` string in your config's `Disease` block selects
> the class (via the model registry). The bundled `example-config.json` uses
> `example_disease_parameter_uncertainty`, so it runs the **base** model. Change it to
> `example_disease_ramped_intervention` to run the variant.



## Base model — `ExampleDiseaseParameterUncertaintyModel`

A classic three-compartment SIR model:

- **Compartments:** `S` (susceptible), `I` (infected, `infective=True`), `R` (recovered).
- **Transmission:** `S → I` at rate `beta` (frequency-dependent, i.e. scaled by `I/N`),
`I → R` at rate `gamma`.
- **Interventions:** a single `my_intervention` that reduces `beta` while active, applied
through the built-in `_apply_interventions` helper.
- **Demographics:** five age bands (0–4, 5–17, 18–49, 50–64, 65+) with a contact matrix.
- **Parameter uncertainty:** `beta` and `gamma` expose default/min/max ranges so the
simulator can run multiple parameter draws and report a median with an interval.

The `equation()` method leans on framework helpers — `_apply_interventions` and
`_compute_equations` — so it stays short and declarative.

## Variant — `ExampleDiseaseChangeEquation`

Same SIR structure, but written the "manual" way to illustrate customization:

- **Hand-written** `equation()`**:** computes `dS/dt`, `dI/dt`, `dR/dt` directly instead of
calling `_compute_equations`.
- `custom_intervention()`**:** replaces `_apply_interventions`. Instead of switching on/off
instantly, adherence **ramps up linearly** over `ramp_up_days` from the start date, holds
at full effect, then **ramps down linearly** over `ramp_down_days` after the end date
(a trapezoid over time). At full effect it matches the framework formula
`beta * (1 - adherence * transmission_reduction)`.
- **Extra config-driven parameters:** `ramp_up_days` (default 14) and `ramp_down_days`
(default 21), tunable from the config's `Disease` block.



## Nuances users should know

- `_total` **compartments are auto-generated.** The framework adds cumulative `I_total`
and `R_total` compartments for transmission-edge targets. Don't declare them by hand; in
a custom `equation()` you must still initialize them (to zero) and add only their inflows.
- **Units are converted for you.** `gamma` is declared as `DAYS` (a recovery *period*) and
converted to a rate internally. Likewise `PERCENTAGE` params arrive as e.g. `20.0`, not
`0.2` — convert with `self._to_rate(...)` if you use them in math.
- **The ramp parameters only affect the variant.** `ramp_up_days` / `ramp_down_days` are
harmless if present in a config that runs the base model — the base model simply ignores
them.
- **Custom interventions must honor the control run.** `custom_intervention()` checks
`id in self.intervention_dict` so the "without interventions" baseline run correctly
skips it, matching the built-in helper's behavior.
- **Keep custom code JAX-traceable.** `t` is a traced value, so time-dependent logic uses
`jnp.clip` (not Python `if`/comparisons) to build the ramp.
- **Parameter uncertainty is UNIFORM-only.** To vary a parameter, set `has_variance: true`
with a `UNIFORM` distribution and a `min`/`max` in that field's `FieldConfigs`.
- **Travel is identity by default.** No inter-zone mobility unless you declare travel
parameters and override `build_travel_matrix()` (both are commented out in `model.py`).

