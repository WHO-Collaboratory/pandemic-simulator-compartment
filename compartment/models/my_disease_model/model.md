# Example Disease (Declarative Parameter Uncertainty) — Model Notes

This folder contains a teaching example: a minimal **SIR** model authored the
**declarative** way — you define the schema and let the framework build the
equations. It also demonstrates parameter uncertainty, where the simulator runs
many parameter draws and reports a median with an interval.


## What it does

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

## Nuances users should know

- **`_total` compartments are auto-generated.** The framework adds cumulative `I_total`
  and `R_total` compartments for transmission-edge targets — don't declare them by hand.
- **Units are converted for you.** `gamma` is declared as `DAYS` (a recovery *period*) and
  converted to a rate internally. Likewise `PERCENTAGE` params arrive as e.g. `20.0`, not
  `0.2` — convert with `self._to_rate(...)` if you use them in math.
- **Parameter uncertainty is UNIFORM-only.** To vary a parameter, set `has_variance: true`
  with a `UNIFORM` distribution and a `min`/`max` in that field's `FieldConfigs`.
- **Travel is identity by default.** No inter-zone mobility unless you declare travel
  parameters and override `build_travel_matrix()` (both are commented out in `model.py`).
