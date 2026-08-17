# Example Disease (Custom Equation + Parameter Uncertainty) — Model Notes

A teaching example: a minimal **SIR** model authored the **custom** way — the
`equation()` method builds the derivatives by hand and applies a
**hand-written, gradually ramped intervention** instead of the framework's
built-in helper. It also demonstrates parameter uncertainty, where the simulator
runs many parameter draws and reports a median with an interval band. It is a
reference implementation for authoring custom-equation models, not a calibrated
model of any real disease.

This is the custom-equation counterpart to the
`example_parameter_uncertainty_declarative_model`: same SIR structure and
parameter-uncertainty setup, but here you can see exactly what the equations and
intervention do because they are written out explicitly.

## What it does

A classic three-compartment SIR model:

- **Compartments:** `S` (susceptible), `I` (infected, `infective=True`),
  `R` (recovered), plus auto-generated cumulative `I_total` and `R_total`
  trackers.
- **Transmission:** `S → I` at rate `beta`, frequency-dependent (scaled by
  `I/N`); `I → R` at recovery rate `gamma` (declared as a `DAYS` period and
  converted to a rate internally).
- **Custom equation:** `equation()` computes the SIR derivatives explicitly
  rather than delegating to `_compute_equations`, and seeds every compartment
  (including the `_total` trackers) at zero before stacking in
  `compartment_list` order.
- **Custom ramped intervention:** `custom_intervention()` replaces the built-in
  `_apply_interventions` helper. Instead of switching on/off instantly it ramps
  adherence **linearly up over `ramp_up_days`**, holds at full effect, then
  **ramps back down over `ramp_down_days`** after the end date — producing a
  trapezoidal effect over time. Full effect matches the framework formula
  `beta * (1 - adherence * transmission_reduction)`, scaled by the ramp.
- **Parameter uncertainty:** `beta` exposes default/min/max ranges and is driven
  with `has_variance: true` (UNIFORM) in the config, so the simulator runs
  multiple parameter draws and reports a median with an interval band.

## Parameters

- **`beta`** — transmission rate (`S → I`), default `0.3`, varied uniformly in
  the example config.
- **`gamma`** — recovery period in days (`I → R`), default `10` days.
- **`ramp_up_days`** — days for the intervention to climb from baseline to full
  adherence (default `14`).
- **`ramp_down_days`** — days for adherence to fall back to baseline after the
  intervention ends (default `21`).
  

## Strengths

- **Transparent, hand-written equations** — the full SIR dynamics are spelled
  out in `equation()`, making this a clear template for models whose dynamics do
  not fit the declarative edge/FOI patterns.
- **Realistic intervention timing** — the trapezoidal ramp captures gradual
  adoption and relaxation of measures rather than an unrealistic step change.
- **JAX-traceable custom logic** — the intervention uses `jnp.clip` on the
  traced time value, so hand-written time-dependent behavior stays compatible
  with the `odeint` solver.
- **Uncertainty quantification built in** — reports a median with an interval
  band across parameter draws instead of a single deterministic curve.
- **Control run handled correctly** — the intervention lookup checks
  `id in self.intervention_dict`, so the "without interventions" run skips it
  just like the built-in helper.

## Limitations

- **Closed population** — no births, deaths, or importation.
- **Permanent immunity** — no waning (`R` never returns to `S`).
- **UNIFORM-only uncertainty** — parameter variance uses a uniform distribution
  with a `min`/`max` per field; other distributions are not modeled here.
- **No age structure or spatial travel** by default — the travel matrix is the
  identity, and the demographic/contact-matrix hooks are stubbed (commented out)
  in `model.py`.
- **More authoring surface than the declarative version** — writing `equation()`
  and `custom_intervention()` by hand means more code to maintain and more places
  for compartment-ordering or intervention-window mistakes.
- **Illustrative parameter defaults** — not calibrated to any real disease.