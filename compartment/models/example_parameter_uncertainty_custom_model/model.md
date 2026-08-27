# Example Disease (Custom Equation + Parameter Uncertainty) — Model Notes

A teaching example: a minimal **SIR** model authored the **custom** way — the
`equation()` method builds the derivatives by hand and applies a
**hand-written, gradually ramped intervention** instead of the framework's
built-in helper. It also demonstrates **parameter uncertainty**, where the simulator
runs many parameter draws and reports a median with an interval band. It is a
reference implementation for authoring custom-equation models, not a calibrated
model of any real disease.

This is the custom-equation counterpart to
`example_parameter_uncertainty_declarative_model`: same SIR structure and
parameter-uncertainty setup, but here the equations and intervention are written
out explicitly, and this model includes **five age demographic groups** with
contact-matrix mixing — the declarative example does not.

## What it does

A classic three-compartment SIR model, age-stratified across five bands:

- **Compartments:** `S` (susceptible), `I` (infected, `infective=True`),
  `R` (recovered), plus auto-generated cumulative `I_total` and `R_total`
  trackers.
- **Demographics:** five age groups declared in `define_parameters()` —
  `age_0_4`, `age_5_17`, `age_18_49`, `age_50_64`, `age_65_plus` — with
  default weights 6, 16, 42, 19, and 17. Because every group declares an
  `age_range`, the framework auto-loads a synthetic Prem 2021 contact matrix
  (global average when the config sets no `admin_unit_id`).
- **State shape:** *(compartments × age groups × zones)*.
  `prepare_initial_state()` calls `_prepare_demographic_state()` to split each
  zone's population across the five bands before the solver starts.
- **Transmission:** `S → I` is frequency-dependent and **age-mixed** —
  `equation()` computes the infectious fraction per age group and zone, applies
  the contact matrix, then multiplies by `S`. Recovery `I → R` is at rate
  `gamma` (declared as a `DAYS` period and converted to a rate internally).
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

**Demographic weights** (percent of population per age band, must sum to 100):
`age_0_4` 6, `age_5_17` 16, `age_18_49` 42, `age_50_64` 19, `age_65_plus` 17.
Omit a `demographics` block from the config and these schema defaults apply.

## Strengths

- **Transparent, hand-written equations** — the full SIR dynamics are spelled
  out in `equation()`, making this a clear template for models whose dynamics do
  not fit the declarative edge/FOI patterns.
- **Age-stratified contact mixing in custom code** — shows how to wire
  `self.contact_matrix` into a hand-written force of infection, the same pattern
  used in `covid_jax_model` but without `_compute_equations`.
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
- **No inter-zone mobility** — the travel matrix is the identity; zones evolve
  independently unless you uncomment and wire up `travel_sigma`.
- **Uncertainty runs report population-wide values only** — even though the
  model is age-stratified internally, multi-run output collapses to median and
  interval summaries rather than per-band series.
- **More authoring surface than the declarative version** — writing `equation()`
  and `custom_intervention()` by hand means more code to maintain and more places
  for compartment-ordering or intervention-window mistakes.
- **Illustrative parameter defaults** — not calibrated to any real disease.
