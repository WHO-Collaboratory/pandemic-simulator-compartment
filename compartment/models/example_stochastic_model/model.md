# Example Disease (Stochastic) — Model Summary

A teaching example of a **stochastic SIR** model. It splits the infectious
population into **asymptomatic** and **symptomatic** compartments and uses
**tau-leaping** to add demographic randomness, running many trajectories to
report a median with an uncertainty band. It is intended as a reference
implementation for authoring stochastic models in the framework rather than as a
calibrated model of any real disease.

## Model structure

- **Compartments:** `S`, `A` (asymptomatic, infectious), `Sym` (symptomatic,
  infectious), `R`, plus a cumulative `I_total` tracker and `R_total`. `A` and
  `Sym` are combined into one **Infected** curve for graphing via
  `COMPARTMENT_DELTA_GROUPING`, and `I_total` feeds the cumulative "total
  infected" figure.
- **Stochastic integration (`STOCHASTIC = True`):** integrated with fixed-step
  Euler; each step draws new infections and recoveries from Poisson
  distributions (tau-leaping) around the deterministic rates.
- **Transmission:** frequency-dependent force of infection (`beta * S * I/N`),
  where `I` is the sum of both infectious compartments. Recovery at rate
  `gamma`.
- **Asymptomatic split:** a configurable fraction (`asymptomatic_fraction`,
  default 40%) of new infections are asymptomatic; the rest are symptomatic.
- **Intervention:** `my_intervention` reduces `beta` while active, applied
  through the built-in `_apply_interventions` helper.
- **Runs:** `num_runs` (default 30) controls how many trajectories are averaged
  into the median + interval band.

## Strengths

- **Captures demographic stochasticity** — Poisson event draws produce
  run-to-run variability, so the model can represent chance extinction, delayed
  take-off, and outbreak-size spread that a deterministic ODE model cannot.
- **Uncertainty quantification built in** — reports a median with an interval
  band across `num_runs` trajectories instead of a single deterministic curve.
- **Same expected trajectory as the ODE version** — tau-leaping means are the
  deterministic rates, so the average behavior stays interpretable.
- **Reproducible** — pass a `seed` in the config for deterministic trajectories;
  otherwise it seeds from system entropy so each run differs.
- **Symptom-status detail** — separate asymptomatic/symptomatic compartments
  allow questions about the role of silent transmission while still graphing a
  single combined "Infected" curve.
- **Clear authoring reference** — hand-written `equation()`, a custom
  `get_initial_population()` seeding two infectious compartments, suppressed
  per-edge `_total` compartments with a hand-declared aggregate, and PRNG
  handling make it a good template for new stochastic models.

## Limitations

- **Closed population** — no births, deaths, or importation.
- **Permanent immunity** — no waning (`R` never returns to `S`).
- **Equal infectiousness** — asymptomatic and symptomatic cases transmit at the
  same rate, and the same `gamma` governs recovery for both.
- **Fixed asymptomatic fraction** — the split does not vary by age or over time,
  and the split of new infections between `A` and `Sym` is applied
  deterministically rather than drawn stochastically.
- **Approximate tau-leaping** — Poisson means use per-day rates scaled by the
  step size, so results are approximate when the step is well below one day.
- **No age structure or spatial travel** by default (identity travel matrix).
- **Illustrative parameter defaults** — not calibrated to any real disease.

## Other notes

- **Solver:** `STOCHASTIC = True` routes the model through the fixed-step Euler
  integrator in the `SimulationManager` instead of the adaptive `odeint` solver.
- **Config:** see `example-config.json` in this directory for a runnable local
  configuration. Optional spatial-travel and age-stratified demographics hooks
  are stubbed (commented out) in `model.py`.
