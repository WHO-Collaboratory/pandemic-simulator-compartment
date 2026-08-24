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
- **Differential infectiousness:** the two infectious compartments transmit at
  their own rates, and each contributes to a single frequency-dependent force of
  infection in proportion to its own prevalence:
  `FOI = beta * A/N + beta_sym * Sym/N`. With `beta` at `0.12` and `beta_sym` at
  `0.18`, a symptomatic case is 1.5x as infectious as an asymptomatic one.
- **Independent recovery:** each infectious compartment recovers on its own
  clock — `gamma` for `A → R` and `gamma_sym` for `Sym → R`, both declared as
  `DAYS` and defaulting to 10 days. Equal defaults reproduce a single shared
  recovery period, so the two rates only diverge once you change one.
- **Asymptomatic split:** a configurable fraction (`asymptomatic_fraction`,
  default 40%) of new infections are asymptomatic; the rest are symptomatic.
  Whether a new case is asymptomatic is independent of who infected it, so the
  split is applied to the infection total rather than by source compartment.
- **Effective transmission depends on the case mix.** Because the rates are
  prevalence-weighted, the aggregate rate is roughly
  `0.4 * 0.12 + 0.6 * 0.18 = 0.156`/day at the default 40/60 mix — about half
  what a single `0.3` rate would give, so this configuration produces a
  noticeably slower epidemic than a plain SIR with `beta = 0.3`.
- **Intervention:** `my_intervention` reduces **both** `beta` and `beta_sym`
  while active, applied through the built-in `_apply_interventions` helper.
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
- **`asymptomatic_fraction` sets incidence, not prevalence** — it fixes the
  asymptomatic share of *new* cases, but the standing share of infectious people
  also depends on how long each compartment stays infectious. Whenever `gamma`
  and `gamma_sym` differ, the observed mix drifts away from the configured
  fraction: at `gamma_sym = 20` days the asymptomatic share of prevalence falls
  to about 28% even though 40% of new cases are still asymptomatic.
- **Fixed asymptomatic fraction** — the split does not vary by age or over time,
  and is applied deterministically to the infection total rather than drawn
  stochastically per case.
- **The two rates are not interchangeable with a single beta** — because they are
  weighted by prevalence, the effective transmission rate moves with the case
  mix. Changing `asymptomatic_fraction` or either recovery period therefore
  shifts overall transmissibility too, not just the composition of cases. The
  model is sensitive to this: at the default rates, moving `gamma_sym` from 10 to
  20 days takes the final attack rate from 62% to 91%, while dropping it to 4
  days pushes the outbreak below threshold so it dies out entirely.
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
