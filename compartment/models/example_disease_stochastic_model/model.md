# Example Disease (Stochastic) — Model Summary

A teaching example of a **stochastic SIR** model. It splits the infectious
population into **asymptomatic** and **symptomatic** compartments and uses
tau-leaping to add demographic randomness, running many trajectories to report
a median with an uncertainty band.

## Key details

- **Compartments:** `S`, `A` (asymptomatic, infectious), `Sym` (symptomatic,
infectious), `R`. `A` and `Sym` are combined into one **Infected** curve for
graphing via `COMPARTMENT_DELTA_GROUPING`.
- **Stochastic (**`STOCHASTIC = True`**):** integrated with fixed-step Euler; each
step draws new infections and recoveries from Poisson distributions
(tau-leaping) around the deterministic rates.
- **Transmission:** frequency-dependent force of infection (`beta * S * I/N`),
where `I` is the sum of both infectious compartments. Recovery at rate `gamma`.
- **Asymptomatic split:** a configurable fraction (`asymptomatic_fraction`,
default 40%) of new infections are asymptomatic; the rest are symptomatic.
- **Intervention:** `my_intervention` reduces `beta` while active, applied
through the built-in `_apply_interventions` helper.
- **Runs:** `num_runs` (default 30) controls how many trajectories are averaged.



## Limitations

- **Closed population** — no births, deaths, or importation.
- **Permanent immunity** — no waning (`R` never returns to `S`).
- **Equal infectiousness** — asymptomatic and symptomatic cases transmit at the
same rate, and the same `gamma` governs recovery for both.
- **Fixed asymptomatic fraction** — the split does not vary by age or over time.
- **Approximate tau-leaping** — Poisson means use per-day rates scaled by the
step size, so results are approximate when the step is well below one day; the
asymptomatic/symptomatic split of new infections is deterministic, not drawn.
- **No age structure or spatial travel** by default (identity travel matrix).
- Illustrative parameter defaults — not calibrated to any real disease.

