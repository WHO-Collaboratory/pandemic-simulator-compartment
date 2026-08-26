# Ebola (SEIHFR, Community/Hospital/Burial) — Legrand 2007 [stochastic]

A **stochastic** SEIHFR model of Ebola virus disease — the demographic-noise
counterpart of the deterministic `ebola_test` model. Both are ported from
`ebola_SEIHFR_model.ipynb` and based on Legrand et al. (2007), which represents
the three settings in which Ebola spreads with distinct transmission
intensities: the **community**, the **hospital**, and **traditional burial** of
the deceased.

The paper's model is itself stochastic — *"Simulations of the model were
performed using Gillespie's first reaction method"* — so this model restores the
stochasticity that the deterministic ODE reduction averages away. The most
important thing it adds is **variability**: a distribution of outbreak sizes and
the chance that a small introduction fades out, neither of which the
deterministic model can show.

## Model structure

Identical compartments, force of infection, routing algebra, parameters and
interventions as `ebola_test` (see that model's `model.md`). In brief:

- **Compartments:** `S`, `E`, `I` (community-infectious), `H` (hospitalised),
  `F` (dead, awaiting burial — infectious), `R` (removed). Cumulative trackers
  `E_total, I_total, H_total, F_total, R_total`; `I_total` is cumulative symptom
  onsets (the notebook's `C`).
- **Force of infection (frequency-dependent):** `λ = (βI·I + βH·H + βF·F) / N`.
- **Progression:** `S → E → I`, then `I` splits to `H`/`R`/`F`, `H` splits to
  `R`/`F`, `F → R`. The split rates `θ1, δ1, δ2` are derived at each step from
  `θ`, `δ` and the raw durations, exactly as in the deterministic model, so the
  realised hospitalisation proportion and case-fatality ratio reproduce the
  observed `θ = 80%` and `δ = 81%`.

## How the stochasticity is implemented

The Pandemic Simulator integrates on a **fixed daily step**, so it cannot run
the paper's exact event-driven Gillespie algorithm (which advances by variable,
continuous inter-event times). Setting `STOCHASTIC = True` switches the framework
to fixed-step Euler and runs many trajectories; this model's `equation()`
supplies a **chain-binomial tau-leap** for each daily step:

- For every compartment, the number of individuals making a transition is a
  **binomial** draw with the exact competing-risk probability
  `1 − exp(−hazard)` over the day.
- Individuals leaving a compartment that has several possible destinations
  (`I → H/R/F`, `H → R/F`) are split **multinomially** in the ratio of the
  destination hazards.

Because every draw is bounded by its source compartment, the population is
**conserved exactly** and no compartment can go negative. Because the split uses
the destination-hazard ratios, the routing proportions `θ1, δ1, δ2` are
preserved on expectation — so the realised CFR and hospitalisation fraction
converge to `δ` and `θ`, and the ensemble-mean trajectory matches the
deterministic model.

The binomial draws use NumPy (`numpy.random.Generator.binomial`), which is
exact. The framework's Euler path runs this model in NumPy end-to-end, so no JAX
is involved. (An earlier JAX-based draft used `jax.random.binomial`, whose ~1%
per-draw bias compounds over the ~10³ draws in a run and inflated outbreak size
by ~20%; the NumPy version matches the deterministic ODE and the exact-Gillespie
reference.)

### Tau-leap vs. exact Gillespie

This framework model and the paper are **not the same algorithm**, and it matters
in one regime. The chain-binomial tau-leap evaluates a full day's hazards from
the start-of-day state; the exact Gillespie method resolves every event in
continuous time. For the trajectory of an established outbreak the two agree, but
for a **small introduction the daily step understates the fine-grained
early-extinction probability**: with the DRC seed of three cases, the tau-leap
gives on the order of ~2–3% fadeout where the exact Gillespie gives ~9%. If the
fadeout probability of a small introduction is the quantity of interest, use the
standalone exact Gillespie first-reaction implementation
(`ebola_seihfr_gillespie.py` / `ebola_SEIHFR_stochastic.ipynb`) rather than this
framework model. For outbreak-size variability and intervention comparison on an
established epidemic, this model is appropriate.

## Output

With `STOCHASTIC = True` the framework runs `num_runs` independent trajectories
and reports, for every compartment at every timestep, the **median** with a
**95% interval band** (`lower`/`upper`). It also emits the usual automatic
no-intervention **control run** (`control_run: true`). `num_runs` defaults to 100
(class `NUM_RUNS`) and is configurable in the disease config; lower it for a
quick look or raise it for smoother bands (roughly 1 s per trajectory per
scenario locally, so ~3 min for the default 100 runs × 2 scenarios).

Reproducibility: each trajectory is seeded from system entropy so the runs
differ. Passing a fixed `seed` in the config makes **every** trajectory identical
and collapses the band — use it only for a single reproducible trajectory.

## Defaults and data — DRC (Kikwit) 1995

Same as `ebola_test`, at full β precision:
`N = 200,000`, seed `I0 = 3` (`infected_population = 0.0015`, a **percentage**),
`βI, βH, βF = 0.084, 0.1134286, 1.0932857 /day` (weekly Table 4 ÷ 7),
incubation 7 d, onset→hospitalisation 5 d, community infectious period 10 d,
onset→death 9.6 d, death→burial 2 d, `θ = 80%`, `δ = 81%`. Pre-intervention
`R0 ≈ 2.69` (burial-dominant: community 0.499 + hospital 0.424 + funeral 1.771).
The example scenario applies hospital isolation at week 4, safe burial at week 5,
and a 50% community-transmission reduction at week 7.

Verified behaviour (DRC controlled scenario; driver run with 100 trajectories
and an independent 120-run harness): median cumulative onsets ≈ 66–72 with a
wide band (~10–160), median deaths ≈ 56–62 — bracketing the deterministic 66
onsets / ~56 deaths, and the takeoff-mean (~73 onsets) also agrees with the
exact-Gillespie reference. The realised case-fatality ratio (≈ 0.808) and
hospitalisation fraction (≈ 0.797) converge to `δ = 0.81` and `θ = 0.80`; the
population is conserved exactly and no compartment goes negative on any
trajectory.

Other outbreak presets from the source notebook (alternative DRC parameter sets,
Isiro 2012 Bundibugyo EBOV, Gulu 2000 Sudan EBOV, Bundibugyo 2007–08) can be
reproduced by editing the parameter values in the configuration.

## Known framework behaviours affecting output

Same as `ebola_test` (they are framework properties, not model bugs):

- Cumulative `_total` columns are stripped from the per-timestep time series;
  their final medians survive in `compartment_deltas`, **re-keyed onto the base
  compartment names** (so `compartment_deltas["F"]` is cumulative deaths, not the
  `F` compartment, and `compartment_deltas["I"]` is cumulative onsets).
- Horizon endpoint depends on the framework version: older builds stopped one
  day short of `end_date` (last row 1995-06-29), current builds include the
  endpoint (176 rows, last row 1995-06-30). Immaterial for the resolved
  controlled scenario either way.
- The no-intervention counterfactual is generated automatically as a second
  result with `control_run: true`.

## Limitations

- Homogeneous mixing in a single closed population; no age or spatial structure,
  no background births or deaths.
- Interventions are step functions (fully efficient from their start date), not
  gradual ramps.
- **Daily-step tau-leap**, not exact Gillespie — understates the early-extinction
  probability of small introductions (see "Tau-leap vs. exact Gillespie").
- **The control run is not an epidemiological forecast.** Legrand et al. fitted
  these β values with a stochastic model; running that parameterisation as an
  unbounded epidemic yields a ~69% attack rate in Kikwit, where the real 1995
  outbreak had 315 cases. Additionally, because the counterfactual is still
  growing at the horizon, its cumulative figures are sensitive to step
  discretisation and differ from the deterministic model by more than the
  controlled scenario does. Treat it as a relative baseline only.
- Intended for scenario exploration, not real-time forecasting.
