# Ebola (SEIHFR, Community/Hospital/Burial) — Legrand 2007

A deterministic **SEIHFR** model of Ebola virus disease, ported from the
`ebola_SEIHFR_model.ipynb` notebook and based on Legrand et al. (2007). It
represents the three settings in which Ebola spreads with distinct
transmission intensities: the **community**, the **hospital**, and
**traditional burial** of the deceased.

## Model structure

- **Compartments:** `S` (susceptible), `E` (exposed / incubating, not yet
  infectious), `I` (infectious in the community), `H` (hospitalised,
  infectious), `F` (deceased awaiting burial, infectious during funeral rites),
  `R` (removed — recovered or safely buried). Cumulative trackers
  (`E_total`, `I_total`, `H_total`, `F_total`, `R_total`) record incidence;
  `I_total` is cumulative symptom onsets.
- **Force of infection (frequency-dependent):**
  `λ = (βI·I + βH·H + βF·F) / N`. All three of `I`, `H`, `F` are infectious.
- **Progression:** `S → E → I`, then `I` splits three ways to `H`
  (hospitalisation), `R` (community recovery), or `F` (community death); `H`
  splits to `R` (hospital recovery) or `F` (hospital death); `F → R` (burial).
- The three-way/two-way exit rates (`θ1`, `δ1`, `δ2`) are **derived at each
  step** from the raw durations (`1/γh`, `1/γi`, `1/γd`) and the observed
  hospitalisation proportion `θ` and case-fatality ratio `δ`, following the
  paper's Appendix.

That derivation is exact, not approximate: substituting `δ1`, `δ2` and `θ1`
into the competing-hazards exit probabilities recovers the observed targets
identically, so the simulated cohort's realised hospitalisation proportion is
`θ = 80%` and its realised case-fatality ratio is `δ = 81%` to machine
precision. This matters when reading the deaths output — see below.

## Interventions

Each transmission route has its own control, which switches on at its start
date and (with no end date) stays on:

- **Community Transmission Control** — reduces `βI`.
- **Hospital Isolation / Barrier Nursing** — reduces `βH`.
- **Safe Burial Practices** — reduces `βF`.

In the notebook these are multipliers `z ∈ [0, 1]` on each `β`; here they are
expressed as transmission reductions (`reduction = 1 − z`).

The framework carries this value in the config under the key
`transmission_percentage`, which is the same quantity as the model's
`transmission_reduction` argument (`parameters.py` assigns one to the other)
and is divided by 100 at load. With `adherence_min = 100`, the applied rate is
`β · (1 − adherence · reduction)`. So the config's `transmission_percentage` of
50 / 100 / 100 gives `z = 0.50 / 0.0 / 0.0`, matching the notebook.

## Defaults and data

Defaults reproduce the notebook's **DRC (Kikwit) 1995** preset (Zaire EBOV):
`N = 200,000`, seed `I0 = 3`, incubation 7 d, onset→hospitalisation 5 d,
community infectious period 10 d, onset→death 9.6 d, death→burial 2 d,
hospitalisation proportion 80%, CFR 81%.

Transmission coefficients come from the paper's weekly Table 4 estimates
divided by 7, at **full precision**:

| Route | Table 4 (week⁻¹) | Stored (day⁻¹) |
|---|---|---|
| Community `βI` | 0.588 | 0.084 (exact) |
| Hospital `βH` | 0.794 | 0.1134286 |
| Burial `βF` | 7.653 | 1.0932857 |

Do not round these to three decimals. `0.113` understates `βH` by 0.38% and
moves pre-intervention R₀ from 2.694 to 2.692; over a long uncontrolled run
that compounds to several hundred cases.

`infected_population` in the config is a **percentage**, not a fraction and not
a count: the framework computes `round(infected_population / 100 * population, 2)`.
The example config's `0.0015` therefore means 0.0015% of 200,000 = **3** seed
cases, matching the notebook's `I0`. Writing `3` there would seed 6,000 cases.

The example scenario applies hospital isolation at week 4, safe burial at week
5, and a 50% community-transmission reduction at week 7.

Other outbreak presets in the source notebook (alternative DRC parameter sets,
Isiro 2012 Bundibugyo EBOV, Gulu 2000 Sudan EBOV, Bundibugyo 2007–08) can be
reproduced by editing the parameter values in the configuration.

## Basic reproduction number

The model does not emit R₀ — the framework has no output channel for scalar
diagnostics — so the decomposition is recorded here. Using the Legrand Appendix
formulation with the stored defaults above:

```
Δ    = γh·θ1 + γd·(1−θ1)·δ1 + γi·(1−θ1)·(1−δ1)
R0I  = βI / Δ                                        = 0.499
R0H  = (γh·θ1 / (γdh·δ2 + γih·(1−δ2))) · βH / Δ      = 0.424
R0F  = δ · βF / γf                                   = 1.771
R0   = R0I + R0H + R0F                               = 2.694
```

Derived routing quantities at these defaults: `θ1 = 0.673945`,
`δ1 = 0.803638`, `δ2 = 0.796835`.

Burial is the dominant route, contributing 66% of R₀ — which is why the
safe-burial intervention alone pushes Rₑ below 1 in the example scenario.

## Reading the deaths output

`F_total` is the **realised** cumulative death count: the integrated flow into
`F` up to the current time step. Two properties distinguish it from the
notebook's `total_deaths = δ × C`, and both are easy to misread:

1. **It includes the seed cases.** `I_total` accumulates only the `α·E` inflow,
   so the `I0 = 3` seeds placed directly into `I` never appear in it. They do
   appear in `F_total`. The identity is `F_total → δ × (I_total + I0)`, not
   `δ × I_total`.

2. **It is a realised count, not a projection.** `δ × C` implicitly assumes
   every case counted so far has already reached its outcome. `F_total` counts
   only deaths that have happened. While an epidemic is still growing, cases
   sitting in `E`, `I` and `H` have not died yet, so `F_total` is strictly
   lower than the eventual total.

Both effects are visible in the example run:

| Scenario | `I_total` | `F_total` | `δ × (I_total + I0)` | ratio |
|---|---|---|---|---|
| With interventions (resolved by ~wk 10) | 66.17 | 56.02 | 56.02 | **0.810** |
| Control run at day 174 (still growing) | 135,555 | 88,015 | 109,802 | 0.649 |

The controlled scenario has finished, so the identity holds exactly. The
control run has not — roughly 48,000 people are still in `E`/`I`/`H`/`F` at the
end of the window. Integrating the same system to day 400 drives the ratio to
exactly 0.81000, with `F_total = δ × (I_total + I0)` to seven significant
figures.

`F_total` is the more accurate of the two figures, and no `δ × I_total` output
is provided. If you need projected eventual deaths for an unfinished epidemic,
compute `δ × (I_total + I0)` and label it as a projection.

## Known framework behaviours affecting output

These are properties of the simulator rather than of this model, but they
change what appears in the results file:

- **Cumulative `_total` columns are stripped from the per-timestep time series.**
  `helpers.py` excludes any compartment ending in `_total` from the emitted
  series, so there is no per-day `I_total` curve and the notebook's weekly
  incidence plot cannot be rebuilt from it directly. The final values do
  survive, in the `compartment_deltas` dict — but **re-keyed onto the base
  compartment names**, so `compartment_deltas["F"]` is cumulative deaths, not
  the size of the `F` compartment. A per-day onset curve can be reconstructed
  as `S(0) − S(t) − E(t)`.

- **Integration stops one day before `end_date`.** The solver builds its time
  grid as `arange(0, n_timesteps, step)`, so a config spanning 175 days is
  integrated over days 0–174 and the last emitted row is dated 1995-06-29, not
  the configured 1995-06-30. For the controlled scenario this is immaterial
  (the outbreak is extinct by week 10). For the still-growing control run it
  costs about 2,350 cases against the notebook's 137,908. Setting `end_date`
  one day later works around it, at the cost of the config no longer meaning
  what it says.

- **The no-intervention counterfactual is generated automatically.** Every run
  emits two results, the second flagged `control_run: true`. No config edit is
  needed to produce it.

## Limitations

- Homogeneous mixing in a single closed population; no age or spatial
  structure, and no background births or deaths.
- Interventions are step functions (fully efficient from their start date),
  not gradual ramps.
- **The control run is not an epidemiological forecast.** Legrand et al. fitted
  these β values with a *stochastic* model (Gillespie's first-reaction method,
  likelihoods averaged over 700 runs per parameter set). Running that
  parameterisation as an unbounded deterministic epidemic — no stochastic
  extinction, no behavioural feedback, no saturation of local mixing — yields
  ~137,900 cases and a 69% attack rate in Kikwit, where the real 1995 outbreak
  had 315 cases. The control run is a mechanical "what if these rates never
  changed" baseline, useful only for relative comparison against the
  intervention scenario. Do not report "cases averted" from it as a real-world
  quantity.
- This implementation is intended for scenario exploration, not real-time
  forecasting.
