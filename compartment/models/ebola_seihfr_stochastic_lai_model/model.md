# Ebola – SEIHFR, stochastic (Lai 2026)

**Author:** Yingsi Lai.
**Based on:** Legrand J, Grais RF, Boëlle PY, Valleron AJ, Flahault A.
Understanding the dynamics of Ebola epidemics. *Epidemiol Infect.*
2007;135(4):610–621. doi:10.1017/S0950268806007217

A **stochastic** SEIHFR model of Ebola virus disease, built by Yingsi Lai — the
demographic-noise counterpart of the deterministic `ebola_seihfr_burial_legrand_model`. Both are
ported from `ebola_SEIHFR_model.ipynb`. The model structure and the DRC 1995
parameter values are taken from Legrand et al. (2007), which represents the
three settings in which Ebola spreads with distinct transmission intensities:
the **community**, the **hospital**, and **traditional burial** of the
deceased. The stochastic implementation (chain-binomial tau-leap on the
framework's daily step) is the author's own; see below.

The paper's model is itself stochastic — *"Simulations of the model were
performed using Gillespie's first reaction method"* — so this model restores the
stochasticity that the deterministic ODE reduction averages away. The most
important thing it adds is **variability**: a distribution of outbreak sizes and
the chance that a small introduction fades out, neither of which the
deterministic model can show.

## Model metadata

- **authors:** Yingsi Lai.
- **license:** Apache-2.0.
- **citations:**
  - Legrand J, Grais RF, Boëlle PY, Valleron AJ, Flahault A. Understanding the dynamics of Ebola epidemics. *Epidemiol Infect.* 2007;135(4):610–621. doi:10.1017/S0950268806007217 — model structure and default parameter values.
  - Khan AS, et al. The reemergence of Ebola hemorrhagic fever, Democratic Republic of the Congo, 1995. *J Infect Dis.* 1999;179(Suppl 1):S76–S86. doi:10.1086/514306 — primary data for the DRC 1995 defaults (via Legrand et al.).
- **model_type:** Compartmental (SEIHFR), stochastic: chain-binomial tau-leap on a fixed daily step, reported as the median and 95% interval across trajectories; single, well-mixed population.
- **diseases:** Ebola virus disease (default preset: Zaire ebolavirus, Kikwit, DRC 1995).
- **transmission_routes:**
  - Direct contact with symptomatic cases in the community (βI).
  - Nosocomial contact with hospitalised cases, including the isolation ward (βH).
  - Contact with the body during traditional burial (βF).
- **questions_answered:**
  - How much do community, hospital, and burial transmission each contribute to R0?
  - How do the timing and strength of hospital isolation, safe burial, and community measures change epidemic size?
- **key_assumptions:**
  - Homogeneous mixing in a single closed population; no age or spatial structure.
  - No background births or deaths; Ebola deaths pass through F (dead, awaiting burial) to R, so the population is conserved.
  - Frequency-dependent force of infection, λ = (βI·I + βH·H + βF·F) / N.
  - Entire population susceptible at the start apart from the seed cases, which are placed in I.
  - Exponentially distributed durations; individuals leaving I or H are split between destinations in proportion to the destination hazards, so the hospitalised proportion and case-fatality ratio equal θ and δ on expectation.
  - Interventions are step functions: each scales its β from its start date and, with a null end date, stays active.
  - Demographic stochasticity only; parameter values are fixed within a run.
- **applicability:**
  - Scenario exploration of the effect and timing of hospital isolation, safe burial, and community measures on an Ebola outbreak in one population.
  - Showing the spread of plausible outbreak sizes for a given parameter set, for an outbreak that has become established.
- **not_for:**
  - Real-time forecasting or predicting case counts for a current outbreak.
  - Estimating parameters from data; the model is not fitted.
  - Estimating the early-extinction (fade-out) probability of a small introduction; use an exact Gillespie implementation instead (see "Tau-leap vs. exact Gillespie").
  - Spatial spread, population mobility, or age-specific outcomes.
  - Vaccination, therapeutics, or other interventions not represented in the model.
  - Reading the automatic no-intervention control run as a forecast; it is a relative baseline only.
- **constraints:**
  - The mean durations 1/γi and 1/γd must exceed 1/γh.
  - Fixed daily time step (step size 1 for horizons up to 365 days).
  - `num_runs` between 10 and 100 (default 30); runtime is roughly 1 s per trajectory per scenario.
  - Each trajectory is seeded from operating-system entropy, so runs are not reproducible.
  - Each intervention has one start date and full strength from that date; gradual roll-out is not represented.
- **biases:**
  - The daily tau-leap understates the early-extinction probability of small introductions compared with exact Gillespie.
  - The default case-fatality ratio (50%) is lower than the paper's DRC value (81%), which lowers deaths and the burial component of R0; see "Differences from Legrand 2007".
  - The transmission coefficients are maximum-likelihood estimates from a single outbreak, with wide confidence intervals (Legrand 2007, Table 4).
  - Homogeneous mixing in a closed population overstates spread in an unmitigated epidemic: the no-intervention run reaches a far larger share of Kikwit's 200,000 people than the real 1995 outbreak's 315 cases.
- **validation:**
  - Population is conserved exactly and no compartment goes negative on any trajectory.
  - The realised hospitalisation fraction and case-fatality ratio converge to θ and δ.
  - The ensemble-mean trajectory agrees with the deterministic `ebola_seihfr_burial_legrand_model` and with a standalone exact-Gillespie implementation for an established outbreak.
  - Exact NumPy binomial draws are used because `jax.random.binomial` was found to inflate outbreak size by about 20%.
  - Not validated against observed Kikwit 1995 incidence at the current defaults.

## Model structure

Identical compartments, force of infection, routing algebra, parameters and
interventions as `ebola_seihfr_burial_legrand_model`, which rounds the β defaults to 3 decimal places (see that model's `model.md`). In brief:

- **Compartments:** `S`, `E`, `I` (community-infectious), `H` (hospitalised),
  `F` (dead, awaiting burial — infectious), `R` (removed). Cumulative trackers
  `E_total, I_total, H_total, F_total, R_total`; `I_total` is cumulative symptom
  onsets (the notebook's `C`).
- **Force of infection (frequency-dependent):** `λ = (βI·I + βH·H + βF·F) / N`.
- **Progression:** `S → E → I`, then `I` splits to `H`/`R`/`F`, `H` splits to
  `R`/`F`, `F → R`. The split rates `θ1, δ1, δ2` are derived at each step from
  `θ`, `δ` and the raw durations, exactly as in the deterministic model, so the
  realised hospitalisation proportion and case-fatality ratio reproduce the
  targets `θ` and `δ` (defaults 80% and 50%).

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
gave on the order of ~2–3% fadeout where the exact Gillespie gave ~9% (in checks
run at δ = 81%, the paper's value). If the
fadeout probability of a small introduction is the quantity of interest, use the
standalone exact Gillespie first-reaction implementation
(`ebola_seihfr_gillespie.py` / `ebola_SEIHFR_stochastic.ipynb`) rather than this
framework model. For outbreak-size variability and intervention comparison on an
established epidemic, this model is appropriate.

## Output

With `STOCHASTIC = True` the framework runs `num_runs` independent trajectories
and reports, for every compartment at every timestep, the **median** with a
**95% interval band** (`lower`/`upper`). It also emits the usual automatic
no-intervention **control run** (`control_run: true`). `num_runs` defaults to 30
(allowed range 10–100) and is configurable in the disease config; lower it for a
quick look or raise it for smoother bands (roughly 1 s per trajectory per
scenario locally, so about 1 min for the default 30 runs × 2 scenarios).

Reproducibility: each trajectory is seeded from operating-system entropy, so the
runs differ and repeated simulations give slightly different results.

## Parameters

The parameters users can change, with their default (as set in `define_parameters()`), what each one means, and where the value comes from. "Legrand 2007" refers to Legrand J, Grais RF, Boëlle PY, Valleron AJ, Flahault A. Understanding the dynamics of Ebola epidemics. *Epidemiol Infect.* 2007;135(4):610–621. doi:10.1017/S0950268806007217. The paper's numbered references ([3], [18], …) are listed below the table. The "Range" column is the user-facing slider range and is an assumption unless noted.

| Parameter | Default | Range | Meaning | Reference |
|---|---|---|---|---|
| `alpha` (α) | 7 days | 5–9 | Mean duration of the incubation (exposed, non-infectious) period. | Legrand 2007, Table 3 (7 days, mean), citing [18]–[20]. Range: assumption. |
| `gamma_f` (γf) | 2 days | 1–3 | Mean duration between death and (safe or traditional) burial. | Legrand 2007, Table 3 and Methods assumption (e) (2 days, mean). No primary data source; the paper describes it as "approximately the average duration from death to burial". Range: assumption. |
| `betaI` (βI) | 0.084 per day | 0.064–0.104 | Transmission coefficient for contact with symptomatic cases in the community, before interventions (per day). | Legrand 2007, Table 4: 0.588 per week (95% CI 0.420–2.191), fitted by approximate maximum likelihood; ÷ 7 = 0.084 per day. Range: assumption (the paper's CI ÷ 7 is 0.060–0.313). |
| `betaH` (βH) | 0.113429 per day | 0.07–0.15 | Transmission coefficient for contact with hospitalised cases (including isolation ward), before interventions (per day). | Legrand 2007, Table 4: 0.794 per week (95% CI 0.001–4.091); ÷ 7 = 0.1134286 per day. Range: assumption (the paper's CI ÷ 7 is 0.0001–0.584). |
| `betaF` (βF) | 1.09329 per day | 0.8–1.4 | Transmission coefficient for contact with the body during traditional burial, before interventions (per day). | Legrand 2007, Table 4: 7.653 per week (95% CI 0.001–9.997); ÷ 7 = 1.0932857 per day. Range: assumption (the paper's CI ÷ 7 is 0.0001–1.428). |
| `gamma_h` (γh) | 5 days | 1–9 | Mean duration from symptom onset to hospitalisation, for cases who are hospitalised. | Legrand 2007, Table 3 (5 days, mean), citing [3]. Range: assumption (the paper's sensitivity analysis varied it from 1 to 5 days). |
| `gamma_i` (γi) | 10 days | 7–15 | Mean duration from symptom onset to end of infectiousness for cases who survive in the community. Must exceed 1/gamma_h. | Legrand 2007, Table 3 and Methods assumption (e) (10 days, mean), citing [20], [33]. Range: assumption. |
| `gamma_d` (γd) | 9.6 days | 7–12 | Mean duration from symptom onset to death for cases who die without being hospitalised. Must exceed 1/gamma_h. | Legrand 2007, Table 3 (9.6 days, mean), citing [3]. Range: assumption. |
| `theta_target` (θ) | 80% | 50–100 | Target proportion of infectious cases who are hospitalised. The per-step hospitalisation rate (theta1) is derived from this target together with gamma_h, gamma_i, gamma_d, and delta — not set directly. | Legrand 2007, Table 3 (80%; the paper's derived θ1 = 0.67), citing [3]. Range: assumption. |
| `delta_target` (δ) | 50% | 10–90 | Target overall case-fatality ratio across hospitalised and non-hospitalised cases. The per-compartment death-split rates (delta1, delta2) are derived from this target. | Model default, not taken from Legrand 2007. Legrand 2007, Table 3 reports 81% for DRC 1995 (the paper's derived δ1 = δ2 = 0.80), citing [3]; Gulu, Uganda 2000 was 53% [4]. Set this to 81 to reproduce the paper's DRC parameter set. Range: assumption. |

References cited by Legrand 2007 (the paper's numbering):

- [3] Khan AS et al. The reemergence of Ebola hemorrhagic fever, Democratic Republic of the Congo, 1995. *J Infect Dis.* 1999;179(Suppl 1):S76–S86. doi:10.1086/514306
- [4] World Health Organization. *Weekly Epidemiological Record.* 2001;76:41–46 (Gulu, Uganda 2000 outbreak).
- [18] Bwaka MA et al. Ebola hemorrhagic fever in Kikwit, Democratic Republic of the Congo: clinical observations in 103 patients. *J Infect Dis.* 1999;179(Suppl 1):S1–S7. doi:10.1086/514308
- [19] Ndambi R et al. Epidemiologic and clinical aspects of the Ebola virus epidemic in Mosango, Democratic Republic of the Congo, 1995. *J Infect Dis.* 1999;179(Suppl 1):S8–S10. doi:10.1086/514297
- [20] Dowell SF et al. Transmission of Ebola hemorrhagic fever: a study of risk factors in family members, Kikwit, Democratic Republic of the Congo, 1995. *J Infect Dis.* 1999;179(Suppl 1):S87–S91.
- [29] Muyembe-Tamfum JJ et al. Ebola outbreak in Kikwit, Democratic Republic of the Congo: discovery and control measures. *J Infect Dis.* 1999;179(Suppl 1):S259–S262. doi:10.1086/514302
- [33] Rowe AK et al. Clinical, virologic, and immunologic follow-up of convalescent Ebola hemorrhagic fever patients and their household contacts, Kikwit, Democratic Republic of the Congo. *J Infect Dis.* 1999;179(Suppl 1):S28–S35.

### Differences from Legrand 2007 (DRC 1995)

Some defaults in this model differ from the paper's DRC 1995 values. Check these before you compare output with the paper:

- **Case-fatality ratio:** the default is 50%; the paper reports 81%. Changing δ changes the burial part of R0, because R0F = βF × (1/γf) × δ. At δ = 81% this gives R0F ≈ 1.77, matching the paper's Table 4 value of 1.8; at δ = 50% it is ≈ 1.09.
- **Intervention strength:** all three interventions default to 50% adherence and 100% transmission reduction, so each halves its targeted β once active (β × (1 − 0.5 × 1.0)); the example config uses these defaults. The paper instead assumes hospital isolation and safe burial remove hospital and burial transmission completely (βH and βF set to 1.09329), and estimates the community reduction for DRC at 12% (95% CI 0–78), and 88% for Uganda.

## Known framework behaviours affecting output

Same as `ebola_seihfr_burial_legrand_model` (they are framework properties, not model bugs):

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
- Interventions are step functions (at full strength from their start date), not
  gradual ramps.
- **Daily-step tau-leap**, not exact Gillespie — understates the early-extinction
  probability of small introductions (see "Tau-leap vs. exact Gillespie").
- **The control run is not an epidemiological forecast.** Legrand et al. fitted
  these β values with a stochastic model; running that parameterisation as an
  unbounded epidemic reaches a far larger share of Kikwit's 200,000 people than
  the real 1995 outbreak's 315 cases. Additionally, because the counterfactual is still
  growing at the horizon, its cumulative figures are sensitive to step
  discretisation and differ from the deterministic model by more than the
  controlled scenario does. Treat it as a relative baseline only.
- Intended for scenario exploration, not real-time forecasting.

## Change history

- Sep 2026 — Created by Yingsi Lai (laiyingsilaiyingsi) as `ebola_seihfr_stochastic_lai_v1_model`; PR #154.
- Oct 2026 — Renamed from `ebola_seihfr_stochastic_lai_v1_model` (`disease_type` `ebola_seihfr_stochastic_lai_v1`, label "ebola_seihfr_stochastic_lai_v1") to `ebola_seihfr_stochastic_lai_model` (`disease_type` `ebola_seihfr_stochastic_lai`, label "Ebola – SEIHFR, stochastic (Lai 2026)"); by Yingsi Lai (laiyingsilaiyingsi).
- Oct 2026 — Added the parameter reference table (sources from Legrand et al. 2007, Tables 3–4) and a list of differences from the paper's DRC 1995 values; model metadata now lists Yingsi Lai as author, with Legrand et al. (2007) as the cited source. Removed the unused config `seed` lookup: each trajectory is now seeded directly from operating-system entropy. Corrected the documented `num_runs` default to 30. Intervention defaults set to 50% adherence and 100% transmission reduction for all three interventions (previously 90% reduction for community and hospital), and the example config updated to match; no change to disease-parameter defaults; by Yingsi Lai (laiyingsilaiyingsi).
