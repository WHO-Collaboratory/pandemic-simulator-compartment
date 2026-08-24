# Audit — `ebola_test_model` vs. `ebola_SEIHFR_model.ipynb`

**Model audited:** `compartment/models/ebola_test_model/` (`model.py`, `model.md`, `example-config.json`, `main.py`, `__init__.py`)
**Source of truth:** `ebola_SEIHFR_model.ipynb` (Legrand et al. 2007 SEIHFR, deterministic ODE reduction)
**Framework:** `WHO-Collaboratory/pandemic-simulator-compartment` (`compartment.*`)
**Date:** 2026-08-19

## How this audit was performed

The notebook is treated as the reference ("source"). Every facet below is checked
against the notebook text/equations/code and, for Part 2, against the actual model
code plus the framework code it depends on. To move beyond static reading I:

1. Reconstructed the model directory inside a fresh clone of the repository and
   **executed it end-to-end** through the framework's real config-loading,
   validation, and `odeint` integration path (bypassing only the cloud/boto3
   I/O wrapper, which is unrelated to the model and, as shipped, does not import
   under Python 3.11).
2. Compared `model.py` line-by-line against the repository's already-shipped
   precedent `ebola_seihfr_burial_legrand_model` (the file the model's own
   docstring cites). The `equation()` body is **identical** apart from comments;
   the only functional differences are the class/`disease_type` names and the
   community intervention reduction (12 % in the precedent → 50 % here, to match
   the notebook scenario).
3. Independently recomputed R₀ and the routing algebra from the stored parameters.

**Headline numerical result (direct evidence):**

| Scenario | Notebook | Framework model (this audit) |
|---|---|---|
| Controlled — total cases (cumulative onsets) | 66 (0.03 % attack rate) | **66.05 (0.033 %)** — match |
| Controlled — pre-intervention R₀ | 2.694 | **2.692** (from stored params) |
| Counterfactual (z=1) — total cases | 137,908 (68.95 %) | 135,182 (67.59 %) — ~2 % low |

The controlled scenario matches to the integer. The counterfactual is ~2 % lower,
fully explained by two low-severity framework conventions documented in Part 2
(a one-day-shorter horizon and 3-decimal β rounding) acting on an epidemic that is
still near peak growth at the end of the window.

Convention used in the tables: **direct evidence** = observed in code or in a run;
**[inference]** marks any interpretive claim.

---

# Part 1 — Source-to-model fidelity

Does the ported model represent each facet the notebook defines? Assessment is
whether the facet is faithfully represented (true), contradicted (false),
under-determined by the source (ambiguous), or absent from the source
(unsupported).

| # | Model facet or assertion | Assessment | Exact quotation from the source (notebook) | Location in source | Notes and interpretation |
|---|---|---|---|---|---|
| 1 | Six epidemiological compartments S, E, I, H, F, R | true | "**Compartments:** S → E → I → (H or F) → R" | §Header markdown; §1 compartment table | Model declares exactly S,E,I,H,F,R plus cumulative `_total` trackers. |
| 2 | I, H, F are all infectious (three transmission settings) | true | "A compartmental model … with **three transmission routes** … Community βI … Hospital βH … Funeral βF" | §Header table | Model marks I, H, F `infective=True`. |
| 3 | Frequency-dependent force of infection λ=(βI·I+βH·H+βF·F)/N | true | "$\lambda(t) = \frac{\beta_I\,I + \beta_H\,H + \beta_F\,F}{N}$" | §1 Model Equations | Model computes this exact FOI manually (multi-β). |
| 4 | dS/dt = −λS | true | "$\frac{dS}{dt} = -\lambda S$" | §1 | Reproduced. |
| 5 | dE/dt = λS − αE | true | "$\frac{dE}{dt} = \lambda S - \alpha E$" | §1 | Reproduced (α from E→I edge). |
| 6 | dI/dt = αE − (γh·θ1 + γi(1−θ1)(1−δ1) + γd(1−θ1)δ1)·I | true | "$\frac{dI}{dt} = \alpha E - \bigl(\gamma_h\,\theta_1 + \gamma_i(1-\theta_1)(1-\delta_1) + \gamma_d(1-\theta_1)\delta_1\bigr)\,I$" | §1 | Reproduced term-for-term. |
| 7 | dH/dt = γh·θ1·I − (γdh·δ2 + γih(1−δ2))·H | true | "$\frac{dH}{dt} = \gamma_h\,\theta_1\,I - (\gamma_{dh}\,\delta_2 + \gamma_{ih}(1-\delta_2))\,H$" | §1 | Reproduced. |
| 8 | dF/dt = γd(1−θ1)δ1·I + γdh·δ2·H − γf·F | true | "$\frac{dF}{dt} = \gamma_d(1-\theta_1)\delta_1\,I + \gamma_{dh}\,\delta_2\,H - \gamma_f\,F$" | §1 | Reproduced. |
| 9 | dR/dt = γi(1−θ1)(1−δ1)·I + γih(1−δ2)·H + γf·F | true | "$\frac{dR}{dt} = \gamma_i(1-\theta_1)(1-\delta_1)\,I + \gamma_{ih}(1-\delta_2)\,H + \gamma_f\,F$" | §1 | Reproduced. |
| 10 | α = 1/d_E, mean incubation | true | "$\alpha = 1/d_E$ … Rate out of exposed class; $d_E$ = mean incubation period" | §1 parameter table | E→I edge, 7 d. |
| 11 | γh=1/d_h, γi=1/d_i, γd=1/d_d, γf=1/d_f | true | "$\gamma_h = 1/d_h$ … $\gamma_i = 1/d_i$ … $\gamma_d = 1/d_d$ … $\gamma_f = 1/d_f$" | §1 parameter table | 5, 10, 9.6, 2 d respectively. |
| 12 | γdh = 1/(d_d − d_h), γih = 1/(d_i − d_h) (in-hospital residuals) | true | "$\gamma_{dh} = 1/(d_d - d_h)$ … $\gamma_{ih} = 1/(d_i - d_h)$" | §1 parameter table | Model derives these each step (with an ε floor — see Part 2 #12). |
| 13 | δ1 = δγi/(δγi+(1−δ)γd) | true | "delta1 = (delta * gamma_i) / (delta * gamma_i + (1 - delta) * gamma_d)" | §2 Parameters code cell | Identical formula. |
| 14 | δ2 = δγih/(δγih+(1−δ)γdh) | true | "delta2 = (delta * gamma_ih) / (delta * gamma_ih + (1 - delta) * gamma_dh)" | §2 Parameters code cell | Identical formula. |
| 15 | θ1 = θ[γi(1−δ1)+γd·δ1] / (θ[γi(1−δ1)+γd·δ1] + (1−θ)γh) | true | "theta1 = (theta * (gamma_i*(1-delta1) + gamma_d*delta1) / (… + (1 - theta) * gamma_h))" | §2 Parameters code cell | Identical formula. |
| 16 | θ (hosp. proportion) and δ (CFR) are observed targets; θ1/δ1/δ2 derived | true | "$\theta_1$ … Probability of hospitalisation given symptom onset *(derived from θ, δ, rates)*" | §1 routing-probabilities table | Model exposes `theta_target`/`delta_target`, derives the rest. |
| 17 | Transmission rates supplied weekly, converted to per-day (÷7) | true | "bI = bI_wk / 7 … bH = bH_wk / 7 … bF = bF_wk / 7" | §2 Parameters code cell | Model stores per-day (0.084, 0.113, 1.093). See Part 2 #17 (rounding). |
| 18 | DRC preset: βI,βH,βF = 0.588, 0.794, 7.653 wk⁻¹ | true | "bI_wk = 0.588 … bH_wk = 0.794 … bF_wk = 7.653" | §2, `PRESET=='DRC'` | ÷7 → 0.084, 0.113, 1.093 /day. |
| 19 | DRC durations d_E=7, d_h=5, d_d=9.6, d_i=10, d_f=2 | true | "inv_alpha = 7.0 … inv_gamma_h = 5.0 … inv_gamma_d = 9.6 … inv_gamma_i = 10.0 … inv_gamma_f = 2.0" | §2, DRC block | Matches config/defaults exactly. |
| 20 | DRC θ=0.80, δ=0.81 | true | "theta = 0.80 … delta = 0.81" | §2, DRC block | Config `theta_target=80`, `delta_target=81`. |
| 21 | Population N=200,000, seed I0=3 | true | "N = 200_000 … I0 = 3" | §2, DRC block | Config `population=200000`, `infected_population=0.0015` (%) → 3. |
| 22 | Seed goes into the community-infectious class I; E,H,F,R=0 | true | "y0 = [N - I0, 0, I0, 0, 0, 0, 0]" | §3 `run_model` | Framework seeds `I`; verified init vector `[199997,0,3,0,0,0,0,…]`. |
| 23 | Cumulative onset tracker C, dC/dt = αE | true | "C = cumulative symptom onsets (dC/dt = α·E)" | §3 `seihfr_ode` | Model's `I_total`, `derivs["I_total"]=alpha*E`. |
| 24 | Simulation horizon T_end = 25 weeks = 175 days | true | "T_end = 25 * 7" | §2, DRC block | Config 1995-01-06 → 1995-06-30 = 175 steps (see Part 2 #24, endpoint). |
| 25 | Intervention: per-route multiplier z∈[0,1] on each β, from that route's start day | true | "Each transmission route is independently scaled by a factor in **[0, 1]** (0 = completely eliminated, 1 = unchanged)." | §Header; §2 Section B | Model uses 3 step interventions, reduction = 1−z. |
| 26 | Per-route timing independent (no ordering assumed) | true | "Each route has its own start date and effect size -- no ordering assumed." | §2 Section B comment | Three independent interventions with own start dates. |
| 27 | DRC scenario: hospital wk4 (z=0), funeral wk5 (z=0), community wk7 (z=0.5) | true | "T_int_community = 7 * 7; z_community = 0.50 … T_int_hospital = 4 * 7; z_hospital = 0.0 … T_int_funeral = 5 * 7; z_funeral = 0.0" | §2 Section B, else-branch | Config dates 02-24 / 02-03 / 02-10 = days 49/28/35; reductions 50/100/100 %. Match. |
| 28 | Interventions persist after onset (step, not window) | true | "$z_{\rm com}\in[0,1]$ … Post-intervention multiplier" (applied for `T_int ≤ t`) | §1 intervention table; §3 `phase_betas` (`_Ti <= t_start`) | Model uses null end-date ⇒ persistent (Part 2 #28). |
| 29 | Deterministic ODE (mean-field), not Gillespie | true (intended) | "**ODE system (Legrand 2007)** … from scipy.integrate import solve_ivp" | §1; §imports | Matches modeler's declared decision; framework uses `odeint`. |
| 30 | Pre-intervention R₀ decomposition R₀ = R₀I+R₀H+R₀F | **unsupported (in model)** | "$R_0 = \frac{\beta_I}{\Delta} + … + \frac{\delta\,\beta_F}{\gamma_f}$" | §2 Basic Reproduction Number | Notebook computes R₀ & components; **model does not emit R₀** (Part 2 #30). Diagnostic only; dynamics unaffected. |
| 31 | Time-varying effective Rₑ(t) across change-points | **unsupported (in model)** | "def Re_at_time(t): … return compute_R0(bI_eff, bH_eff, bF_eff)[0]" | §2 R₀ code cell | Not reproduced in the ported model. |
| 32 | Weekly incidence (new onsets/week) as primary curve | ambiguous | "def weekly_incidence(t, sol, …): … inc[w] = max(np.interp(...C...) …)" | §3 `weekly_incidence` | Model emits continuous `I_total`; weekly binning is a post-processing/UI step, not in the model. Underlying quantity (cumulative onsets) is present. |
| 33 | Deaths reported as δ × cumulative cases | **false (as an output)** | "total_deaths = delta * total_cases" | §4 simulate cell | Model instead tracks **mechanistic** `F_total` (actual flow into F). Numbers differ (Part 2 #33). Most material output difference. |
| 34 | Counterfactual = no intervention (z=1 all routes) | true | "run_model(z_community=1.0, z_hospital=1.0, z_funeral=1.0)" | §5 Summary cell | Reproducible by emptying `intervention_dict` (verified). |
| 35 | Homogeneous mixing, closed population, no births/deaths | true [inference] | Implicit: single N, no vital-dynamics terms in the ODEs | §1 equations (no birth/death terms) | Model states this explicitly in `key_assumptions`; consistent with source. |
| 36 | Alternative presets (DRC_2/3, Uganda/2/3) | true (intended not-wired) | "PRESET = 'DRC'" with 5 further branches documented | §2 preset block | Per modeler's decision, only DRC is wired; others documented but need manual config edits (Part 2 #36). |

---

# Part 2 — Code-implementation audit

Each facet is linked to the code that implements it and to the framework code
that governs its behavior. "true" = verified correct (by reading and/or by the
executed run); "false" = conflicts with source or does not execute as intended;
"ambiguous" = evidence cannot settle it; "unsupported" = no implementation.

| # | Model facet | Code location (file : symbol / line) | Assessment | Explanation and notes | Alternative approaches |
|---|---|---|---|---|---|
| 1 | Compartments S,E,I,H,F,R (+ `_total`) | `model.py:146-171` `define_parameters`/`add_compartment`; `_TOTAL_COMPARTMENTS` `model.py:59-71` | true | Six compartments declared in order; five explicit cumulative trackers declared and wired (`model.py:485-489`). Verified runtime `compartment_list = [S,E,I,H,F,R,E_total,I_total,H_total,F_total,R_total]`. | Could let the framework auto-generate `_total`s, but auto-gen only covers edge targets (I, R). Manual declaration is the only way to get E/H/F totals — effectively the only reasonable approach. |
| 2 | I, H, F infectious | `model.py:153-167` (`infective=True` on I,H,F) | true | Flags set. **Note:** because the FOI is computed by hand (#3), these flags do *not* drive the FOI; they are read only by `_compute_equations` for declared frequency-dependent edges (none here) and are otherwise inert. Harmless. | Could route the FOI through a declared frequency-dependent edge, but `_compute_equations` (`model.py:1057-1059`) supports only one shared `infective_sum` — cannot mix three βs. Manual FOI is required. |
| 3 | FOI λ=(βI·I+βH·H+βF·F)/N | `model.py:439-443` (`foi = (betaI*I+betaH*H+betaF*F)/(N_total+1e-10)`) | true | Uses per-region `N_total` = sum of non-`_total` compartments. Population is conserved (no births/deaths, deaths pass F→R), so `N_total ≡ N = 200,000`; equivalent to the notebook's constant `N`. Verified: `S+…+R` stays at 200,000. | Frequency- vs density-dependent is a modeling choice; notebook is frequency-dependent, matched. Only reasonable approach given the constraint. |
| 4-9 | ODE terms dS…dR | `model.py:437` `_compute_equations` (E→I, F→R) + `model.py:443,466-477` manual flows; framework `model.py:964-1072` `_compute_equations`, `1074-1110` `_apply_flow` | true | Verified algebraically that edge-flows + manual flows reconstruct the six notebook ODEs exactly (S→E; αE; the 3-way I split; the 2-way H split; F→R). All `_apply_flow`/`_compute_equations` calls resolve and executed without error. | The "declarative edges for simple flows + manual `_apply_flow` for derived splits" split is the framework-idiomatic pattern (mirrors `hantavirus_jax_model`, `dengue_jax_model`). Effectively the only clean approach in this framework. |
| 10-11 | α, γh, γi, γd, γf | edges `model.py:177-198` (alpha, gamma_f, DAYS); params `model.py:254-292` (gamma_h/i/d, FLOAT); rate conversion `model.py:446-448` `_to_rate(...,DAYS)` and framework `model.py:512-521` | true | α, γf converted DAYS→rate at load by `_load_transmission_params` (`model.py:548-554`). γh/γi/γd kept native and converted in `equation()`. Verified `transmission_dict={'alpha':7.0,'gamma_f':2.0}`. | γh/i/d could be edges too, but each feeds *derived* split rates, not a single `rate*source` flow — so they are correctly plain parameters. |
| 11a | **`gamma_d=9.6` uses `ValueType.FLOAT` not `DAYS`** | `model.py:247-292`; framework `schema_generator.py:84-100` `_VALUE_TYPE_TO_PYTHON` | true | Confirmed `_VALUE_TYPE_TO_PYTHON[ValueType.DAYS] = int`. A `DAYS` `add_parameter` would coerce the disease-config field to `int` and reject 9.6. The model's `FLOAT` choice (with manual `_to_rate(...,DAYS)`) is **correct and necessary**, and the code comment accurately explains why. | No cleaner alternative within this framework; the comment documents the pitfall well. |
| 12 | γdh=1/(d_d−d_h), γih=1/(d_i−d_h) | `model.py:455-456` (`jnp.maximum(1/γi−1/γh, eps)`) | true | Formula matches; adds an ε=1e-10 floor the notebook lacks. For all DRC values residuals are positive (d_i−d_h=5, d_d−d_h=4.6) so behavior is identical; the floor only guards degenerate configs (e.g. d_i≤d_h). Defensive improvement, not a deviation. | Could validate `d_i>d_h`, `d_d>d_h` at config load and drop the floor. Equivalent numerically for valid inputs. |
| 13-15 | δ1, δ2, θ1 derived | `model.py:459-463` | true | Character-for-character equal to the notebook's formulas (with `+eps` in denominators). Verified numerically: at DRC values the derivation yields R₀=2.692. | These are the paper's algebraic relations; no alternative. |
| 16 | θ, δ as observed targets | params `model.py:293-320` (`theta_target`, `delta_target`, PERCENTAGE); conversion `model.py:449-450` `_to_rate(...,PERCENTAGE)` | true | 80.0/81.0 → 0.80/0.81. Verified in run. | Only reasonable approach. |
| 17 | Weekly β ÷7 → per-day | disease params `model.py:205-243` (defaults 0.084/0.113/1.093); config `example-config.json:7-9` | true, with rounding caveat | Stored **rounded to 3 decimals**: 0.794/7=0.11343→`0.113` (−0.4 %), 7.653/7=1.09329→`1.093` (−0.03 %), 0.588/7=0.084 (exact). Notebook uses full precision. Effect: R₀ 2.692 vs 2.694 (−0.07 %). Low severity. | Store `0.11343`, `1.09329` (or the weekly value with a `/7` at load) to remove the rounding gap. |
| 18-21 | DRC preset values (β, durations, θ, δ, N, I0) | `example-config.json:4-14,62-70`; defaults in `model.py` | true | All match the notebook DRC block. `infected_population=0.0015` is a **percentage** (0.0015 % × 200,000 = 3), applied by framework `model.py:1209` `get_initial_population` (`infected = round(infected_population/100*population,2)`). Verified seed = 3. | The 0.0015 %-as-percentage encoding is correct but easy to misread as a fraction; a config comment would help. |
| 22 | Seed into I | framework `model.py:1189-1214` `get_initial_population` (default) + `prepare_initial_state` `model.py:388-398` | true | Model does not override `get_initial_population`, so the default routes `infected→I`, `susceptible→S`, all else 0 — matching the notebook's `y0`. Verified. | An override could seed E instead, but the notebook seeds I; default is correct. |
| 23 | Cumulative onsets `I_total`, dC/dt=αE | `model.py:486` (`derivs["I_total"]=params["alpha"]*E`) | true | `I_total` = notebook's `C`. Verified final `I_total=66.05`. Deliberately overwrites the auto-accumulated value to avoid double counting (comment `model.py:479-484`); mechanism confirmed correct against framework `_apply_flow`/`_compute_equations`. | Only reasonable approach given manual flows. |
| 24 | Horizon 175 days | validation `base_simulation.py:123-127` (derive `time_steps` from dates); solver `simulation_manager.py:22` `ts=arange(0,n_timesteps,step)`; `helpers.py:1034-1037` `step=ceil(175/365)=1` | true, off-by-one caveat | `time_steps=175` derived correctly. But `arange(0,175,1)` yields days **0…174** — the run covers 174 days, one short of the notebook's integration to t=175. Negligible for the (extinct-by-wk-10) controlled run; contributes ~1 day of missed growth to the still-growing counterfactual. Framework-level convention, not model-specific. | Use `arange(0, n_timesteps+step, step)` or `linspace(0, n_timesteps, …)` to include the endpoint. Affects all models equally. |
| 25-27 | Per-route step interventions, DRC timing/strength | `model.py:332-367` `add_intervention` (targets betaI/betaH/betaF); config `example-config.json:37-60`; runtime `runtime.py:229-249` `apply_to_rates`; `helpers.py:916-975` `create_intervention_dict` | true | `reduced = rate*(1 − adherence·reduction)`. Config adherence 100 → 1.0, `transmission_percentage` {50,100,100} ÷100 → {0.5,1.0,1.0} ⇒ effective z {0.5,0,0}. Dates 02-03/02-10/02-24 → days 28/35/49 = wk 4/5/7. **All match the notebook.** Verified in run. | Note: the config key is `transmission_percentage` but semantically it is the *reduction* (1−z), divided by 100 at load. Naming is a framework convention; correct but non-obvious. |
| 28 | Interventions persist after start | `runtime.py:174-183` `check_date_activation` (`end_date_ordinal is None ⇒ in_window = day ≥ start`) | true | Null `end_date` ⇒ persistent step, exactly the notebook's `T_int ≤ t` semantics. Confirmed by reading the branch and by the matching controlled-run result. | A windowed intervention (with end date) would bounce rates back — not wanted here; null end date is correct. |
| 29 | Deterministic `odeint` integration | `simulation_manager.py:38-58` (default `odeint`, tight tolerances; `euler` only if STOCHASTIC/SOLVER) | true (intended) | Model sets neither `STOCHASTIC` nor `SOLVER`, and `disease_type="ebola_test"` is not VECTOR_BORNE ⇒ default-tolerance `jax.experimental.ode.odeint`. Matches the modeler's deterministic-ODE decision. **Caveat:** interventions are discontinuous step functions integrated within a *single* adaptive `odeint` call, whereas the notebook integrates **piecewise** between exact change-points (`run_model`, §3). `odeint` handles the discontinuities but with slightly different local error; controlled result still matched to the integer. | For exact parity with the notebook, integrate phase-by-phase between change-points (as `run_model` does) or hand `odeint` the change-point times. Minor. |
| 30 | R₀ decomposition output | *(no symbol)* | unsupported | The notebook's `compute_R0` (R₀I/R₀H/R₀F) is **not implemented** anywhere in the model or emitted by the framework. Independent recomputation from stored params gives 2.692 — so the parameters are right, but the diagnostic output is absent. | Add a `compute_r0()` helper (pure function of the schema params) and surface it in `model_documentation` or an output field. Optional — does not affect the simulated trajectories. |
| 31 | Time-varying Rₑ(t) | *(no symbol)* | unsupported | `Re_at_time` / change-point table not ported. Diagnostic only. | Same as #30. |
| 32 | Weekly incidence curve | `I_total` present; weekly binning absent | ambiguous | The underlying quantity (cumulative onsets) is emitted as `I_total`; converting to weekly new-onset bins is the notebook's `weekly_incidence` post-processing, which lives outside the model. Whether the framework's results view reproduces it was not evaluated (out of scope of the uploaded files). | Post-process `diff(I_total)` into weekly bins in the results layer if needed. |
| 33 | **Deaths output** | `model.py:488` `derivs["F_total"]=flow_I_to_F+flow_H_to_F`; notebook `total_deaths=delta*total_cases` (§4) | **false (semantic mismatch)** | The model reports **mechanistic** cumulative deaths (actual flow into F): run gives `F_total=55.9` (controlled) and `87,687` (counterfactual). The notebook reports `δ×C`: `53.5` and `111,705`. **They differ by ~21 % in the counterfactual** because the realized cohort CFR from the Legrand routing (≈84.7 % controlled, ≈64.9 % counterfactual) is not exactly the target δ=81 % — a property of the model present in *both* implementations; the notebook simply masks it with a flat δ multiplier. The *dynamics* are identical; only the reported death statistic differs. | If comparability with the notebook is the goal, also emit `deaths = δ × I_total`. If model-consistency is the goal, `F_total` is the more defensible figure. At minimum, document which convention `F_total` uses. This is the single most consequential output difference. |
| 34 | Counterfactual (z=1) | disable via empty `intervention_dict`; framework `model.py:1147-1149` (only interventions in `intervention_dict` apply) | true | Reproduced (135,182 cases). The ~2 % gap vs 137,908 = horizon off-by-one (#24) + β rounding (#17) on a near-peak epidemic. | — |
| 35 | Closed population, no vital dynamics | `model.py:134-142` `key_assumptions`; equations have no birth/death terms | true | Conservation verified numerically (Σ compartments constant). | — |
| 36 | Only DRC wired | `example-config.json` (DRC values); `model.md:50-52` | true (intended) | Matches the modeler's stated decision. The other five notebook presets are documented in `model.md` but require manual config edits; no preset switch exists. | Could add named preset configs (one JSON per outbreak) for parity with the notebook's `PRESET` selector. Optional. |
| F1 | All called functions defined & resolve | `_unpack_params`, `_compute_equations`, `_apply_flow`, `_apply_interventions`, `_to_rate` (framework `model.py`); `betaI/H/F`, `gamma_*`, `theta/delta_target` (auto-set from schema, `model.py:226-227`) | true | Every helper resolves; `self.travel_matrix` is populated by `_ensure_travel_matrix` before `equation()` runs (`simulation_manager.py:27`). The full pipeline executed end-to-end with **no NameError/AttributeError/import failure**. `__init__.py` is intentionally empty (package marker) — fine. | — |
| F2 | Documentation matches implementation | `model.md` | true | `model.md` accurately describes N, seed, βs (÷7), durations, θ/δ, R₀≈2.7 (burial-dominant), and the wk 4/5/7 scenario — all consistent with the code and the run. Minor gap: it does not state that `F_total` deaths are mechanistic (≠ δ×C). | Add one sentence to `model.md` clarifying the deaths convention (#33). |

---

# Conclusion

## Overall verdict

The port is **faithful to the notebook in every structural and dynamical respect**.
Compartments, the three-route FOI, all six ODEs, the derived routing algebra
(θ1, δ1, δ2, γih, γdh), parameters, initial conditions, and the per-route step
interventions are reproduced exactly. The `equation()` body is identical to the
repository's already-shipped `ebola_seihfr_burial_legrand_model`, and a full
end-to-end run reproduces the notebook's controlled-scenario result to the integer
(66 cases) and R₀ to three significant figures (2.692 vs 2.694). No undefined,
unresolved, or misused functions were found; the whole pipeline executes cleanly.

The differences are confined to **diagnostic outputs and framework-level numerical
conventions**, not the modeled biology.

## Prioritized discrepancies and defects

1. **[Medium] Deaths output semantics differ (Part 2 #33).** The model emits
   mechanistic `F_total` (cumulative flow into F); the notebook reports
   `δ × cumulative_cases`. These diverge by ~21 % in the uncontrolled scenario
   (87,687 vs 111,705) because the realized cohort CFR from the Legrand routing
   is not exactly the target δ. Not a dynamics error — but anyone comparing
   "deaths" across the two will see materially different numbers.
   *Fix:* decide and document the convention; optionally also emit
   `deaths = δ × I_total` for notebook parity. (`model.py:488`, `model.md`.)

2. **[Low-Medium] R₀ / Rₑ analytics not ported (Part 2 #30-31).** The notebook's
   `compute_R0` decomposition and time-varying `Re_at_time` have no counterpart.
   Diagnostic only; trajectories are unaffected (parameters verified to give the
   right R₀). *Fix:* add an optional `compute_r0()` helper surfaced in
   documentation/outputs.

3. **[Low] β values rounded to 3 decimals (Part 2 #17).** `0.113`, `1.093`
   instead of `0.11343`, `1.09329`. ~0.1 % parameter error (R₀ 2.692 vs 2.694).
   *Fix:* store full precision, or the weekly value with a `/7` at load.
   (`example-config.json:7-9`, `model.py:214,226,239`.)

4. **[Low] Solver horizon off-by-one (Part 2 #24).** `arange(0,175,1)` covers
   days 0-174, one short of the notebook's 175; contributes to the counterfactual's
   ~2 % shortfall on a still-growing epidemic. **Framework-level**, affects all
   models. *Fix (framework):* include the endpoint in
   `simulation_manager.py:22`.

5. **[Low] Step interventions integrated within a single adaptive `odeint`
   (Part 2 #29)** rather than piecewise between exact change-points as the
   notebook does. `odeint` absorbs the discontinuities; controlled result still
   matched exactly. *Fix (optional):* integrate phase-by-phase for exact parity.

## Missing or unsupported features

- R₀ and time-varying Rₑ outputs (#30, #31) — present in the notebook, absent here.
- Weekly-incidence binning (#32) — the underlying `I_total` is emitted; weekly
  bins are left to post-processing.
- A preset selector (#36) — only DRC is wired (an intended decision); the other
  five notebook presets need manual config edits.

## Undefined / unresolved / misused / inappropriate functions

- **None.** All helpers (`_unpack_params`, `_compute_equations`, `_apply_flow`,
  `_apply_interventions`, `_to_rate`) and all schema-derived attributes resolve;
  the pipeline ran end-to-end with no errors. `__init__.py` is intentionally empty.
- One benign redundancy: `infective=True` on I/H/F (#2) does not actually drive
  the hand-computed FOI; it is inert here. Not a defect.

## Documentation and testing gaps

- **No automated tests.** No test in `tests/` references this model (or the
  precedent). A regression test asserting the controlled DRC scenario yields
  ≈66 cumulative onsets and pre-intervention R₀≈2.69 would lock in fidelity.
- **`model.md` deaths convention (#33, F2).** Add one sentence noting `F_total`
  is a mechanistic death count and differs from the notebook's `δ×C`.
- **Config readability (#18-21).** `infected_population: 0.0015` is a percentage
  (→ 3 seeds); a comment would prevent misreading it as a fraction. Likewise the
  `transmission_percentage` key (#25) is really the *reduction* (1−z).

## Specific recommended corrections (file : symbol)

1. `model.py:488` `equation` / `model.md` — document (and, if notebook parity is
   wanted, add) the `δ × I_total` deaths figure alongside `F_total`.
2. `model.py` — add optional `compute_r0()` (pure function of the schema params);
   surface R₀/Rₑ in `model_documentation` or an output field.
3. `example-config.json:7-9` and `model.py:214,226,239` — store full-precision
   βs (`0.11343`, `1.09329`) or apply `/7` at load.
4. `compartment/simulation_manager.py:22` (framework) — include the final
   timestep in `ts` so the integration reaches `T_end`.
5. `tests/` — add a fidelity regression test for the DRC controlled scenario
   (≈66 onsets; R₀≈2.69) and the counterfactual.
6. `example-config.json` — add clarifying comments for `infected_population`
   (percentage) and `transmission_percentage` (= reduction, 1−z).

*Evidence vs interpretation:* all "true"/"false" verdicts above rest on direct
reading of the code and/or the executed run. Items marked **[inference]** in
Part 1 (#35) and the CFR-mechanism explanation in #33 are analytical
interpretations, flagged as such. Repository conventions were **not** treated as
evidence of scientific correctness — the dynamical claims were confirmed by
reproducing the notebook's numbers independently.
