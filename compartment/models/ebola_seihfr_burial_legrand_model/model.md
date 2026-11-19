# Ebola – SEIHFR, burial (Legrand et al. 2007)

A deterministic port of the mean-field equations behind Legrand J, Grais RF,
Boelle PY, Valleron AJ, Flahault A. **"Understanding the dynamics of Ebola
epidemics."** *Epidemiol Infect.* 2007;135(4):610-621.
[doi:10.1017/S0950268806007217](https://doi.org/10.1017/S0950268806007217)
(PMC2870608).

This model's distinguishing feature is explicit separation of
**community, hospital, and traditional-burial** transmission routes, each
with its own transmission coefficient.

## Model structure

- **Compartments:** `S` (susceptible), `E` (exposed, not yet infectious),
  `I` (symptomatic, infectious in the community), `H` (hospitalised,
  infectious), `F` (dead but not yet buried, infectious during traditional
  burial), `R` (removed — recovered or safely buried).
- **Transmission:** the force of infection sums three separate routes,
  each with its own coefficient — `βI` (community), `βH` (hospital),
  `βF` (traditional burial) — following the paper's equations exactly.
- **Branching from I and H:** the proportion of cases hospitalised (θ)
  and the overall case-fatality ratio (δ) are user-set *targets*; the
  underlying per-step branching rates (θ1, δ1, δ2 in the paper's
  notation) are derived from θ, δ, and the mean sojourn times (γh, γi,
  γd) using the paper's own algebraic relationships, and are recomputed
  every simulation step.

## Key assumptions carried over from the source paper

- Homogeneous mixing within the population — no age structure, no
  spatial/travel coupling. The paper itself flags this as a
  simplification (household/network structure and super-spreading are
  not represented).
- The entire population starts susceptible.
- Interventions act at full strength from their start date onward (a step
  function), not gradually — matching the paper's own assumption (b).
- After interventions start, each route's transmission coefficient is
  reduced by its own fixed factor. This is implemented as three independent
  interventions (`community_intervention`, `hospital_intervention`,
  `funeral_intervention`) rather than one shared intervention, since each
  route's reduction magnitude differs. The paper's assumption (d) removes
  hospital and funeral transmission completely and reduces community
  transmission by a smaller factor; the defaults here are weaker (see
  "Differences from Legrand 2007").

## Known limitation vs. the historical outbreaks

Run with the DRC 1995 parameters at a population of 100,000 (the same
population the paper itself uses for its sensitivity analysis) and the
real historical intervention date (4 May 1995), this deterministic,
homogeneously-mixed model produces a substantially **larger** epidemic
than the ~315 cases actually observed in Kikwit. This is a well-documented
property of the *published* mean-field version of this model, not a
translation error here — the paper's own Discussion explicitly attributes
it to the homogeneous-mixing assumption breaking down in reality (contact
structure, household clustering), and independent teaching materials
reproducing this exact model with these exact published parameters note
the same over-prediction ("our best guesses at parameter values are
somewhat pessimistic," J. Drake & P. Rohani, *Sensitivity analysis of
deterministic models... A model for the spread of Ebola virus disease*).
The basic reproduction number produced by these parameters (R0 ≈ 2.7,
decomposed into community/hospital/burial components matching the paper's
own Table 4 almost exactly, with δ = 81%; at this model's 50% default
δ, R0 ≈ 2.0) is correct; it is the size of the well-mixed
population assumed to be exposed that drives the discrepancy in absolute
case counts.

## What this implementation cannot faithfully represent

The paper's own multivariate sensitivity analysis (Figure 3) explores
*improving* care after an intervention date — e.g. a **faster** mean
time to hospitalisation, or a **higher** hospitalisation proportion,
post-intervention. The framework's intervention mechanism only supports
proportional *reductions* to a named rate, not replacing a rate with an
independent post-intervention value or increasing it. Faster/greater
hospitalisation after an intervention therefore is not modelled as a
switchable intervention here; `gamma_h` and `theta_target` are ordinary
user-editable parameters (with uncertainty ranges) that apply for the
whole run, so users can still explore this sensitivity by running the
model multiple times with different fixed values, or via the framework's
built-in parameter-uncertainty (Latin Hypercube) mode — which mirrors the
paper's own LHS-based sensitivity method reasonably closely.

## Parameters

The parameters users can change, with their default (as set in `define_parameters()`, which is also the scenario in `example-config.json`), what each one means, and where the value comes from. "Legrand 2007" refers to Legrand J, Grais RF, Boëlle PY, Valleron AJ, Flahault A. Understanding the dynamics of Ebola epidemics. *Epidemiol Infect.* 2007;135(4):610–621. doi:10.1017/S0950268806007217. The paper's numbered references ([3], [18], …) are listed below the table. The "Range" column is the default uncertainty range (`default_min`–`default_max`) and is an assumption unless noted.

| Parameter | Default | Range | Meaning | Reference |
|---|---|---|---|---|
| `alpha` (α) | 7 days | 5–9 | Mean duration of the incubation (exposed, non-infectious) period. | Legrand 2007, Table 3 (7 days, mean), citing [18]–[20]. Range: assumption. |
| `gamma_f` (γf) | 2 days | 1–3 | Mean duration between death and (safe or traditional) burial. | Legrand 2007, Table 3 and Methods assumption (e) (2 days, mean). No primary data source; the paper describes it as "approximately the average duration from death to burial". Range: assumption. |
| `betaI` (βI) | 0.084 per day | 0.06–0.313 | Transmission coefficient for contact with symptomatic cases in the community, before interventions. | Legrand 2007, Table 4: 0.588 per week, fitted by approximate maximum likelihood; ÷ 7 = 0.084 per day. Range: the paper's 95% CI (0.420–2.191 per week) ÷ 7. |
| `betaH` (βH) | 0.113 per day | 0.0001–0.584 | Transmission coefficient for contact with hospitalised cases (including isolation ward), before interventions. | Legrand 2007, Table 4: 0.794 per week; ÷ 7 = 0.1134, rounded to 3 decimal places. Range: the paper's 95% CI (0.001–4.091 per week) ÷ 7. |
| `betaF` (βF) | 1.093 per day | 0.0001–1.428 | Transmission coefficient for contact with the body during traditional burial, before interventions. | Legrand 2007, Table 4: 7.653 per week; ÷ 7 = 1.0933, rounded to 3 decimal places. Range: the paper's 95% CI (0.001–9.997 per week) ÷ 7. |
| `gamma_h` (γh) | 5 days | 1–9 | Mean duration from symptom onset to hospitalisation, for cases who are hospitalised. | Legrand 2007, Table 3 (5 days, mean), citing [3]. Range: assumption (the paper's sensitivity analysis varied it from 1 to 5 days). |
| `gamma_i` (γi) | 10 days | 7–15 | Mean duration from symptom onset to end of infectiousness for cases who survive. Must exceed 1/gamma_h. | Legrand 2007, Table 3 and Methods assumption (e) (10 days, mean), citing [20], [33]. Range: assumption. |
| `gamma_d` (γd) | 9.6 days | 7–12 | Mean duration from symptom onset to death for cases who die without being hospitalised. Must exceed 1/gamma_h. | Legrand 2007, Table 3 (9.6 days, mean), citing [3]. Range: assumption. |
| `theta_target` (θ) | 80% | 50–100 | Target proportion of infectious cases who are hospitalised. The per-step hospitalisation rate (theta1) is derived from this target together with gamma_h, gamma_i, gamma_d, and delta — not set directly. | Legrand 2007, Table 3 (80%; the paper's derived θ1 = 0.67), citing [3]. Range: assumption. |
| `delta_target` (δ) | 50% | 10–90 | Target overall case-fatality ratio across hospitalised and non-hospitalised cases. The per-compartment death-split rates (delta1, delta2) are derived from this target. | Model default, not taken from Legrand 2007. Legrand 2007, Table 3 reports 81% for DRC 1995 (the paper's derived δ1 = δ2 = 0.80), citing [3]; Gulu, Uganda 2000 was 53% [4]. Set this to 81 to reproduce the paper's DRC parameter set. Range: assumption. |
| `community_intervention` | 50% adherence, 100% reduction | 0–100 each | Reduces βI by adherence × reduction (50% with the defaults) from its start date. | Model default (assumption). Legrand 2007, Table 4 estimates the DRC 1995 community reduction (1 − z) at 12% (95% CI 0–78). |
| `hospital_intervention` | 50% adherence, 100% reduction | 0–100 each | Reduces βH by adherence × reduction (50% with the defaults) from its start date. | Model default (assumption). Legrand 2007, Methods assumption (d), sets hospital transmission to 0 after interventions. |
| `funeral_intervention` | 50% adherence, 100% reduction | 0–100 each | Reduces βF by adherence × reduction (50% with the defaults) from its start date. | Model default (assumption). Legrand 2007, Methods assumption (d), sets burial transmission to 0 after interventions. |

The example config starts all three interventions on 4 May 1995, the DRC 1995 intervention date in Legrand 2007, Table 3, citing [29].

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

- **Case-fatality ratio:** the default is 50%; the paper reports 81%. This lowers deaths and the burial part of R0, because R0F = βF × (1/γf) × δ: at δ = 81%, R0 ≈ 2.69 (community 0.50 + hospital 0.42 + burial 1.77), matching the paper's Table 4 value of 2.7; at δ = 50%, R0 ≈ 2.03 (burial 1.09).
- **Intervention strength:** the defaults (50% adherence, 100% reduction) halve community, hospital and burial transmission. The paper removes hospital and burial transmission completely and estimates a 12% community reduction for DRC. To reproduce the paper's scenario, set adherence to 100% for all three and the reductions to 12% (community), 100% (hospital) and 100% (burial).

## Change history

- Oct 2026 — Label changed from "Ebola (SEIHFR, Community/Hospital/Burial) — Legrand et al. 2007" to "Ebola – SEIHFR, burial (Legrand et al. 2007)"; by laiyingsilaiyingsi.
- Oct 2026 — Added the parameter reference table (sources from Legrand et al. 2007, Tables 3–4) and a list of differences from the paper's DRC 1995 values. β uncertainty ranges set to the paper's 95% CIs ÷ 7 (previously 0.064–0.104, 0.07–0.15 and 0.08–1.4). Example config aligned with the `define_parameters()` defaults: δ = 50% (was 81%), interventions 50% adherence with 100% reduction for all three (was 100% adherence with 12%/100%/100%), incubation uncertainty range 5–9 days (was 4–10); by Yingsi Lai (laiyingsilaiyingsi). Intervention defaults in `define_parameters()` changed from 90%/90%/100% to 100% reduction for all three, keeping 50% adherence.
