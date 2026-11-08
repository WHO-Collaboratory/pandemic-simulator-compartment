# Cholera – SVIRB, reactive OCV (extends Maity et al. 2023)

## Model overview
This model is built on the SVIRB cholera model of **Maity, Saha, Ghosh & Chattopadhyay
(2023), Bull Math Biol 85:55** (Eq. 1), which was fitted to Lusaka's 2017–18 outbreak.
Infected people shed *V. cholerae* into the water (B). Susceptibles are infected in two
ways: from contaminated water (environment-to-human, saturating in B) and from other
infected people (human-to-human).

**What comes from the paper:** the SVIRB-with-water structure (Eq. 1), all parameter values
(Tables 1–2, including the wave-1 fitted β_e, β_h, ρ and ν), and its controls: treatment,
sanitation and bacteria removal, constant from day 0 in the paper.

**What was added here (not in the paper):**
- **The reactive OCV campaign.** The paper models vaccination only as a constant rate
  φ = 0.05/day from day 0; it does not model the January 2018 campaign. The dated campaign
  (doses per day shared between S and R, two modes by dates or by daily speed, the
  7-day time to protection) and the VI and VR compartments it needs were added for this
  model, following Pasetto 2018, Lee 2020, Azman 2015 and Chao 2011, with the campaign's
  dates and doses from WHO AFRO (2018).
- Each of the paper's controls as a separate, dated intervention, with sanitation split
  into a water route and a contact route.
- In the preset, the paper's sanitation (ρ = 16% on both routes) acts from 28 Dec 2017,
  the documented strengthening of the response, instead of from day 0 (see
  [Sanitation from 28 December](#sanitation-from-28-december-preset)).
- The per-week reading of the fitted rates, the fitted N and I(0), and the balanced B(0).

The full list is in [Differences from the source paper](#differences-from-the-source-paper).
With φ = 0.05, no OCV campaign, ρ_e = ρ_h and the paper's controls over the whole run, the
model is exactly the paper's Eq. 1.

**What it is for:** comparing reactive OCV campaigns (earlier, faster, wider) against a
background of water, hygiene and treatment measures, with the paper's transmission values,
in a single, well-mixed urban population. Deterministic ODEs; one or several independent
admin zones. **Not for estimating the cases averted by the 2018 Lusaka campaign:** how much
of the January 2018 decline the campaign explains depends on when, and by how much,
sanitation was strengthened, which the data cannot tell apart (about 560 to 1,440 cases
averted under the settings in [Sensitivity](#sensitivity-timing-of-sanitation)).

Preset: **Lusaka, first wave**, run from 22 Oct 2017 to 4 Feb 2018 (day 0 = 22 Oct 2017),
compared with 15 weeks of reported cases (Sunday–Saturday weeks ending 28 Oct 2017 …
3 Feb 2018; see [Data](#data)).

## Compartments
| Compartment | Meaning | Unit |
|---|---|---|
| S | Unvaccinated susceptible | people |
| V | Vaccinated, not infected (partly protected, leaky) | people |
| I | Unvaccinated infected (cases); infective | people |
| VI | Vaccinated infected (breakthrough cases); infective | people |
| R | Unvaccinated recovered (immune for a while) | people |
| VR | Vaccinated recovered or immune | people |
| B | *V. cholerae* in water | cells/mL |

N = S + V + I + VI + R + VR is constant (births = natural deaths; no cholera deaths).
I + VI is the paper's I, and R + VR is the paper's R. All flows are written by hand in
`equation()`: the force of infection has two routes and the OCV doses depend on S/(S+R),
which a single transmission edge cannot express. The transmission parameters only provide
the rates and the arrows in the platform's diagram.

## Equations
λ = (1 − ρ_e)·β_e·B/(k + B) + (1 − ρ_h)·β_h·(I + VI)

- dS/dt = μN − λS − φS − u(t − T_p)·S/(S+R) + αV + ω(R + VR) − μS
- dV/dt = φS + u(t − T_p)·S/(S+R) − σλV − (α + μ)V
- dI/dt = λS − (τ + γ + μ)I
- dVI/dt = σλV − (τ + γ + μ)VI
- dR/dt = (τ + γ)I − u(t − T_p)·R/(S+R) − (ω + μ)R
- dVR/dt = (τ + γ)VI + u(t − T_p)·R/(S+R) − (ω + μ)VR
- dB/dt = ξ(I + VI) − (δ + ν)B

σ = 1 − vaccine efficacy. u(t) is the number of OCV doses given on day t and T_p the time
to protection (7 days), so u(t − T_p) is the doses taking effect on day t; the S/R split
uses S and R at that time. ρ_e, ρ_h, ν, τ and u are set by the interventions below. With
φ = 0.05, no OCV campaign, ρ_e = ρ_h and the paper's controls over the whole run, the model
is exactly the paper's Eq. 1.

## Parameters
All defaults are the values of `define_parameters()` and the preset, except the zone
fields, where the preset's value is given (the framework's own default for starting cases
is 0.05% of the zone population). Range = the uncertainty range used in uncertainty runs
(`default_min`–`default_max`; see [Uncertainty runs](#uncertainty-runs)); "—" = not varied.

| Parameter | Symbol | Default | Range | Meaning | Reference |
|---|---|---|---|---|---|
| `beta_e` | β_e | 0.0082857 per day (paper 0.058, read per week) | 0.0074–0.0102 | Environment-to-human transmission rate (maximum, B ≫ k) | Maity 2023, Table 2 (fitted; 95% CI ÷ 7, rounded outward) |
| `beta_h` (S → I arrow) | β_h | 2.0×10⁻⁷ per person per day (paper 1.4×10⁻⁶, read per week) | 1.82–2.15×10⁻⁷ | Human-to-human transmission rate per infected person (I + VI) | Maity 2023, Table 2 (fitted; 95% CI ÷ 7, rounded outward) |
| `nu` | ν | 0.257143 per day (paper 1.8, read per week) | 0.23–0.3 | Removal of bacteria from water while the bacteria-removal intervention is on | Maity 2023, Table 2 (fitted; 95% CI 1.6–2.1 ÷ 7 = 0.229–0.3, rounded to 0.23–0.3) |
| `treatment_days` | 1/τ | 3 days | 2.5–3.75 | Time to treatment; treated people stop being infectious | Maity 2023, Table 1 (Miller Neilan 2010); range = τ ± 20%, as in the paper's sensitivity analysis |
| `gamma` (I → R arrow) | 1/γ | 5 days | — | Natural recovery period (also VI → VR) | Maity 2023, Table 1 |
| `omega` (R → S arrow) | 1/ω | 730 days (2 years) | — | Duration of natural immunity (R and VR → S) | Maity 2023, Table 1 |
| `phi` (S → V arrow) | φ | 0 per day | — | Constant background vaccination of S. Paper: 0.05/day; 0 here because the OCV campaign replaces it | Maity 2023, Table 1 (Miller Neilan 2010; Sardar 2013) |
| `xi` | ξ | 10 cells/mL/day per person | — | Shedding of *V. cholerae* by each infected person (I and VI) | Maity 2023, Table 1 |
| `bacteria_lifetime` | 1/δ | 30 days | — | Natural lifetime of *V. cholerae* in water | Maity 2023, Table 1 |
| `half_saturation` | k | 10⁶ cells/mL | — | Concentration giving half the maximum infection risk | Maity 2023, Table 1 (Mukandavire 2011; Hartley 2006; Posny 2015) |
| `life_expectancy` | 1/μ | 43.5 years | — | Life expectancy (births = deaths) | Maity 2023, Table 1 |
| `vaccine_efficacy` | 1 − σ | 67 % | — | OCV vaccine: efficacy (leaky) | Maity 2023, Table 1. Lusaka 2016, one dose: 88.9% (95% CI 42.7–97.8, matched case-control; Ferreras 2020) |
| `time_to_protection` | — | 7 days | — | OCV vaccine: days from a dose until protection | Assumption, following Pasetto 2018 ("immunity one week after administration"); other studies differ (see the Delay note) |
| `alpha` (V → S arrow) | 1/α | 912.5 days (2.5 years) | — | OCV vaccine: duration of protection | Maity 2023, Table 1 |
| `infected_population` (zone) | I(0) | Preset 0.00025535 % (framework default 0.05 %) | — | Infected at the start, % of the zone population (preset: 3.4 people) | Fitted here (not reported in the paper) |
| `vaccinated_population` (zone) | V(0) | 0 % | — | Share of the non-infected population in V at the start: V(0) = share × (N − I(0)) | Assumption: no vaccination before the outbreak |
| `initial_water_mode` (zone) | — | Balanced with starting cases | — | How B(0) is set in the zone (dropdown): "Balanced with starting cases" = computed, ξ·I(0)/(δ + ν); "As entered" = the concentration field below | See Initial conditions |
| `bacteria_concentration` (zone) | B(0) | 0 cells/mL | — | *V. cholerae* in water at the start; used only with "As entered" | Entered by the user (e.g. 0 = clean water) |

The three "OCV vaccine" parameters describe the vaccine and are set together when another
vaccine is modelled. They are model parameters rather than fields of the OCV intervention:
the intervention form only has the framework's generic fields, and the framework reads its
threshold fields as prevalence thresholds when it builds the platform's intervention
timeline, so storing the vaccine there would show the campaign as switched off on its
first day.

**The three fitted rates are read per week and entered per day** (Table 2 value ÷ 7), so
every rate in the model is per day and no unit conversion is hidden in the code. The
per-week reading is ours, as Table 1 prints
"day⁻¹" for β_e, β_h and ν. Two checks support it:
- **ν:** the ratio R_e/R_h of the paper's Eq. 3, ξβ_e/((δ + ν)kβ_h), does not depend on N
  or the initial state. A common unit factor on β_e and β_h cancels, so the ratio fixes
  only ν's unit: with ν read per week it is 1.43 (Table 2: 1.48); read per day, 0.23.
- **β_e and β_h:** the case data cannot decide their unit. Both enter as β·N, so N
  absorbs a factor of 7: read per day, with N and I(0) refitted and the same 98.5%
  coverage, the model fits the data as well (RMSE 80.3, against 81.3 with the per-week
  reading), but only with N ≈ 193,000.
  That is fewer than the 1,319,276 people vaccinated in round 1 and the 1,205,915 target
  population (WHO AFRO, 2 Feb 2018), so the per-day reading is not consistent with the
  campaign. **The paper's own R0 points the other way:** it is computed with φ = 0.05 at
  the disease-free state and an N the paper does not report, and reproducing it needs
  N ≈ 3.9 M with the per-week reading of β_e and β_h, more than the population of Lusaka
  province, but only ≈ 0.56 M with the per-day one.

The per-week reading of all three rates is therefore this model's choice. For ν it is
supported by the R_e/R_h ratio. For β_e and β_h it rests on the campaign size alone, and
the paper's R0 argues against it; the two cannot both hold with the paper's printed values,
so one of the paper's reported quantities is not reproduced either way.

## Interventions
| Intervention | Form | Mechanism | Default (paper / preset) |
|---|---|---|---|
| **Treatment (case management)** | Dates (optional) | Infected people (I, VI) also stop being infectious at τ = 1/`treatment_days`; a case is infectious ~1.9 days instead of 5. Acts throughout, **also in the control run**; dates only restrict it | Whole run |
| **Safe water & sanitation** | Adherence, Transmission reduction (both required), dates | β_e × (1 − Adherence × Transmission reduction); ρ_e = Adherence × Transmission reduction | Paper: 100% × 16% (ρ = 0.16, 95% CI 0.13–0.22) from day 0. Preset: the same from 28 Dec 2017 |
| **Hygiene & contact measures** | Adherence, Transmission reduction (both required), dates | β_h × (1 − Adherence × Transmission reduction); ρ_h likewise | Paper: 100% × 16% from day 0. Preset: the same from 28 Dec 2017 |
| **Bacteria removal from water** | Dates only | Inside the dates bacteria are also removed at ν (`nu`); outside them only natural death δ acts | Whole run |
| **Reactive OCV campaign** | Target coverage, daily vaccination speed, start date, end date | u(t) doses per day (below) | Coverage 98.4892% (1,319,276 doses), 10–20 Jan 2018 (doses on days 80–89, 10–19 Jan; protection 17–26 Jan) |

Treatment differs from the other four: leaving it out of a run does not remove it, and
only its dates can restrict it. This is a choice made for this model: in the paper τ, ρ
and ν are all constants, and the paper does not single treatment out. Keeping treatment
in the control run makes "no interventions" mean no sanitation and no vaccination, while
cases are still treated. The platform's intervention timeline does not show this
treatment, so a control run's timeline is empty although treatment acts in it.

### Sanitation from 28 December (preset)
In the paper, ρ = 16% acts from day 0. In the preset, safe water & sanitation and hygiene
& contact start on **28 Dec 2017** and bacteria removal and treatment act over the whole
run. The paper's 16% is read as the level reached after the response was strengthened in
late December, with no such reduction before. This is this model's reading, made because,
with the data on their true dates, sanitation from day 0 leaves nothing to start the
decline before the vaccine protects (from 17 Jan): the model then peaks two weeks after
the reported peak (RMSE 118.4). With sanitation from 28 Dec it peaks one week after (RMSE
81.3). The date is documented; the size of the step is not:
- Statutory Instrument No. 79 of 2017, the Public Health (Infected Areas) (Cholera)
  Regulations, commenced on 28 Dec 2017 (ZambiaLII). In infected areas it bans public
  ceremonies and gatherings of more than five people (other than families) without
  written permission (reg. 8), and lets officers prohibit food trading in unsanitary
  conditions and close premises without adequate sanitation (regs. 9, 10, 12). Under it
  markets were closed, mass gatherings and street vending of food were prohibited, school
  opening was delayed and premises inspections enhanced (Kapata 2018).
- The ban on street vending and the closure of Soweto and other affected markets were
  announced on 29 Dec (Daily Nation, 30 Dec 2017).
- Chlorine in the municipal water system was raised from late December, and 282
  chlorinated emergency water tanks were placed from December (MMWR 2018).
- An 18:00–06:00 curfew in Kanyama from Sunday 7 Jan (Reuters, January 2018).

The two sanitation interventions carry these measures: water (chlorination, tanks) and
contact (gatherings, vending, markets). The measures started between December and
7 Jan; the single step on 28 Dec stands for all of them (the 6 Jan row in
[Sensitivity](#sensitivity-timing-of-sanitation) shows a later step). Earlier measures from October (case management,
chlorine distribution) are not represented by a separate level, because the framework keeps
one intervention per type and no effect size is reported for them.

### OCV campaign
The campaign is set in the intervention form: target coverage c, daily
vaccination speed v and dates. The framework gives every intervention the same field names
and carries only those values to the model, so the form's fields are relabelled:
"adherence" holds c and "transmission reduction" holds v. The vaccine itself is set in the
three "OCV vaccine" model parameters. Whether an end date is set decides how doses are
given:

> **End date filled → doses spread evenly over (end date − start date) days, and the daily
> speed is ignored. End date empty → doses at the daily speed until the target coverage.**
> The form cannot hide one field when the other is filled, so this rule is written in both
> field labels. With no end date and no speed, the platform fills the speed with 5%/day.

**The end date is not a vaccination day.** Campaign days = end date − start date, as in
the formula below; to vaccinate on 20 Jan as well, enter 21 Jan. The preset (10–20 Jan) is
10 days, matching WHO AFRO's report that round 1 ran 10 days and closed on 20 Jan. At site
level it was less even: sites closed after day 8 (Chipata), day 9 (Chawama, Matero) and
day 11 (Kanyama, so 20 Jan was a vaccination day there) (WHO AFRO, 2 Feb 2018). The other
interventions' end dates are included in their windows.

| | With an end date (preset) | Without an end date |
|---|---|---|
| Inputs | Coverage c, start t₀, end t₁ (speed unused) | Coverage c, speed v (% of population per day), start t₀ |
| Doses per day u | c·N ÷ (t₁ − t₀) | v·N |
| Duration | t₁ − t₀ days | c ÷ v days (stops at the coverage) |
| Literature | Lee 2020 (model 4: about the same number of people each day from start to end date); Pasetto 2018 (doses deployed equally over the campaign days); Azman 2015 (fixed 10-day campaign) | Chao 2011 (a number of people vaccinated per day until the coverage is reached) |

- **Who is vaccinated:** unvaccinated people without symptoms, S and R, in proportion to
  their size, as in Pasetto 2018 (doses u(t)·S/(S+R) and u(t)·R/(S+R) in this file's
  notation, with separate vaccinated compartments). V, I, VI and VR are not vaccinated. Doses to S move people
  to V; doses to R move them to VR (already immune, now marked vaccinated). Azman 2015
  instead gives doses to all unvaccinated states, infected people included.
- **Delay:** a dose given on day t takes effect on day t + time to protection; until
  then the person is infected at the unvaccinated rate. The S/R split is taken when the
  dose takes effect, not when it is given, so a dose given to someone infected during
  the delay goes to another person instead of being wasted; this slightly favours the
  vaccine. The 7 days follow Pasetto 2018 ("immunity one week after administration").
  Other studies differ: Azman 2015 assumes protection immediately after vaccination in its
  main analysis, and Chao 2011 lets efficacy reach 50% of its maximum 10 days after the
  dose. A zero delay is not plausible here, and the 7 days are kept on these grounds,
  although the fit is much better without it: with no delay and N and I(0) refitted
  (doses kept at 1,319,276) the RMSE is 60.4 (N = 1,433,988, I(0) = 0.8) instead of 81.3.
  The zero-delay fit is better only because the campaign then acts from 10 Jan and stands
  in for the decline that the data show before protection began (see Limits).
- The same doses over the same days give identical results (10–20 Jan with an end date =
  no end date with v = c ÷ 10 ≈ 9.85%/day).
- If S + R falls below the planned doses, only the people there are vaccinated (the
  per-capita rate is capped at 5 per day); doses not given are not carried over.
- Coverage can be at most 99%.

The framework gives every intervention the same generic fields. `_keep_intervention_fields`
in `model.py` shows only the fields each intervention uses and relabels them. It edits the
schema builder's private intervention list (no public API yet); if that structure changes,
the generic fields come back with a warning. The rendering of the trimmed forms needs
checking in UAT (Model Approvals → Simulate).

## Initial conditions
Per admin zone, from the zone fields (`get_initial_population()`):
- I(0) = `infected_population` % × N
- V(0) = `vaccinated_population` % × (N − I(0))
- S(0) = N − I(0) − V(0)
- B(0): see below
- VI(0) = R(0) = VR(0) = 0

I(0) is the user's input; **B(0) follows from it** unless the user chooses otherwise. Both
are initial-state fields of the admin zone; the zone's dropdown "Initial V. cholerae in
water" (`initial_water_mode`) says how B(0) is set:
- **Balanced with starting cases (default):** the model computes B(0) = ξ·I(0)/(δ + ν)
  (≈ 34 × the starting cases with the defaults), the concentration at which shedding by
  the starting cases equals natural death plus removal at the paper's rate ν, so
  dB/dt = 0 at day 0 when bacteria removal is on from the start. ν is the model parameter,
  whatever interventions the run contains, so all scenarios start from the same water (in
  a run without bacteria removal the water then fills up from day 0). The concentration
  field is not used. B(0) is recomputed from the current ξ, δ and ν in every
  run, including each sample of an uncertainty run.
- **As entered:** the zone's "Initial V. cholerae concentration" (`bacteria_concentration`)
  is used, e.g. 0 for clean water at the start.

The value computed for the balanced start is applied in `prepare_initial_state()`,
because it needs the model parameters ξ, δ and ν, which `get_initial_population()` does
not receive.

The balanced start is a choice, not a result: bacteria last 3.4 days and a case is
infectious about 1.9 days, so the water is not much faster than infection. Starting with
clean water (B(0) = 0) delays the epidemic: with the preset's I(0) = 3.4 the RMSE becomes
183.8, and the same curve needs about twice as many starting cases (I(0) = 7.2).

The paper reports neither N nor the initial state; in the preset, N and I(0) were fitted
with B(0) tied to I(0) in this way.

## Several admin zones (no mobility)
The model has **no mobility component**: `build_travel_matrix()` is not overridden, the
travel matrix is not used in the equations, and there are no travel parameters or
movement-restriction interventions. With two or more admin zones, each zone is simulated
on its own; the result for a zone is the same as running that zone alone (tested). Points
to watch:
- **No spread between zones.** A zone that starts with no cases and clean water stays
  free of cholera, however large the epidemic next to it. Importations, travel and shared
  water sources are not represented; for those, a model with mobility is needed.
- **Set the zone fields for every zone.** Each zone takes its own population, starting
  cases and initially vaccinated share. With "Balanced with starting cases" (default)
  B(0) is computed in each zone from its own starting cases; with "As entered", enter
  that zone's starting *V. cholerae* concentration. The choice is made per zone.
- **Parameters and interventions are shared.** All zones use the same transmission
  rates and vaccine, and every intervention has the same dates and values in every zone.
  The OCV campaign vaccinates the same coverage in each zone (c × that zone's population),
  so campaigns targeted at some zones only, or run at different times, cannot be set.
- **The fitted values are Lusaka's.** β_e, β_h, ν and ρ were fitted to one city's first
  wave; zones elsewhere may need other values (entered as one set for all zones).
- **Totals** for a parent area are the sums over its zones. This is meaningful for the
  people compartments, but not for B, which is a concentration.

## Outputs
- The seven compartments over time, per zone.
- Cumulative totals, each counting everything that enters the compartment:
  - `I_total`: new cases among unvaccinated people (S → I).
  - `VI_total`: new breakthrough cases (V → VI). **All new cases = I_total + VI_total**;
    this is what is compared with reported cases.
  - `V_total`: vaccinations of susceptibles (S → V; doses to R are not counted).
  - `R_total`: recoveries of I; `VR_total`: recoveries of VI plus doses to R.
  - `S_total`: returns to S as natural and vaccine immunity wane (births not counted).
- The platform's per-compartment summary ("deltas") uses these totals: I and VI are new
  cases (add them for all cases), V vaccinations of susceptibles, R and VR entries, S
  returns to S. For B it is the final concentration (with several zones, the sum of the
  zones' final concentrations, which has no meaning).
- VR returns to S when natural immunity wanes (1/ω = 2 years), so a vaccinated person
  who recovered loses the vaccinated mark then, although vaccine protection lasts
  2.5 years. For recovered breakthrough cases this is needed for the reduction to Eq. 1;
  for doses given to R it is a modelling choice that makes those doses give no
  protection at all. Neither matters over one wave.

## Data
The 15 weekly counts (5, 7, 10, 50, 113, 128, 203, 230, 338, 525, 717, 642, 340, 178, 178;
3,664 cases) are digitised from the paper's Fig. 2a and kept as the paper shows them. They
are **Sunday–Saturday weeks ending 28 Oct 2017 … 3 Feb 2018**: the labels of the figure's
time axis (28/10/17, 16/12/17, 03/02/18, 24/03/18, 12/05/18) are Saturdays, the last day
of Zambia's epidemiological weeks, and the paper's data end on 12 May 2018 (also a
Saturday). Official weekly counts confirm this alignment; read as week starts, the
differences reach 305 cases:

| Week (Sun–Sat) | Official | Digitised | Source |
|---|---|---|---|
| 26 Nov–2 Dec 2017 | 136 | 128 | WHO Disease Outbreak News, 11 Dec 2017 |
| 31 Dec 2017–6 Jan 2018 | 709 | 717 | Mazaba 2018 (Health Press Zambia editorial; see below) |
| 7–13 Jan | 636 | 642 | WHO, quoted on ReliefWeb, 26 Jan 2018 |
| 14–20 Jan | 337 | 340 | WHO AFRO situation report No. 114, 2 Feb 2018 |
| 21–27 Jan | 173 | 178 | WHO AFRO situation report No. 114, 2 Feb 2018 |
| 28 Jan–3 Feb | 140 | 178 | WHO AFRO situation report No. 115, 3 Feb 2018 |

Caveats: the digitised counts differ from the official ones by more than a few cases in
some weeks (last week 178 against 140; weeks 1 and 2 are read as 5 and 7, while WHO
reported fewer than 5 cases a week from 21 Oct to 4 Nov). The official 709 is printed in the editorial as
"709 cases/100,000 population", evidently a misprint for cases; WHO gave "more than 650"
and IFRC "over 700" for that week. The paper describes the data as Lusaka province, while
the situation reports give Lusaka district, so the scope may differ slightly. The
digitised counts are kept, so that the model is compared with the data the paper used.

One statement in the paper points the other way: its Table 2 gives wave 1 as 28 Oct 2017 –
10 Feb 2018 and wave 2 from 11 Feb, which fits reading the labels as week starts. The
official counts above support the week-ending reading for six weeks across the wave (read
as week starts, they differ by up to 305 cases), so they are taken as decisive here.

So day 0 is Sunday 22 Oct 2017 and model week k covers days 7k to 7k + 6. An earlier
version of this model read the labels as week starts, which placed every count one week
too late (and the OCV campaign one week too early relative to the cases).

## Example scenario (example-config.json)
Lusaka's first wave with the paper's controls (sanitation from 28 Dec 2017) and OCV round
1, compared with the 15 weekly counts above. All paper parameters and intervention values
are fixed; only the two values the paper does not report were fitted, by least squares,
with the reported 1,319,276 doses kept:

| Input | Value | Note |
|---|---|---|
| N | 1,339,513 | Population at risk (fitted). The campaign then covers 98.5% of it, which a single homogeneous population needs to match the data although round 1 targeted four sub-districts (see Limits). For comparison, round 1's target population in the four sub-districts was 1,205,915 by head count (WHO AFRO, 2 Feb 2018) |
| I(0) | 3.4 (`infected_population` 0.00025535 %) | Infected on 22 Oct 2017 (fitted) |
| OCV coverage c | 98.4892% | 1,319,276 vaccinated in round 1 (WHO AFRO, 2 Feb 2018) ÷ N |
| Sanitation dates | 28 Dec 2017 – 4 Feb 2018 | Safe water and hygiene & contact, 100% × 16% each; treatment and bacteria removal over the whole run |
| OCV dates | 10–20 Jan 2018 | Launched 10 Jan (WHO AFRO, 10 Jan 2018); 10 days, closed 20 Jan (WHO AFRO, 2 Feb 2018) |
| B(0) | 117.8 cells/mL | Computed ("Balanced with starting cases"), ξ·I(0)/(δ + ν); the concentration field stays at its default 0 and is not used |
| Implied R0 | 1.219 before 28 Dec (R_e 0.716, R_h 0.502); 1.024 from 28 Dec (R_e 0.602, R_h 0.422) | Paper Table 2: 1.0309 (R_e 0.6155, R_h 0.4154). Both R_e and R_h are proportional to N; the paper's value is at its disease-free state with φ = 0.05 (a factor 0.345 here) and an N it does not report |
| V(0), VI(0), R(0), VR(0) | 0 | No vaccination or prior immunity before the outbreak |

Every other value in the config equals the default in the parameter table.

## Validation (wave 1)
- **Implementation:** matches a SciPy solution of the same equations, and the paper's Eq. 1
  with φ = 0.05 (`tests/test_cholera_svirb_ocv_maity_model.py`).
- **Several zones:** a run with three zones gives each zone the same result as running it
  alone; a zone without cases stays at zero.
- **Fit:** RMSE 81.3 cases/week (the paper's own fit, with constant φ = 0.05 from day 0:
  65.8, Fig. 3 caption; this model with sanitation from day 0: 118.4); 3,653 cases vs 3,664
  reported, of which 146 are breakthrough cases (VI). The model peaks in the week ending
  13 Jan (558), one week after the reported peak (717, week ending 6 Jan), and stays at 518
  in 14–20 Jan, when reports had already halved (340).
- **Vaccination:** by 4 Feb the platform gives 1,315,984 doses to S (`V_total`) and about
  3,289 to R, 1,319,273 of the 1,319,276 doses. The platform solves the equations in 32-bit
  floating point, so totals in the millions can be off by a few units; an independent
  64-bit SciPy solution of the same equations gives 1,315,987 and 3,289 (all 1,319,276
  doses). Case totals agree to the decimal (3,653.3).
- **Time to protection:** the 7 days follow Pasetto 2018 and are kept because a zero delay
  is not plausible. Without the delay (doses kept, N and I(0) refitted) the RMSE would be
  60.4; with the preset's N and I(0), 92.5.
- **Platform outputs (tested):** the intervention timeline shows the sanitation
  interventions on from 28 Dec and the OCV campaign on from 10 Jan and off on 20 Jan. The
  deltas (the audit tool's "Compartment deltas" table) are, with interventions: I 3,507 and
  VI 146 (3,653 cases), V 1,315,984, R 3,509, VR 3,426 (about 3,289 doses to R plus about
  137 recovered breakthrough cases), S 18,771 (returns to S as immunity wanes) and B 789
  (final concentration, cells/mL); control run: I 635,032, R 621,676, S 28,123, B
  4,483,087.
- With the paper's own vaccination assumption (φ = 0.05/day from day 0, no campaign, the
  paper's controls from day 0), the Table 2 values fixed, V(0) = 0 and N and I(0) refitted,
  the best RMSE is about 137 (at a degenerate optimum, N ≈ 3.7 M). Without any vaccination
  it is 185.3 with the controls from day 0 and 122.7 with sanitation from 28 Dec. The paper
  reports 65.8 with φ = 0.05, using an N and initial state it does not report.

| Week ending | 30 Dec | 6 Jan | 13 Jan | 20 Jan | 27 Jan | 3 Feb |
|---|---|---|---|---|---|---|
| Reported | 525 | 717 | 642 | 340 | 178 | 178 |
| Model | 540 | 540 | 558 | 518 | 202 | 45 |

## Sensitivity: timing of sanitation
The data cannot tell how much of the January decline came from the late-December measures
and how much from the vaccine. Three settings, each with N and I(0) refitted and the
1,319,276 doses kept (cases by 4 Feb 2018):

| Sanitation (ρ_e = ρ_h) | RMSE | N | Cases | Without OCV | Averted by OCV by 4 Feb |
|---|---|---|---|---|---|
| 16% from day 0 (as in the paper) | 118.4 | 1,448,291 | 3,866 | 5,306 | 1,440 |
| **16% from 28 Dec 2017 (preset)** | **81.3** | **1,339,513** | **3,653** | **4,675** | **1,022** |
| 27% from 6 Jan 2018 (ρ refitted) | 52.0 | 1,340,365 | 3,510 | 4,074 | 564 |

Without OCV the epidemic is still rising on 4 Feb (577, 595 and 614 cases in the last
three weeks of the preset), so the cases averted grow with a longer run.

The last setting fits best (better than the paper's 65.8), but it changes the paper's ρ, and
its date was chosen among several by the fit; it is shown only to give the range. Users
can reproduce these settings by changing the dates and values of the two sanitation
interventions.

## Scenarios (preset, cases by 4 Feb 2018)
| Scenario | Cases |
|---|---|
| All interventions (preset) | 3,653 |
| Without OCV | 4,675 |
| Without hygiene & contact (ρ_h = 0) | 4,596 |
| Without safe water & sanitation (ρ_e = 0) | 4,978 |
| Without bacteria removal (ν = 0) | 511,143 |
| Control run (treatment only) | 635,032 |

The "Without OCV" difference (about 1,020 cases by 4 Feb) is what the model attributes to the
campaign in the preset. It is not an estimate of the cases the 2018 campaign averted: it
depends on the timing and size of the sanitation step (560 to 1,440 in
[Sensitivity](#sensitivity-timing-of-sanitation)). The two sanitation interventions act
only from 28 Dec in the preset, so leaving one out adds about 940 (contact) or 1,330
(water) cases.

**Removing bacteria removal gives a very large epidemic.** The paper fitted β_e and β_h
with bacteria removal and treatment present from day 0, so the outbreak sits just above
its threshold. Without bacteria removal, bacteria survive 30 days instead of 3.4 and the
water route's transmission rises about 8.7-fold. These runs mean "Lusaka with no
sanitation at all", not what the 2018 response achieved. Compare scenarios with the paper's
controls on, and see [Uncertainty runs](#uncertainty-runs) before varying β_e, β_h, ν or τ.

## Key assumptions
- One homogeneous, well-mixed population per zone; constant size; no cholera deaths.
- Treatment ends infectiousness for every infected person at τ = 1/3 per day (literature
  assumption, Miller Neilan 2010; Posny 2015 uses the same structure). In practice
  antibiotics go mainly to hospitalised patients with severe dehydration (GTFCC 2018), so
  treatment of every case at this rate overstates its effect.
- Bacteria removal is constant over the wave, as in the paper. Sanitation is a step on
  28 Dec 2017 (none before, the paper's 16% after), this model's reading; the same efficacy
  on both routes follows the paper's single ρ.
- The vaccine is leaky (each vaccinated person's risk is reduced by the efficacy) and
  protects only after the time to protection.
- Doses are given at a constant number per day during the campaign.

## Limits
- **The January 2018 decline is reproduced only in part, and its cause is assumed.** The
  model peaks one week after the reported peak and stays at 518 in the week of 14–20 Jan
  (reported 340). Protection only began on Wednesday 17 Jan and built up until 26 Jan
  (about 10% of the doses taking effect each day), so it can explain little of that week's
  fall, and the delay from infection to report puts the infections behind the decline
  earlier still. The
  start date and size of the sanitation step are assumptions; the documented measures
  have dates but no reported effect sizes (Kapata 2018). They decide how much of the
  decline is left for the OCV campaign ([Sensitivity](#sensitivity-timing-of-sanitation)).
- Round 1 targeted four sub-districts (Chawama, Kanyama, Matero, Chipata) and Lusaka
  Central Prison (WHO AFRO, 2 Feb 2018); one homogeneous population spreads the coverage
  over everyone, and the fitted N makes the coverage 98.5%. A sub-district model is the
  natural next step.
- **The fit is approximate elsewhere too.** The first three weeks are 3–4 times too high
  (19, 28 and 41 against 5, 7 and 10), the reported peak is matched only by a plateau
  (540, 540 and 558 for the weeks ending 30 Dec, 6 Jan and 13 Jan, against 525, 717 and
  642), and the last week is too low (45 against 178 digitised, 140 official).
- The **second wave** (heavy rain and flooding from mid-February 2018 according to Maity
  2023; MMWR 2018 dates the flooding to mid-March) is not modelled.
- Reported cases are compared with all infections (no reporting fraction, as in the paper).

## Uncertainty runs
β_e, β_h, ν and the time to treatment have uncertainty ranges (the parameter table), so an
uncertainty run samples them. These runs are a **sensitivity check near the epidemic
threshold, not the uncertainty of the fit**: after 28 Dec the preset sits just above
threshold (R0 1.024), the ranges are the paper's separate 95% intervals for values that
were fitted together, the framework samples them independently, and N and I(0) were fitted
at the central values only. Total cases by 4 Feb therefore vary widely. With 200 Latin
hypercube samples over the four ranges (N and I(0) of the preset), the median is about
9,000 cases and the 95% interval about 200 to 110,000, against 3,653 for the preset.
One value at a time, at the ends of its range: β_e 0.0074 / 0.0102 per day gives about
1,100 / 43,000 cases; β_h 1.82 / 2.15 ×10⁻⁷ gives 1,600 / 7,500; ν 0.23 / 0.3 per day gives
9,500 / 950; time to treatment 2.5 / 3.75 days gives 500 / 39,000. To vary these values
consistently with the data, refit N and I(0) for each set.

## Known edge cases
- An intervention whose end date is before its start date is never active (with a
  warning). For treatment this means no treatment at all in that run, unlike leaving
  treatment out, which keeps it on.
- The OCV campaign needs a start date, and either an end date or a daily speed. An end
  date on or before the start date gives no campaign (with a warning). A campaign sent
  with no dates at all (possible through the API, not the form) is given the run's first
  day by the framework and runs at the daily speed. A start date before the run's start
  drops the doses that would fall before day 0.
- "Initial V. cholerae in water" must be one of its two options; any other value (possible
  only through the API) is treated as "Balanced with starting cases", with a warning.
- Threshold-triggered interventions are not supported. The forms show no threshold
  fields; thresholds sent through the API are ignored by the model but would still
  appear as threshold events in the platform's intervention timeline.
- The platform's intervention timeline marks an intervention "off" on its end date. For
  the OCV campaign this matches the model; for the other interventions the model keeps
  them on through that day.
- If both an end date and a daily speed are filled (as in the preset), the end date wins
  and the speed is ignored.
- Adding the same intervention twice keeps only the last one (the framework stores
  interventions by name), so a second OCV round cannot be added in the same run.
- An intervention whose start is on or before day 0, or whose end is at or after the run
  end, stays on to that edge.
- Adherence and transmission reduction of the two sanitation interventions are required:
  left empty, the framework would use 0 and 5% instead of the paper's 100% × 16%.

## Differences from the source paper
| Paper (Maity et al. 2023) | This model | Why |
|---|---|---|
| Constant vaccination φ = 0.05/day from day 0 | φ = 0 and a dated OCV campaign (doses per day to S and R, 7-day time to protection) | Lusaka's 2018 vaccination was a reactive campaign; with the Table 2 values fixed, V(0) = 0 and N and I(0) refitted, φ = 0.05 gives RMSE about 137 |
| Compartments S, V, I, R, B | Adds VI and VR | Tracks vaccinated people after infection, so they are not vaccinated again while in VI or VR; I + VI and R + VR equal the paper's I and R |
| One sanitation efficacy ρ on both routes | Two interventions, ρ_e (water) and ρ_h (contact), same value and dates | Lets users change one route; equal values reproduce the paper |
| Sanitation ρ constant from day 0 | Preset: ρ_e = ρ_h = 16% from 28 Dec 2017 | Documented strengthening of the response; with ρ from day 0 the model peaks two weeks late (RMSE 118.4 vs 81.3) |
| Bacteria removal ν always present | An intervention, default on over the whole run | Lets users switch it off or date it |
| Treatment τ constant from day 0 (like ρ and ν) | An intervention, on by default and also in the control run | A choice for this model, so the control run still treats cases; dates can restrict it |
| N and initial state not reported | N and I(0) fitted; B(0) balanced start | Needed to run the model |
| Rates printed per day | β_e, β_h, ν read per week, entered per day (÷ 7) | ν: only per week reproduces Table 2's R_e/R_h; β_e, β_h: per day would need N ≈ 193,000, fewer than the people vaccinated |

## Related models
No other cholera model is on the repository's main branch. Two are being developed
alongside this one: `cholera_siwr_ocv_model` (two-route SEIWR with reactive OCV, Lusaka
2023–24) and `cholera_mobility_finger_model` (spatial SIRB with human mobility and mass
gatherings, Finger et al. 2016), which is the one to use when spread between zones matters.

## Development notes
This model was developed with the support of an AI assistant (Claude, Anthropic), working
with the modeller in conversation. The assistant extracted the equations and parameter
values from the paper, searched the literature for how OCV doses are allocated (each
reference was checked against the full text supplied by the modeller), fitted N and I(0),
and wrote `model.py`, the tests and this file. Six independent AI reviews (separate agents
that had not seen the development) checked the code, documentation and references against
the paper, the cited sources and the checklist; their findings were fixed or documented.
The fifth found that the digitised data had been placed one week too late; the official
weekly counts were then collected from WHO AFRO and Zambian sources to confirm the dates.
The sixth recomputed the results independently and checked the sources for the response
measures. The modeller (Yingsi Lai) set the
design choices, checked the references and reviewed the code, the fit and the
documentation.

## Change history
- October 2026: created by Yingsi Lai (WHO Hub for Pandemic and Epidemic Intelligence) as
  `cholera_svirb_maity_model` (disease_type `cholera_svirb_maity`).
- November 2026 (Yingsi Lai): VI and VR compartments added; OCV campaign changed to doses
  per day with two modes (by dates or by daily speed); sanitation split into water and
  contact interventions; bacteria removal made an intervention; travel matrix removed from
  the equations (zones are independent); metadata and documentation completed against the
  model checklist.
- November 2026 (Yingsi Lai): renamed from `cholera_svirb_maity_model` (disease_type
  `cholera_svirb_maity`, class `CholeraSvirbMaityModel`) to
  `cholera_svirb_ocv_maity_model` (disease_type `cholera_svirb_ocv_maity`, class
  `CholeraSvirbOcvMaityModel`), to add the feature part of the name (checklist A1).
- November 2026 (Yingsi Lai), after an independent review: vaccine efficacy and time to
  protection moved from the OCV form to model parameters (the platform timeline read them
  as thresholds); cumulative totals redefined as inflows, with VI_total and VR_total added,
  so the summary reports breakthrough cases; OCV coverage set to the reported 1,319,276
  doses (87.3026%, superseded); bounds added to the arrows' durations; sanitation fields made
  required; tests added for the dose split, platform outputs and scenario table.
- November 2026 (Yingsi Lai), after a second independent review: documented that the fit
  cannot confirm the 7-day time to protection (kept: a zero delay is not plausible), the
  implied R0, the site-level campaign dates and how B(0) follows from I(0); an OCV end date
  not after the start date is now ignored with a warning.
- November 2026 (Yingsi Lai): zone field `initial_water_mode` added (dropdown, next to
  the starting cases); by default B(0) is computed from the starting cases, so it follows
  I(0) when users change it.
- November 2026 (Yingsi Lai): label changed to "Cholera – SVIRB, reactive OCV (extends Maity
  et al. 2023)", and the overview, metadata and docstring now state that the reactive OCV
  campaign is an addition, not part of the paper.
- November 2026 (Yingsi Lai), after a third independent review: the per-week reading and
  the treatment choice are stated as this model's, with their reasons; the equations show
  the protection delay (u(t − T_p)); the zero-delay refit is reported under both
  conventions; the treatment uncertainty range is τ ± 20% (2.5–3.75 days).
- November 2026 (Yingsi Lai): the unit switch `rate_unit` removed; β_e, β_h and ν are
  entered per day (Table 2 values ÷ 7), so all rates are per day. Results unchanged.
- November 2026 (Yingsi Lai), after a fourth independent review: the reason for reading
  β_e and β_h per week corrected (the case data cannot decide; the per-day reading needs N
  ≈ 217,000 at the time, since superseded, fewer than the people vaccinated); the R0
  comparison and the φ = 0.05 fit
  stated with their conditions; treatment in the control run stated as this model's
  choice.
- November 2026 (Yingsi Lai), after a fifth independent review: the 15 weekly counts moved
  to their true dates (Sunday–Saturday weeks ending 28 Oct 2017 … 3 Feb 2018, checked
  against WHO AFRO situation reports); day 0 = 22 Oct 2017; N and I(0) refitted (1,448,291
  and 15.6); the documentation states that the preset does not reproduce the January
  decline and should not be used to estimate cases averted.
- November 2026 (Yingsi Lai): in the preset, safe water and hygiene & contact start on
  28 Dec 2017 (the documented strengthening of the response) instead of day 0; N and I(0)
  refitted (1,339,513 and 3.4; RMSE 81.3); sensitivity to the timing of sanitation added
  (cases averted by OCV about 560 to 1,440); after a sixth independent review, sources for
  the response measures corrected (MMWR only for chlorination and tanks; Kapata 2018,
  ZambiaLII, Daily Nation and Reuters added), the title of SI No. 79 corrected, data
  caveats added.
- November 2026 (Yingsi Lai), after a seventh independent review: the uncertainty range of
  ν set to 0.23–0.3 (the paper's interval, rounded) and uncertainty runs documented as a
  near-threshold sensitivity check; interventions with an end date before the start date
  are never active (they made rates negative); the per-day β argument corrected (as good
  a fit at 98.5% coverage, N ≈ 193,000) and the paper's R0 stated as pointing against the
  per-week reading of β; the paper's own wave dates, the gatherings ban in SI No. 79 and the
  remaining misfit stated; related models listed.

## References
1. Maity B, Saha B, Ghosh I, Chattopadhyay J (2023) Model-based estimation of expected time to cholera extinction in Lusaka, Zambia. Bull Math Biol 85:55. doi:10.1007/s11538-023-01149-0
2. Posny D, Wang J, Mukandavire Z, Modnak C (2015) Analyzing transmission dynamics of cholera with public health interventions. Math Biosci 264:38–53.
3. Miller Neilan RL, Schaefer E, Gaff H, Fister KR, Lenhart S (2010) Modeling optimal intervention strategies for cholera. Bull Math Biol 72(8):2004–2018.
4. Sardar T, Mukhopadhyay S, Bhowmick AR, Chattopadhyay J (2013) An optimal cost effectiveness study on Zimbabwe cholera seasonal data from 2008–2011. PLoS ONE 8(12):e81231.
5. Mukandavire Z, Liao S, Wang J, Gaff H, Smith DL, Morris JG Jr (2011) Estimating the reproductive numbers for the 2008–2009 cholera outbreaks in Zimbabwe. PNAS 108(21):8767–8772.
6. Hartley DM, Morris JG Jr, Smith DL (2006) Hyperinfectivity: a critical element in the ability of V. cholerae to cause epidemics? PLoS Med 3(1):e7.
7. Azman AS, Luquero FJ, Ciglenecki I, Grais RF, Sack DA, Lessler J (2015) The impact of a one-dose versus two-dose oral cholera vaccine regimen in outbreak settings: a modeling study. PLoS Med 12(8):e1001867 (main text and S1 Text).
8. Pasetto D, Finger F, Camacho A, Grandesso F, Cohuet S, Lemaitre JC, et al. (2018) Near real-time forecasting for cholera decision making in Haiti after Hurricane Matthew. PLoS Comput Biol 14(5):e1006127 (main text, Methods).
9. Lee EC, Chao DL, Lemaitre JC, Matrajt L, Pasetto D, Perez-Saez J, et al. (2020) Achieving coordinated national immunity and cholera elimination in Haiti through vaccination: a modelling study. Lancet Glob Health 8(8):e1081–89 (appendix, model 4).
10. Chao DL, Halloran ME, Longini IM Jr (2011) Vaccination strategies for epidemic cholera in Haiti with implications for the developing world. PNAS 108(17):7081–7085 (and SI).
11. Ferreras E, et al. (2020) Alternative observational designs to estimate the effectiveness of one dose of oral cholera vaccine in Lusaka, Zambia. Epidemiol Infect 148:e78. doi:10.1017/S095026882000062X
12. Global Task Force on Cholera Control, Case Management Working Group (2018) Technical note: use of antibiotics for the treatment and control of cholera. May 2018.
13. WHO Regional Office for Africa (2018) Zambia cholera situation report No. 114, 2 February 2018.
14. WHO Regional Office for Africa (2018) WHO supports the immunization of 1 million people against cholera in Zambia. News release, 10 January 2018.
15. WHO Regional Office for Africa (2018) Zambia cholera situation report No. 115, 3 February 2018.
16. World Health Organization (2017) Cholera – Zambia. Disease Outbreak News, 11 December 2017.
17. Mazaba ML (2018) Hopeful: towards the tail end of the Zambia cholera 2017/2018 outbreak? (Editorial). Health Press Zambia Bull 2(5):1–3.
18. Sinyange N, Brunkard JM, Kapata N, et al. (2018) Cholera epidemic — Lusaka, Zambia, October 2017–May 2018. MMWR Morb Mortal Wkly Rep 67(19):556–559.
19. Republic of Zambia (2017) Public Health (Infected Areas) (Cholera) Regulations, 2017. Statutory Instrument No. 79 of 2017, commenced 28 December 2017 (ZambiaLII).
20. Kapata N, Sinyange N, Mazaba ML, et al. (2018) A multisectoral emergency response approach to a cholera outbreak in Zambia: October 2017–February 2018. J Infect Dis 218(Suppl 3):S181–S183. doi:10.1093/infdis/jiy490
21. ReliefWeb (2018) Zambia: Cholera Outbreak – Oct 2017 (EP-2017-000178-ZMB), WHO update of 26 January 2018.
22. Daily Nation (Zambia) (2017) Lungu bans street vending, closes Soweto market. 30 December 2017.
23. Reuters (Mfula C) (2018) Zambia declares curfew in Lusaka slum struck by cholera. January 2018 (curfew in Kanyama 18:00–06:00 from Sunday 7 January 2018).
