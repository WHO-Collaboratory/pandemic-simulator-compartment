# Ebola – SEIHFR, contact tracing (Lai, after Legrand et al. 2007)

A deterministic compartmental model of Ebola disease. **The untraced chain is
Legrand et al. (2007) as published** — the same three competing exits from I,
the same two from H, the same derived routing probabilities θ₁, δ₁, δ₂ — with
its three transmission routes: the community (I), isolation (H) and traditional
burial (F). Defaults describe the **2026 Bundibugyo virus disease outbreak in
DRC**.

It adds three things:

1. **R and D are separate.** Legrand's single "removed" compartment holds
   survivors and buried dead together; here recoveries go to R and burials to D.
   Pure bookkeeping — both are absorbing and non-infectious.
2. **Safe burial of deaths in isolation.** Legrand sends every death in H to F;
   a share `s_iso` here goes straight to D. Set it to 0 to reproduce Legrand.
3. **Source-based contact tracing and case finding**, as a parallel chain.
   Legrand has no tracing component; "contact tracing" appears once in his
   paper, in the Discussion, as the suggestion implemented here.

Legrand J, Grais RF, Boelle PY, Valleron AJ, Flahault A. *Understanding the
dynamics of Ebola epidemics.* Epidemiol Infect. 2007;135(4):610-621.
doi:10.1017/S0950268806007217

> **Full derivations, parameter-by-parameter evidence, identifiability
> arguments and the validation analysis are in [`evidence.md`](./evidence.md),
> beside this file.** This page is the summary.

## Structure

```
                            ┌──────────────► R
 S ─λ─► E ─(incubation)─► I ─┼─► H ─┬─► R
                            │      ├─► D   (safe burial, share s_iso)
                            │      └─► F
                            └─► F ─(burial)─► D

 S ─traced─► E_T ─────────► I_T ─► H_T ─► same three outcomes,
                                          with θ_T, Dh_T, δ_T, κ_T, p_notify_T
```

| Compartment | Meaning |
|---|---|
| S | Susceptible |
| E / E_T | Incubating: untraced / listed contact under follow-up |
| I / I_T | Symptomatic, not yet isolated — community, households, and general health facilities before diagnosis |
| H / H_T | In an isolation unit (ETU or transit centre) |
| F | Dead, awaiting traditional burial (infectious) |
| R | Recovered |
| D | Buried (safely, or after a traditional funeral) |

The three exits from I compete, at Legrand's rates `γ_h·θ₁`,
`γ_d·δ₁·(1−θ₁)` and `γ_i·(1−θ₁)(1−δ₁)`, with θ₁, δ₁ and δ₂ derived so that a
share θ is isolated and the case-fatality ratio is δ.

**Why H is split between the chains.** A traced case is admitted earlier, so its
*remaining* course in isolation is longer — 7.06 days against 4.71 at the
defaults. One shared H cannot hold two different stays, so H_T is needed whether
or not the CFRs differ.

## Initial conditions

Seeded from `infected_population` in the admin zone, which is a **percentage of
the zone population**, placed in I. All other non-susceptible compartments start
empty, and the traced chain starts empty — index cases are by definition not
traced. N is the whole population including the dead and is held fixed, as in
Legrand.

**Starting mid-epidemic.** Because the whole seed goes into I while E, H and F
start empty, a run that begins partway through an outbreak opens with a
transient of roughly three weeks while those stages refill, and the seed that
reproduces an observed incidence is therefore larger than the current prevalence
in I — about 2.6× at these defaults. Seed such a run by matching the model's
confirmed incidence to the observed one after the transient (day 30 or so)
rather than by entering a prevalence estimate directly. The shipped
`example-config.json` is built that way.

### The shipped example: Ituri, from the first confirmations

`example-config.json` runs the 2026 outbreak in **Ituri province** from 14 May
2026, the day INRB confirmed the first eight cases, through to 14 May 2027, in
one zone of 5.7 million. Ituri carries 69% of the outbreak's cases and is the
single focus the defaults are calibrated to. The seed is 345 infections, fitted;
interventions start on the dates the response records them, with the community
response ramping over 95 days.

**Every intervention starts on 15 May 2026**, the day the outbreak was declared
— the response was not assembled in stages weeks apart, it was declared at once
and built up gradually, which is what the ramp represents. The record supports
this: field hospitals on 17 May, a provincial ban on funeral wakes on 22 May,
the first 65-bed treatment centre on 1 June and three by 16 June, with tracing
under way by 2 June at about 20% of contacts located.

Two values are fitted, the seed and the ramp length; R0 and the community
reduction come from Ituri's two growth regimes (see *Validation*), not from this
curve. Moving the tracing start date between 15 May and 1 July changes the fit
by less than 0.005 in log-RMSE, so all four share one date.

| | Model, shifted +14 d | Ituri observed |
|---|---|---|
| Confirmed per day, 20 June | 33 | 32 |
| Confirmed per day, 25 July | 62 | 115 |
| Confirmed per day, 15 August | 62 | 68 |
| Confirmed per day, 15 September | 44 | 43 |
| Confirmed per day, 20 October | 31 | 35 |

**The model column is the daily increment of `Confirmed` shifted forward by 14
days, and the comparison is only valid with that shift.** `Confirmed` is dated
at the *moment of infection*, because that is where the model knows the
detection probability; an observed case is dated when it is reported. The gap is
the incubation period plus onset-to-detection, which at the defaults is 8.0 d +
about 6 d. 14 days is the best integer shift, not a derived constant. Unshifted,
the same run reads 46 / 64 / 53 / 39 / 27 and the 20 June point is 44% high.
The model publishes no detection-dated series, so anyone reproducing this table
has to apply the shift by hand — a limitation, not a feature; see *Known edge
cases*.

Log-RMSE 0.178 over 18 June – 22 October, i.e. a typical error near 19%, against
the full daily Ituri series (digitised from the public INSP situation reports and
not redistributed here, so this figure cannot be recomputed from this repository).
Over the five printed points alone the shifted model scores 0.285 and the
unshifted 0.348. The one clear miss is the late-July peak, which a single zone
cannot make as sharp as the data; see *Validation*.

**Bed occupancy is not reproduced.** The isolation stay is the remainder of the
course while real discharge requires two negative PCR tests, so H understates
the number of patients in isolation several-fold. Use this configuration for
transmission, cases and deaths; not for bed planning.

For a run that starts from current conditions instead, set the start date to the
latest data cut-off and put every intervention's start date on it with
the ramp parameters small. Note that the whole seed enters I while E, H and F
start empty, so such a run opens with a transient of about three weeks and needs
a seed well above the current prevalence in I; match the model's confirmed
incidence to the observed one at day 30 rather than entering a prevalence
estimate directly.

## Outputs

Ten population compartments, plus four cumulative series published as outputs of
their own: `Confirmed`, `KnownContact`, `Deaths` and `Listed`. Each population
group also has a matching cumulative total: `E_total` / `E_T_total`
(infections), `I_total` / `I_T_total` (onsets), `H_total` / `H_T_total`
(isolations), `F_total` (deaths given a traditional funeral), `R_total`
(recoveries), `D_total` (all burials = deaths, lagged).

What to read for each quantity:

| Quantity | Read |
|---|---|
| Prevalence, infectious in the community | `I + I_T` |
| Beds occupied | `H + H_T` |
| Cumulative infections | `E_total + E_T_total` |
| Cumulative onsets | `I_total + I_T_total` |
| Cumulative isolations (admissions) | `H_total + H_T_total` |
| **Cumulative confirmed cases** | **`Confirmed`** — confirmed only; compare with a reported *confirmed*-case count |
| **Cumulative deaths** | **`Deaths`** — *all* deaths, so ~1.47× a sitrep's confirmed-death count; `D` is bodies already buried and lags it by ~1.4 d |
| Confirmed cases who were known contacts | `KnownContact / Confirmed` — 0.15 at the defaults, the sitrep indicator `c` was calibrated to |
| Infections in listed contacts | `Listed` — of which the share actually reached is `E_T_total` |
| Contact-listing yield | `E_T_total / Listed` = the follow-up rate `f` |
| Share of infections in the traced chain | `E_T_total / (E_total + E_T_total)` — 0.07 at the defaults |

All four — `Confirmed`, `KnownContact`, `Deaths` and `Listed` — are running
totals, not populations, so they are not part of the
S + E + I + H + E_T + I_T + H_T + F + R + D sum; `CUMULATIVE_OUTPUTS` on the
model class names them, and anything checking that sum has to skip them. They
are named without the `_total` suffix on purpose: the framework drops every
`_total` column from the delivered time series, and these four are meant to be
visible. `Confirmed`, `KnownContact` and `Listed` are counted at the moment of
infection, `Deaths` at death.

**`Deaths` is not the situation-report death count, and the two published series
do not divide into a CFR.** `Deaths` is *all* Ebola deaths; a sitrep counts
deaths among confirmed cases, and the 43% of community deaths that are never
alerted and swabbed never become confirmed. At the defaults the model implies
0.290 deaths per symptomatic case against 0.198 confirmed deaths, so `Deaths`
runs about **1.47×** the reported figure, and `Deaths / Confirmed` = 0.71 rather
than the confirmed-case CFR of 0.484 quoted below. Multiply by
θ + (1 − θ)·`p_notify` = 0.682 to get the confirmed-death equivalent.

**Onsets are not confirmed cases.** An untraced case is detected with
probability 0.41 at the defaults and a traced case with 0.84, which over the
shipped example run leaves `Confirmed` at 0.44 of cumulative onsets: most
symptomatic cases never become confirmed cases. Compare `Confirmed`, not
`I_total`, with a reported case count.

**`Confirmed` is also two weeks early.** It is dated at the moment of infection,
not when the case is detected, because that is where the model knows the
detection probability. Against a reported daily series it therefore needs
shifting forward by about 14 days — the incubation period plus onset-to-detection
— before the curves line up; the level caveat above and this phase caveat are
separate, and both apply. Cumulative totals at the end of a long run are
unaffected.

`Confirmed` counts **confirmed cases only**. All three detection routes end in
laboratory confirmation and the model has no probable-case compartment, so a
situation report's headline case total — confirmed *plus* probable — is larger
than this series by construction. Every identity the defaults were pinned to
uses the same confirmed-only denominator (intra-CTE deaths per confirmed case,
confirmed-case CFR, confirmed cases ever isolated), so the comparison is
like-for-like only against the confirmed count. The severity defaults reproduce
four published 2026 indicators — community share of confirmed deaths 0.619
(reported 0.621), confirmed-case CFR 0.484 (0.486), confirmed cases ever
isolated 0.636 (bracket 0.556–0.698), intra-CTE deaths per confirmed case 0.1845
(0.1843) — but **only the first two are independent of each other**; see
*Validation* for why the other two follow as algebra.

`reproduction_numbers()` returns R_I, R_H, R_F, R0 and Rc from the two-type
next-generation matrix, plus the listed and traced shares of new infections, the
confirmed share of each chain, δ_T, and the known-contact share. Called with no
arguments it returns the **unmitigated** value in the `Rc` key (2.02, i.e. R0):
the post-intervention 0.83 requires passing the reduced coefficients and the two
tracing fields explicitly, as the docstring shows.

## Interventions

Five are declared. Three are Legrand's transmission interventions; two act on
the structure of case detection rather than on a transmission coefficient.

The framework gives every intervention the same two numeric fields, *adherence*
and *transmission reduction*, and multiplies each targeted rate by
`1 − adherence × transmission_reduction`. This model uses those two fields in
two different ways, so read the table by the column headings, not by the field
names on screen:

| Intervention | Transmission reduction means | Adherence means | What it changes |
|---|---|---|---|
| Community Transmission Control | Decrease in community transmission (1 − z) | *not shown; pinned at 100%* | βI × (1 − z) |
| Isolation-Unit Infection Control | Efficacy of isolation infection control (1 − zH) | *not shown; pinned at 100%* | βH × (1 − zH) |
| Safe and Dignified Burials | Efficacy of interventions after death (1 − zF) | *not shown; pinned at 100%* | βF × (1 − zF) |
| Contact Tracing and Follow-up | Contact listing completeness (c) | Contact follow-up rate (f) | Splits new infections into traced (E_T) and untraced (E) |
| Case Finding and Isolation | Isolated share after intervention (θ′) | Reduction in onset-to-isolation delay (r) | θ → θ′ and delay → delay × (1 − r) |

Legrand gives each of his three transmission interventions a **single** effect
size, so exposing a second field would misrepresent the paper. For those three
the model removes `adherence_min` from the form in `define_parameters()` and
pins it to 100% in `__init__()`, before the base class builds its runtime
intervention objects. The formula then collapses to `rate × (1 − z)`, as in the
paper, and a config that omits adherence — as the shipped `example-config.json`
does — still gets the full effect size rather than silently disabling the
intervention. Both halves are needed: pinning alone would leave a field on the
form that does nothing, and hiding alone would make the framework read the
missing value as zero.

Contact tracing and case finding declare `target_rates=[]`, so the framework's
own rate arithmetic does nothing: the model reads the two field values by hand
in `_tracing_and_case_finding()`. They change *who* is traced and *how fast*
cases are isolated rather than scaling a coefficient, which is why their two
fields carry the meanings given above. The field names themselves cannot be
changed, so each intervention's description in the schema spells out which
field means what.

All five interventions are **date-activated only**. Every effect in this model
is applied by `_ramp()`, which keys off the start date, and all five declare
`target_rates=[]`, which makes the framework's own threshold branch a no-op — so
a prevalence trigger could never fire. The model therefore does not declare
`start_threshold` or `end_threshold` on any of them. **An intervention given a
threshold and no start date never activates, silently**: it behaves exactly as
if it had been deleted.

Two warnings about what that declaration does and does not buy.
`add_intervention()` always emits the whole shared field set — the two generic
fields, the two dates and the two thresholds — and offers no way to declare a
subset, so the model drops the unwanted entries from the built schema afterwards.
And the results UI renders a fixed form per intervention rather than reading the
declared list, so **"By Threshold" still appears there**; undeclaring it states
what the model supports without hiding the control.

**Both generic fields are used by all five, and relabelled.** The framework
calls them "Adherence" and "Transmission Reduction", which fits the three
transmission interventions and not the other two, so the model renames them per
intervention and each description spells the meaning out:

| Intervention | first field | second field | what the model does with them |
|---|---|---|---|
| Community Transmission Control | Coverage | Effectiveness | βI × (1 − coverage × effectiveness) |
| Isolation-Unit Infection Control | Coverage | Effectiveness | βH × (1 − coverage × effectiveness) |
| Safe and Dignified Burials | Safe-Burial Completion | Effectiveness | βF × (1 − completion × effectiveness) |
| Contact Tracing and Follow-up | Contact Follow-up Rate (f) | Contact Listing Completeness (c) | routes new infections into the traced chain |
| Case Finding and Isolation | Onset-to-Isolation Delay Reduction | Isolated Share After Intervention (θ′) | raises θ and shortens Dh for the untraced chain |

For the three transmission interventions only the **product** matters, so the
split is a presentational choice except where the evidence supplies both factors
— safe burial, where 56% is 70% completion × 80% effectiveness. Community
transmission control is a calibrated residual and isolation IPC has no basis for
a split, so both carry 100% coverage and put the whole effect in effectiveness.
An omitted coverage is read as full coverage, not as zero, so a configuration
that leaves the field out still runs the published effect size.

- **Community Transmission Control** is a *residual* covering community
  engagement, risk communication, household precautions, earlier care-seeking
  and infection control in general facilities before diagnosis. It is therefore
  the natural quantity to calibrate.
- **Isolation-Unit Infection Control** is proportional to βH, so with βH = 0 it
  does nothing whatever value is entered.
- **Safe and Dignified Burials** is the *share of burial transmission
  prevented* — coverage × effectiveness — not safe-burial coverage. Deaths
  inside isolation are handled by `s_iso`, not by this intervention.
- **Contact Tracing** is bounded by case detection: only infections caused by a
  detected source can be listed, and only if listing happens while the contact
  is still incubating. Raising c and f to 100% still leaves most transmission
  untraced. That ceiling is structural, not a tuning choice.
- **Case Finding** reaches what tracing cannot, because it does not require the
  source to have been detected. θ′ *replaces* θ rather than adding to it.

**Timing.** All three transmission interventions **ramp** linearly, each over
its own scale-up period — `community_ramp_days`, `hospital_ramp_days` and
`funeral_ramp_days`, all 95 days by default — because an outbreak response is
declared in a day but built over weeks: burial teams, treatment centres and
community engagement all arrive gradually, and not necessarily at the same
speed. Only the community one is identified by the data; the other two are
separate so that an unevenly built response can be represented. This is a deliberate departure from Legrand,
whose interventions are steps; setting the parameter to 0 restores his
behaviour. The practical consequence is that the three usually share one start
date — the day the response began — rather than being staggered to stand in for
a scale-up the model could not otherwise represent.
Tracing and case finding ramp over `ramp_days` — partly realism, partly
numerical, since a hard step in the traced share stalls the ODE solver at high
coverage. Different start dates per intervention are supported and tested. The
framework runs every simulation twice, with and without interventions; the
control run is the counterfactual.

## Parameters and defaults (DRC 2026 BVD)

Evidence grades: **A** direct 2026 measurement · **B** derived from 2026 data ·
**C** earlier Bundibugyo outbreak · **D** other Ebola species or outbreaks ·
**E** assumption, no measurement found. The 2026 indicators come from different
dates — delays from May–June, the known-contact share from July, θ, CFR, burials
and follow-up from October — so the defaults are a composite, not one snapshot.
One row per parameter, with the default exactly as `define_parameters()` sets it.
The reference column is a one-line summary; the full derivation and sources for
every row are in [`evidence.md`](./evidence.md). Where a row names an
intervention in brackets, the key is the framework's generic field name for that
intervention and the meaning column says what it carries here.

| Parameter (key) | Default | Range | Meaning | Grade | Reference |
|---|---|---|---|---|---|
| R0 | 2.02 | 1.8–3.7 | Basic reproduction number, used to set the three β values | B | **Inverted from Ituri's own climb**, 15 Jun – 25 Jul 2026: R = 1.81 with safe burial, isolation IPC and contact tracing already running. Bounded 1.81–2.02 (the lower end assumes those early measures did nothing). The four 2026 literature estimates (1.83 Chen; 1.9 epiforecasts; 2.51 CDC MMWR 75(22); 3.67 Oraby) are all case-based and inflated by the May–July testing scale-up; their centre of 2.5 is above what this outbreak's own climb supports |
| `betaI` | 0.1638 /day | 0.1459–0.3 | Community transmission coefficient | B level, E split | R0 inverted through the route integrals at a community share of 85%; the split is a prior, not a measurement — no 2026 study identifies it |
| `betaH` | 0.0247 /day | 0–0.0494 | Isolation-unit transmission coefficient | B level, E split | Set as an isolation *effectiveness*, βH = (1 − ε)·βI with ε = 0.85, at an isolation share of 1.5%. See `evidence.md` for why ε = 1 (βH = 0) is not defensible |
| `betaF` | 0.6354 /day | 0.5661–1.1637 | Traditional-burial transmission coefficient | B level, E split | R0 inverted at a burial share of 13.5%; historical anchors only. Run 8% and 25% |
| `alpha` (edge) | 8.0 d | 6.3–12 | Incubation period | C | Wamala et al. 2010, Uganda 2007 median 7 d (n = 116); pooled BDBV 7.65 d (Oraby et al. 2026). No usable 2026 estimate |
| `gamma_h` | 7.0 d | 5–10 | Onset to isolation, untraced (a route rate, not the realised wait) | B | Akilimali et al., NEJM 2026: DRC onset-to-sampling mean 7.4 d (n = 129), an upper bound; onset-to-reporting median 7 d with no shortening Jul–Sep |
| `gamma_i` | 12 d | 10–17 | Onset to end of infectiousness, survivors | C | Wamala et al. 2010 median 10 d; Isiro 2012 onset-to-discharge 11.85 d. Shorter than a real treatment-centre stay |
| `gamma_d` | 11 d | 8–14 | Onset to death | C | Rosello et al. 2015, Isiro 2012 mean 11.37 d — the informative prior used by both Imperial and epiforecasts in 2026 |
| `gamma_f` (edge) | 2 d | 1–4 | Death to traditional burial | D | Legrand et al. 2007; Yambuku 0.99 d (Camacho et al. 2014); West African funerals 2.0–4.5 d (Rivers et al. 2014). No Bundibugyo or 2026 estimate |
| `theta_target` | 26% | 22–29 | Share of untraced cases ever isolated (θ) | B | Inverted from the cumulative community share of confirmed deaths, 62.1% (CDC MMWR 75(35) blended with INSP sitreps). Identified only jointly with `p_notify`; falling over time as that share rises |
| `delta_target` | 29% | 27–33 | Case-fatality ratio, isolated arm (δ) | B | Pinned by the identity (intra-CTE deaths / confirmed) ÷ (confirmed ever isolated) = 0.1843 / 0.636, INSP sitrep 167. **Not** the crude confirmed CFR of 48.6%, which the model reproduces as an output |
| `cfr_ratio_comm` | 100% | 100–150 | Community CFR as a percentage of the isolated-arm CFR | E | Direction evidenced (ETU admission HR 0.5, Lindblade 2015; odds ×1.11/day, Rosello 2015), magnitude not. Not free: θ/(1−θ) scales with it, so 130% implies θ = 31.2% and the βs need re-inverting |
| `p_notify` | 57% | 47–67 | Share of untraced community deaths notified and swabbed | B | P(alerted) × P(swabbed \| alerted); the second is measured at 66.9% (INSP, Oct 2026), a hard ceiling. P(alerted) is measured nowhere |
| `p_notify_t` | 67% | 57–67 | Same, for deaths among traced contacts | B | A contact seen daily is effectively certain to have their death alerted, so this collapses onto the measured 66.9% |
| `p_detect_rec` | 5% | 3–12 | Share of community survivors ever detected | B | Capped near 0.10–0.12 by the 14.25% of confirmed cases with no recorded outcome (INSP sitrep 167), which such a survivor must sit inside. The range stops at 12% (implied no-outcome 0.1416); 13% breaches the ceiling at 0.1516 |
| `cfr_or_day` | 1.11 | 1.00–1.21 | Odds ratio of death per extra day not hospitalised, used to derive δ_T | D (partly C) | Rosello et al. 2015, pooling Kikwit 1995, **Isiro 2012 BDBV** and Boende 2014; gives δ_T = 24.3%. Set to 1.0 to disable |
| `s_iso` | 100% | 80–100 | Share of deaths in isolation given a safe burial | E | Standard practice and assumed in the CDC 2026 scenarios, but never reported in the 2026 sitreps. Set to 0 to reproduce Legrand |
| `gamma_h_t` | 4.7 d | 3.0–5.5 | Onset to isolation, traced | D | Two independent routes agree: Uganda 2022's 4-vs-6-day ratio (Wanyana et al. 2024) scaled onto 7.0 d, and the standard-follow-up arm of Keita et al. 2023 at 4.8 d. Shorter figures are pre-symptomatic contact isolation, which DRC does not do |
| `theta_t_target` | 80% | 60–100 | Share of traced cases ever isolated (θ_T) | E | Never measured in 2026. Floor from 62.6% of living validated suspects transferred (INSP sitreps 164–167). Moves Rc little: 80% → 100% gives 0.831 → 0.823 |
| `kappa_t` | 100% | 24–100 | Per-day infectiousness of a traced case before isolation (κ_T) | E | No per-day estimate exists for any filovirus. The 0.24 floor is Uganda 2022's IRR 0.16 with the exposure-time ratio divided out. 100% is deliberately conservative |
| `listing_delay` | 1 d | 0.5–3 | Detection of a case to listing of its contacts | E | Never reported in 2026. Ituri teams pre-list 14–37 contacts per case before confirmation (INSP sitreps) |
| `ramp_days` | 7 d | 3–21 | Scale-up period for tracing and case finding | E | Assumption; also numerically necessary, since a hard step stalls the ODE solver at high coverage |
| Contact tracing: follow-up rate (f) | 86% | 78–93 | Share of listed contacts actually visited until onset | A | **Ituri's own rate**, averaged over the fitted period: 78.1% of 10,183 contacts on 15 Jul (WHO DON613), 88% in the week to 23 Aug (WHO AFRO weekly sitrep 15), 92.8% (6,382/6,879) on 20–23 Oct (WHO DON622). Ituri runs *above* the national rate (79.7% on the same late-October day), so the former national default of 82% understated it. Never reached the 95% target. Other provinces that day: Nord-Kivu 73.3%, Tshopo 83.7%, Bas-Uélé 98.9% |
| Contact tracing: listing completeness (c) | 36.3% | — | Share of a detected case's contacts who are listed | B calibrated | Calibrated jointly with the community reduction so that Rc = 0.83 and 15% of new confirmed cases are known contacts (DRC 15–20%, CDC MMWR 75(35)). The indicator has not been published since 12 July 2026, so the target is four months stale. It trades off almost exactly against `f`: at f = 65% it would be 41.0%, at f = 93% 35.0%, with Rc moving 0.834 → 0.830 |
| Safe burials: completion × effectiveness | 70% × 80% = **56%** | 40–75 (product) | Share of burial transmission prevented (1 − zF) | B × E | The two fields now carry the two factors separately. Completion 70% from the INSP alert-completion ratios (66–102% by province and month, falling to about 55% in late October; 21% of Ituri alerts failed, mostly refusal; only 49% of health zones have a burial team). Effectiveness 80% allows for exposure before the team arrives — no 2026 measurement, grade E |
| Isolation IPC: coverage × effectiveness | 100% × 40% = **40%** | — | Share of in-unit transmission prevented (1 − zH) | E | Facility infection-control scorecards averaged 17.6–53.3% of target, Aug–Oct 2026 (Legrand 2007 assumed 100%, sweeping 50–100%). No basis for splitting coverage from effectiveness, so coverage is left at 100%. Small by construction, since isolation carries 1.5% of R0 |
| Community: coverage × effectiveness | 100% × 58.3% = **58.3%** | — | Decrease in community transmission (1 − z) | B calibrated | Calibrated so Rc = 0.83, matching Ituri's observed plateau R over 1 Aug – 25 Oct 2026. A residual absorbing everything not modelled explicitly, so there is nothing to split: coverage stays at 100% and the whole effect sits in effectiveness |
| `community_ramp_days` | 95 d | 70–120 | Days for community transmission control to reach full effect | B fitted | Fitted to the Ituri confirmed-case curve with every intervention starting the day the outbreak was declared; 90–100 days fit about equally well. A response is declared in a day but built over weeks; set to 0 for Legrand's step |
| `hospital_ramp_days` | 95 d | 70–120 | Same, for isolation infection control | E | Not identified by the Ituri curve, which the community term dominates; defaults to the community value |
| `funeral_ramp_days` | 95 d | 70–120 | Same, for safe and dignified burial | E | Not identified by the Ituri curve; defaults to the community value |
| Case finding: isolated share after (θ′) | 80% | — | Share of all symptomatic cases ever isolated, replacing θ | scenario | Legrand et al. 2007 Table 5; Rivers et al. 2014. A target, not a 2026 observation |
| Case finding: delay reduction | 25% | — | Proportional cut in the onset-to-isolation route time | scenario | Rivers et al. 2014. A target, not a 2026 observation |

### What the defaults imply

- R0 = 2.02, mean generation interval 18.8 days; Rc = 0.83 after the default
  interventions, with an untraced case causing 0.85 infections and a traced case
  0.53.
- 15% of new confirmed cases are known contacts; 8.5% of new infections are
  listed and 7.3% traced.
- 41% of untraced cases are ever detected, against 84% of traced cases, and the
  traced-arm CFR is 24.3% against 29%.
- With 100% listing and follow-up, Rc falls to 0.77 — the ceiling, set by how
  much transmission comes from sources never detected.
- Case finding at θ′ = 80% with a 25% shorter delay takes Rc to 0.55, far more
  than tracing can do at any coverage.

Built up in the order the response deployed, and removed one at a time:

| | Cumulative Rc | | Removed from the default set | Rc |
|---|---|---|---|---|
| no interventions | 2.020 | | community transmission control | 1.810 |
| + community transmission control | 1.019 | | safe burial | 0.969 |
| + isolation infection control | 1.007 | | contact tracing | 0.854 |
| + safe burial | 0.854 | | isolation infection control | 0.845 |
| + contact tracing | **0.831** | | | |
| + case finding (θ′ 80%, delay −25%) | 0.552 | | | |

Perfect version of each, others at default: community at 80% → 0.466, safe
burial at 100% → 0.721, tracing at c = f = 1 → 0.766, isolation IPC at 100% →
0.810. Community transmission control carries about 93% of the achieved
reduction, which restates the route split rather than adding to it; safe burial
is the only other lever whose removal puts Rc back above 1.

**Safe burial is the most uncertain lever**, because the burial share of R0 is an
assumption. Re-fitting the community reduction and listing completeness to the
same two targets for each assumed share, Rc at 100% safe burial is 0.765 at an
8% burial share, 0.720 at the default 13.5%, and 0.624 at 25%, while the fitted
community reduction barely moves (58.3 / 58.4 / 58.6%). The calibration absorbs
the assumed split almost entirely, so it matters little for *current* dynamics
but a great deal for how much safe burial can achieve. **Run 8% and 25%.**

## Validation

Three checks, all detailed in [`evidence.md`](./evidence.md).

**Against Legrand.** `reproduction_numbers()` reproduces his Table 4 — Kikwit
R0 = 2.69 (community 0.50, hospital 0.42, burial 1.77) and Gulu R0 = 2.67 (2.60,
0.007, 0.07) — and a test checks it against an independently written
implementation of his Table 2 transitions. The Gulu preset sets
all three ramp parameters to 0, because Legrand's interventions are steps.

**Against the 2026 outbreak, in Ituri.** The defaults are calibrated to Ituri,
not to the national curve, because the model is one homogeneously mixing
population and Ituri is the single focus that carries 69% of cases. Ituri's
confirmed-case series splits into two regimes:

| Regime | r (/day) | Implied R | Model |
|---|---|---|---|
| Climb, 15 Jun – 25 Jul 2026 | +0.0366 | 1.81 | sets R0 = 2.02 |
| Plateau, 1 Aug – 25 Oct 2026 | −0.0093 | 0.83 | Rc = 0.831, r = −0.0095 |

**The national curve gives a different and lower answer** — climb R = 1.52,
plateau R = 0.93, implying R0 = 1.56 — because it superimposes eight provinces
at different stages, and a composite climbs more slowly than any single focus.
A single-zone model should be calibrated to a single focus, which is why Ituri
is used here.

Fitting the whole Ituri curve rather than the two regimes separately agrees:
R0 = 2.02 with a response ramp of 90–100 days gives a log-RMSE near 0.18 over
18 June – 22 October. The fit is flat between R0 2.0 and 2.3, so the climb-based
inversion is the firmer of the two.

What the model still does not reproduce is the sharp peak: 62 modelled against
115 observed confirmed cases per day on 25 July. One zone with one ramped
community term cannot make a peak that sharp, and no parameter choice does.

**Against the 2026 severity indicators — and what that check is really worth.**
Two of the four indicators are fitted: the community share of confirmed deaths
(which pins θ given `p_notify`) and the confirmed-case CFR (which pins
`p_detect_rec` given δ and θ). The other two are **not independent agreement**.
Writing A for the community share of confirmed deaths and B for the
confirmed-case CFR, the model satisfies exactly

```
intra-CTE deaths per confirmed case   ≡  B · (1 − A)
confirmed cases ever isolated         ≡  B · (1 − A) / δ
no recorded outcome                   ≡  1 − B − (1 − δ) · (B · (1 − A) / δ)
```

for *any* values of θ, δ, `p_notify` and `p_detect_rec` — verified to machine
precision over 400 random draws. So once A and B are matched, the 0.1845 against
0.1843 is arithmetic, and δ itself was defined as 0.1843/0.636, which is the
same statement again. An earlier version of this document called those two
indicators "not fitted"; that was wrong and has been withdrawn.

What survives as genuinely external:

| Check | Model | Constraint | Independent? |
|---|---|---|---|
| Confirmed ever isolated | 0.636 | bracket 0.556–0.698 | yes, but the bracket is 25% wide |
| No recorded outcome | 0.064 | ≤ 0.1425 ceiling | yes, one-sided |
| Ascertainment | 0.409 | 0.41–0.50 (epiforecasts) | yes, and it rests on `p_notify`, which comes from measured swab coverage |
| Swab coverage ceiling | `p_notify` 0.57 | ≤ 0.669 measured | yes, one-sided |

That is a weaker claim than four reproduced indicators, and it is the honest
one. The two-regime growth check above is the load-bearing validation.

## Assumptions and limitations

- **One population:** homogeneous mixing, one health zone at a time, no mobility.
- **Course of illness:** exponential stages with competing exits, as in Legrand.
  The entered onset-to-isolation value is a **route rate, not the realised wait**
  — 7.0 entered gives 10.49 realised, because every exit shares one clock and
  that clock is dominated by the 74% never isolated. Shortening the entered delay
  can therefore only shorten the infectious period of those actually isolated:
  over Legrand's 5 → 1 day range the realised period falls about 10% at θ = 26%,
  against 54% at the θ = 80% he worked with. The 2026 data imply a shape
  parameter of 4–5 for incubation and onset-to-death, so an Erlang version would
  be more faithful.
- **Detection and tracing:** only infections caused by detected cases can be
  traced; contacts already ill when listed are not; the listing delay is fixed;
  follow-up is applied once at listing rather than as a daily hazard, so there is
  no E_T → E leak although 18% of listed contacts are never seen.
- **Tracing is home follow-up, not quarantine.** There is no compartment for
  listed-but-uninfected people, so a contact list never changes a susceptible
  person's risk and nobody is isolated before onset. This is the DRC 2026 regime
  by design, but it is **wrong for a quarantine-based response** such as Kenya's,
  where 92–98.5% of listed contacts were quarantined; there this model would
  badly overstate spread.
- **Outcomes:** the CFR is the same in the community and in isolation unless
  `cfr_ratio_comm` is raised. No capacity limits on beds or tracing teams,
  although Nord-Kivu centres ran at 90–143% occupancy in 2026.
- **Isolation stays are shorter than real ones** (4.71 d untraced, 7.06 traced),
  because the stay is the remainder of the course while real discharge needs two
  negative PCR tests. Transmission is barely affected, but **bed demand is
  understated** — treat H as infectious isolated patients, not bed occupancy.
- **No escape from isolation.** Absconding is real and measured (8 of 116
  discharges on one Ituri report-day; Oraby et al. fit 0.034/day and find it their
  largest reproduction-number component). An explicit H → I leak remains the
  single most valuable structural extension.
- **Case finding reduces transmission but not deaths.** δ is fixed for the
  untraced chain and δ_T derives from `Dh_T − Dh` with `Dh_T = min(gamma_h_t, Dh)`,
  so shortening Dh closes the gap and δ_T rises back towards δ while untraced
  mortality never moves. Read the case-finding scenario as a transmission result
  only.
- **Generation interval** is 18.8 days here (community route 18.5, isolation
  23.2, burial 20.5, weighted by the route shares of R0) against the 15.3 days
  every 2026 analysis borrows from West Africa, no Bundibugyo serial interval
  having ever been measured. Growth rates therefore convert to higher R0 here.
- **Deterministic:** no outbreak-size distribution or fadeout probability.
- **Not a forecast.** The route split is an assumption, and the community
  reduction and listing completeness are calibrated to Ituri's plateau R and
  the known-contact share. For a health zone, refit the β scale and the community
  reduction to that zone's recent incidence.

## Known edge cases

- **Onset-to-isolation at or beyond the onset-to-outcome time** is accepted
  rather than rejected, with the isolation stay floored at 1 day, so sampled
  parameters never abort a batch. In that region the routing no longer exactly
  realises the requested θ and δ.
- **βH = 0** makes Isolation-Unit Infection Control inert whatever value is set.
- **Tracing can raise Rc** when isolation transmits more per day than the
  post-intervention community, i.e. βH_eff ≳ 1.2 × βI_eff, as in Kikwit before
  barrier nursing. A test checks this never happens when βH_eff ≤ βI_eff.
- **Very long runs** come back on a coarser grid: the framework caps the number
  of output points, so a span beyond about a year is returned at two- or
  three-day steps rather than daily.
- **Small introductions** are outside the model's scope; a deterministic run
  grows from day 0 and never fades out.
- **No detection-dated series.** `Confirmed`, `KnownContact` and `Listed` are all
  counted at the moment of infection, so a daily curve has to be shifted forward
  by about 14 days before it lines up with reported incidence, and the shift has
  to be applied by the reader. Dating them at detection would need a distributed
  delay the model does not carry. Cumulative totals are unaffected.
- **No prevalence-threshold activation, and the control is still on the form.**
  All five interventions are activated by date: every effect routes through the
  date-activated ramp and `target_rates=[]` makes the framework's own threshold
  branch a no-op. The model does not declare `start_threshold` / `end_threshold`,
  but the results UI renders a fixed form and shows "By Threshold" regardless.
  **Choosing it and leaving the start date empty disables the intervention
  silently** — in the shipped Ituri scenario that takes cumulative confirmed
  cases from 9,452 to 1.81 million, exactly as if community transmission control
  had been deleted. Supporting thresholds properly would need a state variable
  carrying time since activation, so the ramp has something to key off.

## Differences from the source model

Separate R and D; the three transmission interventions each ramped over their
own scale-up period (set 0 for Legrand's steps); safe burial of deaths in isolation via `s_iso` (set 0
for Legrand); the parallel traced chain with its own θ_T, Dh_T, δ_T, κ_T and
p_notify_T; δ_T derived from the survival benefit of earlier isolation rather
than entered; a percentage reduction of the onset-to-isolation delay where
Legrand's Table 5 sweeps absolute days (at Dh = 7.0 d the 25% default gives
5.25 d, just outside his 1–5 day range, so it is the more conservative lever).
Everything else is his.

## Related models

- `ebola_seihfr_burial_legrand_model` — Legrand without the tracing extension,
  when the parallel chain is not wanted.
- `ebola_seihfr_stochastic_lai_v1_model` — stochastic, for small introductions
  where fadeout probability matters.
- `ebola_importation_model` — for cross-border introduction risk rather than
  within-population dynamics.

## Version history

| Date | Change | By |
|---|---|---|
| 2026-10-20 | First version. Legrand et al. (2007) SEIHFR with a parallel contact-tracing chain, defaults for the 2026 DRC Bundibugyo outbreak. | Yingsi Lai |
| 2026-11-19 | Recalibrated to Ituri rather than the national curve: R0 2.5 → 2.02, community reduction 62.1% → 58.3%, listing completeness 36.3%, Rc 0.950 → 0.831. The contact follow-up rate moved from the national 82% to Ituri's own measured 86% (WHO DON613/DON622 and the AFRO weekly sitreps), and the community reduction and listing completeness were then solved jointly to hold Rc = 0.83 and the 15% known-contact share. The three transmission interventions now ramp over their own new scale-up periods (`community_ramp_days`, `hospital_ramp_days`, `funeral_ramp_days`, 95 d) instead of stepping, and share one start date. Cumulative confirmed cases, confirmed known contacts, deaths and infections in listed contacts are now published as the output series `Confirmed`, `KnownContact`, `Deaths` and `Listed` (previously `Confirmed_total` / `KnownContact_total` / `Deaths_total` / `Listed_total`, all of which the framework filtered out of the delivered time series). | Yingsi Lai |

This is the first version; there are no later releases and the model has not been
renamed, so there is no earlier folder name or `disease_type` to record. The
model has not yet been published on the platform.

Changes made between 20 October and 17 November 2026, before first submission:
the traced chain was reverted to Legrand's competing-risk routing after an
interim branching formulation; R and D were split out of Legrand's single
removed compartment; the traced-chain defaults and the detection parameters were
re-derived from 2026 sources; `cfr_ratio_comm` was added so the equal-CFR
assumption can be relaxed; the documentation was split into this file and
[`evidence.md`](./evidence.md); and the example config was moved from Legrand's
Gulu 2000 scenario to the DRC 2026 one, with Gulu kept as a test preset. Claims
that did not survive re-examination are listed in `evidence.md` under "Claims
withdrawn on re-examination".

## AI support

The model code, the parameter search, the documentation and the validation
analysis were supported by Claude (Anthropic), through Claude Code; parts of the
git history are authored by it. Yingsi Lai directed the work, reviewed every
change, and is responsible for the model and this submission.

## Key references

- Legrand J, et al. Understanding the dynamics of Ebola epidemics. Epidemiol Infect. 2007;135:610–621.
- Akilimali A, et al. Clinical characteristics of patients infected with Bundibugyo virus, DRC 2026. N Engl J Med. 2026. doi:10.1056/NEJMc2608070.
- Kabasele K, et al. Bundibugyo virus disease outbreak response, DRC, 2026. MMWR Morb Mortal Wkly Rep. 2026;75(35).
- Oraby T, Falay D, Ndeffo-Mbah ML. Detection, isolation and leakage in the 2026 Bundibugyo virus outbreak, Ituri. medRxiv 2026.08.25.26361360.
- Rosello A, et al. Ebola virus disease in the Democratic Republic of the Congo, 1976–2014. eLife 2015;4:e09015.
- Wamala JF, et al. Ebola hemorrhagic fever associated with novel virus strain, Uganda, 2007–2008. Emerg Infect Dis. 2010;16(7):1087–1092.

The full reference list is in [`evidence.md`](./evidence.md).
