# Novel Respiratory (SEIHDR) — Model Summary

A deterministic compartmental model for novel respiratory diseases. People move
from susceptible through an exposed (infected but not yet infectious) stage to
infectious, and from there either recover, die, or pass through hospital. The
population is split into three age bands that mix at rates taken from the
country's synthetic contact matrix, and zones are linked by daily travel, so an
outbreak in one region seeds its neighbors.

Seven reduced variants of this model are registered alongside it — see
[Related models](#related-models).

## Compartment and state definitions

- **S** — Susceptible.
- **E** — Exposed. Infected but not yet infectious.
- **I** — Infected and infectious. The only compartment that transmits.
- **H** — Hospitalized.
- **D** — Deceased.
- **R** — Recovered and immune.

Every state is tracked separately for each of the three age bands and each zone.
Alongside these, the framework keeps running cumulative counters — `E_total`,
`I_total`, `H_total`, `D_total`, `R_total` — recording everyone who has ever
entered that compartment, since the daily curves above only show who is in it
right now.

## Inputs and parameters

**Per zone:** name, latitude, longitude, population, and the initial infected
percentage.

**Disease parameters**, with defaults and the ranges offered for uncertainty
runs:

| Parameter | Flow | Default | Uncertainty range |
| :--- | :--- | :--- | :--- |
| Transmission rate | S→E | 0.25 /day | 0.2–0.3 |
| Incubation period | E→I | 5 days | 2–14 |
| Hospitalization rate | I→H | 4% | 4–40 |
| Infection fatality rate | I→D | 0.0001 /day | 0.0001–0.001 |
| Hospital fatality rate | H→D | 0.001 /day | 0.001–0.005 |
| Recovery period | I→R | 7.14 days | 4–10 |
| Hospital recovery period | H→R | 7.14 days | 3–14 |

**Travel rate (σ)** — 20% of each zone's population away from home per day,
distributed by an inverse-square gravity model weighted by population and
distance. Setting it to 0 isolates every zone.

**Age bands** — children 0–17 (33.3% of the population), adults 18–55 (44.4%),
elderly 56+ (22.3%). The percentages are editable per simulation.

**Interventions**, all of which lower the transmission rate while active:

- **Mask Wearing** — 20% adherence, 35% transmission reduction.
- **Social Isolation** — 40% adherence, 50% reduction.
- **Lockdown** — 80% adherence, 70% reduction, and additionally halts all
  inter-zone travel while active.
- **Vaccination** — 60% adherence (read as coverage), 80% reduction (read as
  vaccine efficacy).

The two figures multiply, so the transmission rate is scaled by
`1 - adherence × reduction`. Mask wearing at 20% adherence and 35% reduction
therefore cuts transmission by 7%, not 35%.

## Initial conditions

Each zone's initial infected percentage sets **I**; everyone else starts in
**S**. Both are then split across the three age bands by the demographic
weights. **E**, **H**, **D**, and **R** all start empty, as do the cumulative
counters, so the opening of a run is a short build-up through the exposed stage
before infections take off.

## Outputs

Daily counts for all six compartments plus the five cumulative counters,
reported per zone and summed across the region. Every run is produced twice —
once with the interventions applied and once without — so the two can be
compared directly. In uncertainty mode the parameters above are sampled from
their ranges and the output carries a median with confidence bands instead of a
single trajectory.

## Model nuances

- **Only the infectious compartment transmits.** Exposed and hospitalized
  people contribute nothing to the force of infection.
- **Transmission is frequency-dependent** — it scales with the *proportion*
  infectious rather than the count, so a large and a small zone with the same
  prevalence have the same per-person risk. The deceased are excluded from the
  population used for that proportion.
- **Two mixing steps, in order.** Prevalence is first pooled across zones
  through the travel matrix, then across age bands through the contact matrix.
  A susceptible person's risk therefore depends on where their zone's residents
  travel and on which age groups they meet there.
- **The contact matrix is country-aware.** Because all three age bands declare
  an age range, the framework loads the Prem 2021 synthetic matrix for the
  country in `admin_unit_id` and aggregates it to these bands. Countries absent
  from that dataset fall back to their World Bank income-group average, then to
  a global average.
- **The hospitalization and fatality parameters are daily hazards, not
  lifetime shares.** The 4% hospitalization rate is converted to 0.04 per day,
  meaning 4% of those currently infected are hospitalized each day — not 4% of
  all infections overall. The same applies to both fatality rates.
- **Hospitalization does not vary by age.** The same rate applies to all three
  bands, so the model will not reproduce the steep age gradient in severe
  outcomes seen in real respiratory epidemics.
- **Vaccination carries no immunity forward.** It lowers transmission only while
  switched on; the moment it ends, transmission returns to baseline and nobody
  retains protection from the campaign.
- **Recovery is permanent.** There is no waning immunity and no route from R
  back to S, so a single epidemic wave cannot recur.
- **The population is closed** — no births or natural deaths. Only disease
  deaths remove anyone, and they are retained in D rather than dropped.
- **Travel restrictions are all-or-nothing.** Lockdown stops inter-zone travel
  completely; a partial reduction is not supported.

## Known edge cases

- **Travel is silently disabled for admin-2 zones** regardless of the travel
  rate, so a sub-national run will not show spatial spread.
- **Very small zones or very low initial infected percentages** can round to
  under one person and fail to start an outbreak.
- **Stacking interventions** applies each reduction in turn to the already
  reduced rate, so their effects compound rather than add. All four active at
  once can drive transmission low enough to suppress the epidemic outright.

## Deliberate simplifications

This model is not a reimplementation of a specific published model; the
structure is a conventional SEIHDR and the parameter defaults are illustrative
rather than fitted to a particular pathogen. Notable simplifications:

- Age enters only through contact mixing. Susceptibility, severity, and
  mortality are identical across bands.
- Interventions act as a flat multiplier on transmission rather than being
  targeted at specific ages, settings, or zones.

## Related models

Seven variants share this model's equations and differ only in which
compartments exist. Pick the smallest structure that answers the question, since
each removed compartment also removes its parameters:

| Variant | Compartments |
| :--- | :--- |
| `COVID_SEIHDR` | Full model (this one) |
| `COVID_SEIHR` | No deceased |
| `COVID_SEIDR` | No hospital |
| `COVID_SEIR` | No hospital, no deceased |
| `COVID_SIHDR` | No exposed stage |
| `COVID_SIHR` | No exposed, no deceased |
| `COVID_SIDR` | No exposed, no hospital |
| `COVID_SIR` | Susceptible, infected, recovered only |