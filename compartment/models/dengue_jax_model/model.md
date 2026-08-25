# Dengue (4-Serotype) — Model Summary

A dengue model that tracks all four serotypes (DENV-1 to DENV-4) in people
alongside an explicit mosquito population whose size and behavior are driven by
local temperature. Recovering from a first infection gives temporary protection
against every serotype; once that fades, a person can catch one of the other
three, and this second infection carries a raised risk of hospitalization
(antibody-dependent enhancement).

## Compartment and state definitions

**People**

- **S** — Susceptible. Never infected with any serotype.
- **E** — Exposed. Infected but not yet infectious.
- **I** — Infected (first infection). Can pass the virus to mosquitoes.
- **C** — Cross-protected. Recovered from a first infection and temporarily
  immune to all four serotypes.
- **Snot** — Partially susceptible. Cross-protection has faded; still immune to
  the serotype already had, susceptible to the other three.
- **E2 / I2** — Exposed / infected with a second, different serotype.
- **H** — Hospitalized with a severe second infection.
- **R** — Recovered from a second infection.

**Mosquitoes** — charted alongside the human curves, but these are mosquitoes,
not people.

- **SV** — Susceptible. **EV** — Carrying the virus but not yet able to pass it
  on. **IV** — Infectious.

Internally each curve is split by serotype, giving 66 compartments (58 disease
states plus 8 cumulative counters). The charts group them back together.

## Inputs and parameters

**Per zone:** population, initial infected percentage, seroprevalence
(percentage with prior dengue exposure), and minimum, maximum, and mean annual
temperature. The hemisphere is taken from the zone's latitude and decides which
half of the year is warm.

**Disease parameters**, with defaults:

- Cross-immunity period — 240 days
- Host latent period — 5.9 days; infectious period — 5.0 days
- Hospitalization rate for second infections — 1%; hospital stay — 4.9 days
- Maximum mosquito carrying capacity — 1.5 × the human population
- Reference temperature — 29 °C, with an activation energy of 0.05 controlling
  how sharply mosquito numbers fall away from it
- Travel rate (σ) — 20% of each zone's population away from home per day
- Number of serotypes — fixed at 4

**Interventions**

- **Bite Reduction** (bed nets, repellent) — lowers the mosquito biting rate.
- **Vector Control** (spraying, larvicide) — lowers mosquito survival.

## Initial conditions

Seroprevalence is spread evenly across the four partially susceptible states (Snot),
the initial infected percentage is spread evenly across the four first-infection
states, and everyone else starts fully susceptible. Nobody starts out exposed,
hospitalized, or recovered.

**Mosquitoes start at zero.** Mosquito abundance is not an input. Instead the
model works out a carrying capacity for each zone — the largest mosquito
population that zone can support, taken from the maximum carrying capacity above
and reduced as the temperature moves away from the reference temperature — and
grows the population up to it. In warm conditions this takes about a day and a
half, so the opening of a run is a brief mosquito build-up rather than a
meaningful part of the epidemic. The
same self-starting behavior is what lets the population rebuild after it
collapses in the cold season.

## Outputs

Daily counts for every curve above, plus cumulative totals for exposures, first
and second infections, hospitalizations, and recoveries. Results are reported
per zone and summed across the region, and every run is produced twice — once
with the interventions applied and once without, for comparison.

## Model nuances

- **Temperature drives the mosquitoes.** Biting rate, survival, development
  speed, incubation, and the chance a bite passes the virus in either direction
  are all recalculated from a seasonal temperature curve that swings between the
  zone's minimum and maximum.
- **Mosquito numbers scale with people.** Carrying capacity is a multiple of the
  human population in the zone, so a small town supports proportionally fewer
  mosquitoes than a city.
- **Only second infections can be severe.** Hospitalization applies to second
  infections only; a first infection never leads to a hospital stay.
- **Zones are connected by travel**, using a gravity model weighted by
  population and distance. Setting the travel rate to 0 isolates every zone.
- **Births and deaths** run at a fixed rate matching a 73-year life expectancy.
- **Age groups have no effect.** Demographics can be supplied, but this model
  mixes the whole population together and does not use them.

## Known edge cases

- **Seasonal collapse is normal.** Mosquito numbers crash when the seasonal
  curve drops below the growth threshold and rebuild when it rises again, so
  epidemics restart in the following warm season.
- **Very small zones or very low initial infected percentages** can round to
  under one person, and infected counts below one person are cleared to zero, so
  no outbreak starts.

## Sources and deliberate simplifications

The temperature responses follow Huber et al. (2018) and the serotype and
immunity structure follows García-Carreras et al. (2022). Simplifications made here:

- The serotype count is fixed at 4 and cannot be changed.
- Mosquito carrying capacity is a straightforward multiple of the human
  population rather than an ecologically derived figure.
- Seroprevalence and initial infections are split evenly across the four
  serotypes, since zone-level data rarely distinguishes them.
- A very small constant is added to the infectious mosquito compartments each
  step, keeping them from reaching exactly zero and letting transmission resume
  after a seasonal collapse.