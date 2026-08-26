# Age-Stratified SIR (per-age transmission) — Model Notes

A schema-based **SIR** model in which **each age group has its own transmission rate**. It is built the declarative way: compartments, transmission edges, per-age parameters, interventions, and age bands are described in `define_parameters()`, and the framework derives the equations, config validator, and artifact from that schema. It extends the framework's SIR example so that susceptibility/acquisition differs by age band, while age groups still infect one another through a contact matrix.

## Model overview

Susceptible people become infected through contact with infectious people, then recover and stay immune (no waning). Transmission is **frequency-dependent**: the force of infection scales with the *proportion* infectious, not the raw count. The population is closed — no births, deaths, or arrivals — so the compartment totals always sum to the starting population.

The population is split into four age bands (0–17, 18–49, 50–64, 65+). There is no single "average" transmission rate — the four per-age rates are the only transmission parameters. Two things make transmission age-specific:

1. **Per-age transmission rate.** Each band has its own `beta_age_*` parameter — the per-day rate at which a *susceptible* in that band acquires infection per unit of contact-weighted infectious pressure. Raising one band's rate makes that band's epidemic larger and earlier without touching the others.
2. **Age mixing.** Who contacts whom is set by a contact matrix (Prem 2021 synthetic matrix, auto-loaded for the selected country and aggregated to these four bands via each group's `age_range`). This is what carries infection from a high-transmission band into the rest of the population.

Regions are coupled by a gravity **travel matrix**. The model is deterministic: it integrates an ODE system with JAX's `odeint`; parameter uncertainty re-runs the same equations across sampled parameter values.

## Compartment and state definitions

| Compartment | Meaning |
|---|---|
| `S` | Susceptible — can be infected |
| `I` | Infected and infectious (`infective=True`, drives the force of infection) |
| `R` | Recovered and immune (no route back to `S`) |
| `I_total` | Cumulative count of everyone who has ever entered `I` |
| `R_total` | Cumulative count of everyone who has ever entered `R` |

The two `_total` compartments are generated automatically for each transmission-edge target — do not declare them by hand. Internally every compartment is stratified by age group and administrative zone, so the state array is *(compartments × age groups × zones)*. `equation()` receives it already shaped that way and indexes it positionally in `compartment_list` order.

## Inputs and parameters

Transmission edges, set under `TransmissionEdges` in the config:

| Parameter | Edge | Unit | Default | Role |
|---|---|---|---|---|
| `gamma` | `I → R` | days (`DAYS`) | `10` | Mean infectious period. Declared as `DAYS`, so the framework converts to a `0.1`/day rate at load time — enter it in days. |

The `S → I` transmission is age-specific and applied directly from the per-age rates below (there is no scalar/average `beta` edge). Cumulative infections are tracked in an explicitly declared `I_total` compartment.

Per-age transmission rates, set under `Disease` in the config (one editable rate per band):

| Parameter | Band | Unit | Default |
|---|---|---|---|
| `beta_age_0_17` | Children (0–17) | per day (`RATE`) | `0.45` |
| `beta_age_18_49` | Young adults (18–49) | per day (`RATE`) | `0.35` |
| `beta_age_50_64` | Older adults (50–64) | per day (`RATE`) | `0.30` |
| `beta_age_65_plus` | Seniors (65+) | per day (`RATE`) | `0.25` |

Each is an absolute per-day rate with its own default, valid range (`0`–`2`), and uncertainty range (±25% by default), and each surfaces as a custom field in the platform UI. Setting a band's rate to `0` makes that band fully protected (it acquires no infection), because the rate gates the *susceptible* band's acquisition.

Other inputs: `travel_sigma` (percentage of each region's population away from home per day, feeds the gravity travel matrix) and the `demographics` block (population share per band; also determines the age axis order).

A single intervention (`social_isolation`) targets every per-age rate (`beta_age_0_17` … `beta_age_65_plus`) directly; while it is active each band's transmission rate is reduced by its `adherence × transmission_reduction` factor.

## Force of infection

For susceptibles in age band *a* and region *r*:

```
lambda[a, r] = beta[a] * sum_b  M[a, b] * ( T @ (I_b / N_b) )[r]
```

where `beta[a]` is the band's per-age rate, `M[a, b]` is the contact matrix (mean daily contacts of an *a*-person with *b*-people), `T` is the region travel matrix, and `I_b / N_b` is the proportion infectious in band *b*. The `S → I` flow is computed manually so it can mix age groups through `M` and regions through `T`; the `I → R` edge is handled by the framework.

## Outputs

Two runs are returned — an intervention ("business as usual") run and an interventionless control — each as a per-compartment, per-region time series with a median and a confidence band when parameter uncertainty is on. Cumulative incidence per band comes from `I_total`.

## Assumptions and limitations

Closed population; homogeneous mixing within an (age band, region) cell; full immunity after recovery (no `R → S`). Following the framework's flagship age-stratified (COVID) model, the infectious proportion is normalised by each region's *total* population, with the per-contact scale carried by the contact-matrix magnitudes — so per-age rates should be read relative to that convention rather than as standalone within-band reproduction numbers. Contact patterns come from the Prem 2021 synthetic matrices for the selected country and are static over the run. This is a reference model, not a calibrated model of any specific disease.
