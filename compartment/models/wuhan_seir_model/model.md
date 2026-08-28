# Age-structured SEIcIscR (Wuhan COVID-19)

An age-structured, deterministic, discrete-time (daily) compartmental model of the Wuhan COVID-19 outbreak. It is a translation of the SEIcIscR variant of Prem et al. (2020), *The effect of control strategies to reduce social mixing on outcomes of the COVID-19 epidemic in Wuhan, China* (Lancet Public Health). Infections are split into fully infectious clinical cases and less-infectious subclinical cases, with transmission mediated by an age contact matrix over 16 five-year age bands.

## Compartments

| id | meaning |
|----|---------|
| S | Susceptible |
| E | Exposed (latently infected, not yet infectious) |
| Ic | Clinical infectious cases (fully infectious) |
| Isc | Subclinical infectious cases (reduced infectiousness) |
| R | Recovered / removed |

## Parameters

| name | meaning | default | unit |
|------|---------|---------|------|
| beta | Per-contact transmission probability, S→E (normally reverse-engineered from R0) | 0.025 | per day |
| alpha_c | Mean latent period, E→Ic | 6.4 | days |
| alpha_sc | Mean latent period, E→Isc | 6.4 | days |
| gamma_c | Mean infectious period, Ic→R | 7.0 | days |
| gamma_sc | Mean infectious period, Isc→R | 7.0 | days |
| subclinical_infectiousness | Infectiousness of subclinical relative to clinical cases | 0.25 | fraction |
| clinical_fraction_child | Fraction of infections becoming clinical, ages 0–19 | 0.4 | fraction |
| clinical_fraction_adult | Fraction of infections becoming clinical, ages 20+ | 0.8 | fraction |

## Dynamics

Susceptibles acquire infection through an age-structured force of infection: for each age band, `lambda_a = beta * sum_b C[a,b] * (Ic_b + 0.25*Isc_b)/N_b`, where `C` is the aggregated contact matrix and subclinical cases contribute at 25% infectiousness. Exposed individuals progress at rate `alpha` and split into clinical (`Ic`) with age-specific probability `rho` (0.4 for ages 0–19, 0.8 otherwise) and subclinical (`Isc`) with probability `1 - rho`; both recover at rate `gamma`. Transitions use the exact discrete-time hazard `1 - exp(-1/duration)` under a fixed-step Euler solver, reproducing the source difference equations. Two interventions — school closure and lockdown — are represented as reductions applied to `beta`.

## Assumptions and limitations

- Discrete-time daily updates with `1 - exp(-1/duration)` transition hazards, matching the source rather than continuous `1/duration` rates.
- Subclinical infections are fixed at 25% of clinical infectiousness.
- Clinical fraction `rho` is age-specific and hard-coded by age band (0.4 for the first 4 bands, 0.8 for the rest), as in the source.
- Mean latent period 6.4 days and mean infectious period 7 days, per the source.
- The source's location-decomposed contact structure (home/work/school/others) is **not** represented; the framework's single aggregated (Prem 2021) contact matrix is used instead.
- The source's time-varying `pWorkOpen` schedule and matrix-scaling interventions are **approximated** as flat reductions on `beta` (school closure ≈ 20%, lockdown ≈ 60%); the exact date-driven staggered relaxation schedule and per-location scaling are not reproduced.
- `beta` is exposed as a direct parameter (default 0.025); the source's reverse-engineering of `beta` from R0 and the leading eigenvalue of the contact matrix is not performed within this model.