# Example SIR

A simple deterministic SIR (Susceptible–Infected–Recovered) compartmental model. It is a translation of a standard three-equation ODE SIR model (`sir.py`) that integrated `dS = -beta*S*I`, `dI = beta*S*I - gamma*I`, `dR = gamma*I` on a normalized population.

## Compartments

| id | meaning |
|----|---------|
| S | Susceptible to infection |
| I | Currently infectious (infective) |
| R | Recovered and immune |

## Parameters

| name | meaning | default | unit |
|------|---------|---------|------|
| beta | Transmission rate on the S→I edge (frequency-dependent) | 0.1 | per day |
| gamma | Per-capita recovery rate on the I→R edge | 0.05 | per day |

## Dynamics

Susceptibles move to the infected compartment at rate `beta` scaled by the proportion of the population that is infective, and infected individuals recover to `R` at per-capita rate `gamma`. Transmission is coded as frequency-dependent (`beta * S * I / N`); because the source normalized the population to `N = 1`, this coincides with the source's `beta * S * I` term. The framework's intervention hook is applied to `beta` at each time step, but no interventions are configured in this model.

## Assumptions and limitations

- Deterministic, single-population model with no age, spatial, or risk strata.
- No births, deaths, or waning immunity; recovery confers permanent immunity.
- Transmission is expressed in frequency-dependent form; this only matches the source because the source's population is normalized to `N = 1`. For any non-normalized population the two forms differ.
- The intervention mechanism is wired to `beta` but is inactive by default, so behavior matches the plain source model unless interventions are added.