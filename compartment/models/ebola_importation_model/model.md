# Ebola SEI1I2RD

A deterministic compartmental model of Ebola transmission used to estimate the importation risk of Ebola from the Democratic Republic of the Congo (DRC) to the EU/EEA. It was translated from the R Monte-Carlo model "Ebola 2026 Modelling" (Bons, Gomes Dias, Hansson, Kuhlmann Berenzon, Prasse), parameterised for the combined Ituri and Nord Kivu population. Exposed cases branch into two infectious strata — those who will recover and those who will die — reflecting different dwell times to their respective outcomes.

## Compartments

| id | meaning |
|----|---------|
| S | Susceptible population |
| E | Exposed (latent, not yet infectious) |
| IR | Infectious individuals who will recover |
| ID | Infectious individuals who will die |
| R | Recovered and immune |
| D | Deceased |

## Parameters

| name | meaning | default | unit |
|------|---------|---------|------|
| beta | Frequency-dependent transmission rate (S→E), from both infectious strata | 0.124 | per day |
| gamma | Infectious period to recovery (IR→R) | 10.0 | days |
| mu | Symptom-to-death period (ID→D) | 10.0 | days |
| incubation_period | Time from infection to symptom onset (E dwell time, σ = 1/incubation_period) | 10.0 | days |
| p_death | Case-fatality probability (branching fraction of E outflow) | 0.43 | fraction |

## Dynamics

Infection follows a frequency-dependent term `beta * S * (IR+ID) / N`, moving susceptibles into E. Exposed individuals leave E at rate `sigma = 1/incubation_period`, branching to IR with probability `1-p_death` and to ID with probability `p_death`. Recovering cases leave IR at rate `gamma` into R, and dying cases leave ID at rate `mu` into D. The S→E, IR→R and ID→D flows are standard schema transmission edges; the shared-σ split of the E outflow is applied manually. Interventions, if configured, are applied to the transmission rate via the framework's intervention hook.

## Assumptions and limitations

- Well-mixed population of Ituri + Nord Kivu (~13,392,200 in the source).
- Frequency-dependent transmission, with both infectious strata (IR and ID) equally infectious.
- Exposed cases are partitioned into recovering vs. dying at symptom onset by the case-fatality probability `p_death`.
- The Monte-Carlo parameter sampling of the source (drawing R0, p_death, incubation, infectious, and death-delay from their distributions) is **not** reproduced; only the deterministic ODE core is translated, and the framework supplies uncertainty handling.
- The eigenvalue-based seeding of initial compartment values from observed cumulative deaths in the source is **not** reproduced; the framework supplies the initial state.
- `beta` is exposed as a directly-specified default (0.124, derived from the source mean parameters) rather than being recomputed each simulation from R0, p_death, gamma and mu as in the source.
- Travel, detection-probability, and dark-factor parameters from the source `parameters.R` are not part of this transmission model and are not represented here.
- No inter-region travel is modelled; `prepare_initial_state` returns the identity population matrix.