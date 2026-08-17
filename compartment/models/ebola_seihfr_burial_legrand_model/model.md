# Ebola (SEIHFR, Community/Hospital/Burial) — Model Summary

A deterministic port of the mean-field equations behind Legrand J, Grais RF,
Boelle PY, Valleron AJ, Flahault A. **"Understanding the dynamics of Ebola
epidemics."** *Epidemiol Infect.* 2007;135(4):610-621.
[doi:10.1017/S0950268806007217](https://doi.org/10.1017/S0950268806007217)
(PMC2870608).

This is a distinct model from the repository's other `ebola_jax_model`
(an unrelated port of Li et al. 2019 / Getz & Dougherty 2018 with an
Erlang linear-chain structure). This model's distinguishing feature is
explicit separation of **community, hospital, and traditional-burial**
transmission routes, each with its own transmission coefficient.

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
- Interventions are modelled as fully efficient from their start date
  onward (a step function), not gradual — matching the paper's own
  assumption (b).
- After interventions start, hospital and funeral transmission are
  assumed to be reduced toward zero and community transmission is
  reduced by a smaller fixed factor — matching assumption (d). This is
  implemented as three independent interventions (`community_intervention`,
  `hospital_intervention`, `funeral_intervention`) rather than one shared
  intervention, since each route's reduction magnitude differs.

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
own Table 4 almost exactly) is correct; it is the size of the well-mixed
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
