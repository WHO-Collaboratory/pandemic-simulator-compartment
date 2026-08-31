# Example Disease (Spatial Mobility + Age Mixing) — Model Notes

An **age-stratified SIR** model that combines two kinds of structure:

- **Spatial** — you pick one of **four** inter-zone mobility mechanisms by
  number, and the model synthesises the origin–destination matrix from each
  admin zone's **population and coordinates** (no movement data required).
- **Age** — the population is split into **four age bands**, which mix through a
  country **contact matrix** (Prem 2021, auto-loaded from each band's
  `age_range`). Transmission still uses a **single scalar `beta`** — there is no
  per-age transmission rate; the contact matrix is what makes the age groups
  behave differently.

It also keeps parameter uncertainty (median + interval over many draws).

## What it does

A three-compartment SIR model over multiple admin zones **and** age groups:

- **Compartments:** `S` (susceptible), `I` (infected, `infective=True`), `R` (recovered). State is shaped `(compartments, age groups, zones)`.
- **Transmission:** `S → I` driven by a single `beta` (frequency-dependent), `I → R` at rate `gamma`.
- **Spatial coupling:** a mobility matrix `T` (`T[i,j]` = fraction of zone *i*'s residents present in zone *j* per day; rows sum to 1) couples the zones.
- **Age coupling:** a contact matrix `M` (`M[a,b]` = relative contact intensity between age groups *a* and *b*) mixes the age bands within each zone.
- **Interventions:** a single `my_intervention` that reduces `beta` while active.
- **Parameter uncertainty:** `beta` and `gamma` expose default/min/max ranges, so the simulator runs multiple draws and reports a median with an interval.

## Choosing the mobility mechanism (by number)

Set `travel_model` in the config's `Disease` section to an **integer 1–4**:

| `travel_model` | Mechanism | Flow weight i → j | Knob |
|:---:|---|---|---|
| **1** | gravity | `pop_j / d_ij^alpha` | `travel_alpha` |
| **2** | exponential | `pop_j · exp(-d_ij / scale_km)` | `travel_scale_km` |
| **3** | radiation | Simini et al. 2012 (parameter-free) | — |
| **4** | uniform | equal to every other zone | — |

All four are built from population + great-circle distances (no data needed) and
return a **row-stochastic** matrix, so each zone's population is conserved.
`travel_sigma` (percent, default 20) is shared by all four: it sets the outbound
mass per day — each row's off-diagonal sums to `sigma`, the diagonal (stay-home)
is `1 − sigma` — and the chosen mechanism only decides **how that `sigma` is split
across destinations**. Setting `travel_sigma: 0` gives the identity matrix (no
movement between areas), whichever mechanism is selected.

### 1 — Gravity  `pop_j / d_ij^alpha`

The classic spatial-interaction law: a destination pulls travellers in proportion
to its **population** and in inverse proportion to a **power of the distance**. The
exponent `travel_alpha` (`2.0` = inverse-square) sets how fast pull falls with
distance. Good general-purpose default, but it has a heavy long-distance tail and
needs that exponent calibrated.

### 2 — Exponential  `pop_j · exp(-d_ij / scale_km)`

Same population pull, but distance enters through an **exponential decay** with a
characteristic range `travel_scale_km` (call it *L*). Attraction is roughly flat
within *L*, then falls off quickly — very few trips beyond ~2–3 *L*. Lighter
long-range tail than gravity; a better fit when travel has a definite commuting
radius.

### 3 — Radiation  (Simini et al. 2012)  — *parameter-free*

A mechanistic commuting model with **no decay knob to tune**. The flow from *i* to
*j* depends on the two populations **and** on `s_ij`, the "intervening
opportunities" (the population living closer to *i* than *j*). Flows emerge from
the population landscape itself; it can, however, underestimate long, sparse
flows. `travel_sigma` still sets the outbound mass.

### 4 — Uniform  (distance-agnostic baseline)

Each zone spreads its `travel_sigma` **equally across all other zones**, ignoring
distance and destination population. Useful as a simple, assumption-light
baseline or a sanity check.

`travel_scale_km` is only read for mechanism 2, and `travel_alpha` only for
mechanism 1; they are ignored by the others but should stay within their valid
ranges (`travel_scale_km ≥ 1`, `travel_alpha ≥ 0.1`).

## Age structure and the contact matrix

The four demographic groups (`age_0_17`, `age_18_49`, `age_50_64`, `age_65_plus`)
are declared **with `age_range`**, so the framework auto-loads the country's Prem
2021 synthetic contact matrix, aggregated to those four bands. Calling
`self._prepare_demographic_state()` in `prepare_initial_state()` expands the state
across age bands, so **results are tracked per age group** and the age groups can
be exposed differently.

**Single `beta`, contact matrix does the differentiation.** There is only one
transmission rate. To keep `beta` meaning the same thing it did in the non-age
model (an overall transmission scale, ≈ R0·γ), the contact matrix `M` is
**normalised by its spectral radius** once at setup, so `M` only *redistributes*
contact intensity across ages rather than inflating the total. Age bands that mix
more (e.g. children) therefore experience a higher force of infection and a higher
attack rate, even though every band shares the same `beta`.

## How the couplings enter the dynamics

State axes are `(age, zone)`, with the zone axis last. The force of infection for
susceptibles of age *a* in zone *r* applies **spatial presence first, then age
mixing**:

```
N_present[b,j] = Σ_i N[b,i] · T[i,j]        (age-b people present in zone j)
I_present[b,j] = Σ_i I[b,i] · T[i,j]        (age-b infectious present in zone j)
phi[b,j]       = I_present[b,j] / N_present[b,j]     (age-b prevalence in zone j)
exposure[a,j]  = Σ_b M[a,b] · phi[b,j]      (age mixing within each zone)
lambda[a,r]    = beta · Σ_j T[r,j] · exposure[a,j]
new_inf[a,r]   = S[a,r] · lambda[a,r]
```

Residents of zone *i* are exposed to the (age-mixed) prevalence of every zone they
visit. Two limiting cases recover simpler models exactly:

- `M = I` (identity) and one age group → the original zone-only spatial SIR.
- `travel_sigma = 0` (`T = I`) → independent, age-mixed SIRs per zone.

## Nuances users should know

- **Mechanism is a number, not a string.** `travel_model` is an integer 1–4;
  values outside that range fail config validation.
- **`_total` compartments are auto-generated.** The framework adds cumulative
  `I_total` and `R_total` (per age group) for transmission-edge targets — don't
  declare them.
- **Units are converted for you.** `gamma` is declared as `DAYS` and converted to
  a rate internally. `PERCENTAGE` params (e.g. `travel_sigma`) arrive as `20.0`,
  not `0.2`; the model converts with `self._to_rate(...)` before building the
  matrix.
- **Parameter uncertainty is UNIFORM-only.** Set `has_variance: true` with a
  `UNIFORM` distribution and `min`/`max` in a field's `FieldConfigs`. The example
  varies `beta` and `gamma`; the results carry a `median` with `lower`/`upper`.
- **Multiple zones matter for space; multiple ages matter for the contact
  matrix.** Mobility only has an effect with two or more admin zones; age mixing
  only differentiates outcomes when the demographics declare more than one band.
- **The standard results JSON sums over age.** The dynamics are age-resolved
  internally (state `(compartment, age, zone)`), but the framework's output
  post-processor aggregates the age axis, so the standard JSON shows
  population-wide `S/I/R`. To read per-age results, capture the solver's
  age-resolved array before that sum (see `extract_age_results.py`).

## Running

```
python -m compartment.models.example_spatial_mobility_v2_model.main \
    --mode local \
    --config_file compartment/models/example_spatial_mobility_v2_model/example-config.json \
    --output_file results/example_spatial_mobility-test.json
```
