# Example Disease (Selectable Spatial Mobility) — Model Notes

A minimal **SIR** model that demonstrates **inter-zone mobility**: you pick one of
**four** mobility mechanisms by number, and the model synthesises the
origin-destination matrix from each admin zone's **population and coordinates** —
no movement data required. It also keeps parameter uncertainty (median +
interval over many parameter draws).

## What it does

A classic three-compartment SIR model over multiple admin zones:

- **Compartments:** `S` (susceptible), `I` (infected, `infective=True`), `R` (recovered).
- **Transmission:** `S → I` at rate `beta` (frequency-dependent), `I → R` at rate `gamma`.
- **Spatial coupling:** a mobility matrix `T` (`T[i,j]` = fraction of zone *i*'s
  residents present in zone *j* per day; rows sum to 1) couples the zones in the
  force of infection.
- **Interventions:** a single `my_intervention` that reduces `beta` while active.
- **Parameter uncertainty:** `beta` and `gamma` expose default/min/max ranges,
  so the simulator runs multiple draws and reports a median with an interval.

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
to its **population** and in inverse proportion to a **power of the distance**. Big
places attract more trips; the pull weakens with distance at a rate set by the
exponent `travel_alpha` (`2.0` = inverse-square, the textbook default; larger →
more local, smaller → more long-range). Good general-purpose default, but it has a
heavy long-distance tail and needs that exponent calibrated.

### 2 — Exponential  `pop_j · exp(-d_ij / scale_km)`

Same population pull, but distance enters through an **exponential decay** with a
characteristic range `travel_scale_km` (call it *L*). Attraction is roughly flat
within *L*, then falls off quickly — very few trips beyond ~2–3 *L*. Compared with
gravity it has a **lighter long-range tail**, so it's a better fit when travel is
dominated by a definite commuting radius. Tune *L* to that radius.

### 3 — Radiation  (Simini et al. 2012)  — *parameter-free*

A mechanistic commuting model with **no decay knob to tune**. The flow from *i* to
*j* depends on the two populations **and** on `s_ij`, the population living closer
to *i* than *j* is — the "intervening opportunities". Intuitively, people are less
likely to travel to *j* when many equally-good closer destinations already sit
between them. Flows emerge from the population landscape itself, which often
matches real commuting better than gravity without any fitting; it can, however,
underestimate long, sparse flows. `travel_sigma` still sets the outbound mass.

### 4 — Uniform  (distance-agnostic baseline)

Each zone spreads its `travel_sigma` **equally across all other zones**, ignoring
distance and destination population entirely. It encodes no geography — useful as
a simple, assumption-light baseline or a sanity check, or when zones are small and
well-mixed. Every off-diagonal entry in a row is identical.

Example `Disease` block (mechanism 2 = exponential):

```json
"Disease": {
    "disease_type": "example_spatial_mobility",
    "travel_model": 2,
    "travel_sigma": 20.0,
    "travel_scale_km": 150.0,
    "travel_alpha": 2.0
}
```

`travel_scale_km` is only read for mechanism 2, and `travel_alpha` only for
mechanism 1; they are ignored by the others but should stay within their valid
ranges (`travel_scale_km ≥ 1`, `travel_alpha ≥ 0.1`).

## How the coupling enters the dynamics

Unlike the base SIR template — whose `equation()` uses
`_compute_equations`, which computes a **per-zone** force of infection and never
reads `self.travel_matrix` (so any mobility matrix would be a no-op) — this model
applies the matrix explicitly, as a metapopulation *presence* force of infection:

```
N_present_j = Σ_i N_i · T[i,j]        I_present_j = Σ_i I_i · T[i,j]
phi_j       = I_present_j / N_present_j          (prevalence experienced in zone j)
new_inf_i   = S_i · beta · Σ_j T[i,j] · phi_j
```

Residents of zone *i* are exposed to the prevalence of every zone they visit. With
`travel_sigma: 0` (`T = I`) this reduces exactly to the standard independent-zone
SIR.

## Nuances users should know

- **Mechanism is a number, not a string.** `travel_model` is an integer 1–4;
  values outside that range fail config validation.
- **`_total` compartments are auto-generated.** The framework adds cumulative
  `I_total` and `R_total` for transmission-edge targets — don't declare them.
- **Units are converted for you.** `gamma` is declared as `DAYS` and converted to
  a rate internally. `PERCENTAGE` params (e.g. `travel_sigma`) arrive as `20.0`,
  not `0.2` — convert with `self._to_rate(...)`; the model does this before
  building the matrix.
- **Parameter uncertainty is UNIFORM-only.** Set `has_variance: true` with a
  `UNIFORM` distribution and `min`/`max` in a field's `FieldConfigs`. The example
  varies `beta` and `gamma`; the results carry a `median` with `lower`/`upper`.
- **Multiple zones matter.** Mobility only has an effect with two or more admin
  zones. The example config uses five Madagascar zones and seeds infection only in
  Antananarivo — every other zone's outbreak is driven entirely by mobility (they
  stay at zero under `travel_sigma: 0`).
- **Demographics are declared but the dynamics are not age-stratified.** As in the
  base template, the age groups are declared for schema/UI parity; the SIR
  dynamics here run per zone (state `S/I/R` per zone). To engage age structure,
  expand the state with `self._prepare_demographic_state()` and add contact-matrix
  mixing to the force of infection (see `covid_jax_model` for the pattern).

## Running

```
python -m compartment.models.example_spatial_mobility_model.main \
    --mode local \
    --config_file compartment/models/example_spatial_mobility_model/example-config.json \
    --output_file results/example_spatial_mobility-test.json
```
