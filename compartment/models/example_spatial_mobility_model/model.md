# Example Disease (Selectable Spatial Mobility) — Model Notes

A minimal **SIR** model that demonstrates **inter-zone mobility**: the user picks
the mobility mechanism with a single `travel_model` parameter, and the model
synthesises the origin-destination matrix from each admin zone's **population and
coordinates** — no movement data required. It also keeps the declarative-style
parameter uncertainty (median + interval over many parameter draws).

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

## Choosing the mobility mechanism

Set `travel_model` in the config's `Disease` section. No movement data is needed —
every mechanism is built from population + coordinates (great-circle distances):

| `travel_model` | Weight of flow i → j | Extra knob |
|---|---|---|
| `gravity` | `pop_j / d_ij^alpha` | `travel_alpha` (default 2.0 = inverse-square) |
| `exp` | `pop_j · exp(-d_ij / scale)` | `travel_scale_km` (default 150) |
| `radiation` | Simini et al. 2012 (parameter-free) | — |
| `uniform` | equal to all other zones | — |
| `none` | identity — no inter-zone travel | — |

`travel_sigma` (percent, default 20) sets the outbound mass per day: each row's
off-diagonal sums to `sigma` and the diagonal (stay-home) is `1 − sigma`. The
mechanism only decides how that `sigma` is distributed across destinations. All
mechanisms return a **row-stochastic** matrix, so each zone's population is
conserved.

Example `Disease` block:

```json
"Disease": {
    "disease_type": "example_spatial_mobility",
    "travel_model": "exp",
    "travel_sigma": 20.0,
    "travel_scale_km": 150.0,
    "travel_alpha": 2.0
}
```

## How the coupling enters the dynamics

Unlike the base declarative template — whose `equation()` uses
`_compute_equations`, which computes a **per-zone** force of infection and never
reads `self.travel_matrix` (so any mobility matrix would be a no-op) — this model
applies the matrix explicitly, as a metapopulation *presence* force of infection:

```
N_present_j = Σ_i N_i · T[i,j]        I_present_j = Σ_i I_i · T[i,j]
phi_j       = I_present_j / N_present_j          (prevalence experienced in zone j)
new_inf_i   = S_i · beta · Σ_j T[i,j] · phi_j
```

Residents of zone *i* are exposed to the prevalence of every zone they visit.
With `travel_model: "none"` (`T = I`) this reduces exactly to the standard
independent-zone SIR.

## Nuances users should know

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
  zones. The example config uses two Madagascar zones and seeds infection only in
  Antananarivo — Toamasina's outbreak is driven entirely by mobility (it stays at
  zero under `travel_model: "none"`).
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
