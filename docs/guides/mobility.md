# Mobility

Mobility describes how people move between administrative zones. It is important when calculating the force of infection for diseases spread through contact, because people may be exposed while visiting another zone.

The simulator represents this movement with a **travel matrix**, which records the fraction of each zone's residents present in every other zone. People do not permanently relocate, and each zone's population stays constant.

## The travel matrix

For a simulation with R zones, the travel matrix **T** is an R×R grid of fractions:

**T[i, j] = the fraction of zone i's population present in zone j on a given day.**

**Sigma (σ)** is the fraction of each zone's population that travels. For example, σ = 0.2 means 20% travel and 80% remain in their home zone.

- **Rows are origins, columns are destinations.** Row *i* describes where the residents of 
zone *i* go.
- **Every row sums to 1.** This accounts for the zone's entire population, no one is created or lost.
- **The diagonal is `1 - sigma`**, the fraction that stays home. The other entries in the row sum to sigma.
- **The matrix is not symmetric.** T[i, j] ≠ T[j, i]. For example, a village sends a large share of its population to the nearby city each day; the city sends a tiny share of its much larger population to the village.
- **Rows and columns follow the order of admin zones** in the case file and population matrix.

## When the matrix is built

**A travel matrix always exists, but it is not automatically a mobility model.**

This distinction matters, so it's worth being precise. Before every simulation, [`SimulationManager.run_simulation()`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/simulation_manager.py) calls `Model._ensure_travel_matrix()`, which calls your model's `build_travel_matrix()` and stores the result on `self.travel_matrix`:

```python
# compartment/simulation_manager.py
self.model._ensure_travel_matrix()
init_state = self.model.prepare_initial_state()
```

This happens for every model on every run, so `self.travel_matrix` is guaranteed to be present by the time `equation()` runs. But the base class implementation of `build_travel_matrix()` returns the identity matrix:

```python
# compartment/model.py — the default
def build_travel_matrix(self, admin_zones):
    return np.eye(len(admin_zones))
```

So a model that does nothing gets a matrix that has no effect. **There is no framework-level default gravity model, and nothing computes mobility for you.** Mobility is model-owned: a model that travels declares its own parameters and defines how they become a matrix.

Three further details are worth knowing:

- **It is built once per run, not once per timestep.** The matrix is fixed for the whole simulation unless an intervention changes it.
- **It is rebuilt from scratch for each uncertainty sample.** This is how a sampled `travel_sigma` actually takes effect in `UNCERTAINTY` runs — each Latin Hypercube draw reconstructs the model and rebuilds the matrix.
- **The ordering is load-bearing.** `_ensure_travel_matrix()` runs after `__init__` and before `prepare_initial_state()`. Assigning `self.travel_matrix` in either of those will be silently overwritten. Override `build_travel_matrix()` instead.

## Building the matrix is not enough — you must use it

This is the most common reason mobility appears to do nothing. The framework builds the matrix and hands it to you, but it has no idea what your force of infection looks like, so it cannot apply the matrix for you. **If `travel_matrix` does not appear in your `equation()`, your simulation has no mobility** regardless of what `travel_sigma` is set to.

The change is usually one line. Without mobility, susceptibles in a zone are exposed only to their own zone's prevalence:

```python
foi = beta * (I / N)
```

With mobility, they are exposed to the average prevalence across the zones they visit, weighted by how much time they spend in each:

```python
I_frac = I / N                            # infectious fraction per zone, shape (R,)
mixed_frac = self.travel_matrix @ I_frac  # travel-weighted prevalence, shape (R,)
foi = beta * mixed_frac
```

Written out, `foi[i] = beta * sum_j T[i, j] * I[j] / N[j]`. If zone *i* sends 5% of its residents to zone *j*, then 5% of zone *i*'s exposure comes from zone *j*'s prevalence.

The ebola model is the clearest example in the repository ([`ebola_jax_model/model.py`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/models/ebola_jax_model/model.py)):

```python
I_eff = (I1 + I2) + self.etu_risk * (H1 + H2) + self.funeral_risk * Funeral
I_frac = I_eff / N                         # shape (R,)
mixed_frac = self.travel_matrix @ I_frac   # shape (R,)
current_rate = beta * mixed_frac
```

Mpox and hantavirus_human do the same thing with `jnp.einsum("ij,j->i", self.travel_matrix, I_over_N)`, which is equivalent to the `@` form and just makes the index bookkeeping explicit.

### Combining mobility with age structure

In an age-stratified model the state is shaped `(R, A)` — zones by age groups — and there are two separate mixing structures to apply: the travel matrix mixes across zones, and the contact matrix mixes across age groups. They are applied one after the other. From [`covid_jax_model/model.py`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/models/covid_jax_model/model.py):

```python
rates, travel_matrix = self._apply_interventions(t, rates, prop_infective_scalar)

BETA = ((rates["beta"] * travel_matrix) @ I_frac.T).T   # mix across zones  -> (R, A)
omega = self.contact_matrix @ BETA                      # mix across ages   -> (R, A)
flow_foi = S * omega
```

The `.T` transposes exist only because the travel matrix needs to multiply the zone axis while the contact matrix needs the age axis. See [contact-matrices.md](./contact-matrices.md) for the age side.

Note that covid uses the travel matrix **returned by `_apply_interventions()`**, not `self.travel_matrix` directly. That matters when interventions can restrict travel — see below.

## Adding mobility to a model

Two steps.

### 1. Declare the parameters

Mobility parameters are ordinary disease parameters, declared in `define_parameters()` alongside everything else. There is no separate mobility configuration block. **Any parameter your kernel needs goes here** — this is what makes it configurable and visible to users.

```python
@classmethod
def define_parameters(cls, schema):
    ...
    schema.add_parameter(
        name="travel_sigma",
        label="Travel Rate (σ)",
        description="Percentage of each zone's population away from home on a given day.",
        value_type=ValueType.PERCENTAGE,
        default=20.0,
        min_value=0.0,
        max_value=100.0,
        unit="%",
    )
```

Declaring it this way means it flows through the whole pipeline for free: it appears in the model artifact's `custom_fields`, renders as an editable field on the simulation's Disease step, shows up in the results' Custom Parameters section, and participates in uncertainty sampling. Only parameters a model actually declares are accepted — the generated config validation rejects anything else.

Two conventions to follow:

- **Name it `travel_sigma`, never plain `sigma`.** During uncertainty runs, `build_overridden_config()` routes transmission-edge names to the transmission dictionary and everything else to `Disease`. A mobility parameter whose name collides with an edge is misrouted silently. Ebola's `sigma` is already its E→I incubation rate, which is why its mobility parameter carries the prefix.
- **Prefix any extras the same way**, e.g. `travel_alpha`, `travel_scale_km`.

### 2. Override `build_travel_matrix()`

```python
def build_travel_matrix(self, admin_zones):
    sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
    return get_gravity_model_travel_matrix(admin_zones, sigma)
```

`admin_zones` arrives in population-matrix column order, each zone carrying `center_lat`, `center_lon` and `population`. Every parameter you declared is already an attribute on `self` — `Model.__init__` sets one per declared disease parameter, so `self.travel_sigma` is available with no extra plumbing.

One easy mistake: **`PERCENTAGE` parameters arrive in native units**, so `self.travel_sigma` is `20.0`, not `0.2`. Only transmission edges are auto-converted. Convert with `self._to_rate(..., ValueType.PERCENTAGE)` before handing the value to a kernel, all of which expect a 0–1 fraction. Skipping this gives you a sigma of 20, and a matrix with a diagonal of −19.

Then use `self.travel_matrix` in `equation()` as shown above. Nothing happens until you do.

## Kernels available in the repository

A "kernel" is just the rule that decides how the travelling population splits across destinations. All three below take population and distance and differ only in how sharply attraction falls off with distance. Pick the one that matches the kind of travel your disease spreads through, or write your own.

| Kernel | Attraction from i to j | Where | Used by |
|---|---|---|---|
| Inverse-square gravity | `pop_i * pop_j / d²` | `get_gravity_model_travel_matrix()` in [`compartment/helpers.py`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/helpers.py) | covid, dengue, ebola |
| Power-law gravity, exponent configurable | `pop_j / d^α` | `gravity()` in `hantavirus_human_jax_model/model.py` | hantavirus_human (`travel_alpha`, default 1.5) |
| Exponential decay | `pop_j * exp(-d / scale_km)` | `mobility()` in `mpox_jax_model/model.py` | mpox (`travel_scale_km`, default 500 km) |

The shared helper in `helpers.py` is one option, not a default. It is the classic gravity model, named after Newton's law of gravitation because attraction grows with the two "masses" (populations) and falls with the square of the distance. It uses geodesic distances via `geopy`; the two per-model kernels use the Haversine great-circle formula, which is faster and accurate enough at these scales.

Choosing between them:

- **Power-law** decays slowly, so distant large cities still attract meaningful flow. Good for air travel and long-distance movement. A larger α makes it more local: 1.0 is quite flat, 2.0 clearly favours neighbours.
- **Exponential** decays much faster and effectively imposes a horizon around `scale_km`. Good for commuting and ground transport, where travel beyond a few hours' drive is genuinely rare.

Two useful facts. First, because rows are normalised, the origin population cancels out — `pop_i * pop_j / d²` and `pop_j / d²` produce **identical** matrices. The origin term only scales absolute flow volume, which sigma already sets. Second, `α = 2` in the power-law kernel is the inverse-square kernel, so the first two rows of that table are the same model with a fixed versus configurable exponent.

### Writing your own

Expected, not exotic — reach for it whenever your disease has behavioural assumptions the generic kernels don't capture, such as seasonal labour migration or pilgrimage. Declare the parameters you need, then follow the same five steps every kernel in the repo uses. From mpox's `mobility()`:

```python
# 1. Edge cases first: one zone, or nobody travelling, means no mobility.
if n <= 1 or sigma == 0.0:
    return onp.eye(n)

# 2. Pairwise great-circle distances in km. `a` is the Haversine intermediate
#    term, built from the pairwise lat/lon differences; clip it to [0, 1] so
#    floating-point drift can't push arcsin out of its domain.
dist_km = 2 * 6371.0 * onp.arcsin(onp.sqrt(onp.clip(a, 0.0, 1.0)))

# 3. Attraction, with no self-flow. Clamp small distances so coincident
#    zone centroids don't divide by zero.
attraction = pops[None, :] * onp.exp(-dist_km / scale_km)
onp.fill_diagonal(attraction, 0.0)

# 4. Normalise each row so the destinations share out one unit of travel.
row_sums = attraction.sum(axis=1, keepdims=True)
row_sums = onp.where(row_sums == 0.0, 1.0, row_sums)   # guard empty rows
T = attraction / row_sums

# 5. Scale to sigma and put the stay-at-home remainder on the diagonal.
travel_matrix = sigma * T
onp.fill_diagonal(travel_matrix, 1.0 - sigma)
```

Steps 4 and 5 are what produce the invariants above. Build the matrix with plain NumPy, not `jax.numpy`. It is constructed once, outside the solver, so JAX buys you nothing — and in-place operations like `fill_diagonal` don't work on JAX arrays anyway. The framework converts the result for you.

## A model with no mobility

Do nothing. Declare no travel parameters and don't override `build_travel_matrix()`; the base class supplies the identity matrix and the zones evolve independently. This is the right choice whenever the zones genuinely don't interact, or when inter-zone spread isn't what the model is about.

Say so explicitly, though, because silence looks like an oversight. [`hantavirus_jax_model`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/models/hantavirus_jax_model/model.py) is the reference — it models rodent populations that don't migrate between zones, and records the decision in `__init__`:

```python
# No region-to-region travel — each admin zone is an independent
# territory containing all three sectors internally, so this model
# declares no travel parameters and inherits the base class's identity
# build_travel_matrix(). Movement between the u/a/f sectors *within* a
# zone is modelled explicitly as ODE flows in equation().
```

That last sentence is the useful distinction: this model has plenty of movement, just not movement *between zones*. Within-zone structure is ordinary compartment flow and has nothing to do with the travel matrix.

You can also switch mobility off in an existing travel model without touching code, by setting `travel_sigma` to 0 in the config. Every kernel returns the identity matrix for a zero sigma. That's the first thing to try when you want to isolate the effect of spatial spread.

## Interventions that restrict travel

An intervention declared with `modifies_travel=True` replaces the travel matrix with the identity matrix while it is active — a full stop on inter-zone movement. Covid's lockdown is the only one in the repository:

```python
schema.add_intervention(
    id="lock_down",
    label="Lockdown",
    description="Severe movement restrictions: halts inter-regional travel and ...",
    target_rates=["beta"],
    modifies_travel=True,
    adherence=80.0,
    transmission_reduction=70.0,
)
```

This is the one case where the matrix changes during a run, and it's why `_apply_interventions()` returns a travel matrix as its second value:

```python
rates, travel_matrix = self._apply_interventions(t, rates, prop_infective)
foi = beta * (travel_matrix @ I_frac)   # use the returned matrix, not self.travel_matrix
```

If you use `self.travel_matrix` in the force of infection instead of the returned value, travel-restricting interventions will have no effect on travel. Note also that the effect is all-or-nothing: there is currently no way to declare a partial travel reduction.

## Configuration

Mobility parameters live inside the `Disease` block, like every other model parameter. The zone coordinates and populations that feed the kernel come from `case_file.admin_zones`:

```json
{
  "admin_unit_id": "USA",
  "Disease": {
    "disease_type": "COVID_SEIHDR",
    "travel_sigma": 20.0
  },
  "case_file": {
    "admin_zones": [
      { "name": "New York",    "center_lat": 40.7128, "center_lon": -74.0060,  "population": 8336817 },
      { "name": "Los Angeles", "center_lat": 34.0522, "center_lon": -118.2437, "population": 3979576 },
      { "name": "Chicago",     "center_lat": 41.8781, "center_lon": -87.6298,  "population": 2693976 }
    ]
  }
}
```

`travel_sigma` is a percentage, so 20% travel is `20.0`, not `0.2`. Models that need more declare it alongside — mpox adds `travel_scale_km`, hantavirus_human adds `travel_alpha`. Each model's `example-config.json` shows its own set.

Mobility needs no dataset. Every kernel reads only the zone coordinates and populations already in the case file.

## Which models use mobility today

| Model | Mobility |
|---|---|
| covid | Inverse-square gravity, travel + age contact mixing, lockdown restricts travel |
| ebola | Inverse-square gravity, single-line spatial force of infection — **the clearest starting point** |
| mpox | Exponential decay, minimal SIR — clearest example of a custom kernel |
| hantavirus_human | Power-law gravity with configurable α; person-to-person spread mixes across zones, rodent spillover stays local |
| dengue | Inverse-square gravity via the legacy intervention path; both the human and vector force of infection are spatially mixed |

Worth knowing: **none of the `example_*` models use mobility.** They ship with the travel block commented out (see `example_stochastic_model/model.py`), as does the `new_model` template. Those comments are a useful skeleton to uncomment, but for working code read ebola or mpox.

Also note that `ebola_seihfr_burial_legrand_model` calls `_apply_interventions()` but keeps its force of infection local, so it has a travel matrix it never uses. Holding the matrix is not the same as using it.

## Checking your matrix

Nothing validates the matrix while a simulation runs. A malformed one produces quietly wrong output rather than an error, which is why the invariants are worth checking deliberately. The three failures that produce plausible-looking but wrong results:

- **Rows that don't sum to 1** — the force of infection gains or loses population every step.
- **Row or column order mismatched to the admin zones** — infection is routed into the wrong zones.
- **Swapped latitude and longitude** — no error, just distances that make no sense and an odd travel pattern.

[`tests/test_travel_matrix.py`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tests/test_travel_matrix.py) enforces these for every registered model, and a new kernel is covered automatically as soon as the model declares `travel_sigma`. It checks shape, finiteness, non-negativity, row sums, the `1 - sigma` diagonal, and row/column ordering, and it asserts that each travel model's example config has a non-zero sigma so the smoke tests actually exercise mixing.

To inspect a matrix by hand:

```python
print("Row sums:  ", travel_matrix.sum(axis=1))             # want all 1.0
print("Diagonal:  ", np.diag(travel_matrix))                # want 1 - sigma
print("Travelling:", travel_matrix.sum(axis=1) - np.diag(travel_matrix))  # want sigma
```

## Troubleshooting

**Results are identical with and without travel.** In order of likelihood: `travel_matrix` never appears in `equation()`; `travel_sigma` is 0; only one admin zone is configured; or every zone was seeded with the same prevalence, which leaves nothing for mixing to move. Seed one zone only and watch the others.

**All zones send their travellers to one dominant zone.** Expected behaviour for a gravity model when one zone is far larger — attraction scales with destination population. If it's too strong, increase the decay (a higher α, or a smaller `scale_km`) so distance matters more, or switch to the exponential kernel for a sharper cutoff.

**Nobody appears to stay home.** Check that sigma was converted from a percentage. An unconverted 20.0 produces a diagonal of −19.

**Infection appears in the wrong zones.** Almost always row/column ordering. `admin_zones` order is the matrix order; if you pivot or group by zone id anywhere in your kernel, pass an explicit ordering, because pandas sorts its labels and real zone ids are UUIDs.

## Related documentation

- [contact-matrices.md](./contact-matrices.md) — age-group mixing, the demographic counterpart to spatial mixing
- [interventions.md](./interventions.md) — how interventions are declared and applied, including travel restrictions
- [developing-models.md](./developing-models.md) — building a model end to end
- [`compartment/model.py`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/model.py) — the `build_travel_matrix()` hook and `_ensure_travel_matrix()`
- [`compartment/helpers.py`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/helpers.py) — the shared gravity kernel

## References

- **Zipf, G.K. (1946).** "The P1 P2/D Hypothesis: On the Intercity Movement of Persons." *American Sociological Review* 11(6): 677-686. The original gravity model for human movement.
- **Sattenspiel, L., & Dietz, K. (1995).** "A structured epidemic model incorporating geographic mobility among regions." *Mathematical Biosciences* 128(1-2): 71-91. The presence-based coupling used here.
- **Balcan et al. (2009).** "Multiscale mobility networks and the spatial spreading of infectious diseases." *PNAS* 106(51): 21484-21489. Validates gravity models for epidemic spread against global air travel data.
- **Viboud et al. (2006).** "Synchrony, waves, and spatial hierarchies in the spread of influenza." *Science* 312(5772): 447-451. Empirical support for α ≈ 2 in seasonal influenza.
- **Kraemer et al. (2020).** "The effect of human mobility and control measures on the COVID-19 epidemic in China." *Science* 368(6490): 493-497.

Where empirical mobility data is available (census commuting flows, mobile-phone or platform mobility data, flight bookings), it is worth fitting the decay shape rather than assuming one. Typical starting points: daily commuting α 1.5–2.0 or `scale_km` 30–100 with σ 5–15%; weekly travel α 1.0–1.5 or `scale_km` 200–500 with σ 10–30%.
