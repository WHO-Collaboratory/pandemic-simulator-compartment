# Gravity Model for Spatial Mobility in the Pandemic Simulator

This document explains how the gravity model is used to create spatial mobility matrices for multi-region compartmental disease models in the Pandemic Simulator.

## Overview

The gravity model is a spatial interaction model that estimates movement flows between geographic regions based on:

1. Population sizes (the "mass" of each region)
2. Geographic distance between regions
3. A distance-decay parameter (how quickly travel drops off with distance)

The model produces a **travel matrix** that describes what fraction of each region's population is present in every other region during a given timestep. This enables models to capture spatial disease spread through human mobility.

## The Physics Analogy

The name "gravity model" comes from Newton's law of universal gravitation:

```
F = G * (m₁ * m₂) / d²
```

In the mobility context, the "gravitational attraction" between two regions is proportional to their populations and inversely proportional to distance:

```
Flow(i → j) ∝ (population_i * population_j) / distance_ij^α
```

Where:

- **population_i, population_j** = populations of origin and destination regions
- **distance_ij** = geographic distance between regions (great-circle distance in km)
- **α** (alpha) = distance-decay exponent (controls how quickly travel drops off)

## Mobility is model-owned

There is **no framework-level default mobility model**. Nothing is built for you automatically: whatever mobility a model has, that model declares the parameters for it and defines how they become a matrix. Models with no inter-zone travel declare nothing and inherit an identity matrix from the base class.

This means a modeler is never fighting a hidden default, and σ shows up in the UI as an ordinary editable field for whichever kernel the model actually uses.

### The two pieces you write

**1. Declare the parameters as custom fields** in `define_parameters()`. Convention is `travel_sigma` as a `PERCENTAGE` in 0–100, plus whatever else your kernel needs:

```python
schema.add_disease_parameter(
    name="travel_sigma",
    label="Travel Rate (σ)",
    description="Percentage of each zone's population away from home on a given day.",
    value_type=ValueType.PERCENTAGE,
    default=20.0, min_value=0.0, max_value=100.0, unit="%",
)
```

Because it's an ordinary disease parameter it flows through the whole existing pipeline for free: it lands in `custom_fields` in the model artifact, renders as an editable field (with a variance range) on the simulation's Disease step, appears in the results' Custom Parameters section, and participates in Latin-Hypercube uncertainty sampling.

> **Never name it plain `sigma`.** `build_overridden_config()` routes transmission-edge `variable_name`s to the transmission dict and everything else to `Disease`. A mobility parameter that collides with an edge name — ebola's `sigma` is its E→I incubation rate — silently misroutes during uncertainty runs.

**2. Override `build_travel_matrix()`** on the model:

```python
def build_travel_matrix(self, admin_zones):
    # PERCENTAGE disease params arrive in native units (20.0, not 0.2).
    # Only transmission edges are auto-converted.
    sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
    return get_gravity_model_travel_matrix(admin_zones, sigma)
```

The framework calls this via `Model._ensure_travel_matrix()` immediately **before** `prepare_initial_state()`, and assigns the result to `self.travel_matrix`. That ordering is load-bearing: `_apply_interventions()` reads `self.travel_matrix` from inside `equation()`. Don't assign `self.travel_matrix` yourself — it will be overwritten.

`admin_zones` arrives in population-matrix column order, each zone carrying `center_lat`, `center_lon`, and `population`.

### Available kernels

Pick the decay shape that fits your disease; these are the three in the repo today.

| Kernel | Where | Attraction from i to j |
|---|---|---|
| Inverse-square gravity (geopy geodesic) | `compartment/helpers.py` `get_gravity_model_travel_matrix()` — used by covid, dengue, ebola | `pop_i * pop_j / d²` |
| Power-law gravity, α configurable (Haversine) | `hantavirus_human_jax_model/model.py` `gravity()` | `pop_j / d^α` |
| Exponential distance decay | `mpox_jax_model/model.py` `mobility()` | `pop_j * exp(-d / scale_km)` |

The shared helper is one option, not a default. Writing your own is expected when none of these fit:

```python
def gravity_model(df, mass_origin_col, mass_dest_col, distance_col, k=1):
    """k * pop_origin * pop_dest / distance²"""
    df["gravity"] = k * df[mass_origin_col] * df[mass_dest_col] / df[distance_col] ** 2
    return df
```

Note that with row normalisation the origin mass cancels out, so `pop_i * pop_j / d²` and `pop_j / d²` produce the *same* matrix — the origin term only changes the absolute flow scale, which σ sets anyway.

## How Travel Matrices Are Built

Every kernel in the repo follows this pattern, and yours should too — steps 3-5 are what produce the invariants the tests enforce:

### Step 1: Calculate Pairwise Distances

Use the **Haversine formula** to compute great-circle distances from latitude/longitude coordinates:

```python
# Haversine formula (vectorized)
R_earth = 6371.0  # Earth radius in km
lat_r = np.radians(lats)
lon_r = np.radians(lons)
dlat = lat_r[:, None] - lat_r[None, :]
dlon = lon_r[:, None] - lon_r[None, :]

a = (np.sin(dlat / 2) ** 2 +
     np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2)
     
distance_km = 2 * R_earth * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
```

### Step 2: Compute Attraction Matrix

Apply the gravity model formula to get relative attraction from each origin to each destination:

```python
# Classic gravity (α = 2)
attraction = (pops[None, :] * pops[:, None]) / (distance_km ** 2)

# OR exponential decay
attraction = pops[None, :] * np.exp(-distance_km / scale_km)

# OR power-law with custom α
attraction = pops[None, :] / (distance_km ** alpha)
```

Key operations:

- Set diagonal to zero (no "self-flow" in the attraction calculation)
- Clamp very small distances (< 1 km) to avoid division by zero

### Step 3: Normalize Rows

Convert raw attractions to **flow fractions** — each row must sum to 1.0:

```python
# Replace zero row-sums to avoid division by zero
row_sums = attraction.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0.0, 1.0, row_sums)

# Normalize: each row sums to 1
normalized_flow = attraction / row_sums
```

After normalization:

- `normalized_flow[i, j]` = fraction of zone i's **outbound travelers** going to zone j
- Each row sums to 1.0 (all travelers go somewhere)

### Step 4: Apply Sigma (Leaving Fraction)

**Sigma (σ)** controls the overall mobility rate:

- **σ = 0.0** → nobody travels (identity matrix)
- **σ = 0.2** → 20% of each region's population travels elsewhere
- **σ = 1.0** → everyone leaves their home region

```python
travel_matrix = sigma * normalized_flow
np.fill_diagonal(travel_matrix, 1.0 - sigma)
```

**Final matrix semantics:**

- `travel_matrix[i, j]` = fraction of region i's population **present** in region j
- Row i sums to 1.0 (the entire population of region i is accounted for)
- Diagonal `travel_matrix[i, i] = 1 - sigma` = fraction staying home

### Step 5: Handle Edge Cases

```python
# Single region: no travel needed
if n_regions == 1:
    return np.array([[1.0]])

# Zero mobility: identity matrix
if sigma == 0.0:
    return np.eye(n_regions)
```

## Travel Matrix Semantics

The resulting travel matrix **T** is an R×R array where R = number of regions.

### Interpretation

**T[i, j]** = fraction of region *i*'s population that is **present** in region *j* during the timestep

### Properties

1. **Row sums = 1.0** — Each region's entire population is accounted for:
   ```
   sum_j T[i, j] = 1.0  for all i
   ```

2. **Diagonal = (1 - σ)** — Fraction staying home:
   ```
   T[i, i] = 1 - sigma
   ```

3. **Off-diagonal sum = σ per row** — Total fraction traveling:
   ```
   sum_{j ≠ i} T[i, j] = sigma  for all i
   ```

4. **Generally asymmetric** — Flow from i→j ≠ flow from j→i:
   ```
   T[i, j] ≠ T[j, i]  in general
   ```
   This reflects real-world mobility: small rural regions may send many travelers to large cities, but receive few in return.

### Example: 3-Region System

Consider three regions with populations:

- **Region A:** 1,000,000 (major city)
- **Region B:** 100,000 (town)
- **Region C:** 10,000 (village)

With σ = 0.2 (20% travel rate), a typical travel matrix might be:

```
       A      B      C
A   [0.80   0.12   0.08]   (A → mostly stays, some to B, less to C)
B   [0.60   0.80   0.10]   (B → many to A, most stay, few to C)
C   [0.70   0.25   0.80]   (C → many to A, some to B, most stay)
```

**Reading the matrix:**

- Row A: 80% stay in A, 12% present in B, 8% present in C
- Row B: 60% go to A (!), 30% stay in B, 10% go to C
    - Note: 60% + 30% + 10% = 100% ✓
    - Large fraction goes to city A despite distance
- Row C: 70% go to A, 25% go to B, 5% stay in C
    - Small villages send most people to nearby town/city

## Using Travel Matrices in Disease Models

Travel matrices enable spatial mixing of infections across regions. The force of infection must account for both **where people are infected** and **where they bring infections**.

### Pattern 1: Direct Spatial Mixing (SIR/SEIR without demographics)

In a simple SIR model with R regions, the travel matrix directly scales the force of infection:

```python
# Compute infectious fraction by region
I_frac = I / (N + 1e-9)  # Shape: (R,)

# FOI accounts for spatial travel mixing
# Someone from region i gets infected by the weighted-average prevalence
# across all regions j, weighted by how much time they spend in j
foi = beta * (travel_matrix @ I_frac)  # Shape: (R,)

# New infections
new_infections = foi * S
```

**What's happening mathematically:**
```
foi[i] = beta * sum_j T[i,j] * (I[j] / N[j])
```

- Susceptibles in region i are exposed to infections in region j proportionally to `T[i,j]`
- If region i sends 20% of its population to region j, those travelers face region j's infection risk

### Pattern 2: Spatial + Demographic Mixing (Age-Structured Models)

For age-stratified models (e.g., COVID-19 with age groups), **both** travel and contact matrices contribute to mixing:

```python
# State dimensions: (R regions, A age groups)
S.shape = (R, A)
I.shape = (R, A)
N.shape = (R, A)

# Compute age-specific infectious fraction by region
I_frac = I / (N + 1e-9)  # Shape: (R, A)

# Step 1: Apply travel matrix (spatial mixing)
# For each age group, account for where people travel
BETA = ((beta * travel_matrix) @ I_frac.T).T  # Shape: (R, A)

# Step 2: Apply contact matrix (age mixing)
# For each region, account for age-specific contact patterns
omega = contact_matrix @ BETA  # Shape: (R, A)

# Force of infection combines both spatial and demographic structure
foi = S * omega  # Shape: (R, A)
```

**Step-by-step breakdown:**

1. **`travel_matrix @ I_frac.T`** (shape R×R × R×A^T = R×A):
   - For each region i and age group a, computes the spatially-mixed prevalence
   - Accounts for infections encountered during travel to other regions

2. **`contact_matrix @ BETA`** (shape A×A × A×R^T = A×R, then transposed):
   - For each age group a and region r, applies age-specific contact patterns
   - Accounts for differential mixing between age groups (children contact children more than elderly)

3. **`S * omega`**:
   - Susceptibles in each (region, age) cell get infected at rate determined by both spatial and demographic structure

### Pattern 3: A Bespoke Kernel

When none of the existing kernels fit, declare the parameters your formula needs and implement it in `build_travel_matrix()`. Everything the model needs is already on `self` — `Model.__init__` sets an attribute per declared disease parameter:

```python
class MyDiseaseModel(Model):
    @classmethod
    def define_parameters(cls, schema):
        schema.add_disease_parameter(
            name="travel_sigma", label="Travel Rate (σ)",
            description="Percentage of each zone's population away from home on a given day.",
            value_type=ValueType.PERCENTAGE, default=20.0,
            min_value=0.0, max_value=100.0, unit="%",
        )
        schema.add_disease_parameter(
            name="travel_scale_km", label="Travel Decay Length",
            description="Characteristic distance of the exponential mobility decay.",
            value_type=ValueType.FLOAT, default=500.0,
            min_value=1.0, max_value=20000.0, unit="km",
        )

    def build_travel_matrix(self, admin_zones):
        sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
        return self.mobility(admin_zones, sigma, scale_km=self.travel_scale_km)

    def mobility(self, admin_zones, sigma, scale_km=500.0):
        """Exponential distance-decay model."""
        # ... implement the formula, honouring the invariants above ...
        return travel_matrix
```

**When to use:**

- Model-specific mobility assumptions (e.g., commuter vs. leisure travel)
- Disease-specific behavior changes (e.g., reduced travel during outbreak)
- Non-standard distance-decay functions

See `mpox_jax_model` (exponential decay) and `hantavirus_human_jax_model` (power law with a configurable α) for worked examples.

## Configuration

### In the Config JSON

Mobility parameters live inside the `Disease` block, like every other model parameter:

```json
{
  "admin_unit_id": "USA",
  "Disease": {
    "disease_type": "COVID_SEIHDR",
    "travel_sigma": 20.0
  },
  "case_file": {
    "admin_zones": [
      {
        "name": "New York",
        "center_lat": 40.7128,
        "center_lon": -74.0060,
        "population": 8336817
      },
      {
        "name": "Los Angeles", 
        "center_lat": 34.0522,
        "center_lon": -118.2437,
        "population": 3979576
      },
      {
        "name": "Chicago",
        "center_lat": 41.8781,
        "center_lon": -87.6298,
        "population": 2693976
      }
    ]
  }
}
```

**Parameters:**

- **`Disease.travel_sigma`** (float, 0-100): Percentage of each region's population away from home per timestep
    - `0` = no travel (identity matrix)
    - `10` = 10% of people travel (typical for daily commuting)
    - `20` = 20% travel, the default most models ship with
    - It's a `PERCENTAGE`, so the value is `20.0`, **not** `0.2`

Models that need more than σ declare it alongside: mpox adds `travel_scale_km`, hantavirus_human adds `travel_alpha`. Only the parameters a model actually declares are accepted — the auto-generated Pydantic config rejects the rest.

## Validation and Edge Cases

### Handled by the kernels in this repo

1. **Single region** → 1×1 identity matrix `[[1.0]]`
2. **Zero sigma** → R×R identity matrix (no travel)
3. **Model declares no mobility** → identity matrix from the base class's `build_travel_matrix()`
4. **Coincident or very close zones** → distance is clamped (Haversine kernels) or the degenerate pair is dropped and the population stays home (geopy kernel). Either way rows still sum to 1 — no population is lost.

`tests/test_travel_matrix.py` asserts all four for every registered model, plus the row-sum, diagonal, and ordering invariants. If you write a new kernel, it's covered automatically as soon as the model declares `travel_sigma`.

### What is *not* checked at runtime

Nothing validates your matrix while a simulation runs — a malformed one produces quietly wrong output rather than an error. In particular there are no runtime warnings for:

- **Negative or implausible distances** (a swapped lat/lon shows up as an odd travel pattern, not a failure)
- **Row sums ≠ 1.0** — the FOI just silently gains or loses population each step
- **Row/column order mismatched to the population matrix** — infections get routed into the wrong zones

Cover these in tests instead. `tests/test_travel_matrix.py` has an `assert_valid_travel_matrix()` helper and an ordering test you can extend.

### Numerical Stability

Key numerical considerations:


```python
# Avoid division by zero
distance_clamped = np.where(distance_km < 1.0, 1.0, distance_km)

# Avoid NaN from zero row-sums
row_sums = np.where(row_sums == 0.0, 1.0, row_sums)

# Replace inf from zero distance
df["gravity"] = df["gravity"].replace(np.inf, 0.0)

# Clip Haversine intermediate value to valid range
a = np.clip(a, 0.0, 1.0)
```

## Best Practices

### ✅ Do

- **Use realistic σ values** — typical daily mobility is 5-20%
- **Validate coordinates** — ensure lat/lon are in correct order and valid ranges
- **Test with small σ first** — easier to debug spatial mixing with limited travel
- **Check matrix properties** — row sums = 1.0, diagonal ≈ (1 - σ)
- **Write your own kernel** when you have disease-specific behavioral assumptions — that's the expected path, not an escape hatch
- **Name the parameter `travel_sigma`**, never plain `sigma` — a collision with a transmission-edge name misroutes it during uncertainty runs
- **Document your distance-decay choice** — different exponents have different interpretations

### ❌ Don't

- **Don't use σ > 0.5 without justification** — implies more than half the population travels daily
- **Don't assume symmetry** — flow from A→B ≠ flow from B→A in real systems
- **Don't ignore edge cases** — always handle single-region, zero-travel, and coincident-coordinate scenarios
- **Don't set `self.travel_matrix` directly** — override `build_travel_matrix()`; the framework overwrites direct assignments
- **Don't mix up lat/lon order** — standard is (latitude, longitude), but some systems reverse this
- **Don't forget to normalize** — raw gravity values must be converted to fractions

## Troubleshooting

### Unrealistic Travel Patterns

**Symptoms:**

- All regions send travelers to one dominant region
- Small regions have near-zero diagonal (nobody stays home)
- Travel matrix is nearly uniform

**Possible causes:**

1. **σ too high** — Try reducing `leaving` to 0.1-0.2
2. **One region much larger than others** — Gravity models naturally concentrate flow toward large cities
3. **Distance-decay too weak** — Try increasing α (use α = 1.5 or 2.0 instead of 1.0)

**Solutions:**

- Use exponential decay instead of power-law for more gradual falloff
- Cap maximum travel distance (set attraction = 0 beyond threshold)
- Normalize by region size (divide by destination population)

### Matrix Not Row-Normalized

**Symptoms:**

- Runtime errors: "row sums must equal 1.0"
- Numerical instability in ODE solver
- Population not conserved

**Causes:**

- Skipped normalization step
- NaN or inf values in distance calculation
- Incorrect diagonal setting

**Debug:**
```python
print("Row sums:", travel_matrix.sum(axis=1))
print("Diagonal:", np.diag(travel_matrix))
print("Off-diag sum:", travel_matrix.sum(axis=1) - np.diag(travel_matrix))
```

**Fix:**
```python
# Ensure normalization
T = sigma * (attraction / row_sums)
np.fill_diagonal(T, 1.0 - sigma)

# Verify
assert np.allclose(T.sum(axis=1), 1.0), "Rows must sum to 1"
```

### Zero Distance Between Regions

**Symptoms:**

- Inf or NaN in travel matrix
- Crash during matrix construction

**Cause:**

- Duplicate coordinates in admin_zones
- Same region listed twice

**Fix:**
```python
# Clamp minimum distance
dist_km = np.maximum(dist_km, 1.0)

# Or use conditional
dist_clamped = np.where(dist_km < 1.0, 1.0, dist_km)
```

### Spatial Mixing Not Affecting Results

**Symptoms:**

- Results identical with/without travel
- Infections don't spread between regions

**Possible causes:**

1. **Travel matrix not used in equation** — Check that `travel_matrix` appears in FOI calculation
2. **σ = 0** — No travel configured
3. **All regions infected identically** — Initial conditions mask spatial effects

**Debug:**
```python
# In equation(), log the travel-mixed prevalence
print(f"t={t:.1f} | Travel-mixed I_frac: {(travel_matrix @ I_frac)}")
print(f"t={t:.1f} | Raw I_frac: {I_frac}")
# Should see differences when infections are spatially heterogeneous
```

## Empirical Validation

The choice of distance-decay function and exponent should be validated against empirical mobility data when possible:

### Data Sources for Validation

- **SafeGraph mobility data** (commercial)
- **Facebook Data for Good** (research access)
- **Google/Apple mobility reports** (COVID-19 era)
- **Census commuting flows** (US: LODES, EU: EUROSTAT)
- **Flight/train booking data** (long-distance travel)

### Validation Metrics

Compare model predictions to observed mobility:

1. **Distance decay curve** — Plot flow vs. distance, fit power-law or exponential
2. **Total flow volumes** — Sum of all inter-regional flows
3. **Destination ranking** — For each origin, rank destinations by flow volume
4. **Symmetry ratio** — Compare flow(i→j) vs. flow(j→i)

### Typical Parameter Ranges (Empirical)

| Context | α (power-law) | scale_km (exponential) | σ (leaving) |
|---------|---------------|------------------------|-------------|
| Daily commuting | 1.5-2.0 | 30-100 km | 0.05-0.15 |
| Weekly travel | 1.0-1.5 | 200-500 km | 0.1-0.3 |
| Disease-driven (reduced) | 2.0-3.0 | 50-200 km | 0.01-0.1 |
| Historical (pre-modern) | 2.5-4.0 | 10-50 km | < 0.05 |

## Related Documentation

- **[developing-models.md](./developing-models.md)** — How to build custom mobility models in your disease class
- **[contact-matrices.md](./contact-matrices.md)** — Age-specific contact patterns (complementary to spatial mixing)
- **[.claude/MODEL_AUTHORING_REFERENCE.md](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/.claude/MODEL_AUTHORING_REFERENCE.md)** — Internal reference for model development
- **[compartment/model.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/model.py)** — The `build_travel_matrix()` hook and `_ensure_travel_matrix()`
- **[compartment/helpers.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/helpers.py)** — Source code for `gravity_model()` and `get_gravity_model_travel_matrix()`
- **[tests/test_travel_matrix.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tests/test_travel_matrix.py)** — The invariants every model's matrix is held to

## References

### Foundational Papers

- **Zipf, G.K. (1946).** "The P1 P2/D Hypothesis: On the Intercity Movement of Persons." *American Sociological Review* 11(6): 677-686.
    - Original formulation of the gravity model for human mobility

- **Barthélemy, M. (2011).** "Spatial networks." *Physics Reports* 499(1-3): 1-101.
    - Comprehensive review of spatial networks, including gravity models

### Empirical Studies

- **Balcan et al. (2009).** "Multiscale mobility networks and the spatial spreading of infectious diseases." *PNAS* 106(21): 21484-21489.
    - Validates gravity models for epidemic modeling using global air travel data

- **Kraemer et al. (2020).** "The effect of human mobility and control measures on the COVID-19 epidemic in China." *Science* 368(6490): 493-497.
    - Modern application of mobility models to COVID-19 spread in China

### Distance-Decay Exponents

- **Viboud et al. (2006).** "Synchrony, waves, and spatial hierarchies in the spread of influenza." *Science* 312(5772): 447-451.
    - Empirical support for α ≈ 2.0 in seasonal influenza spread (US)

- **Erlander & Stewart (1990).** *The Gravity Model in Transportation Analysis.* VSP.
    - Comprehensive treatment of gravity models in transportation, typical α = 1.5-2.0

---

**Last Updated:** May 20, 2026  
**Version:** 0.1.9
