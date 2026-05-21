# Gravity Model for Spatial Mobility in the Pandemic Simulator

This document explains how the gravity model is used to create spatial mobility matrices for multi-region compartmental disease models in the Pandemic Simulator.

## Overview

The **gravity model** is a spatial interaction model that estimates movement flows between geographic regions based on:
1. **Population sizes** (the "mass" of each region)
2. **Geographic distance** between regions
3. **A distance-decay parameter** (how quickly travel drops off with distance)

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

## Implementation

The Pandemic Simulator provides a gravity model implementation in the framework:

### Classic Gravity Model (Standard Implementation)

**Location:** `compartment/helpers.py` → `gravity_model()` and `get_gravity_model_travel_matrix()`

**Formula:**
```
attraction(i → j) = (pop_i * pop_j) / distance_ij²
```

**Distance-decay:** Inverse-square law (α = 2.0), matching Newtonian gravity

**When used:**
- Default implementation called by the validation post-processor
- Used when `travel_volume` is specified in the config
- Automatically applied unless a model overrides with its own mobility function

**Code:**
```python
def gravity_model(df, mass_origin_col, mass_dest_col, distance_col, k=1):
    """
    Calculates the gravity model for a given dataframe.
    
    Returns:
        pandas dataframe with an additional column 'gravity' containing
        k * pop_origin * pop_dest / distance²
    """
    df["gravity"] = k * df[mass_origin_col] * df[mass_dest_col] / df[distance_col] ** 2
    return df
```

Models can also implement custom mobility functions with different distance-decay characteristics (exponential decay, power-law with custom exponents, etc.) by defining their own `mobility()` or `gravity()` methods.

## How Travel Matrices Are Built

The standard implementation follows this general pattern:

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

### Pattern 3: Custom Mobility Models

Some models define mobility functions directly on the disease class:

```python
class MyDiseaseModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self._admin_zones = config["case_file"]["admin_zones"]
        self._sigma = config.get("travel_volume", {}).get("leaving", 0.0)
    
    def mobility(self, admin_zones, sigma, scale_km=500.0):
        """Custom exponential distance-decay model"""
        # ... implement custom gravity formula ...
        return travel_matrix
    
    def prepare_initial_state(self):
        # Build travel matrix using custom mobility model
        self.travel_matrix = np.array(
            self.mobility(self._admin_zones, self._sigma)
        )
        return self.population_matrix, list(self.compartment_list)
```

**When to use:**
- Model-specific mobility assumptions (e.g., commuter vs. leisure travel)
- Disease-specific behavior changes (e.g., reduced travel during outbreak)
- Non-standard distance-decay functions

## Configuration

### In the Config JSON

Travel behavior is controlled by the `travel_volume` object:

```json
{
  "admin_unit_id": "USA",
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
  },
  "travel_volume": {
    "leaving": 0.2
  }
}
```

**Parameters:**
- **`leaving`** (float, 0.0-1.0): Fraction of each region's population that travels elsewhere per timestep
  - `0.0` = no travel (identity matrix)
  - `0.1` = 10% of people travel (typical for daily commuting)
  - `0.2` = 20% travel (default in many models)
  - Values > 1.0 are normalized to [0, 1]

### In Model Schemas

Models can declare travel parameters in `define_parameters()`:

```python
@classmethod
def define_parameters(cls):
    schema = ParameterSchemaBuilder()
    
    # Set default travel rate
    schema.set_travel_volume(default_leaving=0.2)
    
    # Models can also define custom mobility parameters
    schema.add_disease_parameter(
        "mobility_scale_km",
        description="Distance scale for exponential decay (km)",
        default=500.0,
        value_type=ValueType.RATE
    )
    
    return schema
```

## Validation and Edge Cases

### Automatic Handling

The framework automatically handles common edge cases:

1. **Single region** → Returns 1×1 identity matrix `[[1.0]]`
2. **Zero sigma** → Returns R×R identity matrix (no travel)
3. **Missing travel_volume** → Defaults to σ = 0.0 (no travel)
4. **Very small distances** → Clamped to minimum 1 km to avoid division by zero

### Warnings

The validation post-processor logs warnings for:
- **Negative distances** (indicates coordinate errors)
- **Extremely large distances** (> 20,000 km, possible coordinate swap)
- **Row sums ≠ 1.0** (normalization errors)

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
- **Use custom mobility models** when you have disease-specific behavioral assumptions
- **Document your distance-decay choice** — different exponents have different interpretations

### ❌ Don't

- **Don't use σ > 0.5 without justification** — implies more than half the population travels daily
- **Don't assume symmetry** — flow from A→B ≠ flow from B→A in real systems
- **Don't ignore edge cases** — always handle single-region and zero-travel scenarios
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
1. **Travel matrix not used in derivative** — Check that `travel_matrix` appears in FOI calculation
2. **σ = 0** — No travel configured
3. **All regions infected identically** — Initial conditions mask spatial effects

**Debug:**
```python
# In derivative(), log the travel-mixed prevalence
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

- **[DEVELOPING_MODELS.md](./DEVELOPING_MODELS.md)** — How to build custom mobility models in your disease class
- **[CONTACT_MATRICES.md](./CONTACT_MATRICES.md)** — Age-specific contact patterns (complementary to spatial mixing)
- **[.claude/MODEL_AUTHORING_REFERENCE.md](../.claude/MODEL_AUTHORING_REFERENCE.md)** — Internal reference for model development
- **[compartment/helpers.py](../compartment/helpers.py)** — Source code for `gravity_model()` and `get_gravity_model_travel_matrix()`

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
