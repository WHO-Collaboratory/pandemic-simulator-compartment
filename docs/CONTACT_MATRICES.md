# Contact Matrices in the Pandemic Simulator

This document explains how contact matrices are created, loaded, aggregated, and used within the Pandemic Simulator's compartmental modeling framework.

## Overview

Contact matrices quantify **age-specific social mixing patterns** — how frequently people in different age groups come into contact with each other. These mixing patterns are critical for modeling respiratory and other contact-transmitted diseases, as they determine the force of infection across demographic groups.

The Pandemic Simulator uses **country-specific synthetic contact matrices** from [Prem et al. 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009098), covering 177 countries with 16 five-year age bands (0-4, 5-9, ..., 75+).

## Three Ways to Supply a Contact Matrix

The framework supports three approaches for defining contact matrices, in order of precedence (later wins):

### 1. Country-Aware Prem 2021 Default (Recommended)

**How it works:**
- Declare an inclusive `age_range=(low, high)` on every demographic group in your model schema
- At model instantiation, the framework:
  1. Reads the simulation's `admin_unit_id` (e.g., `"USA"`, `"DEU.1_1"`)
  2. Extracts the ISO3 country code (splits on `.` and takes the first part)
  3. Loads that country's 16×16 synthetic contact matrix from the bundled dataset
  4. Aggregates the matrix down to your declared age bands using fractional-membership weighting

**When to use:** This is the recommended approach for most models. It provides realistic, country-specific mixing patterns without requiring you to manually specify contact rates.

**Example:**
```python
def define_parameters(cls):
    schema = ParameterSchemaBuilder()
    schema.set_model_info(disease_type="RESPIRATORY_AGE_STRUCTURED", ...)
    
    # Declare age ranges for automatic Prem matrix loading
    schema.add_demographic_group(
        id="age_0_17",
        default_population_fraction=25.0,
        age_range=(0, 17)  # This enables Prem auto-loading
    )
    schema.add_demographic_group(
        id="age_18_55",
        default_population_fraction=50.0,
        age_range=(18, 55)
    )
    schema.add_demographic_group(
        id="age_56_plus",
        default_population_fraction=25.0,
        age_range=(56, 120)
    )
```

If the country is not in the Prem dataset, the framework falls back to a **global-average matrix** (mean across all 177 countries) and logs an informational message.

### 2. Schema-Level Overrides

**How it works:**
- Use `schema.set_contact_override(from_group, to_group, value)` in `define_parameters()`
- When **any** schema-level override is declared, the framework **does not** load the Prem matrix
- All unspecified cells default to identity (1.0 on diagonal, 0.0 elsewhere)

**When to use:** 
- Your model has bespoke contact values (e.g., from POLYMOD or other empirical studies) that should be used regardless of country
- You want to bake specific mixing assumptions into the model itself

**Example:**
```python
def define_parameters(cls):
    schema = ParameterSchemaBuilder()
    schema.add_demographic_group(id="children")
    schema.add_demographic_group(id="adults")
    
    # Schema-level overrides suppress Prem auto-loading
    schema.set_contact_override("children", "children", 12.0)
    schema.set_contact_override("children", "adults", 4.0)
    schema.set_contact_override("adults", "children", 2.0)
    schema.set_contact_override("adults", "adults", 8.0)
```

### 3. Per-Run Config Overrides

**How it works:**
- Add `contact_matrix_overrides` to your simulation config JSON
- These overrides beat both Prem defaults and schema overrides for the cells they specify

**When to use:**
- Sensitivity analyses
- Exploring non-default mixing scenarios without modifying the model code
- Testing counterfactual contact patterns (e.g., school closures reducing child-child contacts)

**Example config:**
```json
{
  "admin_unit_id": "USA",
  "Disease": { ... },
  "contact_matrix_overrides": {
    "age_0_17": {
      "age_0_17": 8.0,
      "age_18_55": 2.5
    },
    "age_56_plus": {
      "age_0_17": 0.5,
      "age_56_plus": 6.0
    }
  }
}
```

## How Contact Matrices Work

A contact matrix is an **A×A** array (where A = number of demographic groups) where:
- **`matrix[i, j]`** = mean daily contacts per person in group *i* with all people in group *j*

### Asymmetric Semantics

Contact matrices are **asymmetric by design**:
- **Rows (i):** Each row represents a typical person in group *i* and shows their mean contacts with each group
- **Columns (j):** Each column sums the total contacts flowing into group *j* from all other groups

This asymmetry is critical for force-of-infection calculations because:
- A small group (e.g., elderly) may have fewer total contacts
- But they may contact larger groups (e.g., working-age caregivers) frequently
- The FOI on the elderly depends on the **prevalence** in those working-age groups **times** the contact rate

### Matrix Interpretation Example

For a 3-group model (children, adults, elderly):

```
         Children  Adults  Elderly
Children    12.0     4.0     1.0
Adults       6.0     8.0     2.0  
Elderly      2.0     3.0     5.0
```

**Reading the matrix:**
- Children have 12 contacts/day with other children, 4 with adults, 1 with elderly
- Adults have 6 contacts/day with children, 8 with other adults, 2 with elderly
- The matrix is **not symmetric** — adults contact children at rate 6.0, but children contact adults at rate 4.0

## Aggregation: From 16 Bands to Your Model's Age Groups

The Prem matrices are 16×16 (five-year age bands), but your model may have fewer, broader groups (e.g., 0-17, 18-55, 56+). The framework aggregates using **fractional-membership weighting**.

### Aggregation Algorithm

Given a 16×16 source matrix **M** and target age ranges, the aggregated A×A matrix is:

```
M_agg = W @ M @ U^T
```

Where:
- **W** (A × 16): Row-normalized overlap fractions. Each row sums to 1. This **averages** across source bands within each target band.
- **U** (A × 16): Raw (un-normalized) overlap fractions. This **sums** across source bands for each target band.

### Why This Works

The asymmetric row/column treatment preserves the "mean total contacts per person" semantic:

1. **Row direction (W):** When a target band spans multiple source bands, we take the **mean** contact rate of a typical person sampled from that band
2. **Column direction (U^T):** When a target band spans multiple source bands, we **sum** the total contacts flowing to all people in that band

**Key property:** Aggregating a Prem matrix back to its own 16 bands returns the original matrix exactly.

### Fractional Membership Example

If your model has a group `age_0_17` (0-17 years), it overlaps Prem bands:
- **(0-4):** 5 years out of 18 → weight = 5/18
- **(5-9):** 5 years out of 18 → weight = 5/18  
- **(10-14):** 5 years out of 18 → weight = 5/18
- **(15-19):** 3 years out of 18 (only 15-17 overlap) → weight = 3/18

The aggregator computes these overlaps automatically for every source-target pair.

## Implementation Details

### Code Structure

```
compartment/contact_matrices/
├── __init__.py           # Public API, PREM_BAND_EDGES constant
├── loader.py             # load_country_matrix(), default_matrix()
├── aggregator.py         # aggregate_to_bands()
└── data/
    └── contact_all.npz   # Bundled Prem 2021 matrices (177 countries)
```

### Key Functions

#### `load_country_matrix(iso3: str) -> np.ndarray | None`
- Returns the 16×16 Prem matrix for the given ISO3 code
- Case-insensitive: `"usa"` and `"USA"` both work
- Returns `None` if the country is not in the bundle

#### `default_matrix() -> np.ndarray`
- Returns the global-average 16×16 matrix
- Computed as the mean across all 177 country matrices
- Cached after first call

#### `aggregate_to_bands(matrix: np.ndarray, target_ranges: list[tuple[int, int]]) -> np.ndarray`
- Collapses a 16×16 Prem matrix to A×A bands
- `target_ranges`: list of inclusive (low, high) age tuples, e.g., `[(0, 17), (18, 55), (56, 120)]`
- Returns an A×A matrix in the same units (mean daily contacts per person)

### Model Integration

The `Model` base class method `_build_contact_matrix(config)` orchestrates the entire process:

1. Check if the model declares demographic groups — if not, return `None`
2. Resolve effective group IDs (config demographics take precedence over schema defaults)
3. Start with an identity matrix (A×A)
4. If conditions are met (all groups have `age_range`, no overrides), load and aggregate the Prem matrix
5. Apply schema-level overrides (if any)
6. Apply config-level overrides (if any)
7. Return the final matrix as a JAX/NumPy array

The resulting matrix is stored in `self.contact_matrix` and used by the model's `derivative()` function to compute age-stratified force of infection.

## Using Contact Matrices in Your Model

### In `define_parameters()`

```python
@classmethod
def define_parameters(cls):
    schema = ParameterSchemaBuilder()
    
    # 1. Declare demographic groups with age_range for Prem auto-loading
    schema.add_demographic_group(
        id="age_0_17",
        default_population_fraction=25.0,
        age_range=(0, 17)
    )
    schema.add_demographic_group(
        id="age_18_55",
        default_population_fraction=50.0,
        age_range=(18, 55)
    )
    schema.add_demographic_group(
        id="age_56_plus",
        default_population_fraction=25.0,
        age_range=(56, 120)
    )
    
    # 2. (Optional) Override specific cells if needed
    # schema.set_contact_override("age_0_17", "age_0_17", 10.0)
    
    return schema
```

### In `derivative()`

```python
def derivative(self, y, t, p):
    xp = self._array_module()
    R = self.num_regions
    A = self.num_age_groups
    
    # Reshape state: compartments × regions × age_groups
    state = {comp: y[i].reshape(R, A) for i, comp in enumerate(self.compartment_list)}
    
    # Population by region and age
    N = xp.sum(xp.stack([state[c] for c in self.compartment_list]), axis=0)  # (R, A)
    
    # Force of infection with contact matrix
    # prevalence[r, a] = infectious population fraction in region r, age group a
    prevalence = state["I"] / (N + 1e-9)
    
    # foi[r, a] = sum over age groups: beta * contact_matrix[a, a'] * prevalence[r, a']
    # Shape: (R, A)
    foi = self.beta * (self.contact_matrix @ prevalence.T).T
    
    # Apply foi to susceptibles
    new_infections = foi * state["S"]
    
    # ... rest of derivative logic
```

The key insight: the contact matrix transforms prevalence (infection fraction by age) into an age-specific force of infection.

## Validation and Warnings

The framework validates and warns about common issues:

### Warning: Identity Matrix Default
If demographics are provided but:
- No `age_range` is declared on any group, AND
- No schema overrides are present, AND  
- No config overrides are present

Then the matrix defaults to **identity** (each group only contacts itself), and a warning is logged. This is almost always a bug — real populations have cross-group mixing.

### Warning: Zero-Overlap Bands
If a target age range has **no overlap** with the Prem source bands (0-120), the aggregator logs a warning. The corresponding rows/columns will be zero.

## Best Practices

### ✅ Do

- **Declare `age_range` on all demographic groups** to enable country-specific Prem auto-loading
- **Use inclusive (low, high) tuples** for age ranges, e.g., `(0, 17)` includes ages 0 through 17
- **Test your model with different countries** to ensure mixing patterns are realistic
- **Use config overrides for sensitivity analyses** rather than modifying the model code
- **Check the logs** during model runs — the framework reports which matrix source was used

### ❌ Don't

- **Don't declare age ranges inconsistently** — either all groups have them or none do
- **Don't mix schema overrides and age ranges** — schema overrides suppress Prem loading entirely
- **Don't assume symmetric matrices** — contact patterns are asymmetric by population size
- **Don't declare age ranges outside [0, 120]** — the Prem bands cap at 120 for the open-ended "75+" group

## Troubleshooting

### "Contact matrix defaults to identity" warning

**Cause:** Model has demographic groups but no way to determine cross-group mixing.

**Fix:** Add `age_range=(low, high)` to every demographic group in your schema, or provide explicit `set_contact_override()` calls.

### Country not found in Prem bundle

**Cause:** The `admin_unit_id` ISO3 code is not in the 177-country dataset.

**Effect:** Framework falls back to global-average matrix and logs an info message. The model still runs.

**Fix (if needed):** 
- Check the list of available countries: `from compartment.contact_matrices import available_countries; print(available_countries())`
- Use a neighboring country's ISO3 code in your config for testing

### Force of infection seems wrong

**Check:**
1. Is `self.contact_matrix` being applied correctly in `derivative()`?
2. Are you using the right matrix dimensions (A×A, not R×A)?
3. Is the matrix multiplication `contact_matrix @ prevalence.T` producing the expected shape?
4. Are you scaling by `beta` (transmission rate per contact)?

**Debug snippet:**
```python
print("Contact matrix shape:", self.contact_matrix.shape)
print("Contact matrix:\n", self.contact_matrix)
print("Prevalence shape:", prevalence.shape)
```

## Related Documentation

- **[DEVELOPING_MODELS.md](./DEVELOPING_MODELS.md)** — General guide to authoring new models, includes contact matrix section
- **[.claude/MODEL_AUTHORING_REFERENCE.md](../.claude/MODEL_AUTHORING_REFERENCE.md)** — Internal reference for model development patterns and pitfalls
- **[compartment/contact_matrices/](../compartment/contact_matrices/)** — Source code for loader, aggregator, and bundled data
- **[tests/test_contact_matrices.py](../tests/test_contact_matrices.py)** — Unit tests demonstrating aggregation behavior
- **[tests/test_demographics.py](../tests/test_demographics.py)** — Integration tests for `_build_contact_matrix()`

## References

- **Prem et al. 2021:** ["Projecting contact matrices in 177 geographical regions: An update and comparison with empirical data for the COVID-19 era"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009098), *PLOS Computational Biology*
- **POLYMOD Study:** Original empirical contact survey data from 8 European countries (Mossong et al. 2008)
- **Synthetic Contact Matrices Repository:** [https://github.com/kieshaprem/synthetic-contact-matrices](https://github.com/kieshaprem/synthetic-contact-matrices)

---

**Last Updated:** May 20, 2026  
**Version:** 0.1.9
