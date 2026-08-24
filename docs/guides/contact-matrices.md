# Contact Matrices in the Pandemic Simulator

This document explains how contact matrices are created, loaded, aggregated, and used within the Pandemic Simulator's compartmental modeling framework.

## Table of Contents

- [Overview](#overview)
- [Contact Matrix Methods](#contact-matrix-methods)
  - [Asymmetric Matrices](#asymmetric-matrices)
  - [Matrix Interpretation Example](#matrix-interpretation-example)
- [Aggregation: From 16 Bands to Your Model's Age Groups](#aggregation-from-16-bands-to-your-models-age-groups)
  - [Aggregation Algorithm](#aggregation-algorithm)
  - [Why This Works](#why-this-works)
  - [Example](#example)
- [Three Ways to Add a Contact Matrix in Code](#three-ways-to-add-a-contact-matrix-in-code)
  - [1. Built-in Contact Matrix Default (Recommended)](#1-built-in-contact-matrix-default-recommended)
    - [Altering income classifications](#altering-income-classifications)
  - [2. Custom Contact Matrix in Code](#2-custom-contact-matrix-in-code)
  - [3. Temporarily Testing Custom Contact Matrix](#3-temporarily-testing-custom-contact-matrix)
- [Using Contact Matrices in Your Model](#using-contact-matrices-in-your-model)
  - [In define_parameters()](#in-define_parameters)
  - [In equation()](#in-equation)
- [Validation and Warnings](#validation-and-warnings)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Related Documentation](#related-documentation)
- [References](#references)



## Overview

Contact matrices quantify **age-specific social mixing patterns** — how frequently people in different age groups come into contact with each other. These mixing patterns are critical for modeling respiratory and other contact-transmitted diseases, as they determine the force of infection across demographic groups.

The Pandemic Simulator uses **country-specific synthetic contact matrices** from [Prem et al. 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009098), covering 177 countries with 16 five-year age bands (0-4, 5-9, ..., 75+). When a country's ISO code is not in the Prem dataset, the framework uses a **precomputed income-level average** grouped by [World Bank income classification](https://blogs.worldbank.org/en/opendata/understanding-country-income--world-bank-group-income-classifica), and if no income group is available, a **global average** across all 177 countries is used.

## Contact Matrix Methods

A contact matrix is an **A×A** array (where A = number of demographic groups) where:

- `matrix[i, j]` = mean daily contacts per person in group *i* with all people in group *j*



### Asymmetric Matrices

Contact matrices are **asymmetric by design**:

- **Rows (i):** Each row represents a typical person in group *i* and shows their mean contacts with each group
- **Columns (j):** Each column sums the total contacts flowing into group *j* from all other groups

This asymmetry is critical for force-of-infection calculations because:

- A small group (e.g., elderly) may have fewer total contacts
- But they may contact larger groups (e.g., working-age caregivers) frequently
- The FOI on the elderly depends on the **prevalence** in those working-age groups **times** the contact rate

**Further reading** — using asymmetric contact matrices in force-of-infection calculations:

- Wallinga, Teunis & Kretzschmar 2006, ["Using Data on Social Contacts to Estimate Age-specific Transmission Parameters for Respiratory-spread Infectious Agents"](https://doi.org/10.1093/aje/kwj317), *American Journal of Epidemiology*
- Mossong et al. 2008 (POLYMOD), ["Social Contacts and Mixing Patterns Relevant to the Spread of Infectious Diseases"](https://doi.org/10.1371/journal.pmed.0050074), *PLOS Medicine*
- Franco et al. 2022, ["Inferring age-specific differences in susceptibility to and infectiousness upon SARS-CoV-2 infection based on Belgian social contact data"](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009965), *PLOS Computational Biology*



### Matrix Interpretation Example

For a 3-group model (children, adults, elderly):

```
         Children  Adults  Elderly
Children    12.0     4.0     1.0
Adults       6.0     8.0     2.0  
Elderly      2.0     3.0     5.0
```

**Reading the matrix:**

- On average, children have 12 contacts/day with other children, 4 with adults, 1 with elderly
- On average, adults have 6 contacts/day with children, 8 with other adults, 2 with elderly
- The matrix is **not symmetric** — adults contact children at rate 6.0, but children contact adults at rate 4.0



## Aggregation: From 16 Bands to Your Model's Age Groups

The Prem matrices are 16×16 (five-year age bands), but your model may have fewer, broader groups (e.g., 0-17, 18-55, 56+). The framework will automatically aggregate to your specified age groups.

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

### Example

If your model has a group `age_0_17` (0-17 years), it overlaps Prem bands:

- **(0-4):** 5 years out of 18 → weight = 5/18
- **(5-9):** 5 years out of 18 → weight = 5/18  
- **(10-14):** 5 years out of 18 → weight = 5/18
- **(15-19):** 3 years out of 18 (only 15-17 overlap) → weight = 3/18

The aggregator computes these overlaps automatically for every source-target pair.

This "sum over contact columns, average over participant rows" scheme is a standard published aggregation — essentially `hhh4contacts::aggregateC()` (Meyer & Held 2017) with uniform weights, resting on the methodology of Arregui et al. 2018 (see [References](#references)).

## Three Ways to Add a Contact Matrix in Code

The framework supports three approaches for defining contact matrices. They are listed below from lowest to highest precedence: when more than one approach specifies the same matrix cell, the one listed later overrides the earlier ones.



### 1. Built-in Contact Matrix Default (Recommended)

This approach provides realistic, country-specific mixing patterns without requiring you to manually specify contact rates. This method uses the three-tier matrix lookup detailed below:

**Three-tier matrix lookup:**


| Tier | Condition                                                      | Source                                              |
| ---- | -------------------------------------------------------------- | --------------------------------------------------- |
| 1    | Country code is in the Prem 2021 paper (177 countries)         | Country-specific synthetic 16×16 matrix             |
| 2    | Country code has a World Bank income classification in the CSV | Average of all Prem matrices at that income level   |
| 3    | Neither                                                        | Global average across all 177 Prem country matrices |


- Each country's income-group assignment is defined in the [contact_matrices_economics.csv](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/contact_matrices/data/contact_matrices_economics.csv), which maps every country to its tier and World Bank income classification (**High income**, **Upper middle income**, **Lower middle income**, **Low income**).

**How it works:**

- Declare an inclusive `age_range=(low, high)` on every demographic group in your model schema
- At model instantiation, the framework:
  1. Reads the simulation's `admin_unit_id` (e.g., `"USA"`, `"DEU.1_1"`)
  2. Extracts the ISO3 country code
  3. Resolves a 16×16 source matrix using a **three-tier lookup** (see below)
  4. Aggregates the matrix down to your declared age bands using [aggregation methods detailed above](#aggregation-from-16-bands-to-your-models-age-groups)

**Example:**

```python
@classmethod
def define_parameters(cls, schema):
    schema.set_model_info(disease_type="RESPIRATORY_AGE_STRUCTURED", ...)
    
    # Declare age ranges for automatic Prem matrix loading
    schema.add_demographic_group(
        "age_0_17", "Children (0-17)",
        default_weight=25.0,
        age_range=(0, 17),  # This enables the three-tier matrix lookup
    )
    schema.add_demographic_group(
        "age_18_55", "Adults (18-55)",
        default_weight=50.0,
        age_range=(18, 55),
    )
    schema.add_demographic_group(
        "age_56_plus", "Elderly (56+)",
        default_weight=25.0,
        age_range=(56, 120),
    )
```

The framework logs an informational message indicating which tier was used for every run.



#### Altering income classifications

- To change a location’s income category—for example, to reclassify the Cayman Islands from `High income` to `Upper middle income`—update the location’s `Tier` and `Income` values in the CSV file. Ensure that the capitalization matches the existing category names, and then save the file.

Next, run the following command:

`python tools/build_income_matrices.py`

This command regenerates `income_defaults.npz`, which contains the precomputed average for each income group.



### 2. Custom Contact Matrix in Code

**When to use:**

- Your model has bespoke contact values (e.g., from POLYMOD or other empirical studies) that should be used as defaults

**How it works:**

- Use `schema.set_contact_override(from_group, to_group, value)` in `define_parameters()`
- When **any** schema-level override is declared, the framework **does not** load the Prem matrix
- All unspecified cells default to the identity matrix (1.0 on diagonal, 0.0 elsewhere). This signifies that people in that age group have, on average, 1 contact per day with other people of that age group and no contact with other age groups.

**Example:**

```python
@classmethod
def define_parameters(cls, schema):
    schema.add_demographic_group("children", "Children", default_weight=50.0)
    schema.add_demographic_group("adults", "Adults", default_weight=50.0)
    
    # Schema-level overrides suppress Prem auto-loading
    schema.set_contact_override("children", "children", 12.0)
    schema.set_contact_override("children", "adults", 4.0)
    schema.set_contact_override("adults", "children", 2.0)
    schema.set_contact_override("adults", "adults", 8.0)
```



### 3. Temporarily Testing Custom Contact Matrix

**When to use:**

- Sensitivity analyses
- Exploring non-default mixing scenarios without modifying the model code
- Testing counterfactual contact patterns (e.g., school closures reducing child-child contacts)

**How it works:**

- Add `contact_matrix_overrides` to your simulation config JSON
- These are the highest-precedence source: for any cell they specify, the config value is used instead of the Prem default or a custom contact matrix. Cells that are not specified keep their existing value (from the custom contact matrix or Prem default), so you only need to list the cells you want to change.

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


## Using Contact Matrices in Your Model


### In `define_parameters()`

```python
@classmethod
def define_parameters(cls, schema):
    # 1. Declare demographic groups with age_range for Prem auto-loading
    schema.add_demographic_group(
        "age_0_17", "Children (0-17)",
        default_weight=25.0,
        age_range=(0, 17),
    )
    schema.add_demographic_group(
        "age_18_55", "Adults (18-55)",
        default_weight=50.0,
        age_range=(18, 55),
    )
    schema.add_demographic_group(
        "age_56_plus", "Elderly (56+)",
        default_weight=25.0,
        age_range=(56, 120),
    )
    
    # 2. (Optional) Override specific cells if needed
    # schema.set_contact_override("age_0_17", "age_0_17", 10.0)
```



### In `equation()`

```python
def equation(self, y, t, p):
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
    
    # ... rest of equation logic
```

The key insight: the contact matrix transforms prevalence (infection fraction by age) into an age-specific force of infection.

## Validation and Warnings

The framework validates and warns about common issues:

### Warning: Identity Matrix Default

If demographics are provided but:

- No `age_range` is declared on any group, AND
- No schema overrides are present, AND  
- No config overrides are present

Then the matrix defaults to **identity** (each group only contacts itself), and a warning is logged.


### Warning: Zero-Overlap Bands

If a target age range has **no overlap** with the Prem source bands (0-120), the aggregator logs a warning. The corresponding rows/columns will be zero.


## Related Documentation

- **[compartment/contact_matrices/](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/contact_matrices/)** — Source code for loader, aggregator, and bundled data
- **[tests/test_contact_matrices.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tests/test_contact_matrices.py)** — Unit tests demonstrating aggregation behavior
- **[tests/test_demographics.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tests/test_demographics.py)** — Integration tests for `_build_contact_matrix()`



## References

- **Prem et al. 2021:** ["Projecting contact matrices in 177 geographical regions: An update and comparison with empirical data for the COVID-19 era"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009098), *PLOS Computational Biology*
- **Synthetic Contact Matrices Repository:** [https://github.com/kieshaprem/synthetic-contact-matrices](https://github.com/kieshaprem/synthetic-contact-matrices)
- **Aggregation method —** `hhh4contacts::aggregateC()`**:** Meyer & Held 2017, ["Incorporating social contact data in spatio-temporal models for infectious disease spread"](https://doi.org/10.1093/biostatistics/kxw051), *Biostatistics* (see the `aggregateC` [reference](https://cran.r-project.org/web/packages/hhh4contacts/refman/hhh4contacts.html#aggregateC), which sums over contact columns and averages over participant rows — uniform weights by default, matching this framework's aggregation methods)
- **Aggregation method (foundational):** Arregui et al. 2018, ["Projecting social contact matrices to different demographic structures"](https://doi.org/10.1371/journal.pcbi.1006638), *PLOS Computational Biology*
- **POLYMOD Study:** Original empirical contact survey data from 8 European countries [Mossong et al. 2008](https://doi.org/10.1371/journal.pmed.0050074)
- **Asymmetric contact matrices in force-of-infection calculations:** Wallinga, Teunis & Kretzschmar 2006, ["Using Data on Social Contacts to Estimate Age-specific Transmission Parameters for Respiratory-spread Infectious Agents"](https://doi.org/10.1093/aje/kwj317), *American Journal of Epidemiology*
- **Asymmetric contact matrices (applied, COVID-19):** Franco et al. 2022, ["Inferring age-specific differences in susceptibility to and infectiousness upon SARS-CoV-2 infection based on Belgian social contact data"](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009965), *PLOS Computational Biology*
- **World Bank income classification:** [Understanding country income: World Bank Group income classifications for FY26 (July 1, 2025–June, 2026)](https://blogs.worldbank.org/en/opendata/understanding-country-income--world-bank-group-income-classifica)

---

**Last Updated:** August 10, 2026  
**Version:** 0.2.0