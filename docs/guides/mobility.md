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

By default, `build_travel_matrix()` returns an identity matrix, which means there is no travel between zones. There is no default mobility or gravity model. A model with mobility must declare its parameters and override this method — see [Adding mobility to a model](#adding-mobility-to-a-model) below for how to do both.

- **It is built once per run, not once per timestep.** The matrix is fixed for the whole simulation unless an intervention changes it.
- **It is rebuilt for each uncertainty sample**, so sampled mobility parameters are applied if the modeler or user selects those as parameters to vary.
- **Do not assign `self.travel_matrix` in `__init__` or `prepare_initial_state()`.** The framework will overwrite it. Override `build_travel_matrix()` instead.

## Using the travel matrix in the model equation

The framework creates the travel matrix, but it does not automatically apply it. The model's `equation()` method must use the matrix when calculating the force of infection. Otherwise, travel has no effect on disease transmission.

Without mobility, the force of infection depends only on prevalence in the local zone:

```python
foi = beta * (I / N)
```

With mobility, it depends on prevalence in all zones visited by residents:

```python
I_frac = I / N
mixed_frac = self.travel_matrix @ I_frac
foi = beta * mixed_frac
```

Here, the matrix weights each destination's prevalence by the fraction of residents who visit it. For example, if 5% of zone *i*'s residents visit zone *j*, then 5% of their exposure is based on zone *j*'s prevalence.

See [`ebola_jax_model/model.py`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/models/ebola_jax_model/model.py) for a straightforward implementation. The mpox and hantavirus_human models use the equivalent `jnp.einsum()` operation.

### Combining mobility with age structure

Age-stratified models account for both where people travel and which age groups they contact. Apply the travel matrix first to combine prevalence across zones, then apply the contact matrix to combine exposure across age groups.

The COVID model follows this pattern:

```python
rates, travel_matrix = self._apply_interventions(t, rates, prop_infective_scalar)

BETA = ((rates["beta"] * travel_matrix) @ I_frac.T).T  # mix across zones
omega = self.contact_matrix @ BETA                     # mix across age groups
flow_foi = S * omega
```

The `.T` transposes exist because the travel matrix needs to multiply the 
zone axis while the contact matrix needs the age axis. Use the travel matrix returned by `_apply_interventions()` if you have an intervention that restricts travel. See [contact-matrices.md](./contact-matrices.md) for more about age-group mixing.

## Adding mobility to a model

### 1. Declare parameters, if needed

This step is optional. Use `schema.add_parameter()` in `define_parameters()` only for mobility settings that users should be able to change in the UI. If your mobility calculation has no user-configurable settings, you can skip this step.

For example, a model can expose the percentage of each zone's population that travels:

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

Declared parameters appear in the model configuration and UI and can be varied during uncertainty analysis.

If you declare mobility parameters:

- Name the travel fraction `travel_sigma`, not `sigma`, to avoid conflicts with transmission parameters.
- Use the `travel_` prefix for additional settings, such as `travel_alpha` or `travel_scale_km`.

### 2. Override `build_travel_matrix()`

Override this method to define how the model creates its travel matrix. This example uses the shared gravity function and the optional `travel_sigma` parameter declared above:

```python
def build_travel_matrix(self, admin_zones):
    sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
    return get_gravity_model_travel_matrix(admin_zones, sigma)
```

Each item in `admin_zones` contains the zone's `center_lat`, `center_lon`, and `population`. The list follows the same zone order as the population matrix.

Declared parameters are available on `self`. Percentage values are not automatically converted, so `self.travel_sigma` is `20.0` for 20%. Use `_to_rate()` to convert it to `0.2`, which the travel function expects.

Finally, use `self.travel_matrix` in `equation()` as described in [Using the travel matrix in the model equation](#using-the-travel-matrix-in-the-model-equation).

## Mobility functions available in the repository

A mobility function determines how travellers are distributed across destinations. The available functions use population and distance but give different weights to nearby and distant zones.

| Model | Mobility function | Destination weight from i to j | Implementation | Notes |
|---|---|---|---|---|
| `covid_jax_model` | Inverse-square gravity | `pop_i * pop_j / d²` | `get_gravity_model_travel_matrix()` in [`compartment/helpers.py`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/helpers.py) | Combines mobility with age-group mixing and travel restrictions |
| `dengue_jax_model` | Inverse-square gravity | `pop_i * pop_j / d²` | `get_gravity_model_travel_matrix()` | Applies mobility to human transmission |
| `ebola_jax_model` | Inverse-square gravity | `pop_i * pop_j / d²` | `get_gravity_model_travel_matrix()` | Simplest example of using mobility in the force of infection |
| `hantavirus_human_jax_model` | Power-law gravity | `pop_j / d^α` | `gravity()` in `hantavirus_human_jax_model/model.py` | Applies mobility to person-to-person spread; rodent spillover remains local. Uses `travel_alpha`, default 1.5 |
| `mpox_jax_model` | Exponential decay | `pop_j * exp(-d / scale_km)` | `mobility()` in `mpox_jax_model/model.py` | Clear example of a custom mobility function. Uses `travel_scale_km`, default 500 km |

The shared function in `helpers.py` implements the classic gravity model: larger and closer zones receive more travellers. It is available for reuse but is not applied by default.

## Implementing a custom mobility function

Write a custom function when the available options do not represent the travel relevant to your disease. [`mpox_jax_model`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/models/mpox_jax_model/model.py) is an example. It uses two user-configurable parameters:

- `travel_sigma`: the percentage of the population that travels;
- `travel_scale_km`: the distance over which travel becomes less likely.

The mpox model exposes two settings: the fraction of the population that travels, and the distance at which travel becomes uncommon.

```python
schema.add_parameter(
    name="travel_sigma",
    label="Travel Rate (σ)",
    value_type=ValueType.PERCENTAGE,
    default=20.0, min_value=0.0, max_value=100.0, unit="%",
)
schema.add_parameter(
    name="travel_scale_km",
    label="Travel Decay Length",
    value_type=ValueType.FLOAT,
    default=500.0, min_value=1.0, max_value=20000.0, unit="km",
)
```

The model converts `travel_sigma` to a fraction and passes both parameters to its `mobility()` function:

```python
def build_travel_matrix(self, admin_zones):
    sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
    return self.mobility(admin_zones, sigma, scale_km=self.travel_scale_km)
```

The function then builds the matrix:

```python
def mobility(self, admin_zones, sigma, scale_km=500.0):
    # Return an identity matrix if there is no travel to distribute.
    n = len(admin_zones)
    if n <= 1 or sigma == 0.0:
        return onp.eye(n)

    # Calculate distances and give more weight to large, nearby destinations.
    pops = onp.array([z["population"] for z in admin_zones], dtype=float)
    dist_km = ...  # pairwise distances from zone coordinates
    attraction = pops[None, :] * onp.exp(-dist_km / scale_km)
    onp.fill_diagonal(attraction, 0.0)

    # Convert destination weights to fractions.
    row_sums = attraction.sum(axis=1, keepdims=True)
    row_sums = onp.where(row_sums == 0.0, 1.0, row_sums)
    T = attraction / row_sums

    # Apply the travel fraction and assign the remainder to the home zone.
    travel_matrix = sigma * T
    onp.fill_diagonal(travel_matrix, 1.0 - sigma)
    return travel_matrix
```

To create a different mobility function, change how destinations are weighted while keeping the matrix requirements in [Checking your matrix](#checking-your-matrix). Use NumPy rather than `jax.numpy`; the matrix is created before the simulation begins, and the framework converts it as needed.

## Models without mobility

If a model does not include travel between zones, do not declare travel parameters or override `build_travel_matrix()`. The framework will use an identity matrix, so each zone is simulated independently.

[`hantavirus_jax_model`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/models/hantavirus_jax_model/model.py) follows this pattern. It models movement within each zone, but not between zones.

To temporarily disable mobility in a model that supports it, set `travel_sigma` to `0`.

## Interventions that restrict travel

Set `modifies_travel=True` when declaring an intervention to stop travel between zones while the intervention is active. The COVID lockdown is the only current example:

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
foi = beta * (travel_matrix @ I_frac)
```

Using `self.travel_matrix` instead would ignore the intervention. Travel restrictions currently stop all inter-zone travel; partial reductions are not supported.

## Configuration

User-configurable mobility parameters belong in the `Disease` block. The mobility function reads each zone's coordinates and population from `case_file.admin_zones`.

The [Ebola example configuration](../../compartment/models/ebola_jax_model/example-config.json) uses this structure (shown here with two of its zones):

```json
{
  "admin_unit_id": "MDG",
  "Disease": {
    "disease_type": "EBOLA",
    "travel_sigma": 5.0
  },
  "case_file": {
    "admin_zones": [
      {
        "name": "Antananarivo",
        "center_lat": -18.9359,
        "center_lon": 46.8047,
        "population": 7982937
      },
      {
        "name": "Antsiranana",
        "center_lat": -13.8693,
        "center_lon": 49.4135,
        "population": 2127141
      }
    ]
  }
}
```

`travel_sigma` is a percentage: use `20.0` for 20% travel. Some models expose additional settings, such as mpox's `travel_scale_km` and hantavirus_human's `travel_alpha`.

The existing mobility functions do not require a separate dataset.

## Checking your matrix

The framework does not validate the matrix during a simulation. An invalid matrix may produce incorrect results without an error. Check that:

- its shape is R×R and all values are finite and non-negative;
- every row sums to 1;
- the diagonal equals `1 - sigma`;
- rows and columns follow the order of `admin_zones`; and
- latitude and longitude are in the correct fields.

[`tests/test_travel_matrix.py`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tests/test_travel_matrix.py) checks these requirements for registered mobility models.


## Related documentation

- [contact-matrices.md](./contact-matrices.md) — age-group mixing, the demographic counterpart to spatial mixing
- [interventions.md](./interventions.md) — how interventions are declared and applied, including travel restrictions
- [`compartment/model.py`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/model.py) — the `build_travel_matrix()` hook and `_ensure_travel_matrix()`
- [`compartment/helpers.py`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/helpers.py) — the shared gravity function
