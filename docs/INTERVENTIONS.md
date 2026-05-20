# Interventions in the Pandemic Simulator

This document explains how interventions (public health measures) are defined, configured, and applied in the Pandemic Simulator's compartmental modeling framework.

## Overview

**Interventions** represent public health measures that modify disease transmission or population behavior during a simulation. Examples include:
- **Social distancing** / physical distancing
- **Mask wearing**
- **Vaccination campaigns**
- **Lockdowns** (travel restrictions)
- **Case isolation**
- **Vector control** (for vector-borne diseases)

Interventions can activate based on:
1. **Calendar dates** ("start on day X, end on day Y")
2. **Infection thresholds** ("activate when 5% infected, deactivate when below 1%")
3. **Both** (date window with threshold-based hysteresis)

The framework provides a **declarative intervention system**: models declare supported interventions in their schema, and users configure them at runtime via JSON config files. The framework handles activation logic, rate modification, and tracking automatically.

## Core Concepts

### How Interventions Work

At each timestep during simulation, the model:

1. **Checks activation conditions** for each intervention
   - Date-based: "Are we inside the date window?"
   - Threshold-based: "Is infection proportion above/below thresholds?"

2. **Applies rate reduction** to targeted transmission parameters
   - Formula: `new_rate = rate * (1 - adherence * transmission_reduction)`
   - Example: 40% adherence × 50% reduction = 20% overall reduction

3. **Modifies travel matrix** (for lockdowns)
   - Replaces travel matrix with identity (nobody travels)

4. **Tracks status** (active/inactive) for each intervention
   - Threshold interventions use **hysteresis**: once activated, they stay active until the end threshold is reached (prevents oscillation)

### Two-Pass Application

The framework applies interventions in **two passes** (replicating the original `interventions.py` logic):

**Pass 1: Date-based**
- Check if current day is inside `[start_date, end_date]` window
- Apply rate reduction **only while inside window**
- Rates bounce back once outside the window

**Pass 2: Threshold-based**  
- Check if `prop_infective >= start_threshold` (turn on)
- Check if `prop_infective <= end_threshold` (turn off)
- Apply rate reduction **as long as status is active**
- Persists even after threshold drops (until end_threshold)

**Important:** A single intervention can reduce rates **twice** if it has both date and threshold parameters configured. This matches the original behavior.

## Declaring Interventions in Models

Models declare supported interventions in `define_parameters()`:

### Basic Declaration

```python
@classmethod
def define_parameters(cls):
    schema = ParameterSchemaBuilder()
    
    # ... compartments and edges ...
    
    # Declare interventions
    schema.add_intervention(
        id="social_isolation",
        label="Social Isolation",
        description="Reduces contact rates through physical distancing measures",
        target_rates=["beta"],  # Which transmission rate(s) to modify
        adherence=40.0,  # Default: 40% population adherence
        transmission_reduction=50.0,  # Default: 50% transmission reduction
    )
    
    schema.add_intervention(
        id="mask_wearing",
        label="Mask Wearing",
        description="Reduces transmission via respiratory droplet protection",
        target_rates=["beta"],
        adherence=60.0,
        transmission_reduction=30.0,
    )
    
    return schema
```

### Intervention Parameters

#### Required Parameters

- **`id`** (string): Machine identifier (e.g., `"social_isolation"`, `"vaccination"`)
  - Used to reference the intervention in config files
  - Must be unique within a model

- **`label`** (string): Human-readable name displayed in UIs (e.g., `"Social Isolation"`)

- **`description`** (string): Explanation of what the intervention does

#### Optional Parameters

- **`target_rates`** (list[string]): Transmission edge variable names to modify
  - Example: `["beta"]` for respiratory diseases
  - Example: `["b_V_T", "s_V_T"]` for vector-borne diseases
  - Default: `[]` (no rate modification — used for lockdowns that only affect travel)

- **`modifies_travel`** (bool): If `True`, replaces travel matrix with identity when active
  - Default: `False`
  - Set to `True` for lockdown interventions

- **`adherence`** (float, 0-100): Default population adherence percentage
  - Default: `50.0`
  - What fraction of the population follows the intervention

- **`transmission_reduction`** (float, 0-100): Default transmission reduction percentage
  - Default: `5.0`  
  - How much transmission is reduced among adherent individuals

### Lockdown Example

```python
schema.add_intervention(
    id="lock_down",
    label="Lockdown",
    description="Severe movement restrictions — no inter-regional travel",
    target_rates=["beta"],  # Also reduces beta (less social contact)
    modifies_travel=True,  # KEY: replaces travel matrix with identity
    adherence=80.0,
    transmission_reduction=70.0,
)
```

### Vector Control Example (Dengue)

```python
schema.add_intervention(
    id="physical",
    label="Physical Vector Control",
    description="Remove standing water and breeding sites",
    target_rates=["b_V_T"],  # Targets vector birth rate
    adherence=50.0,
    transmission_reduction=30.0,
)

schema.add_intervention(
    id="chemical",
    label="Chemical Vector Control",  
    description="Insecticide fogging and larvicides",
    target_rates=["s_V_T"],  # Targets vector survival rate
    adherence=60.0,
    transmission_reduction=40.0,
)
```

## Configuring Interventions at Runtime

Users configure interventions in the simulation JSON config:

### Date-Based Intervention

Activate during a specific time window:

```json
{
  "interventions": [
    {
      "id": "social_isolation",
      "adherence_min": 40.0,
      "transmission_percentage": 50.0,
      "start_date": "2025-11-18",
      "end_date": "2025-12-31"
    }
  ]
}
```

**Behavior:**
- Activates on `2025-11-18`
- Remains active through `2025-12-31`
- Deactivates on `2026-01-01`
- Rate reduction applies **only during the window**

**Required fields:**
- `id`: Must match a declared intervention
- `start_date`: ISO date string (YYYY-MM-DD)

**Optional fields:**
- `end_date`: ISO date string (omit for "never ending")
- `adherence_min`: Override model default
- `transmission_percentage`: Override model default

### Threshold-Based Intervention

Activate when infection levels cross thresholds:

```json
{
  "interventions": [
    {
      "id": "mask_wearing",
      "adherence_min": 60.0,
      "transmission_percentage": 35.0,
      "start_threshold": 0.05,
      "end_threshold": 0.01
    }
  ]
}
```

**Behavior:**
- Activates when `prop_infective >= 0.05` (5% infected)
- Remains active until `prop_infective <= 0.01` (1% infected)
- Uses **hysteresis** to prevent oscillation
- Rate reduction persists as long as status is active

**Required fields:**
- `id`
- `start_threshold`: Proportion (0.0-1.0) that triggers activation

**Optional fields:**
- `end_threshold`: Proportion that triggers deactivation (omit for "never deactivate")
- `adherence_min`
- `transmission_percentage`

### Combined Date + Threshold

Use both activation mechanisms:

```json
{
  "interventions": [
    {
      "id": "vaccination",
      "adherence_min": 70.0,
      "transmission_percentage": 80.0,
      "start_date": "2025-03-01",
      "end_date": "2025-12-31",
      "start_threshold": 0.03,
      "end_threshold": 0.005
    }
  ]
}
```

**Behavior:**
- **Pass 1 (date):** Reduces rate while inside `[start_date, end_date]`
- **Pass 2 (threshold):** Reduces rate again if `prop_infective >= start_threshold`
- Result: Double reduction is possible during the date window when threshold is also active
- This matches the original two-pass behavior

### Multiple Simultaneous Interventions

Apply several interventions at once:

```json
{
  "interventions": [
    {
      "id": "social_isolation",
      "adherence_min": 40.0,
      "transmission_percentage": 50.0,
      "start_date": "2025-11-01",
      "end_date": "2025-12-31"
    },
    {
      "id": "mask_wearing",
      "adherence_min": 60.0,
      "transmission_percentage": 30.0,
      "start_date": "2025-11-15",
      "end_date": "2026-01-15"
    },
    {
      "id": "lock_down",
      "adherence_min": 80.0,
      "transmission_percentage": 70.0,
      "start_threshold": 0.10,
      "end_threshold": 0.02
    }
  ]
}
```

**Interaction:**
- Interventions are applied **sequentially** in the order they appear
- Each one modifies the rate(s) from the previous step
- Reductions are **multiplicative**, not additive:

```
beta_initial = 0.3
After social_isolation (40% × 50%): beta = 0.3 * (1 - 0.4*0.5) = 0.24
After mask_wearing (60% × 30%):     beta = 0.24 * (1 - 0.6*0.3) = 0.1968
```

## Reduction Formula

For each active intervention targeting a rate:

```
reduced_rate = rate * (1 - adherence * transmission_reduction)
```

Where:
- **adherence** and **transmission_reduction** are fractions (0.0-1.0), converted from percentages
- **adherence** = fraction of population following the intervention
- **transmission_reduction** = fractional reduction in transmission among adherent individuals

### Example Calculation

**Intervention:** Social distancing  
**adherence_min:** 40.0 (40%)  
**transmission_percentage:** 50.0 (50%)  
**Initial beta:** 0.3

```
adherence_fraction = 0.40
reduction_fraction = 0.50
combined_reduction = 0.40 * 0.50 = 0.20  (20% overall)

new_beta = 0.3 * (1 - 0.20) = 0.24
```

**Interpretation:**
- 40% of the population reduces their contacts by 50%
- Net effect: 20% reduction in transmission rate

### Multiple Interventions (Multiplicative)

With two active interventions:

```
beta_0 = 0.3

After intervention 1 (40% adhere, 50% reduction):
beta_1 = 0.3 * (1 - 0.4*0.5) = 0.3 * 0.8 = 0.24

After intervention 2 (60% adhere, 30% reduction):
beta_2 = 0.24 * (1 - 0.6*0.3) = 0.24 * 0.82 = 0.1968

Total reduction: 1 - (0.1968 / 0.3) = 34.4%
```

This is **not** the same as adding 20% + 18% = 38%.

## Implementation in `derivative()`

Models typically call `_apply_interventions()` in their `derivative()` method:

### Standard Pattern

```python
def derivative(self, y, t, p):
    xp = self._array_module()
    
    # Unpack parameters
    params = self._unpack_params(p)
    states = {c: y[i] for i, c in enumerate(self.compartment_list)}
    
    # Compute proportion infective (for threshold interventions)
    N_total = sum(states[c] for c in self.compartment_list if not c.endswith("_total"))
    I = states["I"]  # or sum of multiple infective compartments
    prop_infective = I.sum() / (N_total.sum() + 1e-10)
    
    # Extract rates that interventions might modify
    rates = {
        "beta": params["beta"],
        # ... other rates ...
    }
    
    # Apply interventions — modifies rates and travel_matrix
    rates, travel_matrix = self._apply_interventions(t, rates, prop_infective)
    
    # Use modified rates in force of infection calculation
    foi = rates["beta"] * I / (N_total + 1e-10)
    
    # ... rest of derivative logic ...
```

### What `_apply_interventions()` Does

1. Converts `t` (solver time) to ordinal day: `current_ordinal_day = start_date_ordinal + t`
2. Loops over all interventions present in `intervention_dict`
3. **Pass 1:** Applies date-based activation and rate reduction
4. **Pass 2:** Applies threshold-based activation and rate reduction
5. Returns `(modified_rates, modified_travel_matrix)`

### Custom Intervention Logic

Models can implement custom intervention methods alongside the framework interventions:

```python
def custom_intervention(self, beta, t, prop_infective):
    """Ring vaccination around detected cases."""
    cfg = self.intervention_dict.get("ring_vaccination")
    if cfg is None:
        return beta
    
    # Custom activation logic
    in_window = (t >= cfg["start_day"]) and (t <= cfg["end_day"])
    above_threshold = prop_infective >= cfg["detection_threshold"]
    
    if in_window and above_threshold:
        return beta * (1 - cfg["coverage"] * cfg["efficacy"])
    return beta

def derivative(self, y, t, p):
    # ... setup ...
    
    # Framework interventions (social distancing, masks, etc.)
    rates, travel_matrix = self._apply_interventions(t, rates, prop_infective)
    
    # Custom disease-specific intervention
    rates["beta"] = self.custom_intervention(rates["beta"], t, prop_infective)
    
    # ... use modified rates ...
```

## Uncertainty Quantification with Interventions

Intervention parameters can vary during uncertainty analysis:

### In Config JSON

```json
{
  "interventions": [
    {
      "id": "mask_wearing",
      "adherence_min": 60.0,
      "transmission_percentage": 35.0,
      "start_date": "2025-11-18",
      "end_date": "2025-12-31",
      "variance_params": [
        {
          "has_variance": true,
          "distribution_type": "UNIFORM",
          "field_name": "adherence_min",
          "min": 40.0,
          "max": 80.0
        },
        {
          "has_variance": true,
          "distribution_type": "UNIFORM",
          "field_name": "transmission_percentage",
          "min": 20.0,
          "max": 50.0
        }
      ]
    }
  ]
}
```

**Behavior in `UNCERTAINTY` mode:**
- Latin Hypercube Sampling draws values from specified ranges
- Each simulation run gets a different `(adherence, transmission_reduction)` pair
- Output includes median and confidence intervals across runs

**Why vary interventions?**
- **Adherence uncertainty:** How many people will actually follow guidelines?
- **Efficacy uncertainty:** How effective is mask-wearing really?
- **Policy scenario analysis:** What's the range of possible outcomes?

## Intervention Status Tracking

The model maintains an `intervention_statuses` dictionary:

```python
self.intervention_statuses = {
    "social_isolation": False,
    "mask_wearing": False,
    "lock_down": False,
}
```

**Updated every timestep:**
- `False` → intervention is inactive
- `True` → intervention is active

**Used for:**
- Hysteresis in threshold interventions (prevent oscillation)
- Logging/debugging intervention activation times
- Conditional logic in custom interventions

**Access in `derivative()`:**
```python
if self.intervention_statuses.get("lock_down", False):
    # Lockdown is currently active
    pass
```

## Validation and Edge Cases

### Missing Intervention

**Config references undeclared intervention:**
```json
{
  "interventions": [
    {"id": "nonexistent_intervention", ...}
  ]
}
```

**Behavior:** Ignored silently (no error). The framework only applies interventions that are declared in the model schema.

### No Activation Criteria

**Intervention with neither dates nor thresholds:**
```json
{
  "id": "mask_wearing",
  "adherence_min": 60.0,
  "transmission_percentage": 30.0
}
```

**Behavior:** Never activates (no start_date or start_threshold). This is valid but useless.

### Overlapping Date Windows

**Two interventions with overlapping dates:**
```json
[
  {"id": "social_isolation", "start_date": "2025-11-01", "end_date": "2025-12-31"},
  {"id": "mask_wearing", "start_date": "2025-11-15", "end_date": "2026-01-15"}
]
```

**Behavior:** Both apply during overlap (Nov 15 - Dec 31). Reductions are multiplicative.

### Zero Adherence or Reduction

```json
{"id": "mask_wearing", "adherence_min": 0.0, "transmission_percentage": 30.0}
```

**Behavior:** No effect (0% adherence means nobody follows it).

```json
{"id": "mask_wearing", "adherence_min": 60.0, "transmission_percentage": 0.0}
```

**Behavior:** No effect (0% reduction means it doesn't reduce transmission).

### Threshold Hysteresis

**Why end_threshold exists:**
```json
{"id": "isolation", "start_threshold": 0.05, "end_threshold": 0.01}
```

Without `end_threshold`, the intervention would:
1. Activate at 5% infected
2. Reduce transmission
3. Infections drop below 5%
4. Deactivate immediately
5. Infections rise above 5% again
6. **Oscillate forever**

With `end_threshold`, it stays active until infections drop to 1%, preventing oscillation.

## Best Practices

### ✅ Do

- **Declare realistic adherence values** (40-80% for most public health measures)
- **Use threshold interventions** for reactive policies (triggered by outbreak severity)
- **Use date interventions** for planned campaigns (vaccination rollout with fixed dates)
- **Set end_threshold < start_threshold** to prevent oscillation
- **Test without interventions first** to establish a baseline
- **Use uncertainty quantification** for adherence and efficacy when confidence is low
- **Verify intervention activation** by checking logs or status tracking
- **Document intervention assumptions** in your model schema descriptions

### ❌ Don't

- **Don't assume 100% adherence** unless modeling a controlled trial
- **Don't use 100% transmission reduction** (even perfect interventions have some failure)
- **Don't set end_threshold = start_threshold** (will oscillate)
- **Don't omit end_threshold for threshold interventions** (they'll never deactivate)
- **Don't forget both passes** when debugging — an intervention can reduce twice
- **Don't modify `target_rates` after schema is built** (affects artifact generation)
- **Don't assume linear combination** — reductions are multiplicative

## Troubleshooting

### Intervention Not Activating

**Symptoms:**
- Output identical with/without intervention in config
- No change in transmission rates

**Possible causes:**

1. **Missing activation criteria**
   ```json
   {"id": "masks"}  // No start_date or start_threshold!
   ```
   **Fix:** Add `start_date` or `start_threshold`

2. **Dates outside simulation window**
   ```json
   {"start_date": "2026-06-01"}  // Simulation ends 2026-03-01
   ```
   **Fix:** Check simulation `end_date` in config

3. **Threshold never reached**
   ```json
   {"start_threshold": 0.15}  // But infections never hit 15%
   ```
   **Fix:** Lower threshold or check model dynamics

4. **Intervention ID mismatch**
   ```json
   {"id": "social-distancing"}  // Model declares "social_isolation"
   ```
   **Fix:** Match exactly the schema ID

### Intervention Too Strong/Weak

**Symptoms:**
- Epidemic completely suppressed (R < 1 immediately)
- Or no noticeable effect

**Possible causes:**

1. **Unrealistic parameters**
   ```json
   {"adherence_min": 95.0, "transmission_percentage": 90.0}  // 85.5% total reduction!
   ```
   **Fix:** Use empirically-validated values (typically 20-60% combined)

2. **Multiple interventions stacking**
   - Two interventions each reducing 50% → 75% total reduction
   **Fix:** Account for multiplicative effect

3. **Wrong target_rate**
   ```python
   target_rates=["gamma"]  # Targeting recovery rate instead of beta!
   ```
   **Fix:** Target transmission rates (usually `beta`)

### Intervention Oscillation

**Symptoms:**
- Intervention repeatedly activates/deactivates
- Unstable output with rapid fluctuations

**Cause:**
```json
{
  "start_threshold": 0.05,
  "end_threshold": 0.05  // Same threshold!
}
```

**Fix:**
```json
{
  "start_threshold": 0.05,
  "end_threshold": 0.01  // Lower end threshold
}
```

Set `end_threshold` significantly below `start_threshold` to create a buffer zone.

### Lockdown Not Stopping Travel

**Symptoms:**
- Spatial spread continues despite lockdown

**Possible causes:**

1. **`modifies_travel` not set**
   ```python
   schema.add_intervention(
       id="lock_down",
       target_rates=["beta"],
       modifies_travel=False,  # WRONG
   )
   ```
   **Fix:** Set `modifies_travel=True`

2. **Travel matrix not used in derivative**
   ```python
   # Missing travel_matrix in FOI calculation
   foi = beta * I / N  # Should be: (travel_matrix @ I)
   ```
   **Fix:** Ensure `travel_matrix` from `_apply_interventions()` is used

## Related Documentation

- **[DEVELOPING_MODELS.md](./DEVELOPING_MODELS.md)** — General model development guide
- **[GRAVITY_MODEL.md](./GRAVITY_MODEL.md)** — How travel matrices work (relevant for lockdowns)
- **[CONTACT_MATRICES.md](./CONTACT_MATRICES.md)** — Age-specific mixing (interventions can target by age)
- **[compartment/interventions.py](../compartment/interventions.py)** — Legacy implementation (being replaced by runtime.py)
- **[compartment/runtime.py](../compartment/runtime.py)** — New `Intervention` class with JAX-compatible logic
- **[compartment/model.py](../compartment/model.py)** — `_apply_interventions()` implementation

## References

### Intervention Effectiveness Studies

- **Flaxman et al. (2020).** "Estimating the effects of non-pharmaceutical interventions on COVID-19 in Europe." *Nature* 584: 257-261.
  - Empirical estimates of NPI effectiveness (lockdowns, school closures, etc.)

- **Chu et al. (2020).** "Physical distancing, face masks, and eye protection to prevent person-to-person transmission of SARS-CoV-2." *The Lancet* 395: 1973-1987.
  - Meta-analysis of mask effectiveness and physical distancing

- **Haug et al. (2020).** "Ranking the effectiveness of worldwide COVID-19 government interventions." *Nature Human Behaviour* 4: 1303-1312.
  - Comparative effectiveness of 6,000 interventions across 79 countries

### Adherence and Compliance

- **Betsch et al. (2020).** "Social and behavioral consequences of mask policies during the COVID-19 pandemic." *PNAS* 117(36): 21851-21853.
  - Adherence rates for mask-wearing mandates

- **Huynh (2020).** "The COVID-19 containment in Vietnam: What are we doing?" *Journal of Global Health* 10(1): 010338.
  - Case study of high-adherence interventions in Vietnam

---

**Last Updated:** May 20, 2026  
**Version:** 0.1.9
