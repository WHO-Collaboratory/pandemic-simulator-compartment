# Uncertainty Quantification in the Pandemic Simulator

This document explains how to quantify and communicate parameter uncertainty in compartmental disease models using the Pandemic Simulator's built-in uncertainty quantification (UQ) system.

## Overview

**Uncertainty quantification** addresses a fundamental challenge in epidemic modeling: **we don't know the true values of model parameters**. Transmission rates, recovery periods, intervention effectiveness, and initial conditions all have inherent uncertainty.

Instead of running a single simulation with point estimates, UQ:
1. **Defines uncertainty ranges** for parameters (e.g., "beta is between 0.2 and 0.4")
2. **Samples parameter combinations** using Latin Hypercube Sampling (LHS)
3. **Runs multiple simulations** with different parameter sets
4. **Aggregates results** to produce median trajectories and confidence intervals

This provides **credible ranges** for model predictions rather than false precision from a single run.

## Why Uncertainty Quantification Matters

### The Problem with Point Estimates

A deterministic simulation with `beta = 0.3` produces a single trajectory:
```
Day 30: 1,234 infections
Day 60: 8,567 infections
```

But what if `beta` is actually 0.25? Or 0.35? The difference can be enormous.

### The UQ Solution

An uncertainty run with `beta ~ Uniform(0.25, 0.35)` produces:
```
Day 30: 1,100 infections (95% CI: 850-1,450)
Day 60: 7,800 infections (95% CI: 5,200-11,500)
```

**Key benefits:**
- **Honest uncertainty:** Reflects what we don't know
- **Decision support:** "In the worst case, we'll have 11,500 infections"
- **Sensitivity analysis:** Which parameters drive uncertainty the most?
- **Credibility:** Avoids overconfident predictions

## Run Modes: DETERMINISTIC vs. UNCERTAINTY

The simulator supports two run modes:

### DETERMINISTIC Mode

```json
{
  "run_mode": "DETERMINISTIC"
}
```

**Behavior:**
- Runs model **once** with parameter point estimates
- Returns a single trajectory for each compartment
- Fast (seconds to minutes)
- Use for: initial exploration, testing, baseline scenarios

### UNCERTAINTY Mode

```json
{
  "run_mode": "UNCERTAINTY",
  "n_simulations": 30
}
```

**Behavior:**
- Runs model **N times** with sampled parameters (default: 30)
- Returns **median, lower bound (2.5%), upper bound (97.5%)** for each compartment
- Slower (minutes to hours, depending on N and model complexity)
- Use for: final results, communicating uncertainty, policy analysis

**Note:** Both modes still run **with and without interventions** in parallel (control run), so you get uncertainty bands for both scenarios.

## Latin Hypercube Sampling (LHS)

The framework uses **Latin Hypercube Sampling** to efficiently explore the parameter space.

### Why LHS?

Compared to random sampling or grid search:
- ✅ **Efficient:** Better coverage with fewer samples
- ✅ **Stratified:** Ensures all parts of the range are represented
- ✅ **Low discrepancy:** Avoids clustering of sample points

### How LHS Works

For each parameter:
1. **Divide the range into N equal-probability intervals**
2. **Sample once from each interval**
3. **Randomly permute** the samples to avoid correlation

**Example:** Sampling beta ~ Uniform(0.2, 0.4) with N=5:

```
Intervals:  [0.20, 0.24] [0.24, 0.28] [0.28, 0.32] [0.32, 0.36] [0.36, 0.40]
Samples:         0.22          0.26          0.30          0.34          0.38
Permuted:        0.34          0.22          0.38          0.26          0.30
```

This guarantees good coverage even with few samples.

## Supported Distributions

The framework supports **five probability distributions** for parameter uncertainty:

### 1. Uniform Distribution

**When to use:** No prior knowledge, all values equally likely within a range

```json
{
  "variance_params": {
    "has_variance": true,
    "distribution_type": "UNIFORM",
    "min": 0.2,
    "max": 0.4
  }
}
```

**Characteristics:**
- Flat probability density
- Simple, interpretable
- Conservative (gives equal weight to extremes)

**Example:** "Beta could be anywhere between 0.2 and 0.4"

### 2. Normal (Gaussian) Distribution

**When to use:** Parameter is approximately bell-shaped around a mean

```json
{
  "variance_params": {
    "has_variance": true,
    "distribution_type": "NORMAL",
    "mean": 0.3,
    "std": 0.05
  }
}
```

**Characteristics:**
- Symmetric bell curve
- Most values near the mean
- Can produce negative values (clamp if needed)

**Example:** "Beta is normally distributed with mean 0.3 and std 0.05"

### 3. Triangular Distribution

**When to use:** You have a best guess (mode) and min/max bounds

```json
{
  "variance_params": {
    "has_variance": true,
    "distribution_type": "TRIANGULAR",
    "min": 0.2,
    "probability_mode": 0.5,
    "max": 0.4
  }
}
```

**Characteristics:**
- Asymmetric triangle
- Mode at `min + (max - min) * probability_mode`
- More realistic than uniform (acknowledges a "best guess")

**Example:** "Beta is most likely 0.3, but could range from 0.2 to 0.4"

**Note:** `probability_mode` must be in [0, 1]. It's the **position** of the peak, not the value.
- `probability_mode=0.0` → peak at min
- `probability_mode=0.5` → peak at midpoint
- `probability_mode=1.0` → peak at max

### 4. Beta Distribution

**When to use:** Parameter is a proportion (0-1), with specific shape

```json
{
  "variance_params": {
    "has_variance": true,
    "distribution_type": "BETA",
    "alpha": 2.0,
    "beta": 5.0
  }
}
```

**Characteristics:**
- Bounded to [0, 1]
- Flexible shapes (U-shaped, bell-shaped, uniform)
- Used for probabilities and proportions

**Example:** "Intervention adherence has a beta(2, 5) distribution"

**Shape guide:**
- `alpha = beta = 1` → Uniform(0, 1)
- `alpha > beta` → Skewed right (most values near 1)
- `alpha < beta` → Skewed left (most values near 0)
- `alpha, beta > 1` → Bell-shaped

### 5. Lognormal Distribution

**When to use:** Parameter is always positive and right-skewed (e.g., incubation periods)

```json
{
  "variance_params": {
    "has_variance": true,
    "distribution_type": "LOGNORMAL",
    "mean": 1.0,
    "sigma": 0.5
  }
}
```

**Characteristics:**
- Always positive
- Right-skewed (long tail)
- Natural for duration and rate parameters

**Example:** "Incubation period is lognormally distributed"

**Note:** `mean` and `sigma` are parameters of the **underlying normal distribution**, not the lognormal itself. The median of the lognormal is `exp(mean)`.

## Declaring Uncertainty in Config

### On Transmission Edges

Add `variance_params` to any transmission edge:

```json
{
  "Disease": {
    "transmission_edges": [
      {
        "source": "susceptible",
        "target": "exposed",
        "data": {
          "transmission_rate": 0.3,
          "variance_params": {
            "has_variance": true,
            "distribution_type": "UNIFORM",
            "min": 0.25,
            "max": 0.35
          }
        }
      },
      {
        "source": "exposed",
        "target": "infected",
        "data": {
          "transmission_rate": 0.2,
          "variance_params": {
            "has_variance": true,
            "distribution_type": "NORMAL",
            "mean": 0.2,
            "std": 0.03
          }
        }
      }
    ]
  }
}
```

**Effect:** Each simulation run draws new values for these rates from the specified distributions.

### On Interventions

Add `variance_params` to intervention parameters:

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

**Effect:** Each run draws new adherence and transmission_percentage values.

**Important:** For interventions, you must specify `field_name` to indicate which parameter varies.

### Choosing Sample Size (n_simulations)

```json
{
  "run_mode": "UNCERTAINTY",
  "n_simulations": 30
}
```

**Guidelines:**
- **10-30:** Quick exploration, rough CI estimates
- **30-100:** Standard for reporting (default: 30)
- **100-500:** High-confidence CIs, sensitivity analysis
- **500+:** Publication-quality, very tight CIs

**Trade-off:** More samples = narrower CIs but longer runtime.

**Rule of thumb:** 30 samples gives ~15% CI width, 100 samples gives ~10% CI width.

## Understanding Output

### DETERMINISTIC Output

```json
{
  "admin_zones": [
    {
      "time_series": [
        {
          "date": "2025-11-18",
          "S": 999500,
          "I": 500,
          "R": 0
        },
        {
          "date": "2025-11-19",
          "S": 998950,
          "I": 550,
          "R": 0
        }
      ]
    }
  ]
}
```

Single value per compartment per timestep.

### UNCERTAINTY Output

```json
{
  "admin_zones": [
    {
      "time_series": [
        {
          "date": "2025-11-18",
          "S": {
            "mean": 999500,
            "lower": 999400,
            "upper": 999600
          },
          "I": {
            "mean": 500,
            "lower": 420,
            "upper": 580
          }
        },
        {
          "date": "2025-11-19",
          "S": {
            "mean": 998950,
            "lower": 998200,
            "upper": 999500
          },
          "I": {
            "mean": 550,
            "lower": 450,
            "upper": 680
          }
        }
      ]
    }
  ]
}
```

Three values per compartment per timestep:
- **mean:** Median across all simulation runs (50th percentile)
- **lower:** Lower bound of 95% CI (2.5th percentile)
- **upper:** Upper bound of 95% CI (97.5th percentile)

**Interpretation:**
- "On day 19, we expect **550 infections** (median)"
- "We're 95% confident the true value is between **450 and 680**"

### Confidence Interval (CI)

The framework uses a **95% confidence interval** by default:
- **lower:** 2.5th percentile of simulation results
- **upper:** 97.5th percentile of simulation results

This means:
- **95% of simulation runs** fall within [lower, upper]
- **2.5% are below** lower bound
- **2.5% are above** upper bound

**CI width** reflects uncertainty:
- **Narrow CI:** Parameters well-constrained
- **Wide CI:** Large parameter uncertainty or high sensitivity

## Workflow: From Deterministic to Uncertainty

### Step 1: Deterministic Baseline

Start with a deterministic run to validate your model:

```json
{
  "run_mode": "DETERMINISTIC",
  "Disease": {
    "transmission_edges": [
      {
        "source": "susceptible",
        "target": "infected",
        "data": {"transmission_rate": 0.3}
      }
    ]
  }
}
```

**Check:**
- Does the model run without errors?
- Are trajectories reasonable?
- Do interventions have the expected effect?

### Step 2: Identify Uncertain Parameters

Ask: **Which parameters am I most uncertain about?**

**Good candidates:**
- **Transmission rates (beta)** — varies by setting, behavior, season
- **Initial infections** — often poorly known early in an outbreak
- **Intervention effectiveness** — depends on adherence and implementation
- **Recovery/removal rates** — clinical heterogeneity

**Poor candidates:**
- **Well-known biological constants** (e.g., human lifespan)
- **Structural parameters** (e.g., number of regions)
- **Fixed policy choices** (e.g., lockdown start date)

### Step 3: Literature Review

Find plausible ranges from literature:

**Example sources:**
- Meta-analyses (e.g., "R0 for COVID-19: 2.5-5.0" → beta ranges)
- Clinical studies (e.g., "Incubation period: 5-7 days")
- Empirical intervention effectiveness (e.g., "Masks reduce transmission by 20-50%")

### Step 4: Add variance_params

```json
{
  "run_mode": "UNCERTAINTY",
  "n_simulations": 30,
  "Disease": {
    "transmission_edges": [
      {
        "source": "susceptible",
        "target": "infected",
        "data": {
          "transmission_rate": 0.3,
          "variance_params": {
            "has_variance": true,
            "distribution_type": "UNIFORM",
            "min": 0.25,
            "max": 0.35
          }
        }
      }
    ]
  }
}
```

### Step 5: Run and Interpret

Run the uncertainty mode:

```bash
python -m compartment.models.covid_jax_model.main \
    --mode local \
    --config_file my-uncertainty-config.json \
    --output_file results/uncertainty-output.json
```

**Examine output:**
- Is the median similar to your deterministic baseline?
- Are CIs reasonable (not too wide or too narrow)?
- Do CIs widen over time (typical — uncertainty compounds)?

### Step 6: Sensitivity Analysis

Which parameters drive uncertainty the most?

**Method:**
1. Run uncertainty with **only beta varying**
2. Run uncertainty with **only gamma varying**
3. Compare CI widths

**Result:** The parameter that produces wider CIs is the **high-sensitivity parameter** — prioritize getting better estimates for it.

## Common Patterns

### Pattern 1: Uniform Priors for All Transmission Rates

**Use case:** Early in an outbreak, minimal data

```json
{
  "transmission_edges": [
    {
      "source": "susceptible",
      "target": "exposed",
      "data": {
        "transmission_rate": 0.3,
        "variance_params": {
          "has_variance": true,
          "distribution_type": "UNIFORM",
          "min": 0.2,
          "max": 0.4
        }
      }
    },
    {
      "source": "exposed",
      "target": "infected",
      "data": {
        "transmission_rate": 0.2,
        "variance_params": {
          "has_variance": true,
          "distribution_type": "UNIFORM",
          "min": 0.15,
          "max": 0.25
        }
      }
    }
  ]
}
```

### Pattern 2: Intervention Adherence Uncertainty

**Use case:** Modeling compliance with public health measures

```json
{
  "interventions": [
    {
      "id": "social_isolation",
      "adherence_min": 50.0,
      "transmission_percentage": 50.0,
      "start_date": "2025-11-01",
      "variance_params": [
        {
          "has_variance": true,
          "distribution_type": "BETA",
          "field_name": "adherence_min",
          "alpha": 3.0,
          "beta": 3.0
        }
      ]
    }
  ]
}
```

**Note:** Beta(3,3) produces a bell-shaped distribution on [0, 1], reflecting that adherence is usually moderate (not 0% or 100%).

### Pattern 3: Mixed Distribution Types

**Use case:** Different parameters have different uncertainty profiles

```json
{
  "transmission_edges": [
    {
      "source": "susceptible",
      "target": "infected",
      "data": {
        "transmission_rate": 0.3,
        "variance_params": {
          "has_variance": true,
          "distribution_type": "NORMAL",
          "mean": 0.3,
          "std": 0.05
        }
      }
    },
    {
      "source": "infected",
      "target": "recovered",
      "data": {
        "transmission_rate": 0.1,
        "variance_params": {
          "has_variance": true,
          "distribution_type": "LOGNORMAL",
          "mean": -2.3,
          "sigma": 0.2
        }
      }
    }
  ]
}
```

## Computational Performance

### Runtime Scaling

**Deterministic mode:** O(1) — single simulation run

**Uncertainty mode:** O(N) — N parallel simulations

**Typical runtimes (M1 Mac, 2 workers):**
- 30 simulations: 2-5 minutes
- 100 simulations: 10-20 minutes
- 500 simulations: 1-2 hours

**Factors affecting runtime:**
- Number of simulations (n_simulations)
- Model complexity (compartments, regions, age groups)
- Simulation duration (time_steps)
- Intervention complexity

### Parallelization

The framework runs simulations in **parallel** using multiprocessing:

```python
# From run_simulation.py
top_level_workers = 2  # with vs. without interventions
low_level_workers = 2  # parallel UQ runs within each
```

**Total parallelism:** Up to `top_level_workers * low_level_workers` cores used.

**Tuning:** On machines with many cores, increase `low_level_workers` in the code for faster UQ runs.

## Best Practices

### ✅ Do

- **Start with deterministic** to validate your model before adding uncertainty
- **Use uniform distributions** when you have minimal prior information
- **Use normal/lognormal** when you have mean and variance estimates from data
- **Vary transmission rates** — they're almost always uncertain
- **Vary intervention adherence** — compliance is highly unpredictable
- **Run 30+ simulations** for reportable results
- **Report CIs alongside point estimates** — never give a single number without uncertainty
- **Document parameter sources** — where did your ranges come from?
- **Check CI widths** — unreasonably wide or narrow suggests bad priors

### ❌ Don't

- **Don't vary everything** — prioritize parameters you're most uncertain about
- **Don't use tiny ranges** (e.g., min=0.299, max=0.301) — be honest about uncertainty
- **Don't ignore literature** — use published estimates when available
- **Don't report only the median** — CIs are the point of UQ
- **Don't assume normality** without justification — many parameters are skewed
- **Don't run <10 simulations** — CIs will be unreliable
- **Don't vary structural parameters** (e.g., compartment list, number of regions)

## Troubleshooting

### CIs Are Too Wide

**Symptoms:**
- Output ranges span orders of magnitude
- CIs include implausible values (e.g., negative populations)

**Possible causes:**

1. **Parameter ranges too broad**
   ```json
   {"min": 0.01, "max": 10.0}  // 1000-fold range!
   ```
   **Fix:** Narrow ranges using literature

2. **Too many varying parameters**
   - Uncertainty compounds multiplicatively
   **Fix:** Vary only high-sensitivity parameters

3. **Long simulation horizon**
   - Uncertainty grows exponentially with time
   **Fix:** Accept this (it's reality) or shorten time horizon

### CIs Are Too Narrow

**Symptoms:**
- CIs barely wider than a deterministic run
- Unrealistic confidence

**Possible causes:**

1. **Parameter ranges too tight**
   ```json
   {"min": 0.29, "max": 0.31}  // Only 7% variation
   ```
   **Fix:** Widen ranges to reflect true uncertainty

2. **Only varying low-sensitivity parameters**
   - E.g., varying recovery rate but not transmission rate
   **Fix:** Identify high-sensitivity parameters via sensitivity analysis

### NaN or Inf in Output

**Symptoms:**
- Output contains `NaN` or `inf` values
- Some simulation runs crash

**Possible causes:**

1. **Distribution produces negative values**
   ```json
   {"distribution_type": "NORMAL", "mean": 0.1, "std": 0.2}
   // Can sample negative rates!
   ```
   **Fix:** Use lognormal or clamp to positive values

2. **Extreme parameter combinations**
   - Very high beta + very low gamma → explosion
   **Fix:** Ensure parameter ranges are biologically plausible

3. **Model instability**
   - Some parameter combinations violate model assumptions
   **Fix:** Add validation in `derivative()` or narrow ranges

### Slow Performance

**Symptoms:**
- UQ runs take hours

**Possible causes:**

1. **Too many simulations**
   **Fix:** Start with n_simulations=30, increase only if needed

2. **Complex model**
   - Many regions, age groups, compartments
   **Fix:** Simplify for exploratory UQ, then scale up

3. **Low parallelism**
   **Fix:** Increase `low_level_workers` in `run_simulation.py`

## Advanced Topics

### Custom Distributions

To add a new distribution (requires code modification):

1. **Add LHS function in `helpers.py`:**
   ```python
   def LHS_gamma(shape, scale, uniform_samples):
       return stats.gamma.ppf(uniform_samples, shape, scale=scale)
   ```

2. **Update `generate_LHS_samples()`:**
   ```python
   elif dist == "gamma":
       samples = LHS_gamma(cfg["shape"], cfg["scale"], base)
   ```

3. **Use in config:**
   ```json
   {
     "distribution_type": "GAMMA",
     "shape": 2.0,
     "scale": 1.5
   }
   ```

### Varying Initial Conditions

Currently, `variance_params` only supports transmission edges and interventions. To vary initial infections:

**Workaround:** Run multiple configs with different `infected_population` values and aggregate externally.

**Future enhancement:** Add `variance_params` support for `case_file` fields.

### Covarying Parameters

LHS assumes **independence** between parameters. To model correlation:

**Workaround:** Use external tools (e.g., Cholesky decomposition) to generate correlated samples, then pass via custom parameter sets.

**Future enhancement:** Add correlation matrix support to LHS.

## Related Documentation

- **[INTERVENTIONS.md](./INTERVENTIONS.md)** — Varying intervention effectiveness
- **[DEVELOPING_MODELS.md](./DEVELOPING_MODELS.md)** — Model development guide
- **[compartment/run_simulation.py](../compartment/run_simulation.py)** — UQ orchestration code
- **[compartment/helpers.py](../compartment/helpers.py)** — LHS implementation (`generate_LHS_samples`)
- **[compartment/batch_simulation_manager.py](../compartment/batch_simulation_manager.py)** — Parallel simulation runner

## References

### Latin Hypercube Sampling

- **McKay et al. (1979).** "A comparison of three methods for selecting values of input variables in the analysis of output from a computer code." *Technometrics* 21(2): 239-245.
  - Original LHS paper

- **Iman & Conover (1982).** "A distribution-free approach to inducing rank correlation among input variables." *Communications in Statistics* 11(3): 311-334.
  - LHS with correlation

### Uncertainty Quantification in Epidemiology

- **Ferguson et al. (2020).** "Report 9: Impact of non-pharmaceutical interventions (NPIs) to reduce COVID-19 mortality and healthcare demand." *Imperial College COVID-19 Response Team*.
  - Influential UQ study with wide CIs

- **Jewell et al. (2020).** "Predictive mathematical models of the COVID-19 pandemic: Underlying principles and value of projections." *JAMA* 323(19): 1893-1894.
  - Discussion of model uncertainty

- **Holmdahl & Buckee (2020).** "Wrong but Useful — What Covid-19 Epidemiologic Models Can and Cannot Tell Us." *New England Journal of Medicine* 383(4): 303-305.
  - Limitations and appropriate use of uncertain models

### Distribution Selection

- **Vose (2008).** *Risk Analysis: A Quantitative Guide* (3rd ed.). John Wiley & Sons.
  - Comprehensive guide to probability distributions for uncertainty

---

**Last Updated:** May 20, 2026  
**Version:** 0.1.9
