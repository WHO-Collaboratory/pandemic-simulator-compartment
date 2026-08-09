# Uncertainty Quantification in the Pandemic Simulator

This document explains how to quantify and communicate parameter uncertainty in compartmental disease models using the Pandemic Simulator's built-in parameter uncertainty quantification (UQ) system.


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
- Use for: final results, communicating parameter uncertainty, policy analysis

**Note:** Both modes still run **with and without interventions** in parallel (control run), so you get parameter uncertainty bands for both scenarios.

## Latin Hypercube Sampling (LHS)

The framework uses **Latin Hypercube Sampling** to efficiently explore the parameter space.


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

> **Visualizing bands:** `python tools/view_results.py results/<output>.json` plots the `mean` line and shades the `lower`–`upper` band for each compartment, with the intervention and control runs side by side. See [tools/README.md](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tools/README.md).

### Simulation-based interval (CI)

The framework uses a **95% simulation-based interval** by default:
- **lower:** 2.5th percentile of simulation results
- **upper:** 97.5th percentile of simulation results


### Parallelization

The framework runs simulations in **parallel** using multiprocessing:

```python
# From run_simulation.py
top_level_workers = 2  # with vs. without interventions
low_level_workers = 2  # parallel UQ runs within each
```

**Total parallelism:** Up to `top_level_workers * low_level_workers` cores used.

**Tuning:** On machines with many cores, increase `low_level_workers` in the code for faster UQ runs.


## Related Documentation

- **[INTERVENTIONS.md](./interventions.md)** — Varying intervention effectiveness
- **[DEVELOPING_MODELS.md](./developing-models.md)** — Model development guide
- **[tools/view_results.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tools/view_results.py)** — Local results viewer; shades the mean/lower/upper parameter uncertainty bands from UNCERTAINTY output
- **[compartment/run_simulation.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/run_simulation.py)** — UQ orchestration code
- **[compartment/helpers.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/helpers.py)** — LHS implementation (`generate_LHS_samples`)
- **[compartment/batch_simulation_manager.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/batch_simulation_manager.py)** — Parallel simulation runner

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

---

**Last Updated:** May 20, 2026  
**Version:** 0.1.9
