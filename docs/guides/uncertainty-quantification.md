# Uncertainty Quantification in the Pandemic Simulator

This document explains how to quantify and communicate uncertainty in compartmental disease models using the Pandemic Simulator. It covers the available run modes, the two methods for estimating uncertainty, what the framework currently supports, how to declare variance, and how to read the output.

## Table of Contents

- [Why Uncertainty Matters](#why-uncertainty-matters)
- [Run Modes](#run-modes)
- [Two Ways to Estimate Uncertainty](#two-ways-to-estimate-uncertainty)
  - [1. Parameter Uncertainty via Latin Hypercube Sampling (LHS)](#1-parameter-uncertainty-via-latin-hypercube-sampling-lhs)
  - [2. Stochastic Models](#2-stochastic-models)
- [What the Simulator Currently Supports](#what-the-simulator-currently-supports)
- [How the Simulator Chooses a Run Mode](#how-the-simulator-chooses-a-run-mode)
- [Declaring Variance](#declaring-variance)
  - [On Transmission Edges](#on-transmission-edges)
  - [On Interventions](#on-interventions)
  - [On Disease Parameters](#on-disease-parameters)
  - [Enabling a Stochastic Model](#enabling-a-stochastic-model)
- [Number of Runs](#number-of-runs)
- [Understanding Output](#understanding-output)
- [Related Documentation](#related-documentation)
- [References](#references)

## Why Uncertainty Matters

Epidemiologists usually want more than a single "best guess" trajectory, because disease dynamics are inherently variable and model parameters are rarely known exactly. Reporting a median together with a plausible range communicates that uncertainty honestly.

The Pandemic Simulator offers **three run modes**:

- **Deterministic** — runs the model **once** and returns a single trajectory per compartment.
- **Parameter uncertainty** — runs the model **multiple times** (30 by default), each time drawing parameter values via Latin Hypercube Sampling, and reports a median with a 95% interval.
- **Stochastic** — runs a **stochastic model multiple times** (30 by default) and reports a median with a 95% interval.

## Run Modes

<table>
  <thead>
    <tr><th>Mode</th><th>Runs</th><th>Returns</th><th>Speed</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>Deterministic</strong></td><td>1</td><td>A single value per compartment per timestep</td><td>Fast (seconds to minutes)</td></tr>
    <tr><td><strong>Parameter uncertainty</strong></td><td>30 (default)</td><td>Median + 95% interval per compartment per timestep</td><td>Slower (minutes to hours)</td></tr>
    <tr><td><strong>Stochastic</strong></td><td>30 (default)</td><td>Median + 95% interval per compartment per timestep</td><td>Slower (minutes to hours)</td></tr>
  </tbody>
</table>

**Note:** The multi-run modes still run **with and without interventions** in parallel (a control run), so you get uncertainty bands for both scenarios.

## Two Ways to Estimate Uncertainty

There are two ways to estimate uncertainty in the Pandemic Simulator:

1. Use the built-in, efficient functionality that applies **Latin Hypercube Sampling (LHS)** to model parameters to produce a 95% simulation-based interval.
2. Use a **stochastic model**.

### 1. Parameter Uncertainty via Latin Hypercube Sampling (LHS)

Latin Hypercube Sampling is an efficient, stratified sampling method for exploring how uncertain inputs affect a model's output. It divides each parameter's range into equally probable intervals, draws one sample from each interval, and then combines the samples across parameters. Compared with simple random sampling, this guarantees that the full range of every parameter is covered while using far fewer runs — which is why it is widely used for uncertainty and sensitivity analysis of epidemic models.

In practice, the simulator draws a set of parameter values for each run, simulates a trajectory for each set, and summarizes the resulting spread as a median and a 95% interval.

**References:** LHS was introduced by [McKay et al. 1979](https://doi.org/10.1080/00401706.1979.10489755). Its use for uncertainty and sensitivity analysis of epidemic and immunological models is described by [Marino et al. 2008](https://doi.org/10.1016/j.jtbi.2008.04.011), and it remains in common use — for example, [Majeed et al. 2022](https://doi.org/10.3389/fpubh.2022.1086849) apply LHS to analyze a co-circulating influenza/COVID-19 model.

### 2. Stochastic Models

A stochastic model builds randomness directly into the disease dynamics: instead of moving a fixed fraction of the population between compartments at each step, transitions occur as random events. Running the same model many times therefore produces a spread of trajectories, which captures the intrinsic variability of transmission — especially important for small populations and early-outbreak dynamics, where chance events can strongly influence the outcome.

**References:** For an introduction to formulating and simulating stochastic epidemic models, see [Allen 2017](https://doi.org/10.1016/j.idm.2017.03.001); for a broader survey, see [Britton 2010](https://doi.org/10.1016/j.mbs.2010.01.006).

## What the Simulator Currently Supports

- **Option 1 — Parameter uncertainty (LHS).** LHS is applied to input parameters for both the model and its interventions. **Only a uniform distribution is currently supported.**
- **Option 2 — Stochastic model.** The modeler can supply a stochastic model.
- **Option 3 — Combined approach.** Both a stochastic model *and* LHS parameter uncertainty can be used together. This may be overkill and can take a long time to run, but it is an available option.

## How the Simulator Chooses a Run Mode

You do **not** need to select a run mode manually — the simulator detects it automatically from the model and its parameters (any `run_mode` value in the frontend config is intentionally ignored). The logic is:

1. If the model class declares `STOCHASTIC = True` → **stochastic** (always runs the model's configured number of trajectories; any variance parameters are spread across those same runs rather than adding more).
2. Otherwise, if **any** variance parameter is declared (on an edge, intervention, or disease parameter) → **parameter uncertainty**.
3. Otherwise → **deterministic**.

## Declaring Variance

Variance can be added in **three places**: on a transmission edge, as part of an intervention, or as part of a disease parameter. For transmission edges and interventions, the ability to vary parameters is built into the framework — **no extra code is required**, you just add `variance_params` in the config. For disease parameters, variance is controlled by the `enable_variance` option in the model schema.

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
            "distribution_type": "UNIFORM",
            "min": 0.17,
            "max": 0.23
          }
        }
      }
    ]
  }
}
```

**Effect:** Each simulation run draws new values for these rates from the specified (uniform) range.

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

**Effect:** Each run draws new `adherence_min` and `transmission_percentage` values.

**Important:** For interventions, you must specify `field_name` to indicate which parameter varies.

### On Disease Parameters

When you declare a disease parameter in your model schema, it exposes an `enable_variance` option. It defaults to `True`, which lets a user vary that parameter in the UI. Set it to `False` when you do **not** want the parameter to be varied (for example, an integer count field):

```python
@classmethod
def define_parameters(cls, schema):
    # Variance allowed (default) — users can vary this parameter
    schema.add_disease_parameter(
        name="incubation_period",
        default=5.0,
        min_value=2.0,
        max_value=10.0,
        enable_variance=True,
    )

    # Variance disabled — hides the variance option for this parameter
    schema.add_disease_parameter(
        name="num_runs",
        default=30,
        enable_variance=False,
    )
```

### Enabling a Stochastic Model

For a stochastic model, the framework cannot infer intent from parameters alone, so the modeler must explicitly set the class-level flag `STOCHASTIC = True`. This tells the simulator the model must be run multiple times:

```python
class MyStochasticModel(CompartmentModel):
    STOCHASTIC = True  # run multiple trajectories

    @classmethod
    def define_parameters(cls, schema):
        ...
        # Optional: change the default trajectory count (see below)
        schema.set_num_runs(default=30, min_value=1, max_value=100)
```

## Number of Runs

Both multi-run modes default to **30** runs. You can change this:

- **Any run:** set `n_simulations` in the config file to explore how the model behaves with a different number of runs.

```json
{
  "n_simulations": 50
}
```

- **Stochastic models:** call `set_num_runs()` in `define_parameters()` to change the default (30) baked into the model:

```python
schema.set_num_runs(default=50, min_value=1, max_value=100)
```

**Trade-off:** more runs give narrower, more stable intervals but longer runtime.

**Guidelines:**

- **10-30:** Quick exploration, rough interval estimates
- **30-100:** Standard for reporting (default: 30)
- **100-500:** High-confidence intervals, sensitivity analysis
- **500+:** Publication-quality, very tight intervals

## Understanding Output

To see how a model with parameter uncertainty runs locally, add a variance range into the `example-config.json` file (see [Declaring Variance](#declaring-variance) above). Visualizations for **parameter uncertainty** and **stochastic** models look the **same** in the Pandemic Simulator: both display the **median** and a **95% simulation-based interval**.

### Deterministic Output

```json
{
  "admin_zones": [
    {
      "time_series": [
        { "date": "2025-11-18", "S": 999500, "I": 500, "R": 0 },
        { "date": "2025-11-19", "S": 998950, "I": 550, "R": 0 }
      ]
    }
  ]
}
```

A single value per compartment per timestep.

### Multi-Run Output (Parameter Uncertainty or Stochastic)

```json
{
  "admin_zones": [
    {
      "time_series": [
        {
          "date": "2025-11-18",
          "S": { "mean": 999500, "lower": 999400, "upper": 999600 },
          "I": { "mean": 500, "lower": 420, "upper": 580 }
        },
        {
          "date": "2025-11-19",
          "S": { "mean": 998950, "lower": 998200, "upper": 999500 },
          "I": { "mean": 550, "lower": 450, "upper": 680 }
        }
      ]
    }
  ]
}
```

Three values per compartment per timestep:

- **mean:** Median across all runs (50th percentile)
- **lower:** Lower bound of the 95% interval (2.5th percentile)
- **upper:** Upper bound of the 95% interval (97.5th percentile)

**Interpretation:**

- "On day 19, we expect **550 infections** (median)"
- "We're 95% confident the value lies between **450 and 680**"

The framework uses a **95% simulation-based interval** by default: the `lower` and `upper` bounds are the 2.5th and 97.5th percentiles of the simulated runs.

> **Visualizing bands:** `python tools/view_results.py results/<output>.json` plots the `mean` line and shades the `lower`–`upper` band for each compartment, with the intervention and control runs side by side. See [tools/README.md](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tools/README.md).

## Related Documentation

- **[interventions.md](./interventions.md)** — Varying intervention effectiveness
- **[tools/view_results.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tools/view_results.py)** — Local results viewer; shades the mean/lower/upper uncertainty bands from multi-run output
- **[compartment/run_simulation.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/run_simulation.py)** — UQ orchestration code
- **[compartment/helpers.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/helpers.py)** — LHS implementation (`generate_LHS_samples`) and run-mode resolution (`resolve_run_mode`)
- **[compartment/batch_simulation_manager.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/batch_simulation_manager.py)** — Parallel simulation runner

## References

### Latin Hypercube Sampling

- **McKay, Beckman & Conover (1979).** ["A comparison of three methods for selecting values of input variables in the analysis of output from a computer code."](https://doi.org/10.1080/00401706.1979.10489755) *Technometrics* 21(2): 239-245.
    - Original LHS paper
- **Iman & Conover (1982).** "A distribution-free approach to inducing rank correlation among input variables." *Communications in Statistics* 11(3): 311-334.
    - LHS with correlation
- **Marino, Hogue, Ray & Kirschner (2008).** ["A methodology for performing global uncertainty and sensitivity analysis in systems biology."](https://doi.org/10.1016/j.jtbi.2008.04.011) *Journal of Theoretical Biology* 254(1): 178-196.
    - LHS/PRCC methodology for epidemiological and immunological models
- **Majeed et al. (2022).** ["Mitigating co-circulation of seasonal influenza and COVID-19 pandemic in the presence of vaccination: A mathematical modeling approach."](https://doi.org/10.3389/fpubh.2022.1086849) *Frontiers in Public Health* 10: 1086849.
    - Recent epidemiological application of LHS (with PRCC sensitivity analysis)

### Stochastic Epidemic Models

- **Allen (2017).** ["A primer on stochastic epidemic models: Formulation, numerical simulation, and analysis."](https://doi.org/10.1016/j.idm.2017.03.001) *Infectious Disease Modelling* 2(2): 128-142.
    - Introduction to formulating and simulating stochastic epidemic models
- **Britton (2010).** ["Stochastic epidemic models: A survey."](https://doi.org/10.1016/j.mbs.2010.01.006) *Mathematical Biosciences* 225(1): 24-35.
    - Survey of stochastic epidemic model formulations

### Uncertainty Quantification in Epidemiology

- **Ferguson et al. (2020).** "Report 9: Impact of non-pharmaceutical interventions (NPIs) to reduce COVID-19 mortality and healthcare demand." *Imperial College COVID-19 Response Team*.
    - Influential UQ study with wide intervals
- **Jewell, Lewnard & Jewell (2020).** "Predictive mathematical models of the COVID-19 pandemic: Underlying principles and value of projections." *JAMA* 323(19): 1893-1894.
    - Discussion of model uncertainty
- **Holmdahl & Buckee (2020).** "Wrong but Useful — What Covid-19 Epidemiologic Models Can and Cannot Tell Us." *New England Journal of Medicine* 383(4): 303-305.
    - Limitations and appropriate use of uncertain models

---

**Last Updated:** August 11, 2026  
**Version:** 0.2.0
