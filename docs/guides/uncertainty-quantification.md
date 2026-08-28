# Uncertainty Quantification in the Pandemic Simulator

This document explains how to quantify and communicate uncertainty in compartmental disease models using the Pandemic Simulator. It covers the available run modes, the two methods for estimating uncertainty, what the framework currently supports, how to declare variance, and how to read the output.

## Table of Contents

- [Overview](#overview)
- [Run Modes](#run-modes)
- [Two Ways to Estimate Uncertainty](#two-ways-to-estimate-uncertainty)
  - [1. Parameter Uncertainty via Latin Hypercube Sampling (LHS)](#1-parameter-uncertainty-via-latin-hypercube-sampling-lhs)
  - [2. Stochastic Models](#2-stochastic-models)
- [What the Simulator Currently Supports](#what-the-simulator-currently-supports)
- [How the Simulator Chooses a Run Mode](#how-the-simulator-chooses-a-run-mode)
- [Declaring Variance](#declaring-variance)
  - [On Transmission Edges](#on-transmission-edges)
  - [On Interventions](#on-interventions)
  - [On Custom Parameters](#on-custom-parameters)
  - [Enabling a Stochastic Model](#enabling-a-stochastic-model)
- [Number of Runs](#number-of-runs)
- [Understanding Output](#understanding-output)
  - [Overall File Structure](#overall-file-structure)
  - [Deterministic Output](#deterministic-output)
  - [Multi-Run Output (Parameter Uncertainty or Stochastic)](#multi-run-output-parameter-uncertainty-or-stochastic)
- [Related Documentation](#related-documentation)
- [References](#references)
  - [Latin Hypercube Sampling](#latin-hypercube-sampling)
  - [Stochastic Epidemic Models](#stochastic-epidemic-models)
  - [Uncertainty Quantification in Epidemiology](#uncertainty-quantification-in-epidemiology)



## Overview

Epidemiologists usually want more than a single "best guess" trajectory, because disease dynamics are inherently variable and model parameters are rarely known exactly. Reporting a measure of central tendency together with a range communicates the level of uncertainty.

## Run Modes

The Pandemic Simulator offers **three run modes**:

- **Deterministic** — runs the model **once** and returns a single trajectory per compartment.
- **Parameter uncertainty** — runs the model **multiple times** (30 by default), each time drawing parameter values via Latin Hypercube Sampling, and reports a median with a 95% simulation-based interval.
- **Stochastic** — runs a **stochastic model multiple times** (30 by default) and reports a median with a 95% simulation-based interval.

> Throughout this guide, the **95% simulation-based interval** refers to the band between the 2.5th and 97.5th percentiles of the simulated runs, with the median as the central line. See [Understanding Output](#understanding-output) for how these bounds appear in the results.



## Two Ways to Estimate Uncertainty



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
- **Option 3 — Combined approach.** A stochastic model and LHS parameter uncertainty can be combined. This is supported, but combining the two sources of variability tends to widen the simulation-based interval.



## How the Simulator Chooses a Run Mode

You do **not** need to select a run mode manually — the simulator detects it automatically from the model and its parameters.

Variance can be added in **three places**. Transmission edges and interventions support it out of the box (config only), while additional parameters must opt in from the model code:


| Place             | Example                                                                                      | How parameter uncertainty is enabled                                                      |
| ----------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Transmission edge | The `beta` transmission rate (S→I) in `example_parameter_uncertainty_declarative_model`      | Built-in, no code changes                                                                 |
| Intervention      | `my_intervention`'s reduction of `beta` in `example_parameter_uncertainty_declarative_model` | Built-in, no code changes                                                                 |
| Additional parameter | The `asymptomatic_fraction` parameter in `example_stochastic_model`                          | Requires code — set `enable_variance=True` in the model schema. Built-in, no code changes |


1. If the model class declares `STOCHASTIC = True` → **stochastic** (always runs the model's configured number of trajectories; any variance parameters are spread across those same runs rather than adding more).
2. Otherwise, if **any** variance parameter is declared (on an edge, intervention, or additional parameter) → **parameter uncertainty**.
3. Otherwise → **deterministic**.



## Declaring Variance

To test parameter uncertainty locally, declare the variance in the config file you run the simulation with. There are three places to declare it, and each uses a different shape. Every example below is taken from a model in this repository.

**LHS currently supports only the uniform distribution, so every variance declaration must supply a** `min` **and a** `max` **value.**

### On Transmission Edges

Set `has_variance` to `true`. From `[example_parameter_uncertainty_declarative_model/example-config.json](../../compartment/models/example_parameter_uncertainty_declarative_model/example-config.json)`, where the transmission rate varies but the recovery period does not:

```json
"TransmissionEdges": {
  "items": [
    {
      "transmission_edge": {
        "source": "susceptible",
        "target": "infected",
        "value_type": "RATE"
      },
      "value": 0.3,
      "FieldConfigs": {
        "items": [
          {
            "field_key": "value",
            "has_variance": true,
            "distribution_type": "UNIFORM",
            "disease_param": "BETA",
            "min": 0.2,
            "max": 0.4
          }
        ]
      }
    },
    {
      "transmission_edge": {
        "source": "infected",
        "target": "recovered",
        "value_type": "DAYS"
      },
      "value": 10.0,
      "FieldConfigs": {
        "items": [
          {
            "field_key": "value",
            "has_variance": false,
            "distribution_type": "UNIFORM",
            "disease_param": "GAMMA"
          }
        ]
      }
    }
  ]
}
```

`disease_param` names the schema variable the edge maps to — `BETA` resolves to the `beta` declared via `variable_name` in `add_transmission_parameter()`.

**Effect:** Each run draws a new `beta` from a uniform distribution over 0.2–0.4. `gamma` stays fixed at 10 days.

### On Interventions

Interventions use `field_key` to name which intervention field varies. From the same config file:

```json
"Interventions": {
  "items": [
    {
      "Intervention": { "name": "MY_INTERVENTION", "display_name": "My intervention" },
      "adherence_min": 50.0,
      "transmission_percentage": 50.0,
      "start_date": "2026-03-01",
      "end_date": "2026-06-01",
      "FieldConfigs": {
        "items": [
          {
            "field_key": "adherence_min",
            "has_variance": true,
            "distribution_type": "UNIFORM",
            "default": 50.0,
            "min": 40.0,
            "max": 60.0
          },
          {
            "field_key": "transmission_percentage",
            "has_variance": true,
            "distribution_type": "UNIFORM",
            "default": 50.0,
            "min": 45.0,
            "max": 55.0
          }
        ]
      }
    }
  ]
}
```

Three values per compartment per timestep:

- **median:** Median across all simulation runs (50th percentile)
- **lower:** Lower bound of 95% CI (2.5th percentile)
- **upper:** Upper bound of 95% CI (97.5th percentile)

**Interpretation:**

- "On day 19, we expect **550 infections** (median)"
- "We're 95% confident the true value is between **450 and 680**"

> **Visualizing bands:** `python tools/view_results.py results/<output>.json` plots the `median` line and shades the `lower`–`upper` band for each compartment, with the intervention and control runs side by side. See [tools/README.md](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tools/README.md).

### Confidence Interval (CI)

The framework uses a **95% confidence interval** by default:
- **lower:** 2.5th percentile of simulation results
- **upper:** 97.5th percentile of simulation results

This means:

**Important:** `field_key` is required — it identifies which field varies. Each field you want to vary needs its own entry in `items[]`, as shown here for both `adherence_min` and `transmission_percentage`.

### On Custom Parameters

Custom parameters take two steps: the model must allow variance, and the config must request it.

**1. Allow it in the schema.** `add_parameter()` accepts `enable_variance`, which defaults to `True`. Leave it `True` to allow the parameter to be varied; set it to `False` for parameters that should never be sampled. From `[example_parameter_uncertainty_custom_model/model.py](../../compartment/models/example_parameter_uncertainty_custom_model/model.py)`:

```python
schema.add_parameter(
    name="ramp_up_days",
    label="Intervention Ramp-Up (days)",
    description="Days for the intervention to climb from baseline to full adherence.",
    value_type=ValueType.DAYS,
    default=14.0,
    min_value=1.0,
    max_value=180.0,
    unit="days",
    required=False,
    enable_variance=True,   # allow this parameter to be sampled
)
```

**2. Request it in the config.** To add parameter variance you must alter the `variance_params` list inside the `Disease` block. Using `ramp_up_days` from `[example_parameter_uncertainty_custom_model](../../compartment/models/example_parameter_uncertainty_custom_model/model.py)`:

```json
"Disease": {
  "disease_type": "example_parameter_uncertainty_custom",
  "ramp_up_days": 14,
  "ramp_down_days": 21,
  "variance_params": [
    { "param": "ramp_up_days", "dist": "uniform", "min": 7, "max": 21 }
  ]
}
```

**Effect:** Each run draws a new `ramp_up_days` between 7 and 21.

> Note: In the example, `ramp_up_days` already ships with `enable_variance=True` (step 1), so the config above takes effect as-is. If a parameter has `enable_variance=False`, you must change it in `schema.add_parameter()` to `True` first.



### Enabling a Stochastic Model

For a stochastic model, the framework cannot infer intent from parameters alone, so the modeler must explicitly set the class-level flag `STOCHASTIC = True`. This tells the simulator the model must be run multiple times:

```python
from compartment.model import Model, ValueType


class ExampleStochasticModel(Model):
    STOCHASTIC = True  # run multiple trajectories

    @classmethod
    def define_parameters(cls, schema):
        ...
        # Optional: change the default trajectory count (see below)
        schema.add_parameter(
            name="num_runs",
            label="Number of Runs",
            description="Number of stochastic trajectories to simulate.",
            value_type=ValueType.INTEGER,
            default=30,
            min_value=5,
            max_value=50,
            enable_variance=False,
        )
```



## Number of Runs

Both parameter uncertainty and stochastic run modes default to **30** runs. You can change this:

- **Stochastic models:** change the default baked into the model by adjusting the `num_runs` parameter's `default` in `define_parameters()` — the same `add_parameter()` call shown in [Enabling a Stochastic Model](#enabling-a-stochastic-model) above (e.g. set `default=50`).
- **Any run (local only):** set `n_simulations` in your local config file to explore how the model behaves with a different number of runs. This override only applies when running locally — in the hosted Pandemic Simulator you **cannot** change the number of runs for parameter uncertainty; it is fixed at 30.

For example, [`example_parameter_uncertainty_declarative_model/example-config.json`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/models/example_parameter_uncertainty_declarative_model/example-config.json) already sets `n_simulations` at the top level, so it runs 20 simulations instead of the default 30:

```json
{
  "n_simulations": 20,
  "Disease": {
    "disease_type": "example_parameter_uncertainty_declarative"
  },
  "start_date": "2026-01-01",
  "end_date": "2026-12-31"
}
```



## Understanding Output

To see how a model with parameter uncertainty runs locally, add a variance range into the `example-config.json` file (see [Declaring Variance](#declaring-variance) above). Visualizations for **parameter uncertainty** and **stochastic** models look the **same** in the Pandemic Simulator: both display the **median** and a **95% simulation-based interval**.

### Overall File Structure

Every results file is a **JSON array of two runs** — the run with interventions and the control run without them:

```json
[
  { "control_run": false, "admin_zones": [ ... ], "compartment_deltas": { ... } },
  { "control_run": true,  "admin_zones": [ ... ], "compartment_deltas": { ... } }
]
```

Each run also carries `start_date`, `end_date`, `time_steps`, `interventions`, `parent_admin_total` (the whole-population series), and `compartment_deltas` (cumulative totals per compartment). The run mode changes only the shape of the values inside `time_series`.

### Deterministic Output

Each compartment holds an object of demographic keys. `age_all` carries the population-wide value; a model with declared age groups also lists one key per group. For example, [`example_parameter_uncertainty_custom_model`](../../compartment/models/example_parameter_uncertainty_custom_model/model.py) declares five bands, so a **deterministic** run of that model produces keys like this:

```json
"time_series": [
  {
    "date": "2026-01-01",
    "S": { "age_0_4": 0, "age_5_17": 0, "age_18_49": 0, "age_50_64": 0, "age_65_plus": 0, "age_all": 999900.0 },
    "I": { "age_0_4": 0, "age_5_17": 0, "age_18_49": 0, "age_50_64": 0, "age_65_plus": 0, "age_all": 100.0 },
    "R": { "age_0_4": 0, "age_5_17": 0, "age_18_49": 0, "age_50_64": 0, "age_65_plus": 0, "age_all": 0.0 }
  }
]
```

A model with no age stratification produces `age_all` alone — for example, a deterministic run of [`example_parameter_uncertainty_declarative_model`](../../compartment/models/example_parameter_uncertainty_declarative_model/model.py):

```json
{ "date": "2026-01-01", "S": { "age_all": 999900.0 }, "I": { "age_all": 100.0 } }
```

One value per compartment per timestep, always nested under a demographic key.

### Multi-Run Output (Parameter Uncertainty or Stochastic)

Each compartment holds a summary across runs instead. Note that the demographic nesting is **replaced** — multi-run output reports population-wide values only, even for age-stratified models such as `example_parameter_uncertainty_custom_model`. Both uncertainty examples ship with `has_variance` flags, so their local runs use this shape rather than the per-age keys shown above:

```json
"time_series": [
  {
    "date": "2026-01-01",
    "S": { "mean": 999900.0, "lower": 999900.0, "upper": 999900.0 },
    "I": { "mean": 100.0, "lower": 100.0, "upper": 100.0 }
  },
  {
    "date": "2026-01-02",
    "S": { "mean": 999871.2, "lower": 999845.6, "upper": 999889.4 },
    "I": { "mean": 126.9, "lower": 108.7, "upper": 152.1 }
  }
]
```

Three values per compartment per timestep:

- **mean:** Median across all runs (50th percentile)
- **lower:** Lower bound of the 95% simulation-based interval (2.5th percentile)
- **upper:** Upper bound of the 95% simulation-based interval (97.5th percentile)

The framework uses a **95% simulation-based interval** by default: the `lower` and `upper` bounds are the 2.5th and 97.5th percentiles of the simulated runs.

> **Visualizing bands:** `python tools/view_results.py results/<output>.json` plots the `mean` line and shades the `lower`–`upper` band for each compartment, with the intervention and control runs side by side. See [tools/README.md](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tools/README.md).



## Related Documentation

- **[interventions.md](./interventions.md)** — Varying intervention effectiveness
- **[model-integration-documentation.md](./model-integration-documentation.md)** — Model development guide
- **[contact-matrices.md](./contact-matrices.md)** — Age-structured mixing; per-age-group series appear in **deterministic** output only (multi-run output collapses to population-wide summaries)
- **[tools/view_results.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/tools/view_results.py)** — Local results viewer; shades the median/lower/upper uncertainty bands from multi-run output
- **[compartment/run_simulation.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/run_simulation.py)** — UQ orchestration code
- **[compartment/helpers.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/helpers.py)** — LHS implementation (`generate_LHS_samples`) and run-mode resolution (`resolve_run_mode`)
- **[compartment/batch_simulation_manager.py](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/batch_simulation_manager.py)** — Parallel simulation runner



## References



### Latin Hypercube Sampling

- **McKay, Beckman & Conover (1979).** ["A comparison of three methods for selecting values of input variables in the analysis of output from a computer code."](https://doi.org/10.1080/00401706.1979.10489755) *Technometrics* 21(2): 239-245.
  - Original LHS paper
- **Marino, Hogue, Ray & Kirschner (2008).** ["A methodology for performing global uncertainty and sensitivity analysis in systems biology."](https://doi.org/10.1016/j.jtbi.2008.04.011) *Journal of Theoretical Biology* 254(1): 178-196.
  - LHS/PRCC methodology for epidemiological and immunological models
- **Majeed et al. (2022).** ["Mitigating co-circulation of seasonal influenza and COVID-19 pandemic in the presence of vaccination: A mathematical modeling approach."](https://doi.org/10.3389/fpubh.2022.1086849) *Frontiers in Public Health* 10: 1086849.
  - Recent epidemiological application of LHS (with PRCC sensitivity analysis)



### Stochastic Epidemic Models

- **Allen (2017).** ["A primer on stochastic epidemic models: Formulation, numerical simulation, and analysis."](https://doi.org/10.1016/j.idm.2017.03.001) *Infectious Disease Modelling* 2(2): 128-142.
  - Introduction to formulating and simulating stochastic epidemic models
- **Britton (2010).** ["Stochastic epidemic models: A survey."](https://doi.org/10.1016/j.mbs.2010.01.006) *Mathematical Biosciences* 225(1): 24-35.
  - Broader survey of stochastic epidemic models



### Uncertainty Quantification in Epidemiology

- **Ferguson et al. (2020).** ["Report 9: Impact of non-pharmaceutical interventions (NPIs) to reduce COVID-19 mortality and healthcare demand."](https://doi.org/10.25561/77482) *Imperial College COVID-19 Response Team*.
  - Influential UQ study with wide intervals
- **Jewell, Lewnard & Jewell (2020).** ["Predictive mathematical models of the COVID-19 pandemic: Underlying principles and value of projections."](https://doi.org/10.1001/jama.2020.6585) *JAMA* 323(19): 1893-1894.
  - Discussion of model uncertainty
- **Holmdahl & Buckee (2020).** ["Wrong but Useful — What Covid-19 Epidemiologic Models Can and Cannot Tell Us."](https://doi.org/10.1056/NEJMp2016822) *New England Journal of Medicine* 383(4): 303-305.
  - Limitations and appropriate use of uncertain models

---

**Last Updated:** August 27, 2026
