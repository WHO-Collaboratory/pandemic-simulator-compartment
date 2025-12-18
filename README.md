# Pandemic Simulator - Compartmental models in Python

Fast, flexible, and accurate construction of compartmental models to simulate transmission dynamics of respiratory and vector-borne diseases.

## Overview
This repository builds on code developed for the World Health Organization Pandemic Hub’s Pandemic Simulator project. Its purpose is to extend that work by providing accessible, research-based compartmental modeling tools that enable decision makers, epidemiologists, and modelers to assess public health intervention strategies.

## Using This Repository
You can run simulations using this repository by either running commands to the compartment module, or using the provided Dockerfile to build a Docker container with your model.

### To run a command to the compartment module without using Docker

```
# Ensure you are at the root of this project. You should be in the "pandemic-simulator-compartment" folder.
# Initialize a virtual environment using the require packages under uv.
uv venv
# Install the project's dependencies into your virtual environment
uv sync
# Activate the virtual environment which you've created for this project.
source .venv/bin/activate
# Run a covid model using the sample configuration: Madagascar.
python -m compartment.models.covid_jax_model.main --mode local --config_file compartment/models/covid_jax_model/example-config.json --output_file results/example-run.json
```

### To run a command in a container
```
# First, build the Dockerfile while in the root of this project.
docker build . -t compartment 
# TIP: You can build multiple images using different models by passing different values for MODEL_DIR to the build command.

docker run compartment
```

### To build a container with a specific model.
```
# Pass the build-arg parameter to docker build, with your target directory specified.
docker build --build-arg MODEL_DIR=compartment/models/dengue_jax_model/ . -t local-dengue
``` 

### To use your own local config in the reference/ directory and write the output to a custom file in the results/ directory
```
# TIP: The filepaths you reference in the environment variables must be the paths inside the Docker container. You also need to mount those directories to your local filesystem in order to save outputs from the container.

docker run \
  -v $(pwd)/reference:/opt/reference \
  -v $(pwd)/results:/opt/results \
  -e CONFIG_FILE=/opt/reference/my-config-file.json \
  -e OUTPUT_FILE=/opt/results/my-output.json compartment

```

Cloud mode is intended for use in the Pandemic Simulator app, and is not supported for use by the wider community at this time.

## Using Reference Files

Configuration files define the parameters for running simulations. These JSON files specify the disease model, geographic regions, population data, interventions, and simulation settings. Example configuration files are available in the `reference/` directory. We encourage modelers to add their own reference files to this directory when experimenting with models locally.

### Required Fields (covid_jax_model and dengue_jax_model)

All configuration files must include the following shared fields:

- **`admin_unit_0_id`** (string): Identifier for the primary administrative unit (e.g., country code like "DEU")
- **`start_date`** (string): Simulation start date in ISO format (YYYY-MM-DD)
- **`end_date`** (string): Simulation end date in ISO format (YYYY-MM-DD), must be on or after `start_date`
- **`simulation_type`** (string): Must be `"COMPARTMENTAL"`
- **`run_mode`** (string): Either `"DETERMINISTIC"` or `"UNCERTAINTY"` for uncertainty quantification
- **`time_steps`** (integer): Number of time steps to run the simulation (must be > 0)
- **`AdminUnit0`** (object): Primary administrative unit with:
  - `id` (string): Unit identifier
  - `center_lat` (float): Latitude of the unit center (-90 to 90). Used for selecting hemisphere for the dengue model.
- **`Disease`** (object): Disease-specific configuration (see below)
- **`case_file`** (object): Population and geographic data with:
  - `admin_zones` (array): List of administrative zones, each containing:
    - `name` (string): Zone name
    - `center_lat` (float): Latitude (-90 to 90)
    - `center_lon` (float): Longitude (-180.0 to 180.0)
    - `population` (integer): Population count (≥ 0)
    - `infected_population` (float): Initial infected population percentage (0-100)
    - Additional optional fields which are used in the Pandemic Simulator app, but not in local simulations: `id` (string), `admin_code`(string), `admin_iso_code`(string), `admin_level`(integer), `viz_name`(string), `osm_id`(string)

### Optional Fields (covid_jax_model and dengue_jax_model)

- **`admin_unit_1_id`** (string, default: `""`): Secondary administrative unit identifier used by the webapp.
- **`admin_unit_2_id`** (string, default: `""`): Tertiary administrative unit identifier used by the webapp.
- **`AdminUnit1`** (object, optional): Secondary administrative unit (same structure as `AdminUnit0`) a major administrative territory within a country, like a state or prefacture.
- **`AdminUnit2`** (object, optional): Tertiary administrative unit (same structure as `AdminUnit0`) a minor administrative territory within a level 1 admin unit, such as a county or ward.
- **`id`** (string, optional): Simulation identifier, random string used by the web application.
- **`simulation_name`** (string, default: `""`): Friendly name for the simulation, used by the web application for display.
- **`owner`** (string, optional): Owner identifier, used by the web application.
- **`travel_volume`** (object, optional): Travel/mobility parameters:
  - `leaving` (float, default: 0.2): Percentage of population leaving their admin zone, causing mixing across admin zones (0-1, or 0-100 which will be normalized)
- **`case_file.demographics`** (object, optional): Age structure, used for a social mixing function:
  - `age_0_17` (float, default: 25.0): Percentage aged 0-17 (0-100)
  - `age_18_55` (float, default: 50.0): Percentage aged 18-55 (0-100)
  - `age_56_plus` (float, default: 25.0): Percentage aged 56+ (0-100)

### Respiratory Disease Configuration

For respiratory diseases (e.g., COVID-19), the `Disease` object must include:

- **`disease_type`** (string): Must be `"RESPIRATORY"`
- **`compartment_list`** (array of strings): List of disease compartments (e.g., `["S", "E", "I", "R", "H", "D"]` for Susceptible, Exposed, Infected, Recovered, Hospitalized, Dead)
- **`transmission_edges`** (array): List of transitions between compartments used to fill in rates in the jax models, each containing:
  - `source` (string): Source compartment name (e.g., `"susceptible"`, `"infected"`)
  - `target` (string): Target compartment name (e.g., `"exposed"`, `"recovered"`)
  - `data` (object):
    - `transmission_rate` (float): Rate of transition (> 0) from source to target compartment
    - `variance_params` (object, optional): For uncertainty quantification:
      - `has_variance` (boolean): Whether to vary this parameter
      - `distribution_type` (string): `"UNIFORM"`
      - `min` (float): Minimum value for distribution
      - `max` (float): Maximum value for distribution
      - `field_name` (string, optional): Field to vary

**Example:** See `reference/novel-respiratory-basic-example-config.json` for a simple SIR model, or `reference/novel-respiratory-advanced-example-config.json` for a more complex model with multiple compartments and interventions.

### Vector-Borne Disease Configuration

For vector-borne diseases (e.g., Dengue), the `Disease` object must include:

- **`disease_type`** (string): Must be `"VECTOR_BORNE"`
- **`immunity_period`** (integer): Duration of temporary cross-protection between first and second infection in days (≥ 0)

Additionally, `admin_zones` in the `case_file` should include:
- **`seroprevalence`** (float, optional): Percentage of population susceptible to second infection (0-100), defaults to 0
- **`temp_min`** (float, optional): Minimum annual temperature (default: 15)
- **`temp_max`** (float, optional): Maximum annual temperature (default: 30)
- **`temp_mean`** (float, optional): Mean annual temperature (default: 25)

**Example:** See `reference/novel-vector-borne-basic-example-config.json` for a vector-borne disease configuration.

### Interventions

Interventions are optional and can be included for both disease types. Different disease_types have different interventions available.

Interventions can be triggered at a certain date, or when the percentage of population in a particular compartment reaches a certain threshold. If dates and thresholds are both provided, the intervention begins when the first condition which could trigger the intervention is reached, and the other condition trigger is ignored.

 The `interventions` array contains objects with:

- **`id`** (string): Intervention type, must be one of:
  - Respiratory: `"social_isolation"`, `"vaccination"`, `"mask_wearing"`, `"lock_down"`
  - Vector-borne: `"chemical"`, `"physical"`
- **`start_date`** (string, optional): Start date in ISO format (YYYY-MM-DD)
- **`end_date`** (string, optional): End date in ISO format (YYYY-MM-DD)
- **`adherence_min`** (float, optional): Minimum adherence percentage, sets adherence in deterministic simulations and bounds adherence in stochastic simulations.
- **`transmission_percentage`** (float, optional): Percentage reduction in transmission caused by adhering to the intervention.
- **`start_threshold`** (float, optional): Threshold to start intervention
- **`end_threshold`** (float, optional): Threshold to end intervention
- **`variance_params`** (array, optional): List of variance parameters for uncertainty quantification:
  - `has_variance` (boolean): Whether to vary this parameter
  - `distribution_type` (string): `"UNIFORM"` 
  - `field_name` (string): Field to vary (e.g., `"adherence_min"`, `"transmission_percentage"`)
  - `min` (float): Minimum value
  - `max` (float): Maximum value

### Example Files

The `reference/` directory contains example configuration files:

- **`novel-respiratory-basic-example-config.json`**: Simple SIR model with deterministic run mode
- **`novel-respiratory-advanced-example-config.json`**: Complex model with multiple compartments, uncertainty quantification, and interventions
- **`novel-vector-borne-basic-example-config.json`**: Vector-borne disease configuration with temperature parameters and interventions

These files serve as templates for creating your own simulation configurations. All configuration files are validated using Pydantic models defined in `compartment/validation/` to ensure they meet the required structure and constraints.

## Features
Current:

* Simulate respiratory and vector-borne disease dynamics
* Incorporate real-world factors including population mobility, age structure, and intervention strategies
* Run multiple simulations efficiently for uncertainty quantification
* Leverages [JAX](https://github.com/jax-ml/jax) for high-performance, efficient computation

Planned

* Modular modeling framework supporting extensions and variants for diseases.
* Separation of models from simulation runs, and built in support for batch runs of simulations.
* Summary statistics objects.

## Methods

For detail on our methods please refer to our documentation:
* [Respiratory compartmental documentation](https://drive.google.com/file/d/1Ff4gEKu5gu3MuwTdgzRIH1A7jjzZerCj/view?usp=drive_link)
* [Vector-borne compartmental documentation](https://drive.google.com/file/d/1g5wkayJ9dUL4WuZjvTCj8OvRrAdq2LxG/view?usp=drive_link)


## Directory Structure

```
pandemic-simulator-compartment/
├── compartment/                    # Top level contains helpers and classes used across models and to execute models.
│   ├── cloud_helpers/              # Helper files used for the pandemic-simulator web application
│   ├── models/                     # Models which implement the tools in this repository. Likely available in the pandemic-simulator app.
│   │   |── covid_jax_model/        # Respiratory disease model
|   |   |── dengue_jax_model/       # Vector-borne disease model
│   ├── validation/                 # Pydantic syntax models which verify that a config is an acceptable input for a model.
├── reference/                      # Model configurations for running models locally. You can add your own reference configs.
```

## Caveats/Limitations
* This repository is currently under development. Some descriptions reflect planned features or the intended final state.
* Models in this repository may not apply to all scenarios, for example, modeling outbreaks of endemic diseases versus the introduction of a disease into a new region.

## Contributing
We welcome contributions via pull request related to the core features or submitting an example model. 

For major changes, please open an issue first to discuss what you would like to change.
Join our project and provide assistance by:
Checking out the list of open issues where we need help.
If you need new features, please open a new issue or start a discussion. 

Example models submitted to this repository will be reviewed by community subject matter experts. Public release is subject to approval and cannot be guaranteed. Users who contribute models should include example configs for those models.
