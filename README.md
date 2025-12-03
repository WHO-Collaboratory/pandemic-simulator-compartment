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
source .venv/bin/activate
python -m compartment.examples.covid_jax_model.main --config_file pansim
```

### To run a command in that Dockerfile
```
# First, build the Dockerfile while in the root of this project.
docker build . -t compartment 
# TIP: You can build multiple images using different models by passing different values for MODEL_DIR to the build command.

docker run compartment
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
├── compartment/
│   ├── abstract/                   # Abstract base classes and interfaces
│   ├── concrete/                   # Concrete model implementations
│   │   └── respiratory/            # Respiratory disease models
│   ├── examples/                   # Community-vetted example models
│   │   └── README.md
│   └── migration/                  # Legacy code from private repository
│       ├── batch_helpers/          # AWS service utilities (to be moved to extensions)
│       ├── compartment/            # Current core compartment model implementation
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

Example models submitted to this repository will be reviewed by community subject matter experts. Public release is subject to approval and cannot be guaranteed.
