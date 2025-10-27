# Pandemic Simulator - Compartmental models in Python

Fast, flexible, and accurate construction of compartmental models to simulate transmission dynamics of respiratory and vector-borne diseases.

## Overview
This repository builds on code developed for the World Health Organization Pandemic Hub’s Pandemic Simulator project. Its purpose is to extend that work by providing accessible, research-based compartmental modeling tools that enable decision makers, epidemiologists, and modelers to assess public health intervention strategies.

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
* Checking out the list of open issues where we need help.
* If you need new features, please open a new issue or start a discussion. 

Example models submitted to this repository will be reviewed by community subject matter experts. Public release is subject to approval and cannot be guaranteed.
