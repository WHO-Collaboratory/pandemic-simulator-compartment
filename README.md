# Pandemic Simulator - Compartmental models in Python

Fast, flexible, and accurate construction of compartmental models to simulate transmission dynamics of respiratory and vector-borne diseases.

## Overview
This repository builds on code developed for the World Health Organization Pandemic Hub’s Pandemic Simulator project. Its purpose is to extend that work by providing accessible, research-based compartmental modeling tools that enable decision makers, epidemiologists, and modelers to assess public health intervention strategies.

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

## Under Construction
This repository is under construction and verbiage reflects the desired state. Sections with such verbiage have indicative headers.

The code in this module has been maintained in a private repository in a different organization, and is in the process of being refactored to make it easier for the community to experiment with creating and testing models.

Code in the migration folder is a copy from the private repository. Functionality from this code will be moved into the abstract and concrete folders over time. The structure of the [mesa](https://github.com/projectmesa/mesa/tree/main) repository is an influence on this structure.

The examples folder will contain model configurations approved by community experts as acceptable starting points for modeling and using tools available in the package. This allows the package to avoid taking specific opinions on exact model parameters.

## About This Repository
* This section includes verbiage which describes the target state of the repository.

This repository provides epidemiological modelers with a flexible Python package and framework for producing compartment models that implement ordinary differential equations to simulate diseases. We provide tools for quickly editing models, running batches of simulations quickly, and exporting data for analysis with python tools or other resources.

This repository's output is a package which can be used by modelers to create models and simulations which will generate outputs interchangeable with tools in this package's ecosystem. The package will contain examples of vetted models endorsed by the collaboratory as valid uses of the package's capabilities, similar to the educational examples provided by the [mesa agent based modeling library](https://github.com/projectmesa/mesa/tree/main/mesa/examples).

## Planned Features
- Modular modeling framework supporting extensions and variants for diseases.
- Separation of models from simulation runs, and built in support for batch runs of simulations.
- Summary statistics objects.

## Contributing
* This section includes verbiage which describes the target state of the repository.

There are two acknowledged ways to contribute to this package. Working on the core features, or submitting an example model.

Contributing to core features will help the package's flexible core components solve common problems in modeling diseases with ODEs computationally. Contributing to core functionality is evaluated against automated build tools and by maintainers of the repository's technical components.

Contributing an example model to the examples folder is reviewed by disease experts in the community. Models may be rejected because the community does not agree about the realism of the model's parameters, or because the example model does not higlight a different disease or features than existing example models.


## About the Pandemic Simulator
This repository adapts code developed for the World Health Organization's Pandemic Simulator project. The path for this repository is to make compartment modelling a standalone package so that tools built for the app are isolated from application dependencies and available for use in the wider community. The current structure includes some dependencies that tightly couple this package with the application, we will pursue moving those dependencies into extensions to preserve compatibility with the app, but with less baggage for modellers.