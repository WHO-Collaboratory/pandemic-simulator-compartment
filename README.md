# Pandemic Simulator Compartment Models
This repository aims to provide epidemiological modelers with a flexible Python package and framework for producing compartment models that implement ordinary differential equations to simulate diseases. We aim to provide tools for quickly editing models, running batches of simulations quickly, and exporting data for analysis with python tools or other resources.

This repository's output is a package which can be used by modelers to create models and simulations which will generate outputs interchangeable with tools in this package's ecosystem. The package will contain examples of vetted models endorsed by the collaboratory as valid uses of the package's capabilities, similar to the educational examples provided by the [mesa agent based modeling library](https://github.com/projectmesa/mesa/tree/main/mesa/examples).

## Planned Features
- Modular modeling framework supporting extensions and mixins for diseases.
- Separation of models from simulation runs, and built in support for batch runs of simulations.
- Summary statistics objects.
- Geographic elements in compartment models, such as travel between jurisdictions with different demographics.

## Contributing
This package will contain flexible core components intended to solve common problems in modeling diseases with ODEs computationally. It will lean towards flexibility without forcing users to implement every attribute of underlying model compute on their own.

The examples folder will contain more opinionated models which are considered educational examples of both epidemiological modeling and how to use the tools in this package.