<!--
  MkDocs + mkdocstrings shell. The `:::` block expands from docstrings in
  compartment/model.py when the site is built. Read the published page,
  not this file: https://who-collaboratory.github.io/pandemic-simulator-compartment/api/model/
-->

# Base Model Reference

Base class for all compartmental disease models. Subclasses **must**
implement `define_parameters(schema)`. They **should** also override
`prepare_initial_state()` and `equation()`, the per-step function the
solver calls.

::: compartment.model.Model
    options:
      show_root_heading: true
      members_order: source
      show_source: true
