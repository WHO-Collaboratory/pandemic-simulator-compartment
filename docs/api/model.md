# Base Model Reference

Base class for all compartmental disease models. Subclasses **must**
implement `define_parameters(schema)`. They **should** also override
`prepare_initial_state()` and `derivative()`.

::: compartment.model.Model
    options:
      show_root_heading: true
      members_order: source
      show_source: true
