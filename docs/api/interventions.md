<!--
  MkDocs + mkdocstrings shell. The `:::` block expands from docstrings in
  compartment/interventions.py when the site is built. Read the published page,
  not this file: https://who-collaboratory.github.io/pandemic-simulator-compartment/api/interventions/
-->

# Interventions Reference

JAX-compatible intervention functions that modify transmission rates during
simulation. These support both proportion-based triggering (activate when
infection prevalence crosses a threshold) and date-window triggering
(activate within a calendar date range).

::: compartment.interventions
    options:
      show_root_heading: true
      members_order: source
      show_source: true
