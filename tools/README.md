# tools/

Small local utilities for modelers. Not part of the `compartment` package.

## `view_results.py` — simulation results viewer

Plots the whole-population compartment time series (`parent_admin_total`) from a
results JSON file, showing the **with-interventions** run and the
**control** (no-interventions) run side by side.

```bash
# From the repo root (matplotlib is already a core dependency)
python tools/view_results.py results/covid_results.json

# Only some compartments, log y-axis
python tools/view_results.py results/covid_results.json -c S,I,R --log

# Save to an image instead of opening a window
python tools/view_results.py results/uncertainty.json -o /tmp/run.png
```

What it does automatically:

- **With vs. without interventions** — the two panels are the `control_run: false`
  and `control_run: true` runs that `run_simulation` writes for every simulation,
  drawn on a shared y-axis so they're directly comparable.
- **Intervention markers** — if the run used interventions, a green dashed line
  marks each start date and a red dotted line each end date, labeled with the
  intervention id(s). Threshold-triggered interventions (no fixed date) are noted
  in the panel instead.
- **Uncertainty / stochastic bounds** — for `UNCERTAINTY` (or multi-run) output,
  where each compartment record is `{median, lower, upper}`, the median is drawn as a
  line and the `lower..upper` range is shaded.
- **Compartment deltas metric** — a table beneath the chart lists each run's
  `compartment_deltas` (the per-compartment cumulative totals: total ever-infected,
  total deaths, etc.). Compartment names, colours, and number formatting mirror the
  Pandemic Simulator frontend (`getCompartmentLabelsByDiseaseType`), so `E` shows as
  "Exposed", `D` as "Deceased", and so on. Hide it with `--no-deltas`.

It reads only the *parent* admin total, never per-admin-zone series.

Flags: `-c/--compartments`, `--log`, `--no-deltas`, `--title`, `-o/--save`. Run with `-h` for details.
