#!/usr/bin/env python3
"""Local simulation viewer for compartmental-model results.

Renders the ``parent_admin_total`` (whole-population) compartment time series
from a results JSON file as two side-by-side panels:

    left  = "With interventions"      (the run with control_run == False)
    right = "Without interventions"   (the control run, control_run == True)

Features
--------
* Auto-detects the per-compartment record format:
    - deterministic  ->  {"age_0_17": .., "age_all": ..}   (plots age_all)
    - uncertainty    ->  {"median": .., "lower": .., "upper": ..}
      (plots the median as a line and shades the lower..upper band)
* Draws vertical marker lines at each intervention start/stop date, when the
  run actually used interventions. Threshold-triggered interventions (no fixed
  date) are listed in the panel note instead of drawn as a line.
* Consistent colour per compartment across both panels and a shared y-axis so
  the two runs are directly comparable. Compartment names and colours mirror the
  Pandemic Simulator frontend (getCompartmentLabelsByDiseaseType in
  src/data/constants.tsx).
* A ``compartment_deltas`` metric table beneath the time series. For uncertainty
  and stochastic multi-run results, each central value is followed by its grey
  2.5th–97.5th percentile interval, matching the frontend result cards. Hide
  with ``--no-deltas``.

This reads only the *parent* admin total; it never drills into per-admin-zone
series.

Usage
-----
    python tools/view_results.py results/covid_results.json
    python tools/view_results.py results/uncertainty.json --save out.png
    python tools/view_results.py results/covid_results.json --compartments S,I,R --log
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Compartment naming — model artifact metadata with legacy fallbacks
# --------------------------------------------------------------------------- #
# Local simulation output includes the display-related subset of the same model
# artifact consumed by the cloud UI. The maps below remain as fallbacks for old
# result files that predate embedded artifact metadata.

COVID_LABELS = {
    "S": ("Susceptible", "#00bba7"),
    "E": ("Exposed", "#5A2C85"),
    "I": ("Infected", "#A8228E"),
    "R": ("Recovered", "#008ECE"),
    "H": ("Hospitalized", "#80BD01"),
    "D": ("Deceased", "#ef4444"),
    "C": ("Temporary Cross-protection", "#506BF7"),
    "Snot": ("Susceptible to other three serotypes", "#F26728"),
    "E2": ("Exposed with 2nd Serotype", "#A8228E"),
    "I2": ("Infected with 2nd Serotype", "#008ECE"),
}

DENGUE_LABELS = {
    "S": ("Susceptible", "#00bba7"),
    "E": ("Exposed with 1st serotype", "#5A2C85"),
    "I": ("Infected with 1st serotype", "#A8228E"),
    "R": ("Recovered", "#008ECE"),
    "H": ("Hospitalized", "#80BD01"),
    "D": ("Deceased", "#ef4444"),
    "C": ("Temporary Cross-protection", "#506BF7"),
    "SV": ("Susceptible Vectors", "#F26728"),
    "EV": ("Exposed Vectors", "#A8228E"),
    "IV": ("Infected Vectors", "#008ECE"),
    "Snot": ("Susceptible to other three serotypes", "#F26728"),
    "E2": ("Exposed with 2nd Serotype", "#A8228E"),
    "I2": ("Infected with 2nd Serotype", "#008ECE"),
}

# Compartments unique to the vector-borne model; their presence selects the
# dengue label map, matching getCompartmentLabelsByDiseaseType.
_DENGUE_MARKER_KEYS = {"SV", "EV", "IV", "Snot", "E2", "I2"}


def _generate_color(key):
    """Deterministic fallback colour for unknown compartments.

    Mirrors the frontend's generateCompartmentColor: hash the key, then map to
    hsl(hash % 360, 65%, 45%).
    """
    h = 0
    for ch in key:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    if h >= 0x80000000:
        h -= 0x100000000
    hue = (abs(h) % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.65)  # HSL lightness 45%, saturation 65%
    return (r, g, b)


def extract_model_artifact(runs):
    """Return embedded model artifact metadata from a collection of runs.

    New local result files store the artifact subset directly under
    ``model_artifact``. The cloud-shaped ``ModelArtifact.artifact_json`` form is
    also accepted so exported payloads can use the same viewer path.
    """
    for run in runs:
        artifact = run.get("model_artifact") or run.get("ModelArtifact")
        if isinstance(artifact, dict) and "artifact_json" in artifact:
            artifact = artifact["artifact_json"]
        if isinstance(artifact, str):
            try:
                artifact = json.loads(artifact)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        if isinstance(artifact, dict):
            return artifact
    return {}


def order_compartments(compartments, model_artifact=None):
    """Apply the artifact's cloud-UI display order, preserving unknown keys."""
    display_order = (model_artifact or {}).get("compartment_display_order", [])
    rank = {compartment: index for index, compartment in enumerate(display_order)}
    original_order = {
        compartment: index for index, compartment in enumerate(compartments)
    }
    return sorted(
        compartments,
        key=lambda compartment: (
            rank.get(compartment, len(rank)),
            original_order[compartment],
        ),
    )


def resolve_labels(compartments, model_artifact=None):
    """Return {key: (display_name, colour)} for the given compartments.

    Artifact labels take precedence, matching the cloud UI's model metadata.
    Older results fall back to the legacy dengue/COVID maps, then the raw key.
    """
    base = DENGUE_LABELS if (set(compartments) & _DENGUE_MARKER_KEYS) else COVID_LABELS
    artifact_labels = {
        compartment.get("id"): compartment.get("label")
        for compartment in (model_artifact or {}).get("compartments", [])
        if isinstance(compartment, dict)
        and compartment.get("id")
        and compartment.get("label")
    }
    return {
        c: (
            artifact_labels.get(c, base[c][0] if c in base else c),
            base[c][1] if c in base else _generate_color(c),
        )
        for c in compartments
    }


def format_number(value):
    """Thousands-separated integer, matching the frontend's formatNumber."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:,.0f}"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_runs(path):
    """Load the results file and return it as a list of run dicts."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty list of run outputs")
    return data


def split_runs(runs):
    """Return (intervention_run, control_run).

    Uses the ``control_run`` flag when present, and otherwise falls back to
    'the run that has interventions' vs 'the other one'.
    """
    with_iv = next((r for r in runs if r.get("control_run") is False), None)
    control = next((r for r in runs if r.get("control_run") is True), None)

    if with_iv is None:
        # No explicit flags: prefer the run that actually lists interventions.
        with_iv = next((r for r in runs if r.get("interventions")), runs[0])
    if control is None and len(runs) > 1:
        control = next((r for r in runs if r is not with_iv), None)
    return with_iv, control


def extract_value(entry):
    """Return (value, lower, upper) for a single compartment record.

    ``lower``/``upper`` are None for deterministic (non-band) output.
    """
    if isinstance(entry, dict):
        central = entry.get("median", entry.get("mean"))
        if central is not None:
            median = float(central)
            lower = float(entry.get("lower", median))
            upper = float(entry.get("upper", median))
            return median, lower, upper
        if "age_all" in entry:
            return float(entry["age_all"]), None, None
        # Unknown dict shape: fall back to the first numeric leaf.
        nums = [v for v in entry.values() if isinstance(v, (int, float))]
        if nums:
            return float(nums[0]), None, None
    else:
        try:
            return float(entry), None, None
        except (TypeError, ValueError):
            pass
    return math.nan, None, None


def parse_parent_admin_total(run):
    """Extract dates + per-compartment series from a run's parent_admin_total.

    Returns (dates, series, has_bands) where ``series`` maps
    compartment -> {"median": [...], "lower": [...] or None, "upper": [...] or None}.
    """
    pat = run.get("parent_admin_total") or {}
    ts = pat.get("time_series") or []
    if not ts:
        raise ValueError("run has no parent_admin_total.time_series")

    dates = [datetime.strptime(rec["date"], "%Y-%m-%d") for rec in ts]
    comp_keys = [k for k in ts[0].keys() if k != "date"]

    series = {}
    has_bands = False
    for comp in comp_keys:
        medians, lowers, uppers = [], [], []
        comp_has_band = False
        for rec in ts:
            median, lower, upper = extract_value(rec.get(comp, {}))
            medians.append(median)
            if lower is not None:
                comp_has_band = True
            lowers.append(lower)
            uppers.append(upper)
        series[comp] = {
            "median": medians,
            "lower": lowers if comp_has_band else None,
            "upper": uppers if comp_has_band else None,
        }
        has_bands = has_bands or comp_has_band
    return dates, series, has_bands


def intervention_markers(run):
    """Collect intervention start/end dates and threshold-only ids.

    Returns (starts, ends, threshold_only) where ``starts``/``ends`` map a
    parsed date -> sorted list of intervention ids active at that boundary.
    """
    starts, ends, threshold_only = {}, {}, []
    for iv in run.get("interventions") or []:
        iv_id = iv.get("id", "intervention")
        start = iv.get("start_date")
        end = iv.get("end_date")
        if not start and not end:
            # Threshold-triggered with no fixed calendar date.
            if iv.get("start_threshold") is not None or iv.get("end_threshold") is not None:
                threshold_only.append(iv_id)
            continue
        if start:
            starts.setdefault(datetime.strptime(start, "%Y-%m-%d"), []).append(iv_id)
        if end:
            ends.setdefault(datetime.strptime(end, "%Y-%m-%d"), []).append(iv_id)
    for d in starts:
        starts[d].sort()
    for d in ends:
        ends[d].sort()
    return starts, ends, threshold_only


def _delta_mapping(value):
    """Deserialize an AWSJSON delta payload into a mapping, if possible."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return value if isinstance(value, dict) else None


def _has_delta_ranges(deltas):
    """Whether a delta mapping contains at least one complete interval."""
    return any(
        isinstance(value, dict)
        and ("median" in value or "mean" in value)
        and value.get("lower") is not None
        and value.get("upper") is not None
        for value in deltas.values()
    )


def parse_compartment_delta_stats(run):
    """Return central values and optional percentile bounds for a run.

    The v2 AWSJSON field is preferred because current multi-run output keeps
    ``{median, lower, upper}`` there while the legacy ``compartment_deltas``
    field contains only the central value. Older output that stores the nested
    statistics directly in the legacy field remains supported, as does the
    historical ``mean`` central-value key.

    The result is ``{compartment: {value, lower, upper}}``. ``lower`` and
    ``upper`` are ``None`` for deterministic output or incomplete intervals.
    """
    candidates = [
        _delta_mapping(run.get("compartment_deltas_v2")),
        _delta_mapping(run.get("compartment_deltas")),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    source = next((candidate for candidate in candidates if _has_delta_ranges(candidate)),
                  candidates[0] if candidates else {})

    out = {}
    for comp, raw in source.items():
        if comp == "__typename":
            continue
        central = raw
        lower = upper = None
        if isinstance(raw, dict):
            central = raw.get("median", raw.get("mean"))
            if raw.get("lower") is not None and raw.get("upper") is not None:
                try:
                    lower = float(raw["lower"])
                    upper = float(raw["upper"])
                except (TypeError, ValueError):
                    lower = upper = None
        try:
            value = float(central)
        except (TypeError, ValueError):
            continue
        out[comp] = {"value": value, "lower": lower, "upper": upper}
    return out


def parse_compartment_deltas(run):
    """Return central compartment-delta values as {compartment: float}."""
    return {
        comp: stats["value"]
        for comp, stats in parse_compartment_delta_stats(run).items()
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def delta_compartments(panels, selected):
    """Compartments (in `selected` order) that appear in any run's deltas."""
    dsets = [parse_compartment_deltas(run) for _, run in panels]
    return [c for c in selected if any(c in d for d in dsets)]


def plot_deltas_table(ax, panels, selected, labels):
    """Render compartment_deltas as a metric table beneath the time series.

    Mirrors the frontend's per-compartment delta metrics (DiseaseModelTable /
    CompareSimulationsTable): each compartment is a row labelled with its display
    name, each run is a value column, and cells are thousands-formatted totals.
    Compartments are rows (not columns) so long disease names stay readable and
    the table scales to any number of compartments. Returns True if drawn.
    """
    panel_deltas = [(label.split(" (")[0], parse_compartment_delta_stats(run))
                    for label, run in panels]
    comps = delta_compartments(panels, selected)
    if not comps:
        return False

    ax.axis("off")
    show_ranges = any(
        stats.get("lower") is not None and stats.get("upper") is not None
        for _, deltas in panel_deltas
        for stats in deltas.values()
    )

    header = ["Compartment"]
    for label, _ in panel_deltas:
        header.append(label)
        if show_ranges:
            header.append("2.5% → 97.5%")

    cell_text = []
    for comp in comps:
        row = [labels[comp][0]]
        for _, deltas in panel_deltas:
            stats = deltas.get(comp)
            row.append(format_number(stats["value"]) if stats else "-")
            if show_ranges:
                if (stats and stats["lower"] is not None
                        and stats["upper"] is not None):
                    row.append(
                        f'{format_number(stats["lower"])} → '
                        f'{format_number(stats["upper"])}'
                    )
                else:
                    row.append("")
        cell_text.append(row)

    n_val = len(panel_deltas)
    name_w = 0.34 if show_ranges and n_val > 1 else 0.42 if show_ranges else 0.46
    if show_ranges:
        group_w = (1 - name_w) / n_val
        col_widths = [name_w]
        for _ in panel_deltas:
            col_widths.extend([group_w * 0.38, group_w * 0.62])
    else:
        col_widths = [name_w] + [(1 - name_w) / n_val] * n_val

    tbl = ax.table(cellText=cell_text, colLabels=header, colWidths=col_widths,
                   cellLoc="right", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Header row: bold; compartment-name column left-aligned. Percentile-range
    # headers and values are muted grey, matching the frontend result cards.
    for j in range(len(header)):
        cell = tbl[0, j]
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_ha("left" if j == 0 else "right")
        if show_ranges and j > 0 and j % 2 == 0:
            cell.get_text().set_color("0.45")
            cell.get_text().set_fontsize(8)
    if show_ranges:
        for i in range(1, len(comps) + 1):
            for j in range(2, len(header), 2):
                tbl[i, j].get_text().set_color("0.45")
    # Compartment-name column: left-aligned and coloured to match its line.
    for i, c in enumerate(comps):
        name_cell = tbl[i + 1, 0]
        name_cell.get_text().set_ha("left")
        name_cell.get_text().set_color(labels[c][1])
        name_cell.get_text().set_fontweight("bold")

    ax.set_title("Compartment deltas — cumulative totals", fontsize=10, pad=8)
    return True


def plot_panel(ax, run, labels, selected, log_scale, draw_markers):
    """Draw one run onto ``ax``. Returns True if uncertainty bands were drawn."""
    dates, series, _ = parse_parent_admin_total(run)
    drew_band = False

    for comp in selected:
        data = series.get(comp)
        if data is None:
            continue
        display_name, color = labels[comp]
        ax.plot(
            dates,
            data["median"],
            label=display_name,
            color=color,
            linewidth=1.6,
        )
        if data["lower"] is not None and data["upper"] is not None:
            ax.fill_between(dates, data["lower"], data["upper"],
                            color=color, alpha=0.18, linewidth=0)
            drew_band = True

    if draw_markers:
        starts, ends, threshold_only = intervention_markers(run)
        _draw_markers(ax, starts, ends)
        note = []
        if not starts and not ends and not threshold_only:
            note.append("no interventions applied")
        if threshold_only:
            note.append("threshold-triggered: " + ", ".join(threshold_only))
        if note:
            ax.text(0.02, 0.02, "\n".join(note), transform=ax.transAxes,
                    fontsize=7, style="italic", color="0.35", va="bottom")

    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    return drew_band


def _draw_markers(ax, starts, ends):
    """Draw dashed start / dotted end vertical lines with id annotations."""
    for date, ids in starts.items():
        ax.axvline(date, color="#2ca02c", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.annotate("▶ " + ", ".join(ids), xy=(date, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(2, -2), textcoords="offset points",
                    rotation=90, va="top", ha="left",
                    fontsize=7, color="#1b6b1b")
    for date, ids in ends.items():
        ax.axvline(date, color="#d62728", linestyle=":", linewidth=1.1, alpha=0.8)
        ax.annotate("■ " + ", ".join(ids), xy=(date, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(2, -2), textcoords="offset points",
                    rotation=90, va="top", ha="left",
                    fontsize=7, color="#a11")


def build_figure(path, compartments=None, log_scale=False, title=None, show_deltas=True):
    runs = load_runs(path)
    with_iv, control = split_runs(runs)
    model_artifact = extract_model_artifact(runs)

    # Union of compartments (stable order from the with-interventions run first).
    all_comps = []
    for run in (with_iv, control):
        if run is None:
            continue
        _, series, _ = parse_parent_admin_total(run)
        for c in series:
            if c not in all_comps:
                all_comps.append(c)
    all_comps = order_compartments(all_comps, model_artifact)

    if compartments:
        requested = [c.strip() for c in compartments.split(",") if c.strip()]
        missing = [c for c in requested if c not in all_comps]
        if missing:
            raise SystemExit(
                f"Unknown compartment(s): {', '.join(missing)}. "
                f"Available: {', '.join(all_comps)}"
            )
        selected = requested
    else:
        selected = all_comps

    labels = resolve_labels(all_comps, model_artifact)
    panels = [("With interventions", with_iv)]
    if control is not None:
        panels.append(("Without interventions (control)", control))
    ncols = len(panels)

    delta_comps = delta_compartments(panels, selected) if show_deltas else []
    have_deltas = bool(delta_comps)

    ts_h = 5.0
    if have_deltas:
        # Table panel grows with the compartment count so long names stay legible.
        delta_h = 0.55 + 0.32 * (len(delta_comps) + 1)
        fig = plt.figure(figsize=(7.5 * ncols, ts_h + delta_h + 0.9))
        gs = fig.add_gridspec(2, ncols, height_ratios=[ts_h, delta_h])
    else:
        fig = plt.figure(figsize=(7.5 * ncols, 6))
        gs = fig.add_gridspec(1, ncols)

    ts_axes = []
    for c in range(ncols):
        ax = fig.add_subplot(gs[0, c], sharey=ts_axes[0] if ts_axes else None)
        ts_axes.append(ax)

    any_band = False
    for ax, (label, run) in zip(ts_axes, panels):
        # Markers only make sense on the run that used interventions.
        draw_markers = run is with_iv
        drew = plot_panel(ax, run, labels, selected, log_scale, draw_markers)
        any_band = any_band or drew
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Date")
    ts_axes[0].set_ylabel("Population" + (" (log scale)" if log_scale else ""))

    # Single shared legend (compartments + marker key).
    handles, leg_labels = ts_axes[0].get_legend_handles_labels()
    if any(intervention_markers(with_iv)[i] for i in (0, 1)):
        handles.append(plt.Line2D([], [], color="#2ca02c", linestyle="--", label="intervention start"))
        handles.append(plt.Line2D([], [], color="#d62728", linestyle=":", label="intervention end"))
    if any_band:
        handles.append(plt.Line2D([], [], color="0.4", alpha=0.3, linewidth=8,
                                  label="uncertainty band (lower–upper)"))
    leg_labels = [h.get_label() for h in handles]
    fig.legend(handles, leg_labels, loc="upper center", ncol=min(len(leg_labels), 8),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.99))

    if have_deltas:
        delta_ax = fig.add_subplot(gs[1, :])
        plot_deltas_table(delta_ax, panels, selected, labels)

    stem = path.rsplit("/", 1)[-1]
    sim_type = (with_iv or {}).get("simulation_type", "")
    suptitle = title or f"parent_admin_total — {stem}"
    if sim_type:
        suptitle += f"   ({sim_type})"
    fig.suptitle(suptitle, fontsize=13, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="View parent_admin_total compartment time series "
                    "(with vs without interventions).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("file", help="Path to a results JSON file")
    ap.add_argument("--compartments", "-c", default=None,
                    help="Comma-separated subset to plot, e.g. S,I,R (default: all)")
    ap.add_argument("--log", action="store_true", help="Use a log-scale y-axis")
    ap.add_argument("--no-deltas", action="store_true",
                    help="Hide the compartment_deltas metric table")
    ap.add_argument("--title", default=None, help="Override the figure title")
    ap.add_argument("--save", "-o", default=None,
                    help="Save to this image path instead of opening a window")
    args = ap.parse_args()

    fig = build_figure(args.file, args.compartments, args.log, args.title,
                       show_deltas=not args.no_deltas)
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
