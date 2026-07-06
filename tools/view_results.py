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
    - uncertainty    ->  {"mean": .., "lower": .., "upper": ..}
      (plots the mean as a line and shades the lower..upper band)
* Draws vertical marker lines at each intervention start/stop date, when the
  run actually used interventions. Threshold-triggered interventions (no fixed
  date) are listed in the panel note instead of drawn as a line.
* Consistent colour per compartment across both panels and a shared y-axis so
  the two runs are directly comparable.

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
import json
import math
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


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
        if "mean" in entry:
            mean = float(entry["mean"])
            lower = float(entry.get("lower", mean))
            upper = float(entry.get("upper", mean))
            return mean, lower, upper
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
    compartment -> {"mean": [...], "lower": [...] or None, "upper": [...] or None}.
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
        means, lowers, uppers = [], [], []
        comp_has_band = False
        for rec in ts:
            mean, lower, upper = extract_value(rec.get(comp, {}))
            means.append(mean)
            if lower is not None:
                comp_has_band = True
            lowers.append(lower)
            uppers.append(upper)
        series[comp] = {
            "mean": means,
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


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _color_map(compartments):
    cmap = plt.get_cmap("tab10" if len(compartments) <= 10 else "tab20")
    return {c: cmap(i % cmap.N) for i, c in enumerate(compartments)}


def plot_panel(ax, run, colors, selected, log_scale, draw_markers):
    """Draw one run onto ``ax``. Returns True if uncertainty bands were drawn."""
    dates, series, _ = parse_parent_admin_total(run)
    drew_band = False

    for comp in selected:
        data = series.get(comp)
        if data is None:
            continue
        color = colors[comp]
        ax.plot(dates, data["mean"], label=comp, color=color, linewidth=1.6)
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


def build_figure(path, compartments=None, log_scale=False, title=None):
    runs = load_runs(path)
    with_iv, control = split_runs(runs)

    # Union of compartments (stable order from the with-interventions run first).
    all_comps = []
    for run in (with_iv, control):
        if run is None:
            continue
        _, series, _ = parse_parent_admin_total(run)
        for c in series:
            if c not in all_comps:
                all_comps.append(c)

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

    colors = _color_map(all_comps)

    panels = [("With interventions", with_iv)]
    if control is not None:
        panels.append(("Without interventions (control)", control))

    fig, axes = plt.subplots(
        1, len(panels), figsize=(7.5 * len(panels), 6), sharey=True
    )
    if len(panels) == 1:
        axes = [axes]

    any_band = False
    for ax, (label, run) in zip(axes, panels):
        # Markers only make sense on the run that used interventions.
        draw_markers = run is with_iv
        drew = plot_panel(ax, run, colors, selected, log_scale, draw_markers)
        any_band = any_band or drew
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Date")
    axes[0].set_ylabel("Population" + (" (log scale)" if log_scale else ""))

    # Single shared legend (compartments + marker key).
    handles, labels = axes[0].get_legend_handles_labels()
    if any(intervention_markers(with_iv)[i] for i in (0, 1)):
        handles.append(plt.Line2D([], [], color="#2ca02c", linestyle="--", label="intervention start"))
        handles.append(plt.Line2D([], [], color="#d62728", linestyle=":", label="intervention end"))
        labels = [h.get_label() for h in handles]
    if any_band:
        handles.append(plt.Line2D([], [], color="0.4", alpha=0.3, linewidth=8,
                                  label="uncertainty band (lower–upper)"))
        labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 8),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.99))

    stem = path.rsplit("/", 1)[-1]
    sim_type = (with_iv or {}).get("simulation_type", "")
    suptitle = title or f"parent_admin_total — {stem}"
    if sim_type:
        suptitle += f"   ({sim_type})"
    fig.suptitle(suptitle, fontsize=13, y=1.06)
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
    ap.add_argument("--title", default=None, help="Override the figure title")
    ap.add_argument("--save", "-o", default=None,
                    help="Save to this image path instead of opening a window")
    args = ap.parse_args()

    fig = build_figure(args.file, args.compartments, args.log, args.title)
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
