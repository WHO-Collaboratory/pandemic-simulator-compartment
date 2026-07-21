"""Build and save income-level average contact matrices.

Reads contact_matrices_economics.csv to identify which income group each
country belongs to.  For each of the four World Bank income levels, averages
the Prem 2021 synthetic matrices of all countries classified as
"Synthetic POLYMOD" at that income level, then writes the result to
compartment/contact_matrices/data/income_defaults.npz.

Run from the repo root:
    python tools/build_income_matrices.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_CSV = _ROOT / "compartment" / "contact_matrices" / "data" / "contact_matrices_economics.csv"
_NPZ_IN = _ROOT / "compartment" / "contact_matrices" / "data" / "contact_all.npz"
_NPZ_OUT = _ROOT / "compartment" / "contact_matrices" / "data" / "income_defaults.npz"

INCOME_LEVELS = (
    "High income",
    "Upper middle income",
    "Lower middle income",
    "Low income",
)


def main() -> None:
    # Load all synthetic matrices once.
    with np.load(_NPZ_IN) as data:
        synthetic: dict[str, np.ndarray] = {
            k: np.asarray(data[k], dtype=np.float64)
            for k in data.keys()
            if not k.startswith("__")
        }

    # Parse CSV: collect ISO3 codes that have Synthetic POLYMOD + a known income.
    income_to_isos: dict[str, list[str]] = {lvl: [] for lvl in INCOME_LEVELS}
    with open(_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["Matrix Group"] == "Synthetic POLYMOD" and row["Income"] in INCOME_LEVELS:
                income_to_isos[row["Income"]].append(row["ISO"])

    # Compute and report averages.
    result: dict[str, np.ndarray] = {}
    for level in INCOME_LEVELS:
        isos = income_to_isos[level]
        matrices = [synthetic[iso] for iso in isos if iso in synthetic]
        missing = [iso for iso in isos if iso not in synthetic]
        if missing:
            print(f"  WARNING: {level}: {len(missing)} ISO(s) in CSV but not in npz: {missing}")
        if not matrices:
            print(f"  ERROR: {level}: no matrices found — skipping", file=sys.stderr)
            continue
        avg = np.stack(matrices, axis=0).mean(axis=0)
        result[level] = avg
        print(f"  {level}: averaged {len(matrices)} matrices → shape {avg.shape}")

    np.savez_compressed(_NPZ_OUT, **result)
    print(f"\nSaved {len(result)} income-level matrices to {_NPZ_OUT}")


if __name__ == "__main__":
    main()
