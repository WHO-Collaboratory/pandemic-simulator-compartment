"""Load Prem 2021 synthetic contact matrices from the bundled npz."""

from __future__ import annotations

import csv
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np


_DATA_DIR = Path(__file__).resolve().parent / "data"
_DATA_PATH = _DATA_DIR / "contact_all.npz"
_INCOME_PATH = _DATA_DIR / "income_defaults.npz"
_CSV_PATH = _DATA_DIR / "contact_matrices_economics.csv"

_lock = threading.Lock()
_npz_cache: dict[str, np.ndarray] | None = None

_income_lock = threading.Lock()
_income_cache: dict[str, np.ndarray] | None = None

# Four World Bank income tiers used as fallback groups.
INCOME_LEVELS: tuple[str, ...] = (
    "High income",
    "Upper middle income",
    "Lower middle income",
    "Low income",
)

# Built once at import time: ISO3 → income level for countries that lack a
# synthetic matrix but do have an assigned income group in the CSV.
# ISOs with Matrix Group == "Synthetic POLYMOD" are intentionally excluded —
# they are served by load_country_matrix() and never need this lookup.
def _build_iso_income_map() -> dict[str, str]:
    result: dict[str, str] = {}
    if not _CSV_PATH.exists():
        return result
    with open(_CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            matrix_group = row.get("Matrix Group", "")
            if matrix_group in INCOME_LEVELS:
                result[row["ISO"]] = matrix_group
    return result


_ISO_INCOME_MAP: dict[str, str] = _build_iso_income_map()


def _load_npz() -> dict[str, np.ndarray]:
    global _npz_cache
    if _npz_cache is None:
        with _lock:
            if _npz_cache is None:
                if not _DATA_PATH.exists():
                    raise FileNotFoundError(
                        f"contact_all.npz not found at {_DATA_PATH}"
                    )
                with np.load(_DATA_PATH) as data:
                    _npz_cache = {k: np.asarray(data[k], dtype=np.float64) for k in data.keys()}
    return _npz_cache


def _load_income_npz() -> dict[str, np.ndarray]:
    global _income_cache
    if _income_cache is None:
        with _income_lock:
            if _income_cache is None:
                if not _INCOME_PATH.exists():
                    raise FileNotFoundError(
                        f"income_defaults.npz not found at {_INCOME_PATH}. "
                        "Run tools/build_income_matrices.py to generate it."
                    )
                with np.load(_INCOME_PATH) as data:
                    _income_cache = {k: np.asarray(data[k], dtype=np.float64) for k in data.keys()}
    return _income_cache


def load_country_matrix(iso3: str | None) -> np.ndarray | None:
    """Return the 16x16 Prem matrix for ``iso3``, or ``None`` if absent.

    Case-insensitive: ``"usa"`` and ``"USA"`` both match.
    """
    if not iso3:
        return None
    key = iso3.upper()
    data = _load_npz()
    if key in data:
        return data[key].copy()
    return None


def iso_income_group(iso3: str | None) -> str | None:
    """Return the income-level group for an ISO that lacks a synthetic matrix.

    Returns one of the four ``INCOME_LEVELS`` strings, or ``None`` if the ISO
    either has its own synthetic matrix or has no assigned income group.
    """
    if not iso3:
        return None
    return _ISO_INCOME_MAP.get(iso3.upper())


def income_matrix(level: str) -> np.ndarray | None:
    """Return the precomputed average 16x16 matrix for an income level.

    ``level`` must be one of ``INCOME_LEVELS``.  Returns a copy so callers
    cannot corrupt the cache, or ``None`` if the level is not found.
    """
    data = _load_income_npz()
    if level in data:
        return data[level].copy()
    return None


@lru_cache(maxsize=1)
def default_matrix() -> np.ndarray:
    """Return the global-average 16x16 matrix.

    Mean across all available country matrices; computed once and cached.
    """
    data = _load_npz()
    keys = [k for k in data.keys() if not k.startswith("__")]
    stack = np.stack([data[k] for k in keys], axis=0)
    return stack.mean(axis=0)


def available_countries() -> list[str]:
    """Sorted list of ISO3 codes present in the bundle."""
    return sorted(k for k in _load_npz().keys() if not k.startswith("__"))
