"""Load Prem 2021 synthetic contact matrices from the bundled npz."""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np


_DATA_PATH = Path(__file__).resolve().parent / "data" / "contact_all.npz"

_lock = threading.Lock()
_npz_cache: dict[str, np.ndarray] | None = None


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
