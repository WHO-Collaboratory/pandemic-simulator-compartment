"""Load-and-cache access to Bring-Your-Own-Dataset files.

Mirrors ``contact_matrices/loader.py``: a process-global cache guarded by a
lock, populated once per ``(slug, version)``. Datasets are read only through a
small allowlist of deserialization-safe pandas readers — **never**
``read_pickle`` (RCE by design), ``read_hdf`` (documented unsafe against
malicious files), or any other code-executing reader — so a hostile file
cannot achieve code execution on the read path, local or cloud. This loader
discipline is the parser-exploit defense that also holds on the local path,
at zero infra cost.

Configuration (mode, version pins, the model's manifest, cache location) is
supplied once via :func:`configure` from ``run_simulation`` before any model
is constructed. The model's ``load_datasets()`` hook then calls :func:`load`
once — never at import time, never inside the JAX-traced ``derivative()``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pandas as pd

from compartment.datasets.manifest import DatasetDep, load_manifest
from compartment.datasets.resolver import Resolver


# ---------------------------------------------------------------------------
# Deserialization-safe reader allowlist
# ---------------------------------------------------------------------------
# Extension -> logical format. Anything not listed here is refused. There is
# deliberately NO entry for pickle/.pkl, .npy/.npz, .h5/.hdf5, or .xlsx.
_EXT_TO_FORMAT: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".json": "json",
    ".ndjson": "ndjson",
    ".jsonl": "ndjson",
}
ALLOWED_SUFFIXES = frozenset(_EXT_TO_FORMAT)


_READERS = {
    "csv": lambda p: pd.read_csv(p),
    "tsv": lambda p: pd.read_csv(p, sep="\t"),
    "json": lambda p: pd.read_json(p),
    "ndjson": lambda p: pd.read_json(p, lines=True),
}

DEFAULT_CACHE_DIR = "~/.cache/who-collaboratory/datasets"
CLOUD_CACHE_DIR = "/tmp/datasets"


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_cache: dict[tuple[str, str], pd.DataFrame] = {}
_resolver: Resolver | None = None


def _detect_mode() -> str:
    """Fall back to the same signal ``run_simulation`` uses for cloud mode."""
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or os.environ.get("ENVIRONMENT"):
        return "cloud"
    return "local"


def _cache_root(mode: str, cache_root=None) -> Path:
    if cache_root:
        return Path(cache_root).expanduser()
    env = os.environ.get("WHO_DATASET_CACHE")
    if env:
        return Path(env).expanduser()
    base = CLOUD_CACHE_DIR if mode == "cloud" else DEFAULT_CACHE_DIR
    return Path(base).expanduser()


def configure(
    *, mode=None, environment=None, pins=None, model_dir=None, cache_root=None
) -> None:
    """Configure dataset resolution for the current run.

    Called once from ``run_simulation`` right after the config is validated and
    before any model is constructed. Safe to call again — it rebuilds the
    resolver and clears the frame cache so a re-run with different pins or a
    different cache never returns stale data.

    Args:
        mode: ``"local"`` or ``"cloud"``; autodetected from the environment
            when omitted.
        environment: cloud deployment environment (e.g. ``"dev"``).
        pins: the job's ``dataset_pins`` — ``[{slug, version, key, content_hash, ...}]``.
        model_dir: directory holding the running model's ``datasets.yaml``.
        cache_root: override the cache location (else ``WHO_DATASET_CACHE`` or
            the mode default).
    """
    global _resolver
    resolved_mode = mode or _detect_mode()
    _resolver = Resolver(
        mode=resolved_mode,
        environment=environment,
        pins=pins,
        manifest=load_manifest(model_dir),
        cache_root=_cache_root(resolved_mode, cache_root),
        allowed_suffixes=ALLOWED_SUFFIXES,
    )
    with _lock:
        _cache.clear()


def _require_resolver() -> Resolver:
    if _resolver is None:
        raise RuntimeError(
            "datasets.configure() must be called before datasets.load()/"
            "path_for(). run_simulation configures this automatically; call "
            "configure(mode=...) directly to load datasets outside a run."
        )
    return _resolver


def load(name: str, version: str | None = None) -> pd.DataFrame:
    """Return dataset ``name`` as a DataFrame, loading + caching once.

    ``name`` is the logical id ``"<namespace>/<slug>"``. ``version`` overrides
    the pin/manifest version. The return type is inferred from the resolved
    file's extension. Repeated calls for the same ``(slug, version)`` return
    the identical cached frame.
    """
    resolver = _require_resolver()
    resolved = resolver.resolve(name, explicit_version=version)
    key = (resolved.slug, resolved.version)

    with _lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached

    df = read_frame(resolved.path)

    with _lock:
        _cache[key] = df
    return df


def read_frame(path) -> pd.DataFrame:
    """Read a dataset file into a DataFrame via the safe-reader allowlist.

    The single chokepoint for turning dataset bytes into a DataFrame — used by
    :func:`load` and by the ``datasets`` CLI when computing upload metadata, so
    both share one discipline: only ``csv``/``tsv``/``json``/``ndjson`` are
    ever read, never ``read_pickle`` or any code-deserializing reader.
    """
    path = Path(path)
    fmt = _EXT_TO_FORMAT.get(path.suffix.lower())
    if fmt is None:
        allowed = ", ".join(sorted(s.lstrip(".") for s in ALLOWED_SUFFIXES))
        raise ValueError(
            f"Unsupported dataset format '{path.suffix}'. Allowed: {allowed}. "
            "Code-deserializing formats (e.g. pickle) are never read."
        )
    return _READERS[fmt](path)


def path_for(name: str, version: str | None = None) -> Path:
    """Return the local path to dataset ``name`` for non-pandas readers.

    Ensures the file exists (downloading in cloud mode) and enforces the same
    format allowlist as :func:`load`, but does not parse it.
    """
    resolver = _require_resolver()
    return resolver.resolve(name, explicit_version=version).path


def list_declared() -> list[DatasetDep]:
    """Return the dataset dependencies declared in the active model's manifest."""
    return list(_require_resolver().manifest.values())


def _reset() -> None:
    """Test helper: clear all configuration and cached frames."""
    global _resolver
    with _lock:
        _cache.clear()
    _resolver = None
