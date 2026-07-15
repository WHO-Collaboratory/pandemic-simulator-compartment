"""Parse and validate a model's ``datasets.yaml`` dependency manifest.

A model opts into Bring-Your-Own-Dataset by dropping a ``datasets.yaml`` next
to its ``model.py``. It declares which logical datasets the model needs and,
for reproducible published models, the exact version each depends on.

    datasets:
      - name: mobility/kenya
        version: "2026-06-01"   # or "latest"
        required: true
        format: csv

``name`` is the logical id ``"<namespace>/<slug>"``. This module only parses
and validates the declaration; version resolution and file loading live in
``resolver.py`` / ``loader.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


MANIFEST_FILENAME = "datasets.yaml"


@dataclass(frozen=True)
class DatasetDep:
    """A single dataset dependency declared in ``datasets.yaml``."""

    name: str
    version: str = "latest"
    required: bool = True
    format: str | None = None


def parse_manifest(text: str) -> dict[str, DatasetDep]:
    """Parse ``datasets.yaml`` text into ``{name: DatasetDep}``."""
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(
            "datasets.yaml must be a mapping with a top-level 'datasets' key"
        )

    entries = data.get("datasets") or []
    if not isinstance(entries, list):
        raise ValueError("datasets.yaml 'datasets' must be a list")

    deps: dict[str, DatasetDep] = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"datasets.yaml entry {i} must be a mapping")
        name = entry.get("name")
        if not name:
            raise ValueError(f"datasets.yaml entry {i} is missing required 'name'")
        deps[name] = DatasetDep(
            name=name,
            version=str(entry.get("version", "latest")),
            required=bool(entry.get("required", True)),
            format=entry.get("format"),
        )
    return deps


def load_manifest(model_dir: str | Path | None) -> dict[str, DatasetDep]:
    """Load ``<model_dir>/datasets.yaml`` if present; empty mapping otherwise."""
    if not model_dir:
        return {}
    path = Path(model_dir) / MANIFEST_FILENAME
    if not path.exists():
        return {}
    return parse_manifest(path.read_text())
