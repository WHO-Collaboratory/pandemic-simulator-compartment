"""Resolving a declared dataset to a path a model can open.

A model reads its data through :meth:`compartment.model.Model.dataset`, which
delegates here:

.. code-block:: python

    df = pd.read_csv(self.dataset("kenya-contact-matrix"))

The name is looked up in the model's ``datasets.yaml``, and the manifest's
``file:`` path is what gets returned — so the same line of model code works
locally and in the cloud:

* **Locally**, the file arrives via ``datasets pull --dest data/``.
* **In the cloud**, it is baked into the model's container image by the
  ``datasets stage`` step in the release pipeline, at the same relative path.

Resolution is deliberately path-only. The framework never opens the file, so a
modeler can use pandas, numpy, ``json``, geopandas, or anything else without a
framework change.
"""

from __future__ import annotations

import sys
from pathlib import Path

from compartment.datasets.manifest import (
    MANIFEST_FILENAME,
    ManifestError,
    find_manifest,
    load_manifest,
)

# Resolved paths, keyed (model_dir, name). Parsing YAML on every call would be
# wasted work in a warm Lambda, which resolves the same handful of names for
# every invocation.
_CACHE: dict[tuple[str, str], Path] = {}


def model_dir_for_class(model_class) -> Path | None:
    """Directory holding a model class's source, or None if it isn't on disk.

    Variants declared in ``variants.py`` resolve to the same directory as the
    base model, which is correct: a model directory has one ``datasets.yaml``
    and every variant in it shares those datasets.
    """
    module = sys.modules.get(model_class.__module__)
    module_file = getattr(module, "__file__", None)
    return Path(module_file).parent if module_file else None


def dataset_path(model_dir: Path, name: str) -> Path:
    """Absolute path to the dataset ``name`` declared by the model in ``model_dir``.

    Raises ManifestError with an actionable message rather than surfacing a
    bare FileNotFoundError — every failure here has a specific fix, and the
    modeler should be told which one applies.
    """
    model_dir = Path(model_dir)
    cache_key = (str(model_dir), name)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    manifest_path = find_manifest(model_dir)
    if manifest_path is None:
        raise ManifestError(
            f"{name!r} is not available: {model_dir} has no {MANIFEST_FILENAME}. "
            f"Declare the dataset in {model_dir / MANIFEST_FILENAME} before "
            f"reading it. See docs/guides/adding-datasets.md."
        )

    entries = {entry.name: entry for entry in load_manifest(manifest_path)}
    entry = entries.get(name)
    if entry is None:
        declared = ", ".join(sorted(entries)) or "nothing"
        raise ManifestError(
            f"{manifest_path} does not declare {name!r}. It declares: {declared}."
        )

    if not entry.path.is_file():
        raise ManifestError(
            f"{name}@{entry.version} is declared in {manifest_path} but "
            f"{entry.path} does not exist. Download it with:\n"
            f"  python -m compartment.datasets pull {name}@{entry.version} "
            f"--dest {entry.path.parent}"
        )

    _CACHE[cache_key] = entry.path
    return entry.path
