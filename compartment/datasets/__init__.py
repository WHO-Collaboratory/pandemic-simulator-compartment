"""Modeler dataset upload/retrieval.

``manifest`` and ``resolve`` are safe to import from anywhere — artifact
generation uses the former and every running model uses the latter, so neither
may pull in ``requests`` or ``boto3``. The CLI modules are imported lazily by
``__main__`` to keep that true.
"""

from compartment.datasets.manifest import (
    DatasetEntry,
    MANIFEST_FILENAME,
    MAX_DATASET_BYTES,
    ManifestError,
    artifact_refs_for_model_dir,
    find_manifest,
    human_bytes,
    load_manifest,
)
from compartment.datasets.resolve import dataset_path, model_dir_for_class

__all__ = [
    "DatasetEntry",
    "MANIFEST_FILENAME",
    "MAX_DATASET_BYTES",
    "ManifestError",
    "artifact_refs_for_model_dir",
    "dataset_path",
    "find_manifest",
    "human_bytes",
    "load_manifest",
    "model_dir_for_class",
]
