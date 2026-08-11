"""Modeler dataset upload/retrieval.

``manifest`` is safe to import from anywhere (artifact generation uses it).
The CLI modules are imported lazily by ``__main__`` so that importing this
package does not drag in ``requests``.
"""

from compartment.datasets.manifest import (
    DatasetEntry,
    MANIFEST_FILENAME,
    ManifestError,
    artifact_refs_for_model_dir,
    find_manifest,
    load_manifest,
)

__all__ = [
    "DatasetEntry",
    "MANIFEST_FILENAME",
    "ManifestError",
    "artifact_refs_for_model_dir",
    "find_manifest",
    "load_manifest",
]
