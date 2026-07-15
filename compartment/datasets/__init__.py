"""Bring-Your-Own-Dataset access layer.

The public surface mirrors ``contact_matrices``: a small functional API the
model author calls from a one-time ``load_datasets()`` hook — never at import
time and never inside the JAX-traced ``derivative()``. The identical call
resolves from the local cache on a laptop and from S3 (via frozen version
pins) in the cloud, so the same model code runs unchanged in both.

    from compartment import datasets

    df = datasets.load("mobility/kenya")       # DataFrame; version from pin/yaml
    p  = datasets.path_for("mobility/kenya")   # local Path for non-pandas readers
"""

from compartment.datasets.loader import (
    ALLOWED_SUFFIXES,
    configure,
    list_declared,
    load,
    path_for,
)

__all__ = [
    "configure",
    "load",
    "path_for",
    "list_declared",
    "ALLOWED_SUFFIXES",
]
