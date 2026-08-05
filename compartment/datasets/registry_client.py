"""Registry (GraphQL / AppSync) lookups for dataset versions — cloud only.

Resolving "latest published" from the registry at load time is **intentionally
unsupported**, not merely unimplemented. Do not add it.

Trace which paths can reach it:

* **local** — never called; ``_newest_local_version`` resolves ``latest`` from
  what is actually in the cache.
* **cloud, dataset declared in datasets.yaml** — never called either, whatever the
  declared version. The platform already resolved it to a concrete ``PUBLISHED``
  version and froze it onto ``SimulationJob.dataset_pins`` at job creation, so the
  resolver finds a pin.
* **cloud, dataset NOT declared** — the only reachable path, and it means a model
  called ``datasets.load()`` for something it never declared. Silently resolving
  "latest" there would make cloud runs non-reproducible: the same saved job would
  read different bytes after someone published a new version, which is precisely
  the property this feature exists to guarantee.

So the error below is the correct behaviour. It names both remedies.
"""

from __future__ import annotations


def get_latest_published_version(slug: str, *, environment: str | None = None) -> str:
    """Always raises — see the module docstring.

    Intentionally unsupported: reaching this means a cloud run asked for a
    dataset with no frozen pin, and guessing a version there would silently cost
    reproducibility. The raise is the feature.
    """
    raise NotImplementedError(
        f"Resolving 'latest' for dataset '{slug}' from the registry is not "
        "available yet. Pin an exact version in datasets.yaml, or run from a "
        "SimulationJob whose dataset_pins were frozen at job creation."
    )
