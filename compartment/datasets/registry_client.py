"""Registry (GraphQL / AppSync) lookups for dataset versions — cloud only.

Reuses the ``cloud_helpers/gql.py`` request pattern to resolve a dataset
``slug`` to its currently-published ``DatasetVersion``. In Milestone 1 the
local cache and pinned cloud paths cover every supported flow, so the
"latest published" lookup is not wired up yet — it raises an actionable error
rather than silently guessing a version and breaking reproducibility.

Later slices implement this against the ``Dataset`` / ``DatasetVersion``
registry types, selecting scalar metadata only (never the file bytes) to
respect the 1 MB DynamoDB scan-page guardrail.
"""

from __future__ import annotations


def get_latest_published_version(slug: str, *, environment: str | None = None) -> str:
    """Resolve the newest ``PUBLISHED`` DatasetVersion for ``slug``.

    Deferred to a later milestone. Callers that reach this in cloud mode
    without a frozen pin get an error telling them how to make the run
    deterministic.
    """
    raise NotImplementedError(
        f"Resolving 'latest' for dataset '{slug}' from the registry is not "
        "available yet. Pin an exact version in datasets.yaml, or run from a "
        "SimulationJob whose dataset_pins were frozen at job creation."
    )
