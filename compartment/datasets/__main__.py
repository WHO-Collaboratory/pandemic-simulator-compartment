"""Entry point for ``python -m compartment.datasets <push|pull|list>``.

Delegates to ``compartment.datasets_cli`` so the SDK package and its CLI share
one import namespace (mirrors ``python -m compartment.generate_artifact``).
"""

from compartment.datasets_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
