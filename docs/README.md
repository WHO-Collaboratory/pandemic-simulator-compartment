# Documentation source

This folder is the MkDocs *source*, not the published site. The built pages live at
https://who-collaboratory.github.io/pandemic-simulator-compartment/.

| Path | What it is |
|------|------------|
| `index.md`, `guides/` | Human-written pages. Edit these. |
| `api/` | Short shells for the API reference. The `::: module.Class` lines are [mkdocstrings](https://mkdocstrings.github.io/) directives; class and method text is generated from docstrings in `compartment/` at build time. |
| `generate_model_pages.py` | Build hook. Disease model pages and the home-page table are generated from the model registry; they are **not** stored in this repo. |

To preview locally: `uv sync --group docs && uv run mkdocs serve`
