import json
import pathlib
import tempfile

from compartment.run_simulation import run_simulation

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "compartment" / "models"


def _discover_models() -> dict[str, tuple[str, str, pathlib.Path]]:
    """Scan compartment/models/ for directories with model.py + example-config.json."""
    import importlib
    import inspect
    from compartment.model import Model

    found = {}
    for model_dir in sorted(MODELS_DIR.iterdir()):
        config_path = model_dir / "example-config.json"
        model_py = model_dir / "model.py"
        if not config_path.exists() or not model_py.exists():
            continue

        dir_name = model_dir.name
        module_path = f"compartment.models.{dir_name}.model"

        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue

        model_class = None
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Model) and obj is not Model:
                model_class = obj
                break

        if model_class is None:
            continue

        class_path = f"{module_path}.{model_class.__name__}"
        found[dir_name] = (dir_name, class_path, config_path)

    return found


MODEL_CONFIGS = _discover_models()


def _import_class(dotted_path: str):
    """Import a class from a dotted module path."""
    import importlib
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _run_model(model_class, config_path: pathlib.Path) -> list[dict]:
    """Run a model and return the JSON results."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    run_simulation(
        model_class=model_class,
        config_path=str(config_path),
        output_path=output_path,
    )

    with open(output_path) as f:
        return json.load(f)
