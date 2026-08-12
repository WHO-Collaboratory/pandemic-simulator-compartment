"""Tests for resolving a declared dataset to a path a model can open.

Every failure mode here has a specific fix, and the modeler only finds out
which one applies from the error message — so these assert on message content,
not just on the exception type.
"""

import pytest

from compartment.datasets import ManifestError, dataset_path, model_dir_for_class
from compartment.datasets.resolve import _CACHE


@pytest.fixture(autouse=True)
def clear_cache():
    """dataset_path memoizes, and tmp_path dirs differ per test — but a test
    that rewrites the same manifest would otherwise see a stale hit."""
    _CACHE.clear()
    yield
    _CACHE.clear()


def _write_model_dir(tmp_path, manifest: str | None, data_files: dict[str, bytes] | None = None):
    model_dir = tmp_path / "my_model"
    model_dir.mkdir()
    if manifest is not None:
        (model_dir / "datasets.yaml").write_text(manifest)
    for name, content in (data_files or {}).items():
        target = model_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return model_dir


MANIFEST = """\
datasets:
  - name: kenya-contact-matrix
    version: "1"
    file: data/kenya-contacts.csv
"""


def test_resolves_declared_dataset(tmp_path):
    model_dir = _write_model_dir(
        tmp_path, MANIFEST, {"data/kenya-contacts.csv": b"a,b\n1,2\n"}
    )

    resolved = dataset_path(model_dir, "kenya-contact-matrix")

    assert resolved == model_dir / "data" / "kenya-contacts.csv"
    assert resolved.read_bytes() == b"a,b\n1,2\n"


def test_resolution_is_cached(tmp_path):
    """A warm Lambda resolves the same names every invocation; don't re-parse."""
    model_dir = _write_model_dir(
        tmp_path, MANIFEST, {"data/kenya-contacts.csv": b"x"}
    )

    first = dataset_path(model_dir, "kenya-contact-matrix")
    (model_dir / "datasets.yaml").unlink()  # a cold call would now fail
    second = dataset_path(model_dir, "kenya-contact-matrix")

    assert first == second


def test_missing_manifest_names_the_file_to_create(tmp_path):
    model_dir = _write_model_dir(tmp_path, None)

    with pytest.raises(ManifestError) as exc:
        dataset_path(model_dir, "kenya-contact-matrix")

    message = str(exc.value)
    assert "datasets.yaml" in message
    assert "adding-datasets" in message


def test_undeclared_name_lists_what_is_declared(tmp_path):
    model_dir = _write_model_dir(
        tmp_path, MANIFEST, {"data/kenya-contacts.csv": b"x"}
    )

    with pytest.raises(ManifestError) as exc:
        dataset_path(model_dir, "kenya-admin-zones")

    message = str(exc.value)
    assert "kenya-admin-zones" in message
    # The fix is usually a typo, so the available names have to be in the message.
    assert "kenya-contact-matrix" in message


def test_declared_but_absent_file_gives_the_pull_command(tmp_path):
    """The common local case: fresh clone, data/ is gitignored."""
    model_dir = _write_model_dir(tmp_path, MANIFEST)

    with pytest.raises(ManifestError) as exc:
        dataset_path(model_dir, "kenya-contact-matrix")

    message = str(exc.value)
    assert "compartment.datasets pull kenya-contact-matrix@1" in message
    assert str(model_dir / "data") in message


def test_a_directory_at_the_data_path_is_not_a_dataset(tmp_path):
    model_dir = _write_model_dir(tmp_path, MANIFEST)
    (model_dir / "data" / "kenya-contacts.csv").mkdir(parents=True)

    with pytest.raises(ManifestError):
        dataset_path(model_dir, "kenya-contact-matrix")


# ---------------------------------------------------------------------------
# model_dir_for_class
# ---------------------------------------------------------------------------


def test_model_dir_for_class_finds_the_model_directory():
    from compartment.models.mpox_jax_model.model import MpoxJaxModel

    assert model_dir_for_class(MpoxJaxModel).name == "mpox_jax_model"


def test_model_dir_for_class_returns_none_for_a_dynamic_class():
    """Classes built at runtime have no source directory to look in."""
    dynamic = type("DynamicModel", (), {})
    dynamic.__module__ = "not_a_real_module"

    assert model_dir_for_class(dynamic) is None


def test_variants_share_the_base_model_directory():
    """One datasets.yaml per model dir, shared by every variant in it."""
    from compartment.models.covid_jax_model.model import CovidJaxModel
    from compartment.models.covid_jax_model.variants import CovidSIRModel

    assert model_dir_for_class(CovidSIRModel) == model_dir_for_class(CovidJaxModel)


# ---------------------------------------------------------------------------
# Model.dataset() — the modeler-facing entry point
# ---------------------------------------------------------------------------


def test_model_dataset_accessor_resolves_through_the_class(tmp_path, monkeypatch):
    from compartment.model import Model

    model_dir = _write_model_dir(
        tmp_path, MANIFEST, {"data/kenya-contacts.csv": b"x"}
    )

    class FakeModel(Model):
        pass

    # Model.dataset() imports from compartment.datasets at call time, so that
    # is the binding to patch.
    monkeypatch.setattr(
        "compartment.datasets.model_dir_for_class", lambda cls: model_dir
    )

    # Works off the class, so it is callable from classmethods too.
    assert FakeModel.dataset("kenya-contact-matrix").name == "kenya-contacts.csv"


def test_model_dataset_accessor_explains_a_dynamic_class():
    from compartment.model import Model

    dynamic = type("DynamicModel", (Model,), {})
    dynamic.__module__ = "not_a_real_module"

    with pytest.raises(ManifestError) as exc:
        dynamic.dataset("anything")

    assert "source directory" in str(exc.value)
