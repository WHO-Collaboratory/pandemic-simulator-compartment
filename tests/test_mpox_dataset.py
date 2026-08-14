"""Tests for the MPOX model's region-independent example dataset."""

import pandas as pd
import pytest

from compartment.models.mpox_jax_model.model import MpoxJaxModel


def test_transition_schedule_applies_to_arbitrary_regions():
    model = object.__new__(MpoxJaxModel)
    model.admin_units = ["BRA-SP", "Pacific Island 7", "any-other-zone"]
    model.transition_schedule = model._load_transition_schedule()

    assert model._transition_multiplier("infection", 0) == pytest.approx(1.0)
    assert model._transition_multiplier("infection", 22) == pytest.approx(0.975)
    assert model._transition_multiplier("recovery", 45) == pytest.approx(0.925)
    assert model._transition_multiplier("infection", 365) == pytest.approx(1.05)


def test_transition_schedule_requires_both_transitions(tmp_path, monkeypatch):
    dataset = tmp_path / "incomplete.csv"
    pd.DataFrame({"day": [0], "infection": [1.0]}).to_csv(dataset, index=False)
    monkeypatch.setattr(MpoxJaxModel, "dataset", staticmethod(lambda name: dataset))
    model = object.__new__(MpoxJaxModel)

    with pytest.raises(ValueError, match="recovery"):
        model._load_transition_schedule()
