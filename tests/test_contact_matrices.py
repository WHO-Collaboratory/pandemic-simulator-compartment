"""Unit tests for the country-aware contact matrix loader and aggregator."""

import numpy as np
import pytest

from compartment.contact_matrices import (
    PREM_BAND_EDGES,
    aggregate_to_bands,
    available_countries,
    default_matrix,
    load_country_matrix,
)
from compartment.contact_matrices.aggregator import (
    _column_weights,
    _row_weights,
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TestLoader:
    def test_known_country_returns_16x16(self):
        m = load_country_matrix("USA")
        assert m is not None
        assert m.shape == (16, 16)
        assert m.dtype == np.float64
        assert (m >= 0).all()

    def test_case_insensitive(self):
        np.testing.assert_array_equal(
            load_country_matrix("usa"),
            load_country_matrix("USA"),
        )

    def test_unknown_country_returns_none(self):
        assert load_country_matrix("ZZZ") is None
        assert load_country_matrix("") is None
        assert load_country_matrix(None) is None

    def test_available_countries(self):
        countries = available_countries()
        assert len(countries) == 177
        assert "USA" in countries
        assert "MDG" in countries
        assert "ZWE" in countries
        # sorted
        assert countries == sorted(countries)

    def test_default_matrix_equals_mean(self):
        d = default_matrix()
        assert d.shape == (16, 16)
        assert (d > 0).all()
        stack = np.stack(
            [load_country_matrix(c) for c in available_countries()], axis=0
        )
        np.testing.assert_allclose(d, stack.mean(axis=0), rtol=1e-12)

    def test_returned_array_is_copy(self):
        a = load_country_matrix("USA")
        a[0, 0] = -999
        b = load_country_matrix("USA")
        assert b[0, 0] != -999


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class TestAggregator:
    def test_identity_contract(self):
        """Aggregating Prem to its own 16 bands returns Prem exactly."""
        M = load_country_matrix("MDG")
        out = aggregate_to_bands(M, PREM_BAND_EDGES)
        np.testing.assert_allclose(out, M, rtol=1e-12, atol=1e-12)

    def test_identity_contract_on_synthetic_matrix(self):
        M = np.arange(256, dtype=np.float64).reshape(16, 16)
        out = aggregate_to_bands(M, PREM_BAND_EDGES)
        np.testing.assert_allclose(out, M, rtol=1e-12, atol=1e-12)

    def test_single_all_band_with_ones(self):
        """All-ones 16x16, collapsed to [(0, 120)]:
        row direction averages (16 ones -> 1.0);
        column direction sums fractional overlaps. The target (0, 120)
        fully covers the first 15 bands (15.0) and overlaps the 16th band
        (75-120) entirely (1.0) -> total 16.0.
        """
        M = np.ones((16, 16))
        out = aggregate_to_bands(M, [(0, 120)])
        assert out.shape == (1, 1)
        np.testing.assert_allclose(out, [[16.0]], rtol=1e-12)

    def test_three_bucket_widths(self):
        """All-ones M, COVID-style bands. Each output cell = row-mean (1) * col-sum.
        col-sum is the total fractional overlap across 16 source bands.
            (0, 17):  bands 0-4, 5-9, 10-14 (3 full) + 15-19 partial (3/5=0.6) = 3.6
            (18, 55): bands 15-19 partial (2/5=0.4) + 20-24..50-54 (7 full) + 55-59 partial (1/5=0.2) = 7.6
            (56, 120): 55-59 partial (4/5=0.8) + 60-64..70-74 (3 full) + 75-120 full (1.0) = 4.8
        Since rows of all-ones average to 1, every row of the output equals the col-sum vector.
        """
        M = np.ones((16, 16))
        out = aggregate_to_bands(M, [(0, 17), (18, 55), (56, 120)])
        assert out.shape == (3, 3)
        expected_cols = np.array([3.6, 7.6, 4.8])
        for row in out:
            np.testing.assert_allclose(row, expected_cols, rtol=1e-12)

    def test_fractional_membership_15_19(self):
        """Target (0, 17) overlaps source band (15, 19) by 3 years -> weight 3/5."""
        U = _column_weights([(0, 17)])
        # source band index 3 is (15, 19)
        assert U[0, 3] == pytest.approx(0.6, abs=1e-12)
        # source bands 0,1,2 fully inside target -> 1.0
        np.testing.assert_allclose(U[0, :3], [1.0, 1.0, 1.0], rtol=1e-12)
        # source bands >= index 4 disjoint -> 0
        np.testing.assert_allclose(U[0, 4:], np.zeros(12), atol=1e-12)

    def test_row_weights_sum_to_one(self):
        W = _row_weights([(0, 17), (18, 55), (56, 120)])
        for row in W:
            assert row.sum() == pytest.approx(1.0, abs=1e-12)

    def test_disjoint_band_produces_zero_row(self, caplog):
        M = np.ones((16, 16))
        with caplog.at_level("WARNING", logger="compartment.contact_matrices.aggregator"):
            out = aggregate_to_bands(M, [(200, 210)])
        assert out.shape == (1, 1)
        np.testing.assert_allclose(out, [[0.0]], atol=1e-12)
        assert any("no overlap" in r.message for r in caplog.records)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            aggregate_to_bands(np.ones((10, 10)), [(0, 17)])

    def test_rejects_empty_target(self):
        with pytest.raises(ValueError):
            aggregate_to_bands(np.ones((16, 16)), [])

    def test_real_country_aggregation_is_positive_and_asymmetric(self):
        """MDG aggregated to 3 bands should be positive (real-world data)
        and asymmetric (Prem matrices are contactor x contactee)."""
        M = load_country_matrix("MDG")
        out = aggregate_to_bands(M, [(0, 17), (18, 55), (56, 120)])
        assert out.shape == (3, 3)
        assert (out >= 0).all()
        # Asymmetry: not all off-diagonal pairs equal
        assert not np.allclose(out, out.T)
