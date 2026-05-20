"""Collapse a 16x16 Prem contact matrix into the model's declared age bands.

Uses fractional-membership weights: each source band contributes to a target
band in proportion to the inclusive-year overlap divided by the source band's
width.  Row direction averages (mean across source bands within the target
person's age range); column direction sums (total contacts with all people in
the target band).  This preserves the "mean total contacts per person" semantic
that beta multiplies in the force of infection, and aggregating a Prem matrix
back to its own 16 bands returns the original matrix exactly.
"""

from __future__ import annotations

import logging

import numpy as np

from . import PREM_BAND_EDGES

logger = logging.getLogger(__name__)


def _overlap_years(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> int:
    """Inclusive integer-year overlap between [a_lo, a_hi] and [b_lo, b_hi]."""
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    return max(0, hi - lo + 1)


def _overlap_fractions(
    target_ranges: list[tuple[int, int]],
    source_edges: list[tuple[int, int]] = PREM_BAND_EDGES,
) -> np.ndarray:
    """A x S matrix where ``M[t, s] = overlap_years / width(source_band_s)``.

    The cell value is the fraction of source band *s* that lies inside
    target band *t*.  Each cell is in [0, 1].
    """
    A = len(target_ranges)
    S = len(source_edges)
    out = np.zeros((A, S), dtype=np.float64)
    for t, (tlo, thi) in enumerate(target_ranges):
        for s, (slo, shi) in enumerate(source_edges):
            ov = _overlap_years(tlo, thi, slo, shi)
            if ov > 0:
                out[t, s] = ov / (shi - slo + 1)
    return out


def _row_weights(target_ranges: list[tuple[int, int]]) -> np.ndarray:
    """Row-normalized overlap fractions: each target row sums to 1."""
    W = _overlap_fractions(target_ranges)
    for t in range(W.shape[0]):
        row_sum = W[t].sum()
        if row_sum > 0:
            W[t] /= row_sum
    return W


def _column_weights(target_ranges: list[tuple[int, int]]) -> np.ndarray:
    """Raw (un-normalized) overlap fractions; used to SUM across columns."""
    return _overlap_fractions(target_ranges)


def aggregate_to_bands(
    matrix: np.ndarray,
    target_ranges: list[tuple[int, int]],
) -> np.ndarray:
    """Collapse a 16x16 Prem matrix to AxA bands.

    Args:
        matrix: 16x16 array; ``matrix[i, j]`` = mean daily contacts of one
            person in source band *i* with all persons in source band *j*.
        target_ranges: Ordered list of inclusive (low, high) age tuples for
            the model's demographic groups.

    Returns:
        AxA array in the same units (mean daily contacts per person).

    Notes:
        Asymmetric by design: rows are mean-aggregated (the contactor is a
        single typical person sampled from the band), columns are summed
        (total contacts with everyone in the band).  Aggregating Prem to
        its own 16 bands is exact: ``aggregate_to_bands(M, PREM_BAND_EDGES)
        == M``.
    """
    if matrix.shape != (16, 16):
        raise ValueError(
            f"Expected 16x16 source matrix, got shape {matrix.shape}"
        )
    if not target_ranges:
        raise ValueError("target_ranges must be non-empty")

    W = _row_weights(target_ranges)        # A x 16, rows sum to 1 (or 0 if disjoint)
    U = _column_weights(target_ranges)     # A x 16, raw overlap fractions

    # Warn on any target band that has zero overlap with the source range.
    zero_rows = [
        target_ranges[t] for t in range(W.shape[0]) if W[t].sum() == 0
    ]
    if zero_rows:
        logger.warning(
            "aggregate_to_bands: target age ranges %s have no overlap with "
            "Prem source bands [0, 120]; corresponding output rows/cols are zero.",
            zero_rows,
        )

    return W @ matrix.astype(np.float64) @ U.T
