"""
Country-aware contact matrix loading and aggregation.

Provides synthetic age-mixing contact matrices from Prem et al. 2021
(177 countries, 5-year age bands, "overall" setting) and a fractional-
membership aggregator that collapses the source 16x16 matrix down to a
model's declared demographic age bands.

Public API:
    load_country_matrix(iso3)   -> 16x16 numpy array or None
    default_matrix()            -> 16x16 global-average numpy array
    available_countries()       -> list[str]
    aggregate_to_bands(M, ages) -> AxA numpy array
"""

# Inclusive (low, high) tuples for the 16 Prem source bands.
# Top band is open-ended in the source ("75+"); we cap at 120 for math.
PREM_BAND_EDGES: list[tuple[int, int]] = [
    (0, 4), (5, 9), (10, 14), (15, 19),
    (20, 24), (25, 29), (30, 34), (35, 39),
    (40, 44), (45, 49), (50, 54), (55, 59),
    (60, 64), (65, 69), (70, 74), (75, 120),
]

from .loader import load_country_matrix, default_matrix, available_countries
from .aggregator import aggregate_to_bands

__all__ = [
    "load_country_matrix",
    "default_matrix",
    "available_countries",
    "aggregate_to_bands",
    "PREM_BAND_EDGES",
]
