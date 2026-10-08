"""The paper window has to be one definition, used everywhere.

The 2 Oct 2025 -- 28 Apr 2026 slice of the premium series is what the funding
rows of the summary-statistics table, the funding-parameter table
(paper/tables/table3_funding_params.tex) and the funding-dynamics figure
(fig3_funding_dynamics) are all computed on. A single definition guarantees
that the figure is drawn on the same fit as the table beside it; re-slicing
the series inline at each use would not.
"""
import numpy as np
import pytest

from regime_engine.panel import load_paper_window_premium

# Values recomputed from the shipped data/interim/premium_BTC_full.csv.
T_PAPER = 5001
PREMIUM_MIN = -10.5509
PREMIUM_MAX = 23.6132
PREMIUM_MEAN = -3.1478
PREMIUM_STD = 2.8673


@pytest.fixture(scope="module")
def y():
    return load_paper_window_premium()


def test_window_has_the_length_the_paper_reports(y):
    assert len(y) == T_PAPER


def test_series_is_in_basis_points(y):
    """A premium quoted as a raw fraction would be four orders of magnitude out."""
    assert y.min() == pytest.approx(PREMIUM_MIN, abs=5e-4)
    assert y.max() == pytest.approx(PREMIUM_MAX, abs=5e-4)


def test_moments_match_table_one(y):
    assert y.mean() == pytest.approx(PREMIUM_MEAN, abs=5e-4)
    assert y.std(ddof=1) == pytest.approx(PREMIUM_STD, abs=5e-4)


def test_the_october_maximum_is_inside_the_window(y):
    """The stressed premium state is driven by this observation, not by February."""
    assert y.max() > 20.0, "the +23.6 bps excursion of 10 Oct 2025 must be present"


def test_series_is_contiguous_and_finite(y):
    assert np.all(np.isfinite(y))
