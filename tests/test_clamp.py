"""Clamp / funding-floor arithmetic — against the real implementation.

Locks in the Hyperliquid funding mechanics, so that the modelled series stays
the premium index rather than the censored funding rate (Section 3.2 of the
paper).

    F_t = P̄_t + clamp(r - P̄_t, -5e-4, +5e-4),   r = 1e-4 per 8h
    realised hourly = F_t / 8
    floor (premium inside band) => hourly = r/8 = 0.125 bps/h

NOTE: the modelled premium series is the exchange-published premium index P̄_t
(the ``premium`` field of the ``fundingHistory`` endpoint), NOT ``F_t - r/8``.
These tests import the production code in ``regime_engine.funding`` rather
than re-implementing ``funding_8h()``, so a regression in the shipped package
makes them fail.
"""
import numpy as np
import pandas as pd
import pytest

from regime_engine.funding import (
    CLAMP,
    FLOOR_BPS_PER_HOUR,
    R_8H,
    clamp_is_binding,
    clamp_share,
    funding_8h,
    hourly_funding_bps,
)
from regime_engine.paths import DEFAULT_PANEL


def test_floor_value_is_0125_bps_per_hour():
    # premium well inside the clamp band -> funding pins to the interest floor
    f8 = funding_8h(0.0)
    hourly_bps = (f8 / 8) * 1e4
    assert np.isclose(hourly_bps, FLOOR_BPS_PER_HOUR, atol=1e-9)
    assert np.isclose(hourly_funding_bps(0.0), FLOOR_BPS_PER_HOUR, atol=1e-9)


def test_identity_region_collapses_to_interest():
    # for any small premium inside the band, F = r exactly
    for p in (-2e-4, -1e-4, 0.0, 1e-4, 3e-4):
        assert np.isclose(funding_8h(p), R_8H, atol=1e-12)


def test_premium_recovers_offfloor_variation():
    # a large positive premium saturates the clamp on the negative side
    p = 2.36e-3            # ~23.6 bps 8h-equivalent, as in the full-sample max
    f8 = funding_8h(p)
    assert np.isclose(f8, p - CLAMP, atol=1e-12)
    hourly_bps = (f8 / 8) * 1e4
    assert hourly_bps > FLOOR_BPS_PER_HOUR       # off the floor


def test_band_edges_are_asymmetric_in_the_premium():
    """The floor binds for P̄ in (-4, +6) bps 8h-equivalent, not |P̄| < 5 bps."""
    assert np.isclose(funding_8h(-3.9e-4), R_8H, atol=1e-12)     # inside
    assert np.isclose(funding_8h(+5.9e-4), R_8H, atol=1e-12)     # inside
    assert not np.isclose(funding_8h(-4.1e-4), R_8H, atol=1e-12)  # outside, low
    assert not np.isclose(funding_8h(+6.1e-4), R_8H, atol=1e-12)  # outside, high


def test_funding_8h_is_vectorised():
    p = np.array([-1e-3, 0.0, 1e-3])
    out = funding_8h(p)
    assert out.shape == p.shape
    assert np.allclose(out, [p[0] + CLAMP, R_8H, p[2] - CLAMP])


def test_clamp_share_counts_floor_hours():
    funding = np.array([FLOOR_BPS_PER_HOUR, FLOOR_BPS_PER_HOUR, 0.9, -0.4])
    assert clamp_share(funding) == pytest.approx(0.5)
    assert clamp_is_binding(funding).tolist() == [True, True, False, False]


@pytest.mark.skipif(not DEFAULT_PANEL.exists(), reason="shipped panel not present")
def test_shipped_panel_funding_is_reproduced_from_the_premium_column():
    """The real code path, on the real data.

    ``funding_bps`` must be recoverable from the ``premium`` column through the
    clamp — this is what identifies ``premium`` as the exchange premium index
    rather than ``F_t - r/8``.
    """
    panel = pd.read_csv(DEFAULT_PANEL)
    recomputed = hourly_funding_bps(panel["premium"].values)
    assert np.max(np.abs(recomputed - panel["funding_bps"].values)) < 1e-6


@pytest.mark.skipif(not DEFAULT_PANEL.exists(), reason="shipped panel not present")
def test_shipped_panel_clamp_share_is_35_4_percent():
    """Regression lock on the clamp share of the shipped sub-sample panel.

    The paper's 46.0% refers to the full paper window (pinned in
    ``TestPaperWindowCensoringCount`` below) and cannot be computed from this
    panel; on the Dec 2025 - Apr 2026 panel the share is 35.44%.
    """
    panel = pd.read_csv(DEFAULT_PANEL)
    share = clamp_share(panel["funding_bps"].values)
    assert share == pytest.approx(0.35443, abs=5e-5)


class TestPaperWindowCensoringCount:
    """The 46% the manuscript quotes is the *exact* count, not the tolerant one.

    ``clamp_share`` counts hours within 1e-3 bps of the floor and returns 46.11%;
    the paper quotes the exact 2,300 of 5,001 = 45.99%. Both round to the 46% in
    the text, but a replicator running the documented snippet sees the tolerant
    figure, so the two are pinned here together with the identity that justifies
    the paper's reading of the censoring as a property of the premium's law.
    """

    @pytest.fixture(scope="class")
    def window(self):
        import pandas as pd

        from regime_engine.paths import DEFAULT_PREMIUM_FULL, PAPER_WINDOW

        d = pd.read_csv(DEFAULT_PREMIUM_FULL)
        d["time"] = pd.to_datetime(d["time"], utc=True, format="ISO8601")
        lo, hi = (pd.Timestamp(PAPER_WINDOW[0], tz="UTC"),
                  pd.Timestamp(PAPER_WINDOW[1]))
        return d[(d["time"] >= lo) & (d["time"] <= hi)]

    def test_exact_count_on_the_published_funding_rate(self, window):
        n = int((window["fundingRate"].to_numpy(float) == 1.25e-05).sum())
        assert n == 2300
        assert round(100 * n / len(window), 2) == 45.99

    def test_the_floor_hours_are_exactly_the_hours_inside_the_identity_band(self, window):
        """Section 3.2's claim: the censored fraction is P(premium in (-4, +6))."""
        on_floor = window["fundingRate"].to_numpy(float) == 1.25e-05
        y = window["premium"].to_numpy(float) * 1e4
        assert np.array_equal(on_floor, (y > -4) & (y < 6))

    def test_the_tolerant_share_differs_by_exactly_six_hours(self, window):
        from regime_engine.funding import clamp_share, hourly_funding_bps

        y = window["premium"].to_numpy(float)
        tolerant = round(clamp_share(hourly_funding_bps(y)) * len(window))
        assert tolerant - 2300 == 6
