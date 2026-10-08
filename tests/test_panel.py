"""No-lookahead & alignment invariants for the assembled panel.

These exercise ``regime_engine.panel.build_hourly_panel`` — the single
implementation of the alignment rules — and then
assert the same invariants hold for the panel shipped in the repository.
"""
import numpy as np
import pandas as pd
import pytest

from regime_engine.panel import build_hourly_panel, load_panel, panel_fingerprint
from regime_engine.paths import DEFAULT_PANEL


def _candles(n=8, start="2026-01-01 00:00", closes=None):
    t = pd.date_range(start, periods=n, freq="h", tz="UTC")
    c = np.asarray(closes, dtype=float) if closes is not None else \
        100.0 + np.arange(n, dtype=float)
    return pd.DataFrame({
        "time": t, "o": c, "h": c + 1.0, "l": c - 1.0, "c": c,
        "v": np.full(n, 10.0),
    })


def _funding(times):
    times = pd.DatetimeIndex(times)
    return pd.DataFrame({
        "time": times,
        "fundingRate": np.linspace(1e-5, 2e-5, len(times)),
        "premium": np.linspace(-3e-4, 3e-4, len(times)),
    })


# ----------------------------------------------------------------------
def test_hourly_return_uses_only_past_close():
    """r_t must depend only on closes in (t-1h, t] — never on a future close."""
    candles = _candles(n=8)
    funding = _funding(candles["time"])
    base = build_hourly_panel(candles, funding)

    # Perturbing a FUTURE close must leave earlier returns untouched.
    bumped = candles.copy()
    bumped.loc[5, "c"] = bumped.loc[5, "c"] * 1.10
    after = build_hourly_panel(bumped, funding)
    assert np.allclose(base["log_return"].values[:4], after["log_return"].values[:4])

    # ... while r_t and r_{t+1} — the two returns that legitimately involve
    # close_5 — do move.
    assert not np.isclose(base["log_return"].values[4], after["log_return"].values[4])
    assert not np.isclose(base["log_return"].values[5], after["log_return"].values[5])

    # And the value itself is exactly log(c_t / c_{t-1}).
    expected = np.log(candles["c"].values[1:] / candles["c"].values[:-1])
    assert np.allclose(base["log_return"].values, expected)


def test_funding_not_forward_filled():
    """A missing funding stamp is dropped, never imputed from a neighbour."""
    candles = _candles(n=8)
    funding = _funding(candles["time"]).drop(index=4).reset_index(drop=True)  # hole at 04:00
    panel = build_hourly_panel(candles, funding)

    missing = candles["time"].iloc[4]
    assert missing not in set(panel["time"])          # dropped, not filled
    assert len(panel) == 6                            # 7 joined rows, minus the first

    # No value was copied forward across the hole: every surviving funding value
    # still matches its own stamp in the source table.
    src = _funding(candles["time"]).drop(index=4).set_index("time")["fundingRate"]
    for _, row in panel.iterrows():
        assert row["fundingRate"] == pytest.approx(src.loc[row["time"]])

    # The return at 05:00 must still be log(c_05 / c_04) — the true previous
    # hourly close — and NOT log(c_05 / c_03), which is what computing returns
    # after the join would silently produce.
    row = panel.loc[panel["time"] == candles["time"].iloc[5]].iloc[0]
    one_hour = np.log(candles["c"].iloc[5] / candles["c"].iloc[4])
    bridged = np.log(candles["c"].iloc[5] / candles["c"].iloc[3])
    assert row["log_return"] == pytest.approx(one_hour)
    assert not np.isclose(row["log_return"], bridged)


def test_gap_in_the_candle_grid_invalidates_the_return():
    """A missing candle must not produce a two-hour return labelled hourly."""
    candles = _candles(n=8).drop(index=4).reset_index(drop=True)   # no 04:00 candle
    funding = _funding(pd.date_range("2026-01-01 00:00", periods=8, freq="h", tz="UTC"))
    panel = build_hourly_panel(candles, funding)

    stamps = list(panel["time"])
    assert pd.Timestamp("2026-01-01 04:00", tz="UTC") not in stamps  # no candle
    assert pd.Timestamp("2026-01-01 05:00", tz="UTC") not in stamps  # return undefined
    assert pd.Timestamp("2026-01-01 06:00", tz="UTC") in stamps      # defined again


def test_inner_join_drops_first_row():
    """The first joined row has no lagged close, so it cannot survive."""
    candles = _candles(n=6)
    funding = _funding(candles["time"])
    panel = build_hourly_panel(candles, funding)

    assert len(panel) == len(candles) - 1
    assert panel["time"].iloc[0] == candles["time"].iloc[1]
    assert panel["log_return"].notna().all()

    # A funding stamp with no candle is likewise dropped (inner join both ways).
    extra = pd.concat(
        [funding, _funding(pd.DatetimeIndex(["2026-01-02 00:00"]).tz_localize("UTC"))],
        ignore_index=True,
    )
    assert len(build_hourly_panel(candles, extra)) == len(candles) - 1


# ----------------------------------------------------------------------
#  The same invariants, asserted against the panel actually shipped here.
# ----------------------------------------------------------------------
@pytest.mark.skipif(not DEFAULT_PANEL.exists(), reason="shipped panel not present")
class TestShippedPanel:
    @pytest.fixture(scope="class")
    def panel(self):
        return load_panel(DEFAULT_PANEL)

    def test_grid_is_strictly_increasing_hourly_utc(self, panel):
        dt = panel["time"].diff().dropna().unique()
        assert list(dt) == [pd.Timedelta("1h")]
        assert str(panel["time"].dt.tz) == "UTC"
        assert not panel["time"].duplicated().any()

    def test_log_return_matches_own_closes(self, panel):
        """Every row after the first: r_t == log(c_t / c_{t-1}), to machine eps."""
        recomputed = np.log(panel["c"].values[1:] / panel["c"].values[:-1])
        assert np.max(np.abs(panel["log_return"].values[1:] - recomputed)) < 1e-12

    def test_first_row_return_references_a_close_outside_the_file(self, panel):
        """The shipped file is a *slice* of a longer panel.

        Its first row therefore carries a return computed against a close that
        is not in the file. That is not a lookahead violation (the close is in
        the past), but it means row 0 is not self-verifiable — documented here
        so nobody "fixes" it by recomputing row 0 from row 0.
        """
        assert np.isfinite(panel["log_return"].iloc[0])
        implied_prev_close = panel["c"].iloc[0] / np.exp(panel["log_return"].iloc[0])
        assert implied_prev_close > 0
        assert not np.isclose(implied_prev_close, panel["c"].iloc[0])

    def test_no_nans_in_modelled_columns(self, panel):
        assert panel[["c", "log_return", "premium", "funding_bps"]].notna().all().all()

    def test_fingerprint_is_content_sensitive(self, panel):
        """The HMM cache key must change when the data change."""
        original = panel_fingerprint(panel)
        assert original == panel_fingerprint(panel.copy())
        mutated = panel.copy()
        mutated.loc[10, "log_return"] = mutated.loc[10, "log_return"] + 1e-9
        assert panel_fingerprint(mutated) != original
