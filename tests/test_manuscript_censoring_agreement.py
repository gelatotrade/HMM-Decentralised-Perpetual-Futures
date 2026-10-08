"""Pin the cross-venue censoring figures the manuscript prints to the artifact.

Section 3.2, the abstract, the introduction and the conclusion quote shares of
settlements at the funding floor for three venues and two windows, and the
premium means and standard deviations behind them. Each must equal
``output/censoring_comparison.json`` at the precision printed.
"""
import json
import pathlib

import pytest

from regime_engine.paths import DEFAULT_OUT, repo_root


@pytest.fixture(scope="module")
def venues():
    path = pathlib.Path(DEFAULT_OUT) / "censoring_comparison.json"
    if not path.exists():                       # pragma: no cover
        pytest.skip("run `python3 -m regime_engine.censoring` first")
    return json.loads(path.read_text())["venues"]


@pytest.fixture(scope="module")
def tex():
    return (repo_root() / "paper" / "paper.tex").read_text()


def pct(x):
    return f"${100 * x:.1f}\\%$"


def test_shares_at_the_floor_in_the_paper_window(venues, tex):
    for venue in ("hyperliquid", "hyperliquid_at_8h_stamps", "binance", "bybit"):
        assert pct(venues[venue]["paper"]["share_at_floor"]) in tex, venue


def test_shares_at_the_floor_after_the_window(venues, tex):
    for venue in ("hyperliquid", "binance", "bybit"):
        assert pct(venues[venue]["later"]["share_at_floor"]) in tex, venue


def test_counts_quoted_in_section_3(venues, tex):
    b, y = venues["binance"]["paper"], venues["bybit"]["paper"]
    assert f"${b['at_floor']}$ of ${b['n']}$ settlements ({pct(b['share_at_floor'])})" in tex
    assert f"Bybit in ${y['at_floor']}$ ({pct(y['share_at_floor'])})" in tex


def test_shares_below_the_band(venues, tex):
    for venue in ("binance", "bybit"):
        assert pct(venues[venue]["paper"]["share_below"]) in tex, venue
        assert venues[venue]["paper"]["above"] == 0       # "neither exchange settles above"


def test_premium_locations(venues, tex):
    hl = venues["hyperliquid"]["paper"]
    assert f"median ${hl['premium']['median_bps']:.2f}$ bps" in tex
    assert f"lifts its mean to ${hl['premium']['mean_bps']:.2f}$" in tex
    assert pct(hl["share_below"]) in tex
    b, y = venues["binance"]["paper"]["premium"], venues["bybit"]["paper"]["premium"]
    assert f"medians of ${b['median_bps']:.2f}$ and ${y['median_bps']:.2f}$ bps" in tex
    widths = [prem["q75_bps"] - prem["q25_bps"] for prem in (b, y)]
    assert f"interquartile ranges of ${widths[0]:.2f}$ and ${widths[1]:.2f}$ bp" in tex
