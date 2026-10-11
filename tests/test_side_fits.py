"""Two fits the text quotes without a table, pinned to the artifacts behind them.

Section 6.1 says an MS-AR fitted to the headline rate collapses onto a regime of
numerically zero variance; Section 6.2 quotes a two-hourly refit of the return
model over the whole window. scripts/msar_on_rate.py and
scripts/two_hourly_refit.py write the artifacts behind both; the printed numbers
are checked in test_manuscript_numbers.
"""
import json
import pathlib

import pytest

from regime_engine.paths import DEFAULT_OUT


def _load(name):
    path = pathlib.Path(DEFAULT_OUT) / name
    if not path.exists():                       # pragma: no cover
        pytest.skip(f"run the script behind {name} first")
    return json.loads(path.read_text())


def test_the_msar_on_the_rate_collapses_onto_a_zero_variance_regime():
    d = _load("msar_on_rate.json")
    assert d["T"] == 5001
    assert len(d["fits"]) == 4
    assert d["max_collapsed_sigma2"] < 1e-31
    assert {round(phi, 2) for phi in d["ordinary_phi"]} == {0.81}


def test_the_two_hourly_refit_covers_the_whole_window():
    d = _load("two_hourly_refit.json")
    assert d["bars"] == 2500
    assert d["first_bar"].startswith("2025-10-02") and d["last_bar"].startswith("2026-04-28")
    assert [round(v, 1) for v in d["annualised_sigma_pct"]] == [11.0, 31.7, 88.9]
    assert round(100 * d["stationary"][2], 1) == 20.9
    assert d["converged"]
    assert round(d["A"][2][0], 2) == 0.01
