"""Two fits the text quotes without a table, pinned to the artifacts behind them.

Section 3.2 says an MS-AR fitted to the headline rate collapses onto a regime of
numerically zero variance; Section 4.3 quotes a two-hourly refit of the return
model over the whole window. scripts/msar_on_rate.py and
scripts/two_hourly_refit.py write the artifacts behind both numbers.
"""
import json
import pathlib

import pytest

from regime_engine.paths import DEFAULT_OUT, repo_root


def _load(name):
    path = pathlib.Path(DEFAULT_OUT) / name
    if not path.exists():                       # pragma: no cover
        pytest.skip(f"run the script behind {name} first")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def tex():
    return (repo_root() / "paper" / "paper.tex").read_text()


def test_the_msar_on_the_rate_collapses_onto_a_zero_variance_regime(tex):
    d = _load("msar_on_rate.json")
    assert d["T"] == 5001
    assert d["max_collapsed_sigma2"] < 1e-30            # "of order 10^-32"
    assert {round(phi, 2) for phi in d["ordinary_phi"]} == {0.81}
    assert r"$\hat\sigma^2$ of order $10^{-32}$" in tex
    assert r"one ordinary regime with $\hat\phi = 0.81$" in tex


def test_the_two_hourly_refit_matches_section_4_3(tex):
    d = _load("two_hourly_refit.json")
    assert d["bars"] == 2500
    vols = [round(v, 1) for v in d["annualised_sigma_pct"]]
    assert vols == [11.0, 31.6, 88.8]
    assert round(100 * d["stationary"][2], 1) == 21.0
    assert round(d["A"][2][0], 2) == 0.01
    assert r"($2{,}500$ bars, annualised with $\sqrt{12\cdot 365}$)" in tex
    assert r"give $11.0\%$, $31.6\%$ and $88.8\%$, with a high-state mass of $21.0\%$" in tex
