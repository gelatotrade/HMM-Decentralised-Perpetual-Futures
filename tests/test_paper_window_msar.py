"""The paper's headline funding table must have a machine-readable source.

The funding-parameter table (``paper/tables/table3_funding_params.tex``) reports
the MS-AR(1) fit on the paper window -- regime levels, autoregressive
coefficients with standard errors, half-lives with confidence intervals, and
the half-life ratio with its delta-method interval. Every one of those numbers
has to be recoverable from a shipped artifact, or the table is unreplicable.
The fit is computed inside ``run_diagnostics``, which records its parameters
as well as the residuals taken from it.

These tests read ``output/diagnostics.json`` as shipped. They are the contract
between the manuscript table and the replication package.
"""
import json
import pathlib

import pytest

from regime_engine.paths import DEFAULT_OUT

PAPER_LOGL = -4808.53          # the maximum; NOT the -4828.67 attractor
PAPER_RATIO = 11.2


@pytest.fixture(scope="module")
def paper_window_block():
    path = pathlib.Path(DEFAULT_OUT) / "diagnostics.json"
    if not path.exists():                       # pragma: no cover
        pytest.skip("run `make diagnostics` first")
    blocks = json.loads(path.read_text())["tests"]
    hits = [b["msar"] for b in blocks
            if "msar" in b and "paper window" in b["msar"]["window"]]
    assert len(hits) == 1, [b.get("msar", {}).get("window") for b in blocks]
    return hits[0]


def test_the_archived_fit_is_the_maximum_not_the_attractor(paper_window_block):
    """A single-start fit lands on -4828.67 three times in five; that is not this."""
    assert paper_window_block["logL"] == pytest.approx(PAPER_LOGL, abs=0.01)


def test_ratio_matches_the_manuscript(paper_window_block):
    assert paper_window_block["half_life_ratio"] == pytest.approx(PAPER_RATIO, abs=0.05)


def test_block_carries_every_quantity_table_3_quotes(paper_window_block):
    """Levels, sigmas, standard errors and intervals -- not just the point estimates."""
    b = paper_window_block
    for key in ("mu_bps", "sigma_bps", "phi_se", "half_life_ci95",
                "half_life_ratio_se", "half_life_ratio_ci95", "AIC", "BIC"):
        assert key in b, (key, sorted(b))
    assert len(b["mu_bps"]) == 2 and len(b["phi_se"]) == 2
    assert len(b["half_life_ratio_ci95"]) == 2


def test_reported_intervals_bracket_their_point_estimates(paper_window_block):
    b = paper_window_block
    for tau, (lo, hi) in zip(b["half_life_hours"], b["half_life_ci95"]):
        assert lo <= tau <= hi, (tau, lo, hi)
    lo, hi = b["half_life_ratio_ci95"]
    assert lo <= b["half_life_ratio"] <= hi
    assert lo > 1.0, "the manuscript claims the interval excludes unity"


def test_multistart_diagnostic_records_the_inferior_optimum(paper_window_block):
    """The paper's methodological point only stands if the artifact shows it."""
    ms = paper_window_block["multistart"]
    logls = [round(float(o["llf"]), 2) for o in ms["optima_found"]]
    assert len(logls) == 2, ms
    assert min(logls) == pytest.approx(-4828.67, abs=0.01)
    assert max(logls) == pytest.approx(PAPER_LOGL, abs=0.01)
