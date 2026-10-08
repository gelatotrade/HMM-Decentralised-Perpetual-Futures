"""Tests for the winsorisation robustness summary.

``describe_fit`` is what writes ``output/robustness.json``, and the caption of
the winsorisation table (``table5_robustness.tex``) makes a claim about one
specific cell of the transition kernel: that the direct high-to-low-volatility
probability "remains zero in both fits". A summary that identifies that cell
*positionally* cannot support the claim -- the position depends on the
labelling rule -- so the cell has to be named.
"""
import numpy as np

from regime_engine.hmm import GaussianHMM
from regime_engine.robustness import describe_fit


def _fitted(sigma_order):
    """A GaussianHMM carrying the archived sub-sample kernel, permuted.

    ``sigma_order`` chooses the labelling: "ascending" is the paper's rule,
    "descending" is what a mu-ascending sort produces on this fit.
    The economics are identical in both -- only the index of each state moves.
    """
    A = np.array([                       # sigma-ascending: low, moderate, high
        [0.929968, 0.067547, 0.002485],
        [0.021178, 0.886824, 0.091998],
        [0.000000, 0.264988, 0.735012],  # high -> low sits on the boundary
    ])
    mu = np.array([-0.10977, -0.22292, -1.88329]) / 1e4
    sigma = np.array([13.7352, 33.1148, 93.6845]) / 1e4
    if sigma_order == "descending":
        p = np.array([2, 1, 0])
        A, mu, sigma = A[np.ix_(p, p)], mu[p], sigma[p]
    hmm = GaussianHMM(K=3)
    hmm.A, hmm.mu, hmm.sigma = A, mu, sigma
    hmm.pi = np.full(3, 1 / 3)
    hmm.log_likelihood_ = 14323.84356343513      # the archived sub-sample value
    return hmm


def test_summary_names_the_high_to_low_transition():
    """The boundary cell is reported under a name, not a position."""
    d = describe_fit(_fitted("ascending"), T=3507)
    assert "A_high_to_low" in d, sorted(d)
    assert d["A_high_to_low"] == 0.0, d["A_high_to_low"]


def test_named_transition_survives_a_relabelling():
    """The same economic cell, whichever way the states are indexed."""
    asc = describe_fit(_fitted("ascending"), T=3507)
    desc = describe_fit(_fitted("descending"), T=3507)
    assert asc["A_high_to_low"] == desc["A_high_to_low"] == 0.0
    assert np.isclose(asc["A_high_to_moderate"], desc["A_high_to_moderate"])
    assert np.isclose(asc["A_high_to_moderate"], 0.264988)


def test_states_are_reported_in_sigma_ascending_order():
    """Whatever order the caller hands in, the summary is the paper's order."""
    for order in ("ascending", "descending"):
        d = describe_fit(_fitted(order), T=3507)
        assert np.all(np.diff(d["sigma_bps"]) > 0), (order, d["sigma_bps"])
        assert d["sigma_rank"] == [0, 1, 2], (order, d["sigma_rank"])


def test_winsorisation_table_is_written_in_the_papers_order(tmp_path):
    """output/tables/table5_robustness.tex is generated from the fit, never edited by hand."""
    from regime_engine.robustness import write_robustness_table
    payload = {
        "T": 3507,
        "raw": {"ann_vol_pct": [12.86, 30.99, 87.68]},
        "winsorised": {"ann_vol_pct": [12.56, 30.36, 85.36]},
        "winsor_n_clipped": 4,
    }
    out = tmp_path / "tables"
    write_robustness_table(payload, tmp_path)
    tex = (out / "table5_robustness.tex").read_text()
    assert "Crisis" not in tex and "Calm" not in tex, tex
    assert "1 low" in tex and "2 moderate" in tex and "3 high" in tex, tex
    # low column first, high last -- the paper's order
    body = [ln for ln in tex.splitlines() if ln.startswith("Raw")][0]
    assert body.index("12.9") < body.index("87.7"), body
