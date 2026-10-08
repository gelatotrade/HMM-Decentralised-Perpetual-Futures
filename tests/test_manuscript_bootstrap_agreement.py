"""Pin the bootstrap numbers the manuscript prints against the artifact.

Regenerating the bootstrap can move the sub-sample intervals. The generated
table picks the new values up automatically; the sentences in the text that
quote them do not, and would then contradict a table shipped in the same
package.

These are the values the paper prints. If a regeneration moves them, this fails
and the prose has to move with it.
"""
import json
import pathlib

import pytest

from regime_engine.paths import DEFAULT_OUT

# Annualised sigma, 95% bootstrap percentile intervals, sigma-ascending, as
# quoted in Section 4.2 (the discussion of the regime-parameter table,
# paper/tables/table2_regime_params.tex) and used in the sub-period comparison
# of Section 4.3.
FULL_SAMPLE_CI = [(13.6, 15.5), (31.8, 34.9), (83.0, 94.0)]
SUB_SAMPLE_CI = [(11.9, 13.9), (29.4, 32.8), (82.1, 93.7)]

# Parenthesised standard errors of the manuscript's regime-parameter table.
TABLE3_SE = {
    "mu_bps_per_hour": [0.62, 0.78, 3.28],
    "sigma_bps_per_hour": [0.52, 0.87, 2.97],
    "annualised_sigma_pct": [0.49, 0.82, 2.78],
    "expected_duration_hours": [2.08, 0.90, 0.42],
}

LABEL_DISAGREEMENT = 0.402       # Appendix B: "disagree in 40.2% of replications"


@pytest.fixture(scope="module")
def boot():
    path = pathlib.Path(DEFAULT_OUT) / "bootstrap_hmm.json"
    if not path.exists():                       # pragma: no cover
        pytest.skip("run `make bootstrap` first")
    return json.loads(path.read_text())


@pytest.mark.parametrize("block,expected", [
    ("full_sample", FULL_SAMPLE_CI),
    ("sub_sample", SUB_SAMPLE_CI),
])
def test_annualised_intervals_match_the_manuscript(boot, block, expected):
    cells = boot[block]["sigma_ascending"]["annualised_sigma_pct"]
    got = [(round(c["ci95"][0], 1), round(c["ci95"][1], 1)) for c in cells]
    assert got == expected, (block, got, expected)


def test_table3_standard_errors_match_the_manuscript(boot):
    block = boot["full_sample"]["sigma_ascending"]
    for key, expected in TABLE3_SE.items():
        got = [round(c["se"], 2) for c in block[key]]
        assert got == expected, (key, got, expected)


def test_the_drift_that_is_not_distinguishable_from_zero_stays_that_way(boot):
    """The regime-parameter table's mu_3 interval, quoted in the text as [-12.44, +0.27]."""
    cell = boot["full_sample"]["sigma_ascending"]["mu_bps_per_hour"][2]
    lo, hi = round(cell["ci95"][0], 2), round(cell["ci95"][1], 2)
    assert (lo, hi) == (-12.44, 0.27), (lo, hi)
    assert lo < 0 < hi, "the paper says no drift is distinguishable from zero"


def test_label_switching_rate_matches_section_2_1(boot):
    got = boot["full_sample"]["label_switching"]["disagreement_rate"]
    assert round(got, 3) == LABEL_DISAGREEMENT, got


def test_the_two_labelling_blocks_disagree_where_the_paper_says_they_do(boot):
    """Appendix B's contrast: mu-ascending inflates the low-state interval."""
    mu = boot["full_sample"]["mu_ascending"]["sigma_bps_per_hour"]
    sg = boot["full_sample"]["sigma_ascending"]["sigma_bps_per_hour"]
    # sigma-ascending: the 15.55 bps/h state is index 0 and its interval is tight
    lo, hi = sg[0]["ci95"]
    assert (round(lo, 1), round(hi, 1)) == (14.5, 16.5), (lo, hi)
    # mu-ascending: the same component lands at index 1 with a vastly wider one
    lo, hi = mu[1]["ci95"]
    assert (round(lo, 1), round(hi, 1)) == (14.6, 90.7), (lo, hi)
