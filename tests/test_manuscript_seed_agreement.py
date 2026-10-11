"""Pin the seed sentence of Section 6.1 against the artifact behind it.

The manuscript says that each of 240 estimations, six disjoint blocks of 40
seeds, reaches one of two maxima and that the inferior one attracts 20 to 30 of
every 40. ``output/msar_seed_variation.json`` records those runs
(``scripts/msar_seed_variation.py``); this test fails if the artifact and the
prose part company.
"""
import json
import pathlib
import re

import pytest

from regime_engine.paths import DEFAULT_OUT, repo_root


@pytest.fixture(scope="module")
def seeds():
    path = pathlib.Path(DEFAULT_OUT) / "msar_seed_variation.json"
    if not path.exists():                       # pragma: no cover
        pytest.skip("run `PYTHONPATH=src python3 scripts/msar_seed_variation.py` first")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def sentence():
    tex = (repo_root() / "paper" / "paper.tex").read_text()
    m = re.search(r"the inferior one, \$[\d.]+\$ log-likelihood points lower, attracts "
                  r"\$(\d+)\$ to \$(\d+)\$ of every \$40\$ estimations", tex)
    assert m, "the seed sentence is no longer in paper.tex"
    return int(m.group(1)), int(m.group(2))


def test_six_disjoint_blocks_of_forty(seeds):
    runs = seeds["runs"]
    assert len(runs) == 6
    starts = [r["seeds"][0] for r in runs]
    assert all(b - a == 40 for a, b in zip(starts, starts[1:]))


def test_printed_range_matches_the_artifact(seeds, sentence):
    counts = seeds["starts_at_inferior_optimum"]
    assert (counts["min"], counts["max"]) == sentence


def test_every_block_finds_the_same_maximum(seeds):
    assert seeds["best_llf_every_block"] == [-4808.53]


def test_the_first_block_is_the_protocol_the_paper_reports(seeds):
    first = seeds["runs"][0]
    assert first["seed_base"] == 20260802
    assert {o["llf"]: o["count"] for o in first["optima_found"]} == {-4828.67: 24, -4808.53: 16}


def test_the_blocks_ran_the_guarded_scan_and_it_excluded_no_start(seeds):
    """msar.fit_msar_multistart, not the unguarded scan: no start was inadmissible."""
    for run in seeds["runs"]:
        assert run["excluded"] == {"non_finite_llf": 0, "variance_floor": 0,
                                   "singular_hessian": 0}
