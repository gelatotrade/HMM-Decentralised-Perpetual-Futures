"""Pin the Section 5.3 seed sentence against the artifact behind it.

The manuscript says that across six disjoint blocks of 40 seeds the inferior
MS-AR mode attracts between 20 and 30 of the 40 estimations, and that the
ranking of the two modes never changes. ``output/msar_seed_variation.json``
records those runs (``scripts/msar_seed_variation.py``); this test fails
if the artifact and the prose part company.
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
    m = re.search(r"across six disjoint blocks of \$40\$ seeds the inferior mode attracts "
                  r"between \$(\d+)\$ and \$(\d+)\$ of the \$40\$ estimations", tex)
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
