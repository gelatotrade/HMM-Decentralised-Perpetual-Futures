"""scripts/compare_artifacts.py: a regenerated artifact must keep every old value."""
import importlib.util
import json

import pytest

from regime_engine.paths import repo_root


@pytest.fixture(scope="module")
def cmp():
    path = repo_root() / "scripts" / "compare_artifacts.py"
    spec = importlib.util.spec_from_file_location("compare_artifacts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OLD = {"a": 1.5, "b": [1, 2, {"c": None, "d": "x"}], "e": True, "gone": None}


def test_additions_are_counted_not_flagged(cmp):
    new = json.loads(json.dumps(OLD))
    new["b"][2]["extra"] = 3
    new["more"] = {"z": 1}
    res = cmp.compare(OLD, new)
    assert res["changed"] == [] and res["removed"] == []
    assert res["added"] == {"extra": 1, "more": 1}
    assert res["compared"] == 7                 # a, b[0], b[1], c, d, e, gone


def test_a_changed_leaf_is_reported_with_its_path(cmp):
    new = json.loads(json.dumps(OLD))
    new["b"][2]["c"] = 0.0
    res = cmp.compare(OLD, new)
    assert res["changed"] == [("b[2].c", None, 0.0)]


def test_floats_must_match_exactly(cmp):
    new = dict(OLD, a=1.5 + 1e-15)
    assert cmp.compare(OLD, new)["changed"] == [("a", 1.5, 1.5 + 1e-15)]


def test_a_boolean_is_not_an_integer(cmp):
    new = dict(OLD, e=1)
    assert cmp.compare(OLD, new)["changed"] == [("e", True, 1)]


def test_removed_keys_fail_unless_allowed(cmp):
    new = {k: v for k, v in OLD.items() if k != "gone"}
    assert cmp.compare(OLD, new)["removed"] == ["gone"]
    assert cmp.compare(OLD, new, allow_removed={"gone"})["removed"] == []


def test_a_list_that_changes_length_is_a_change(cmp):
    new = dict(OLD, b=[1, 2])
    assert [c[0] for c in cmp.compare(OLD, new)["changed"]] == ["b"]


def test_the_command_line_exit_status(cmp, tmp_path):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(json.dumps(OLD))
    new.write_text(json.dumps(dict(OLD, more=1)))
    assert cmp.main([str(old), str(new)]) == 0
    new.write_text(json.dumps(dict(OLD, a=2.0)))
    assert cmp.main([str(old), str(new)]) == 1
