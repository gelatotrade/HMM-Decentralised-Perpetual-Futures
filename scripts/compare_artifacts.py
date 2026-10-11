"""Check that a regenerated JSON artifact keeps every value of the previous one.

    python3 scripts/compare_artifacts.py OLD.json NEW.json [--allow-removed KEY ...]

Every leaf of OLD (a number, string, boolean or null, addressed by its key
path) must be present in NEW with the identical value; lists must keep their
length. Keys that only NEW has are additions and are counted, not flagged.
``--allow-removed`` names keys (by their last path component) that may be
absent from NEW. The exit status is 0 when nothing changed, 1 otherwise.

Typical use, against the committed version of an artifact:

    git show HEAD:output/oos_robustness.json > /tmp/old.json
    python3 scripts/compare_artifacts.py /tmp/old.json output/oos_robustness.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter


def _same(a, b) -> bool:
    """Identical JSON leaves: exact float equality, and True is not 1."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def compare(old, new, allow_removed=()) -> dict:
    """Walk OLD and report changed and removed key paths, and count added keys."""
    out = {"compared": 0, "changed": [], "removed": [], "added": Counter()}

    def walk(o, n, path):
        if isinstance(o, dict):
            if not isinstance(n, dict):
                out["changed"].append((path, o, n))
                return
            for key, value in o.items():
                sub = f"{path}.{key}" if path else str(key)
                if key not in n:
                    if key not in allow_removed:
                        out["removed"].append(sub)
                    continue
                walk(value, n[key], sub)
            for key in n:
                if key not in o:
                    out["added"][key] += 1
            return
        if isinstance(o, list):
            if not isinstance(n, list) or len(n) != len(o):
                out["changed"].append((path, o, n))
                return
            for i, (a, b) in enumerate(zip(o, n)):
                walk(a, b, f"{path}[{i}]")
            return
        out["compared"] += 1
        if not _same(o, n):
            out["changed"].append((path, o, n))

    walk(old, new, "")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--allow-removed", nargs="*", default=[], metavar="KEY")
    args = ap.parse_args(argv)
    with open(args.old) as fh:
        old = json.load(fh)
    with open(args.new) as fh:
        new = json.load(fh)
    res = compare(old, new, allow_removed=set(args.allow_removed))
    print(f"{res['compared']} values compared; {len(res['changed'])} changed; "
          f"{len(res['removed'])} removed")
    for path, a, b in res["changed"][:50]:
        print(f"  CHANGED {path}: {a!r} -> {b!r}")
    for path in res["removed"][:50]:
        print(f"  REMOVED {path}")
    if res["added"]:
        print("added keys: " + ", ".join(f"{k} x{c}" for k, c in sorted(res["added"].items())))
    return 1 if res["changed"] or res["removed"] else 0


if __name__ == "__main__":
    sys.exit(main())
