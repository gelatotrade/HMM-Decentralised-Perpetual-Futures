"""Compare a rebuilt JSON artifact with the shipped one across platforms.

The shipped artifacts were written on macOS/arm64. Elsewhere (CI runs
Linux/x86-64 with OpenBLAS) sums, regressions and bootstrap means can differ in
the last bits, so floats are compared to a relative tolerance; keys, list
lengths, strings, integers, booleans and nulls must match exactly.
"""
from __future__ import annotations

import math

REL = 1e-9
ABS = 1e-12


def mismatches(got, shipped, rel=REL, abs_=ABS, path="") -> list:
    """Key paths at which GOT and SHIPPED differ, with both values."""
    if isinstance(shipped, dict):
        if not isinstance(got, dict) or set(got) != set(shipped):
            return [(path, "keys", sorted(set(got) ^ set(shipped))
                     if isinstance(got, dict) else type(got).__name__)]
        out = []
        for key in shipped:
            out += mismatches(got[key], shipped[key], rel, abs_, f"{path}.{key}" if path else key)
        return out
    if isinstance(shipped, list):
        if not isinstance(got, list) or len(got) != len(shipped):
            return [(path, got, shipped)]
        out = []
        for i, (a, b) in enumerate(zip(got, shipped)):
            out += mismatches(a, b, rel, abs_, f"{path}[{i}]")
        return out
    if isinstance(shipped, float) and isinstance(got, (int, float)) \
            and not isinstance(got, bool):
        if math.isnan(shipped) and math.isnan(got):
            return []
        return [] if math.isclose(got, shipped, rel_tol=rel, abs_tol=abs_) \
            else [(path, got, shipped)]
    if isinstance(shipped, bool) or isinstance(got, bool):
        return [] if got is shipped else [(path, got, shipped)]
    return [] if got == shipped else [(path, got, shipped)]


def assert_json_close(got, shipped, rel=REL, abs_=ABS):
    """Fail with the first differing key paths, not a diff of the whole file."""
    bad = mismatches(got, shipped, rel, abs_)
    assert not bad, f"{len(bad)} differing values, first: {bad[:5]}"
