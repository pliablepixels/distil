"""Reversible structured compaction (columnar fold) — savings AND reversibility."""

from __future__ import annotations

import json

from distil.compress.structured import fold, is_folded, template_fold
from distil.compress.tier1 import Tier1Reversible
from distil.fidelity import verify_reversible
from distil.trajectory import Block, Kind, Stability

_ARRAY = json.dumps(
    [{"id": i, "name": f"svc-{i}", "status": "active", "ok": True} for i in range(20)], indent=2
)


def test_fold_columnarises_and_shrinks():
    out = fold(_ARRAY)
    assert out is not None and is_folded(out)
    assert len(out) < len(_ARRAY) * 0.7  # meaningful reduction
    # information preserved: every record's name still present in the compact form
    for i in range(20):
        assert f"svc-{i}" in out


def test_fold_skips_non_foldable():
    assert fold("just some prose, not json") is None
    assert fold(json.dumps({"a": 1})) is None  # not an array
    assert fold(json.dumps([1, 2, 3])) is None  # not records
    assert fold(json.dumps([{"a": 1}, {"a": 2}])) is None  # < 3 records
    # never fold a block carrying a decision signal
    assert fold('[{"a":1},{"a":2},{"a":3}]\nDECISION: act') is None


def test_fold_skips_nested_objects():
    nested = json.dumps([{"a": {"deep": 1}} for _ in range(5)])
    assert fold(nested) is None  # only flat scalar records fold


def test_template_fold_collapses_near_identical_lines():
    logs = "\n".join(
        f"2026-06-22T10:{i:02d}:00Z INFO request id=req-{1000 + i} handled in {i}ms"
        for i in range(20)
    )
    out = template_fold(logs)
    assert out is not None and is_folded(out)
    assert len(out) < len(logs) * 0.8  # template stated once → real reduction
    # every varying value is retained (information-preserving): ids 1009 & 1019
    assert "1009" in out and "1019" in out


def test_template_fold_skips_unique_and_decision_lines():
    assert template_fold("a\nb\nc\nd\ne") is None  # no shared template
    assert template_fold("x 1\nx 2\nx 3\nx 4\nx 5\nDECISION: act") is None


def test_tier1_fold_is_byte_reversible():
    b = Block(id="obs", kind=Kind.TOOL_OUTPUT, text=_ARRAY, stability=Stability.VOLATILE)
    result = Tier1Reversible().compress([b])
    assert is_folded(result.blocks[0].text)
    assert len(result.blocks[0].text) < len(_ARRAY)
    # the byte-exact original is recoverable from the restore table
    rep = verify_reversible([b], result)
    assert rep.lossless
    assert _ARRAY in set(result.restore.values())
