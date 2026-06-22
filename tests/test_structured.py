"""Reversible structured compaction (columnar fold) — savings AND reversibility."""

from __future__ import annotations

import json

from distil.compress.structured import fold, is_folded
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


def test_tier1_fold_is_byte_reversible():
    b = Block(id="obs", kind=Kind.TOOL_OUTPUT, text=_ARRAY, stability=Stability.VOLATILE)
    result = Tier1Reversible().compress([b])
    assert is_folded(result.blocks[0].text)
    assert len(result.blocks[0].text) < len(_ARRAY)
    # the byte-exact original is recoverable from the restore table
    rep = verify_reversible([b], result)
    assert rep.lossless
    assert _ARRAY in set(result.restore.values())
