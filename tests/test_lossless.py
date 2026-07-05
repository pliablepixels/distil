"""Tier-0/1 must be reversible (or decision-preserving) by construction."""

import json

from distil.compress.tier0 import Tier0Lossless, collapse_runs, minify_json
from distil.compress.tier1 import Tier1Reversible, digest
from distil.trajectory import Block, Kind, Stability


def _expand_runs(text: str) -> str:
    out = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and lines[i + 1].startswith("<<x") and lines[i + 1].endswith(">>"):
            n = int(lines[i + 1][3:-2])
            out.extend([line] * n)
            i += 2
        else:
            out.append(line)
            i += 1
    return "\n".join(out)


def test_minify_json_roundtrips_to_same_object():
    src = '{"a":  1, "b": [1, 2,  3], "c": "x"}'
    minified = minify_json(src)
    assert minified is not None
    assert " " not in minified.replace('"x"', "")
    assert json.loads(minified) == json.loads(src)


def test_minify_json_returns_none_for_non_json():
    assert minify_json("just a log line: disk at 95%") is None


def test_collapse_runs_is_reversible():
    src = "same\nsame\nsame\nsame\ndifferent\n"
    collapsed = collapse_runs(src)
    assert "<<x4>>" in collapsed
    assert _expand_runs(collapsed.rstrip("\n")) == src.rstrip("\n")


def test_tier0_records_originals_for_recovery():
    blocks = [Block("j", Kind.TOOL_OUTPUT, '{"x":  1,  "y": 2}', Stability.VOLATILE)]
    res = Tier0Lossless().compress(blocks)
    assert res.restore["j"] == blocks[0].text  # original fully recoverable


def test_tier1_digest_preserves_decision_lines():
    text = "\n".join(
        ["header noise line"] * 3 + ["DECISION: this line must survive"] + ["more noise"] * 6
    )
    dtext, changed = digest(text)
    assert changed
    assert "DECISION: this line must survive" in dtext
    assert len(dtext) < len(text)


def test_tier1_digest_keeps_high_salience_failure_lines():
    """FIX 5: the salience net keeps failure/diagnostic lines byte-exact, not just
    DECISION: markers — an agent usually needs these verbatim to react."""
    text = "\n".join(
        ["header noise line"] * 3
        + [
            "Traceback (most recent call last):",
            "ValueError: bad input",
            "WARNING: retry exhausted",
        ]
        + ["more noise"] * 6
    )
    dtext, changed = digest(text)
    assert changed
    assert "Traceback (most recent call last):" in dtext
    assert "ValueError: bad input" in dtext
    assert "WARNING: retry exhausted" in dtext
    assert len(dtext) < len(text)  # still compresses the surrounding noise


def test_tier1_keeps_full_original_behind_handle():
    text = "first\nsecond\nthird\nDECISION: keep\n" + "\n".join(f"noise{i}" for i in range(20))
    blocks = [Block("o", Kind.TOOL_OUTPUT, text, Stability.VOLATILE)]
    res = Tier1Reversible().compress(blocks)
    assert res.restore  # original stored under its content handle
    assert text in res.restore.values()
