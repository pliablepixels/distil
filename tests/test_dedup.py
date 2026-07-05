"""Cross-turn reversible dedup — references recurring inert blocks, reversibly."""

from __future__ import annotations

from distil.compress.dedup import StreamingDedup
from distil.trajectory import Block, Kind, Stability

_BIG = "DESIGN DOC (re-read each turn): " + ("reference background prose. " * 30)


def _block(text, bid="pin", dr=False):
    return Block(
        id=bid, kind=Kind.RETRIEVED, text=text, stability=Stability.VOLATILE, decision_relevant=dr
    )


def test_recurring_block_is_referenced_after_first_sight():
    dd = StreamingDedup()
    out0, r0 = dd.compress([_block(_BIG)], 0)
    out1, r1 = dd.compress([_block(_BIG)], 1)
    assert out0[0].text == _BIG and not r0  # first sight: unchanged
    assert out1[0].text.startswith("«repeat")  # recurrence: referenced
    assert len(out1[0].text) < len(_BIG) * 0.2  # big reduction
    assert _BIG in set(r1.values())  # original recoverable


def test_dedup_never_touches_decision_or_small_blocks():
    dd = StreamingDedup()
    dec = _BIG + "\nDECISION: act"
    dd.compress([_block(dec, dr=True)], 0)
    out, r = dd.compress([_block(dec, dr=True)], 1)
    assert out[0].text == dec and not r  # decision block: never referenced
    small = _block("tiny", bid="s")
    dd.compress([small], 0)
    out2, _ = dd.compress([small], 1)
    assert out2[0].text == "tiny"  # below min_chars: untouched


def test_reset_on_new_pass():
    dd = StreamingDedup()
    dd.compress([_block(_BIG)], 0)
    dd.compress([_block(_BIG)], 1)
    # a non-increasing turn index signals a fresh measurement pass → clean memory
    out, r = dd.compress([_block(_BIG)], 0)
    assert out[0].text == _BIG and not r


def test_repeat_marker_is_expand_recoverable():
    """The «repeat» marker must carry the handle= form the expand path keys on."""
    from distil.replay.expand_runner import _HANDLE_IN_TEXT

    dd = StreamingDedup()
    dd.compress([_block(_BIG)], 0)
    out, restore = dd.compress([_block(_BIG)], 1)
    marker = out[0].text
    assert "repeat of earlier tool output" in marker
    m = _HANDLE_IN_TEXT.search(marker)
    assert m, f"expand regex must match the repeat marker: {marker!r}"
    assert restore[m.group(1)] == _BIG  # handle resolves to the byte-exact original
