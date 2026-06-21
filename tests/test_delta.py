"""Tests for distil/delta.py — delta / append-only context encoding."""

from __future__ import annotations


from distil.trajectory import Block, Kind, Stability
from distil.delta import diff_context, replay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_block(id: str, text: str, kind: Kind = Kind.HISTORY) -> Block:
    return Block(id=id, kind=kind, text=text, stability=Stability.SETTLING)


# ---------------------------------------------------------------------------
# diff_context tests
# ---------------------------------------------------------------------------


def test_identical_prefix_all_references() -> None:
    """An identical prefix should be entirely referenced with zero materialization."""
    prev = [
        make_block("a", "hello world"),
        make_block("b", "foo bar"),
    ]
    curr = [
        make_block("a", "hello world"),
        make_block("b", "foo bar"),
    ]
    delta = diff_context(prev, curr)

    assert delta.references == ["a", "b"]
    assert delta.materialized == []
    assert delta.tokens_materialized == 0
    assert delta.tokens_referenced > 0


def test_changed_block_is_materialized() -> None:
    """A block with the same id but different text must be materialized."""
    prev = [make_block("a", "original text")]
    curr = [make_block("a", "changed text")]

    delta = diff_context(prev, curr)

    assert delta.references == []
    assert len(delta.materialized) == 1
    assert delta.materialized[0].id == "a"
    assert delta.materialized[0].text == "changed text"


def test_new_block_is_materialized() -> None:
    """A block with a brand-new id must be materialized."""
    prev = [make_block("a", "stable")]
    curr = [make_block("a", "stable"), make_block("b", "new block")]

    delta = diff_context(prev, curr)

    assert "a" in delta.references
    materialized_ids = [b.id for b in delta.materialized]
    assert "b" in materialized_ids
    assert "a" not in materialized_ids


def test_round_trip_reconstruction() -> None:
    """replay(prev, diff_context(prev, curr)) must reconstruct curr exactly."""
    prev = [
        make_block("sys", "system prompt"),
        make_block("h1", "turn 1 history"),
    ]
    curr = [
        make_block("sys", "system prompt"),  # unchanged
        make_block("h1", "turn 1 history"),  # unchanged
        make_block("h2", "turn 2 history"),  # new
    ]

    delta = diff_context(prev, curr)
    reconstructed = replay(prev, delta)

    assert len(reconstructed) == len(curr)
    for orig, rec in zip(curr, reconstructed):
        assert rec.id == orig.id, f"id mismatch: {rec.id!r} != {orig.id!r}"
        assert rec.text == orig.text, f"text mismatch for id={orig.id!r}"


def test_round_trip_interleaved() -> None:
    """Round-trip with an interleaved new block (not just appended at the end)."""
    prev = [
        make_block("a", "alpha"),
        make_block("b", "beta"),
        make_block("c", "gamma"),
    ]
    # curr keeps a and c unchanged, replaces b, and adds d at the end.
    curr = [
        make_block("a", "alpha"),
        make_block("b", "CHANGED beta"),  # changed
        make_block("c", "gamma"),
        make_block("d", "delta"),  # new
    ]

    delta = diff_context(prev, curr)
    reconstructed = replay(prev, delta)

    for orig, rec in zip(curr, reconstructed):
        assert rec.id == orig.id
        assert rec.text == orig.text


def test_reduction_between_zero_and_one() -> None:
    """reduction must always be in [0, 1]."""
    prev = [make_block("x", "something")]
    curr = [make_block("x", "something")]
    delta = diff_context(prev, curr)
    assert 0.0 <= delta.reduction <= 1.0


def test_reduction_high_when_little_changed() -> None:
    """reduction should be high when only a small block changed."""
    prev = [make_block(str(i), "stable block content " * 20) for i in range(10)]
    curr = list(prev)  # shallow copy — all unchanged
    # Replace last block with a tiny change.
    curr[-1] = make_block(str(len(prev) - 1), "tiny change")

    delta = diff_context(prev, curr)
    # 9 large stable blocks referenced vs 1 tiny changed block.
    assert delta.reduction > 0.8, f"Expected high reduction, got {delta.reduction:.3f}"


def test_reduction_zero_when_everything_new() -> None:
    """reduction should be 0 when prev is empty (nothing to reference)."""
    prev: list[Block] = []
    curr = [make_block("a", "brand new")]
    delta = diff_context(prev, curr)
    assert delta.reduction == 0.0


def test_reduction_one_when_all_referenced() -> None:
    """reduction should be 1.0 when everything is referenced and nothing materialized."""
    prev = [make_block("a", "hello")]
    curr = [make_block("a", "hello")]
    delta = diff_context(prev, curr)
    assert delta.tokens_materialized == 0
    assert delta.reduction == 1.0


def test_empty_contexts() -> None:
    """Both prev and curr empty is a no-op."""
    delta = diff_context([], [])
    assert delta.references == []
    assert delta.materialized == []
    assert delta.tokens_referenced == 0
    assert delta.tokens_materialized == 0
    assert delta.reduction == 0.0
    assert replay([], delta) == []


def test_order_preserved_in_replay() -> None:
    """replay must return blocks in the exact same order as curr."""
    prev = [
        make_block("a", "aaa"),
        make_block("c", "ccc"),
    ]
    # curr interleaves: reference, new, reference, new
    curr = [
        make_block("a", "aaa"),  # ref
        make_block("b", "bbb"),  # new
        make_block("c", "ccc"),  # ref
        make_block("d", "ddd"),  # new
    ]
    delta = diff_context(prev, curr)
    result = replay(prev, delta)
    ids = [b.id for b in result]
    assert ids == ["a", "b", "c", "d"]


def test_token_counts_non_negative() -> None:
    """Token counts must never be negative."""
    prev = [make_block("a", "text")]
    curr = [make_block("a", "text"), make_block("b", "more text")]
    delta = diff_context(prev, curr)
    assert delta.tokens_referenced >= 0
    assert delta.tokens_materialized >= 0
