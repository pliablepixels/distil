"""Cache stabilization must be lossless and must re-stabilize a churning prefix."""

from distil.compress.cache_aware import simulate
from distil.compress.stabilize import (
    canonicalize_json,
    extract_volatile,
    restore_volatile,
    stabilize_blocks,
)
from distil import pricing
from distil.trajectory import Block, Kind, Stability, Trajectory, Turn


def test_canonicalize_json_is_order_invariant():
    a = canonicalize_json('{"b":1,"a":{"d":4,"c":3}}')
    b = canonicalize_json('{"a":{"c":3,"d":4},"b":1}')
    assert a == b is not None  # same bytes regardless of source key order


def test_extract_volatile_roundtrips():
    text = "Current Date: 2026-06-21T09:14:02Z, req=550e8400-e29b-41d4-a716-446655440000"
    stab, vals = extract_volatile(text)
    assert "2026-06-21" not in stab and "550e8400" not in stab
    assert restore_volatile(stab, vals) == text  # fully reversible


def test_placeholders_are_value_independent():
    # two turns, same structure, different volatile values -> identical stabilized text
    s1, _ = extract_volatile("date=2026-06-21T00:00:00Z id=550e8400-e29b-41d4-a716-446655440000")
    s2, _ = extract_volatile("date=2026-06-22T11:11:11Z id=ffffffff-e29b-41d4-a716-446655440000")
    assert s1 == s2


def _churning_traj() -> Trajectory:
    # a STABLE system block that embeds a per-turn timestamp would normally bust
    # the cache; stabilization should rescue it.
    def turn(i: int, ts: str) -> Turn:
        return Turn(
            i,
            [
                Block(
                    "sys",
                    Kind.SYSTEM,
                    f"You are an agent. Current Date: {ts}",
                    Stability.STABLE,
                    True,
                ),
                Block(f"u{i}", Kind.USER, f"request {i}", Stability.VOLATILE),
            ],
        )

    return Trajectory(
        "churn",
        "claude-opus-4",
        [
            turn(0, "2026-06-21T09:00:00Z"),
            turn(1, "2026-06-21T09:05:00Z"),
            turn(2, "2026-06-21T09:10:00Z"),
        ],
    )


def test_stabilization_restores_cache_hits_on_a_churning_prefix():
    traj = _churning_traj()
    price = pricing.get("claude-opus-4-8")
    plain = simulate(traj, price, strategy="none", caching=True)
    stabilized = simulate(traj, price, strategy="distil", caching=True)
    # the un-stabilized churning prefix never caches; distil re-stabilizes it
    assert plain.cache_hit_tokens == 0
    assert stabilized.cache_hit_tokens > 0


def test_stabilize_blocks_is_noop_without_volatile_tokens():
    blocks = [Block("s", Kind.SYSTEM, "stable instructions, no volatile fields", Stability.STABLE)]
    assert stabilize_blocks(blocks) == blocks
