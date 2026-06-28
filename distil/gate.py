"""The relevance gate — the working-set primitive that wins E8/E11, as a library function.

The relevance-gated reversible tier is distil's highest-accuracy compressor: on a long-horizon
agent it is the only condition statistically non-inferior to full context (E8), and that
result transfers across vendors to a far stronger model (E11). Its mechanism is one decision
made every turn: keep the agent's **working set** — the last ``gate_recent`` user/tool
messages — verbatim, and digest only the aged-out periphery behind a recoverable handle.

This module exposes that decision as a pure, dependency-free function so the winning tier is a
shippable primitive rather than benchmark-only code. ``benchmarks/swe_bench_e2e/compress_proxy``
implements the same selection inline against its proxy state; the contract here is identical
(verified in ``tests/test_gate.py``): given the message roles and ``gate_recent``, return the
set of indices to keep full.

The *operating point* is ``gate_recent``. E11's finding — the safe operating point scales with
agent capability (a weak agent tolerates a small working set; a strong one needs a larger one)
— is why :mod:`distil.calibrate` exists: it selects ``gate_recent`` from a calibration set
instead of leaving it a hand-tuned constant with a silent-loss hazard.
"""

from __future__ import annotations

from collections.abc import Sequence

# Roles whose messages form the agent's working set (its observations and instructions);
# assistant/system turns are not gated periphery.
WORKING_SET_ROLES = ("user", "tool")


def working_set_indices(
    roles: Sequence[str], gate_recent: int, *, working_roles: Sequence[str] = WORKING_SET_ROLES
) -> set[int]:
    """Indices of messages to keep **verbatim** under the relevance gate.

    Args:
        roles: per-message role strings, in conversation order.
        gate_recent: working-set size — keep the last ``gate_recent`` working-role messages
            full; digest the rest. ``gate_recent <= 0`` digests everything (no working set);
            ``gate_recent >= len(working-role messages)`` keeps everything (a no-op gate, the
            regime where short conversations have no periphery to digest).
        working_roles: which roles count as working-set messages (default user/tool).

    Returns the set of indices into ``roles`` that must be kept full. Pure and deterministic.
    """
    if gate_recent <= 0:
        return set()
    ut = [i for i, r in enumerate(roles) if r in working_roles]
    return set(ut[-gate_recent:])


def gate_fraction(
    roles: Sequence[str], gate_recent: int, *, working_roles: Sequence[str] = WORKING_SET_ROLES
) -> float:
    """Fraction of working-role messages the gate would digest (compression aggressiveness).

    1.0 means everything is digested (no working set); 0.0 means nothing is (no-op gate).
    A higher fraction is a more aggressive operating point — the axis E11 calibrates over.
    """
    ut = [i for i, r in enumerate(roles) if r in working_roles]
    if not ut:
        return 0.0
    kept = len(working_set_indices(roles, gate_recent, working_roles=working_roles) & set(ut))
    return (len(ut) - kept) / len(ut)
