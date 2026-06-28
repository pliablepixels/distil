"""The relevance-gate primitive: working-set selection and aggressiveness."""

from __future__ import annotations

from distil.gate import gate_fraction, working_set_indices


def _roles(seq: str) -> list[str]:
    # compact spec: u=user, t=tool, a=assistant, s=system
    m = {"u": "user", "t": "tool", "a": "assistant", "s": "system"}
    return [m[c] for c in seq]


def test_keeps_last_n_working_messages_full():
    roles = _roles("suatuatuat")  # user/tool messages at indices 1,3,4,6,7,9
    ut = [1, 3, 4, 6, 7, 9]
    keep = working_set_indices(roles, gate_recent=2)
    assert keep == set(ut[-2:]) == {7, 9}


def test_ignores_assistant_and_system_for_working_set():
    roles = _roles("saaau")  # only index 4 is a working-set (user) message
    assert working_set_indices(roles, gate_recent=3) == {4}


def test_gate_recent_zero_or_negative_digests_everything():
    roles = _roles("utut")
    assert working_set_indices(roles, gate_recent=0) == set()
    assert working_set_indices(roles, gate_recent=-1) == set()
    assert gate_fraction(roles, gate_recent=0) == 1.0


def test_gate_recent_larger_than_history_is_a_noop():
    roles = _roles("utut")  # 4 working-set messages
    assert working_set_indices(roles, gate_recent=99) == {0, 1, 2, 3}
    assert gate_fraction(roles, gate_recent=99) == 0.0


def test_gate_fraction_is_compression_aggressiveness():
    roles = _roles("uuuu")  # 4 working-set messages
    # keep 1 of 4 full -> digest 3/4
    assert gate_fraction(roles, gate_recent=1) == 0.75
    assert gate_fraction(roles, gate_recent=2) == 0.5


def test_matches_benchmark_proxy_selection_logic():
    # The benchmark proxy computes: ut = [i for i,m ... role in (user,tool)];
    # keep_full = set(ut[-gate_recent:]). The library primitive must agree exactly.
    roles = _roles("utaautuatu")
    ut = [i for i, r in enumerate(roles) if r in ("user", "tool")]
    for g in range(0, 8):
        expected = set(ut[-g:]) if g > 0 else set()
        assert working_set_indices(roles, gate_recent=g) == expected
