"""Cross-family grader ensemble: aggregation semantics + conservatism of the default."""

from __future__ import annotations

import pytest

from distil.ensemble import EnsembleGrader, ensemble_losses
from distil.trajectory import Block, Kind


def _b(text: str) -> list[Block]:
    return [Block(id="x", kind=Kind.USER, text=text)]


class FakeGrader:
    """Decision = a fixed mapping from the block text's first token (per-family canonicalizer)."""

    def __init__(self, mapping):
        self.mapping = mapping  # text -> decision fingerprint

    def decide(self, blocks):
        return self.mapping.get(blocks[0].text, blocks[0].text)


def test_any_aggregation_flags_change_if_one_grader_sees_it():
    base, comp = _b("A"), _b("B")
    g_sees = FakeGrader({"A": "act1", "B": "act2"})  # sees a change
    g_blind = FakeGrader({"A": "act1", "B": "act1"})  # blind to the change
    ens = EnsembleGrader([g_sees, g_blind], aggregate="any")
    assert ens.changed(base, comp) is True
    assert ens.votes(base, comp) == [True, False]


def test_unanimous_requires_all_graders():
    base, comp = _b("A"), _b("B")
    g_sees = FakeGrader({"A": "a", "B": "b"})
    g_blind = FakeGrader({"A": "a", "B": "a"})
    assert EnsembleGrader([g_sees, g_blind], aggregate="unanimous").changed(base, comp) is False
    assert EnsembleGrader([g_sees, g_sees], aggregate="unanimous").changed(base, comp) is True


def test_majority_aggregation():
    base, comp = _b("A"), _b("B")
    sees = FakeGrader({"A": "a", "B": "b"})
    blind = FakeGrader({"A": "a", "B": "a"})
    # 2 of 3 see a change -> majority True
    assert EnsembleGrader([sees, sees, blind], aggregate="majority").changed(base, comp) is True
    # 1 of 3 -> majority False
    assert EnsembleGrader([sees, blind, blind], aggregate="majority").changed(base, comp) is False


def test_any_ensemble_risk_is_conservative_vs_single_grader():
    # The key property: "any" aggregation's measured decision-change rate is >= any single
    # grader's, so a certificate built on it never *under*-reports risk.
    pairs = [(_b(a), _b(b)) for a, b in [("A", "B"), ("C", "D"), ("E", "F")]]
    faithful = FakeGrader({"A": "1", "B": "2", "C": "3", "D": "3", "E": "5", "F": "6"})
    blind = FakeGrader({"A": "1", "B": "1", "C": "3", "D": "3", "E": "5", "F": "5"})
    ens = EnsembleGrader([faithful, blind], aggregate="any")
    ens_rate = sum(ensemble_losses(ens, pairs)) / len(pairs)
    faithful_rate = sum(1.0 for a, b in pairs if faithful.decide(a) != faithful.decide(b)) / len(
        pairs
    )
    blind_rate = sum(1.0 for a, b in pairs if blind.decide(a) != blind.decide(b)) / len(pairs)
    assert ens_rate >= faithful_rate
    assert ens_rate >= blind_rate


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        EnsembleGrader([], aggregate="any")
    with pytest.raises(ValueError):
        EnsembleGrader([FakeGrader({})], aggregate="nonsense")
