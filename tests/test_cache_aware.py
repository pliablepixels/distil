"""Technique #1 — the cache-aware cost ordering must hold."""

from distil import pricing
from distil.compress.cache_aware import simulate
from distil.trajectory import Trajectory

CORPUS = "corpus/sample_trajectory.json"


def _run(strategy, caching):
    traj = Trajectory.load(CORPUS)
    return simulate(traj, pricing.get("claude-opus-4-8"), strategy=strategy, caching=caching)


def test_caching_is_cheaper_than_no_caching():
    assert _run("none", True).total_dollars < _run("none", False).total_dollars


def test_distil_is_cheaper_than_cache_only():
    assert _run("distil", True).total_dollars < _run("none", True).total_dollars


def test_naive_busts_the_cache_and_costs_more_than_distil():
    # naive sends FEWER tokens than cache-only but rewrites the prefix every turn.
    naive = _run("naive", True)
    distil = _run("distil", True)
    assert naive.total_dollars > distil.total_dollars
    assert distil.cache_hit_tokens > naive.cache_hit_tokens  # distil keeps the cache warm


def test_distil_actually_reduces_tokens_sent():
    assert _run("distil", True).total_input_tokens < _run("none", True).total_input_tokens
