"""Compression benchmark — correctness AND honesty (the harness can disqualify Distil)."""

from __future__ import annotations

from distil import benchmark as bm


def _by(report, name):
    return next(r for r in report.results if r.name == name)


def test_builtin_benchmark_runs_and_distil_leads_on_certified_savings():
    rep = bm.run_benchmark()
    names = {r.name for r in rep.results}
    assert {
        "baseline (no compression)",
        "distil-lossless",
        "distil-causal",
        "truncate-tail",
        "summarize",
        "extractive-prune",
    } <= names

    # The winner is the highest CERTIFIED dollar-saver — and it must actually pass.
    w = rep.winner
    assert w is not None and w.certified and w.dollar_savings > 0
    assert w.name.startswith("distil")


def test_lossy_competitors_get_raw_savings_but_fail_the_gate():
    rep = bm.run_benchmark()
    trunc = _by(rep, "truncate-tail")
    summ = _by(rep, "summarize")
    # They genuinely remove tokens (faithful, not strawmen)...
    assert trunc.token_savings > 0 and summ.token_savings > 0
    # ...but drop decisions, so the gate disqualifies them.
    assert not trunc.certified and not summ.certified
    assert trunc.equivalence < 1.0 and summ.equivalence < 1.0


def test_baseline_and_lossless_are_certified():
    rep = bm.run_benchmark()
    assert _by(rep, "baseline (no compression)").certified
    dl = _by(rep, "distil-lossless")
    assert dl.certified and dl.reversible  # byte-exact AND passes


def test_harness_is_honest_destroyer_external_is_disqualified():
    # A real external tool that over-compresses gets high RAW savings but must be
    # rejected by the same gate — proving the harness isn't rigged for Distil.
    destroyer = bm.register_external("destroyer", lambda texts: [t[:20] for t in texts])
    passthrough = bm.register_external("passthrough", lambda texts: list(texts))
    rep = bm.run_benchmark(techniques=bm.builtin_techniques() + [destroyer, passthrough])

    d = _by(rep, "destroyer")
    assert d.token_savings > 0.2 and not d.certified  # big raw cut, but fails the gate
    p = _by(rep, "passthrough")
    assert p.certified and abs(p.token_savings) < 1e-9  # no change → certified, 0 savings
    assert rep.winner.name.startswith("distil")  # Distil still wins on certified savings


def test_load_external_rejects_bad_spec():
    import pytest

    with pytest.raises(ValueError):
        bm.load_external("no-colon-here")


def test_report_renders_table_and_html():
    rep = bm.run_benchmark()
    txt = bm.format_report(rep)
    assert "LEADER (certified $ savings): distil" in txt
    html = bm.render_html(rep)
    assert "distil-causal" in html and "certified" in html.lower()
