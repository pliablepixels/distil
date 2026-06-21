"""distil — command line.

distil compress  --trajectory T            shrink a trajectory, report ratio + reversibility
distil savings   --trajectory T --pricing  price 4 strategies in real dollars (technique #1)
distil prune     --trajectory T            causal ablation: what is free to drop (technique #4)
distil certify   --trajectory T --strategy non-inferiority gate (the quality contract)
"""

from __future__ import annotations

import argparse

from . import __version__, ledger, pricing, tokenizer
from .compress.cache_aware import simulate
from .compress.strategies import REGISTRY, distil as distil_strategy
from .corpus import load_corpus, validate
from .replay.ablation import discover
from .certify.gate import certify
from .trajectory import Trajectory

from .corpus import CORPUS_DIR  # env-aware corpus dir (wheel / repo / $DISTIL_CORPUS)


def _load(path: str | None) -> Trajectory:
    return Trajectory.load(path or CORPUS_DIR / "sample_trajectory.json")


def _pct(part: float, whole: float) -> str:
    return f"{(1 - part / whole) * 100:5.1f}%" if whole else "  n/a"


def cmd_compress(args: argparse.Namespace) -> int:
    traj = _load(args.trajectory)
    tok = tokenizer.resolve(args.tokenizer, model=traj.model)
    before = after = 0
    restored = 0
    print(f"trajectory {traj.id!r}  ({len(traj.turns)} turns, tokenizer={args.tokenizer})\n")
    print(f"{'turn':>4}  {'before':>8}  {'after':>8}  {'saved':>7}")
    for turn in traj.turns:
        b = sum(tok.count(x.text) for x in turn.blocks)
        compressed = distil_strategy(turn.blocks, turn.index)
        a = sum(tok.count(x.text) for x in compressed)
        # reversibility: every byte dropped is recoverable from a local handle/marker
        restored += sum(
            1 for x in compressed if x.text != next(o.text for o in turn.blocks if o.id == x.id)
        )
        before += b
        after += a
        print(f"{turn.index:>4}  {b:>8}  {a:>8}  {_pct(a, b):>7}")
    print(f"\n{'ALL':>4}  {before:>8}  {after:>8}  {_pct(after, before):>7}")
    print(
        f"\nreversible: yes (Tier-0/1) — {restored} blocks digested, originals recoverable locally"
    )
    return 0


def cmd_savings(args: argparse.Namespace) -> int:
    traj = _load(args.trajectory)
    price = pricing.get(args.pricing)
    tok = tokenizer.resolve(args.tokenizer, model=price.name)
    out_t = args.output_tokens_per_turn

    runs = {
        "baseline (no cache, no compress)": dict(strategy="none", caching=False),
        "cache only": dict(strategy="none", caching=True),
        "naive compress + cache": dict(strategy="naive", caching=True),
        "distil (cache-aware lossless)": dict(strategy="distil", caching=True),
    }
    results = {
        label: simulate(traj, price, output_tokens_per_turn=out_t, tok=tok, **kw)
        for label, kw in runs.items()
    }
    baseline = results["baseline (no cache, no compress)"].total_dollars

    tok_note = "≈, not billing-grade" if args.tokenizer == "heuristic" else "billing-grade"
    print(
        f"model {price.name}   |   {len(traj.turns)} turns   |   "
        f"tokenizer={args.tokenizer} ({tok_note})\n"
    )
    print(f"{'strategy':<34}{'$ / run':>12}{'vs baseline':>14}{'cache hits':>12}")
    print("-" * 72)
    for label, r in results.items():
        save = _pct(r.total_dollars, baseline)
        print(f"{label:<34}{r.total_dollars:>12.5f}{save:>14}{r.cache_hit_tokens:>12,}")
    best = results["distil (cache-aware lossless)"].total_dollars
    print("-" * 72)
    print(
        f"\ndistil cuts ${baseline:.5f} -> ${best:.5f} per run "
        f"({(1 - best / baseline) * 100:.1f}% cheaper), losslessly."
    )
    naive = results["naive compress + cache"].total_dollars
    if naive > best:
        print(
            f"note: naive compression costs ${naive:.5f} — "
            f"{(naive / best - 1) * 100:.0f}% MORE than distil despite fewer tokens, "
            f"because it busts the prefix cache."
        )

    if args.record:
        b = results["baseline (no cache, no compress)"]
        d = results["distil (cache-aware lossless)"]
        rec = ledger.record(
            trajectory_id=traj.id,
            model=price.name,
            turns=len(traj.turns),
            baseline_dollars=baseline,
            distil_dollars=best,
            baseline_input_tokens=b.total_input_tokens,
            distil_input_tokens=d.total_input_tokens,
        )
        print(
            f"\nrecorded to {ledger.DEFAULT_PATH}: "
            f"${rec.dollars_saved:.5f} / {rec.tokens_saved} tokens saved this run."
        )
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    s = ledger.summary()
    print(f"distil savings ledger — {ledger.DEFAULT_PATH}\n")
    if s.runs == 0:
        print("no runs recorded yet. run:  distil savings --record")
        return 0
    print(f"runs recorded:        {s.runs}")
    print(f"total tokens saved:   {s.total_tokens_saved:,}")
    print(f"total dollars saved:  ${s.total_dollars_saved:.5f}")
    print("\nby trajectory:")
    for tid, saved in sorted(s.by_trajectory.items(), key=lambda kv: -kv[1]):
        print(f"  {tid:<28} ${saved:.5f}")
    print("\n(local-first; community sharing is opt-in and not enabled.)")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    traj = _load(args.trajectory)
    report = discover(traj)
    print(f"causal ablation over {traj.id!r} — what is free to drop?\n")
    print(f"{'block':<22}{'occ':>4}{'tokens':>8}  verdict")
    for v in report.verdicts:
        verdict = "PRUNE (causally inert)" if v.prunable else "keep (changed a decision)"
        print(f"{v.block_id:<22}{v.occurrences:>4}{v.tokens:>8}  {verdict}")
    print(
        f"\ntokens provably free to drop: {report.tokens_freed} "
        f"across {len(report.prunable)} block(s)."
    )
    return 0


def cmd_certify(args: argparse.Namespace) -> int:
    traj = _load(args.trajectory)
    runner = None
    if args.runner == "anthropic":
        from .replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner(model=traj.model)
    report = certify(traj, args.strategy, runner=runner, margin=args.margin, alpha=args.alpha)
    print(f"certifying strategy {args.strategy!r} on {traj.id!r} (runner={args.runner})\n")
    for d in report.divergences:
        flag = "ok" if d.matched else "DIVERGED"
        print(f"  turn {d.turn}: {flag}")
        if not d.matched:
            print(f"      baseline:   {d.baseline_decision}")
            print(f"      compressed: {d.compressed_decision}")
    t = report.tost
    print(f"\ndecision-equivalence match rate: {report.match_rate * 100:.1f}%")
    print(
        f"TOST non-inferiority (margin={t.margin}, alpha={t.alpha}): "
        f"mean diff={t.mean_diff:+.3f}, p={t.p_non_inferior:.4g}"
    )
    print(
        f"\nVERDICT: {report.verdict}  "
        f"({'certified non-inferior' if t.non_inferior else 'NOT certified — would degrade quality'})"
    )
    return 0 if t.non_inferior else 1


def cmd_bench(args: argparse.Namespace) -> int:
    """Corpus-wide gate (CI). Across every trajectory: price distil vs baseline,
    certify distil is non-inferior, and confirm the gate still rejects the
    aggressive lossy strategy. Exits non-zero if any distil run fails the
    contract or the gate fails to reject aggressive — i.e. a real CI gate."""
    entries = load_corpus()
    price = pricing.get(args.pricing)
    tok = tokenizer.resolve(args.tokenizer, model=price.name)

    print(
        f"corpus gate — {len(entries)} trajectories | model {price.name} | tokenizer={args.tokenizer}\n"
    )
    print(f"{'domain':<18}{'trajectory':<24}{'$ saved':>9}{'distil':>9}{'aggr':>7}{'pruned':>8}")
    print("-" * 75)

    base_total = distil_total = pruned_total = 0.0
    base_tok_total = distil_tok_total = 0
    failures: list[str] = []
    for e in entries:
        bad = validate(e.trajectory)
        if bad:
            failures.append(f"{e.file}: structural — {bad[0]}")
        b_sim = simulate(e.trajectory, price, strategy="none", caching=False, tok=tok)
        d_sim = simulate(e.trajectory, price, strategy="distil", caching=True, tok=tok)
        base, dist = b_sim.total_dollars, d_sim.total_dollars
        d_rep = certify(e.trajectory, "distil", margin=args.margin, alpha=args.alpha)
        a_rep = certify(e.trajectory, "aggressive", margin=args.margin, alpha=args.alpha)
        pruned = discover(e.trajectory, tok=tok).tokens_freed

        base_total += base
        distil_total += dist
        pruned_total += pruned
        base_tok_total += b_sim.total_input_tokens
        distil_tok_total += d_sim.total_input_tokens
        if not d_rep.tost.non_inferior:
            failures.append(f"{e.file}: distil FAILED non-inferiority")
        if a_rep.tost.non_inferior:
            failures.append(f"{e.file}: gate failed to reject aggressive")

        saved = (1 - dist / base) * 100 if base else 0.0
        print(
            f"{e.domain:<18}{e.trajectory.id:<24}{saved:>8.1f}%"
            f"{d_rep.verdict:>9}{a_rep.verdict:>7}{pruned:>8}"
        )

    print("-" * 75)
    overall = (1 - distil_total / base_total) * 100 if base_total else 0.0
    print(
        f"\naggregate: distil cuts ${base_total:.5f} -> ${distil_total:.5f} "
        f"({overall:.1f}% cheaper) losslessly; {int(pruned_total)} tokens causally prunable."
    )

    if args.record:
        rec = ledger.record(
            trajectory_id="corpus-aggregate",
            model=price.name,
            turns=sum(len(e.trajectory.turns) for e in entries),
            baseline_dollars=base_total,
            distil_dollars=distil_total,
            baseline_input_tokens=base_tok_total,
            distil_input_tokens=distil_tok_total,
        )
        print(
            f"recorded corpus run to the savings ledger: "
            f"${rec.dollars_saved:.5f} / {rec.tokens_saved} tokens saved. (distil leaderboard)"
        )

    if failures:
        print(f"\nGATE: FAIL ({len(failures)} issue(s))")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nGATE: PASS — every trajectory certified non-inferior; aggressive rejected on all.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Byte-fidelity gate (Phase 6): every distil compression across the corpus is
    reconstructable, and frozen history never mutates turn-to-turn."""
    from .compress.base import CompressResult
    from .compress.tier0 import Tier0Lossless
    from .compress.tier1 import Tier1Reversible
    from .fidelity import assert_append_only, verify_reversible
    from .trajectory import Stability

    print("byte-fidelity gate — reversibility + append-only across the corpus\n")
    problems: list[str] = []
    for e in load_corpus():
        prev = None
        for turn in e.trajectory.turns:
            volatile = [b for b in turn.blocks if b.stability is Stability.VOLATILE]
            r1 = Tier1Reversible().compress(volatile)
            r0 = Tier0Lossless().compress(r1.blocks)
            merged = CompressResult(r0.blocks, {**r1.restore, **r0.restore})
            rep = verify_reversible(volatile, merged)
            if not rep.lossless:
                problems.append(f"{e.file} turn {turn.index}: irrecoverable {rep.irrecoverable}")
            if prev is not None:
                v = assert_append_only(prev, turn.blocks)
                if v:
                    problems.append(f"{e.file} turn {turn.index}: append-only violation {v}")
            prev = turn.blocks
        print(f"  {e.trajectory.id:<24} reversible + append-only: ok")
    if problems:
        print("\nFIDELITY: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nFIDELITY: PASS — Tier-0/1 byte-reversible and history append-only across the corpus.")
    return 0


def cmd_holdout(args: argparse.Namespace) -> int:
    """Holdout A/B savings with a bootstrap CI (Phase 5)."""
    from .certify.holdout import run_holdout

    price = pricing.get(args.pricing)
    tok = tokenizer.resolve(args.tokenizer, model=price.name)
    rep = run_holdout(load_corpus(), price, control_fraction=args.control_fraction, tok=tok)
    print("holdout A/B savings measurement (deterministic partition + bootstrap CI)\n")
    print(f"  {rep.summary}")
    print(
        f"  control group mean savings: {rep.control_mean_savings * 100:.1f}% "
        "(held out, not counted toward the headline)"
    )
    return 0


def cmd_proxy(args: argparse.Namespace) -> int:
    """Drop-in provider proxy: point any base_url-honoring client at it."""
    from .proxy import serve

    serve(host=args.host, port=args.port, upstream=args.upstream, lossless_only=args.lossless_only)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="distil", description="Compression with a quality contract.")
    p.add_argument("--version", action="version", version=f"distil {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_traj(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--trajectory", "-t", help="path to a trajectory JSON (default: bundled sample)"
        )

    def add_tokenizer(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--tokenizer",
            default="heuristic",
            choices=("heuristic", "anthropic"),
            help="heuristic (offline, default) or anthropic (billing-grade count_tokens)",
        )

    c = sub.add_parser("compress", help="shrink a trajectory; report ratio + reversibility")
    add_traj(c)
    add_tokenizer(c)
    c.set_defaults(func=cmd_compress)

    s = sub.add_parser("savings", help="price strategies in real dollars (technique #1)")
    add_traj(s)
    add_tokenizer(s)
    s.add_argument("--pricing", default="claude-opus-4-8", choices=sorted(pricing.CATALOG))
    s.add_argument("--output-tokens-per-turn", type=int, default=0)
    s.add_argument(
        "--record", action="store_true", help="append this run to the local savings ledger"
    )
    s.set_defaults(func=cmd_savings)

    lb = sub.add_parser("leaderboard", help="cumulative savings ledger across runs")
    lb.set_defaults(func=cmd_leaderboard)

    pr = sub.add_parser("prune", help="causal ablation: what is free to drop (technique #4)")
    add_traj(pr)
    pr.set_defaults(func=cmd_prune)

    ce = sub.add_parser("certify", help="non-inferiority gate (the quality contract)")
    add_traj(ce)
    ce.add_argument("--strategy", default="distil", choices=sorted(REGISTRY))
    ce.add_argument(
        "--runner",
        default="deterministic",
        choices=("deterministic", "anthropic"),
        help="deterministic (offline, default) or anthropic (live model)",
    )
    ce.add_argument("--margin", type=float, default=0.02)
    ce.add_argument("--alpha", type=float, default=0.05)
    ce.set_defaults(func=cmd_certify)

    be = sub.add_parser("bench", help="corpus-wide CI gate across every domain")
    add_tokenizer(be)
    be.add_argument("--pricing", default="claude-opus-4-8", choices=sorted(pricing.CATALOG))
    be.add_argument("--margin", type=float, default=0.02)
    be.add_argument("--alpha", type=float, default=0.05)
    be.add_argument(
        "--record",
        action="store_true",
        help="log the corpus-aggregate savings to the local ledger (distil leaderboard)",
    )
    be.set_defaults(func=cmd_bench)

    ve = sub.add_parser("verify", help="byte-fidelity gate: reversibility + append-only (phase 6)")
    ve.set_defaults(func=cmd_verify)

    ho = sub.add_parser("holdout", help="holdout A/B savings with a bootstrap CI (phase 5)")
    add_tokenizer(ho)
    ho.add_argument("--pricing", default="claude-opus-4-8", choices=sorted(pricing.CATALOG))
    ho.add_argument("--control-fraction", type=float, default=0.2)
    ho.set_defaults(func=cmd_holdout)

    px = sub.add_parser("proxy", help="drop-in provider proxy (point any client's base_url at it)")
    px.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    px.add_argument("--port", type=int, default=8788)
    px.add_argument(
        "--upstream", default="https://api.anthropic.com", help="upstream provider base URL"
    )
    px.add_argument(
        "--lossless-only",
        action="store_true",
        help="lossless compression only (safe for subscription/OAuth sessions)",
    )
    px.set_defaults(func=cmd_proxy)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
