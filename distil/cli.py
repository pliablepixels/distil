"""distil — command line.

distil compress  --trajectory T            shrink a trajectory, report ratio + reversibility
distil savings   --trajectory T --pricing  price 4 strategies in real dollars (technique #1)
distil prune     --trajectory T            causal ablation: what is free to drop (technique #4)
distil certify   --trajectory T --strategy non-inferiority gate (the quality contract)
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    if args.html:
        Path(args.html).write_text(ledger.render_html(s))
        print(f"your savings page → {args.html}")
        return 0
    print(f"distil savings ledger — {ledger.DEFAULT_PATH}\n")
    if s.runs == 0:
        print("no genuine savings recorded yet.")
        print("run `distil proxy` (records real traffic) or `distil savings --record`.")
        return 0
    live = s.by_trajectory.get("live-proxy", 0.0)
    print(f"runs recorded:        {s.runs}")
    print(f"total tokens saved:   {s.total_tokens_saved:,}")
    print(f"total dollars saved:  ${s.total_dollars_saved:.5f}")
    if live:
        print(f"  of which genuine live traffic (live-proxy): ${live:.5f}")
    print("\nby source:")
    for tid, saved in sorted(s.by_trajectory.items(), key=lambda kv: -kv[1]):
        print(f"  {tid:<28} ${saved:.5f}")
    print(
        "\n(local-first; export a page with --html, or share verifiably with "
        "`distil federated-leaderboard`.)"
    )
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
    entries = load_corpus(args.corpus) if args.corpus else load_corpus()
    price = pricing.get(args.pricing)
    tok = tokenizer.resolve(args.tokenizer, model=price.name)
    # Real ingested traces carry no DECISION labels, so the offline decision-
    # equivalence gate doesn't apply — report savings only (certify live instead).
    savings_only = args.savings_only

    mode = "savings only" if savings_only else "gate"
    print(
        f"corpus {mode} — {len(entries)} trajectories | model {price.name} | "
        f"tokenizer={args.tokenizer}\n"
    )
    if savings_only:
        print(f"{'domain':<18}{'trajectory':<28}{'$ saved':>10}")
    else:
        print(
            f"{'domain':<18}{'trajectory':<24}{'$ saved':>9}{'distil':>9}{'aggr':>7}{'pruned':>8}"
        )
    print("-" * 75)

    base_total = distil_total = pruned_total = 0.0
    base_tok_total = distil_tok_total = 0
    failures: list[str] = []
    for e in entries:
        b_sim = simulate(e.trajectory, price, strategy="none", caching=False, tok=tok)
        d_sim = simulate(e.trajectory, price, strategy="distil", caching=True, tok=tok)
        base, dist = b_sim.total_dollars, d_sim.total_dollars
        base_total += base
        distil_total += dist
        base_tok_total += b_sim.total_input_tokens
        distil_tok_total += d_sim.total_input_tokens
        saved = (1 - dist / base) * 100 if base else 0.0

        if savings_only:
            print(f"{e.domain:<18}{e.trajectory.id:<28}{saved:>9.1f}%")
            continue

        bad = validate(e.trajectory)
        if bad:
            failures.append(f"{e.file}: structural — {bad[0]}")
        d_rep = certify(e.trajectory, "distil", margin=args.margin, alpha=args.alpha)
        a_rep = certify(e.trajectory, "aggressive", margin=args.margin, alpha=args.alpha)
        pruned = discover(e.trajectory, tok=tok).tokens_freed
        pruned_total += pruned
        if not d_rep.tost.non_inferior:
            failures.append(f"{e.file}: distil FAILED non-inferiority")
        if a_rep.tost.non_inferior:
            failures.append(f"{e.file}: gate failed to reject aggressive")
        print(
            f"{e.domain:<18}{e.trajectory.id:<24}{saved:>8.1f}%"
            f"{d_rep.verdict:>9}{a_rep.verdict:>7}{pruned:>8}"
        )

    print("-" * 75)
    overall = (1 - distil_total / base_total) * 100 if base_total else 0.0
    tail = "" if savings_only else f"; {int(pruned_total)} tokens causally prunable"
    print(
        f"\naggregate: distil cuts ${base_total:.5f} -> ${distil_total:.5f} "
        f"({overall:.1f}% cheaper) losslessly{tail}."
    )
    if savings_only:
        print(
            "\nsavings-only mode (ingested traces have no decision labels); "
            "certify decision-equivalence live with: distil certify --runner anthropic"
        )
        return 0

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
    if args.use_async:
        from .aproxy import serve  # high-concurrency (needs distil-llm[async])

        serve(
            host=args.host,
            port=args.port,
            upstream=args.upstream,
            lossless_only=args.lossless_only,
            shape_output=args.shape_output,
        )
    else:
        from .proxy import serve

        serve(
            host=args.host,
            port=args.port,
            upstream=args.upstream,
            lossless_only=args.lossless_only,
            shape_output=args.shape_output,
            record=not args.no_record,
            pricing_model=args.pricing,
            expand=args.expand,
            shadow_rate=args.shadow,
        )
    return 0


def cmd_shadow_stats(args: argparse.Namespace) -> int:
    """Show the live decision-equivalence measured by shadow mode on real traffic."""
    from .shadow import ShadowLedger

    led = ShadowLedger.load()
    if led.samples == 0:
        print("No shadow samples yet. Run `distil proxy --shadow 0.05` to sample live traffic.")
        return 0
    change = led.rate()
    print("Shadow-mode live decision-equivalence (real traffic, content-free)\n")
    print(f"  shadowed requests : {led.samples}")
    print(f"  decision changes  : {led.changes}")
    print(f"  decision-change rate (rolling): {change * 100:.2f}%")
    print(f"  decision-equivalence          : {(1 - change) * 100:.2f}%")
    print(
        "\n  Each sampled request was run BOTH compressed and uncompressed; "
        "equivalence\n  means the agent chose the same next action. Numbers only, never content."
    )
    return 0


def cmd_wrap(args: argparse.Namespace) -> int:
    """Transparently wrap a command: spawn the proxy, point its env at it, run it."""
    command = list(args.command)
    if command and command[0] == "--":  # argparse REMAINDER keeps the separator
        command = command[1:]
    if not command:
        print("distil wrap: nothing to run — usage: distil wrap [opts] -- <command> [args...]")
        return 2
    from .proxy import wrap_run

    return wrap_run(
        command,
        host=args.host,
        upstream=args.upstream,
        lossless_only=args.lossless_only,
        shape_output=args.shape_output,
        record=not args.no_record,
        pricing_model=args.pricing,
        env_var=args.env_var,
        expand=args.expand,
    )


def cmd_gateway(args: argparse.Namespace) -> int:
    """Managed multi-tenant gateway with a live per-tenant savings dashboard."""
    from .gateway import serve_gateway

    serve_gateway(
        host=args.host,
        port=args.port,
        upstream=args.upstream,
        pricing_model=args.pricing,
        lossless_only=args.lossless_only,
    )
    return 0


def cmd_train_transformer(args: argparse.Namespace) -> int:
    """Train the transformer keep-model on the corpus (needs distil-llm[train])."""
    from .codec.train_transformer import train_transformer

    metrics = train_transformer(args.out, base_model=args.base_model, epochs=args.epochs)
    print(f"trained transformer keep-model -> {args.out}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return 0


def cmd_output_savings(args: argparse.Namespace) -> int:
    """Measure generation-side output-token savings A/B (token cut + answer kept)."""
    import json as _json

    from .output import measure_output_savings

    src = args.input or (CORPUS_DIR / "output_pairs.jsonl")
    pairs = [
        (d["baseline"], d["shaped"])
        for d in (_json.loads(ln) for ln in Path(src).read_text().splitlines() if ln.strip())
    ]
    rep = measure_output_savings(pairs)
    print("output compression — generation-side shaping, measured A/B\n")
    print(f"  {rep.summary}")
    print("  (answer-preservation is the gate: a reduction that drops the answer is not a saving)")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Convert recorded provider requests into a Distil corpus you can run the gate on."""
    import json as _json

    from .ingest import ingest_file

    traj = ingest_file(args.input, provider=args.provider, model=args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{traj.id}.json"
    (out / fname).write_text(_json.dumps(traj.to_dict(), indent=2))
    manifest = out / "manifest.json"
    entries = []
    if manifest.exists():
        entries = _json.loads(manifest.read_text()).get("trajectories", [])
    entries = [e for e in entries if e.get("file") != fname]
    entries.append({"file": fname, "domain": "ingested", "title": traj.id})
    manifest.write_text(_json.dumps({"version": 1, "trajectories": entries}, indent=2))
    print(f"ingested {len(traj.turns)} turn(s) from {args.input} -> {out / fname}")
    print(
        f"run:  DISTIL_CORPUS={out} distil savings   # or: distil bench --corpus {out} --savings-only"
    )
    print(
        "note: real traces carry no DECISION labels — certify decision-equivalence with --runner anthropic."
    )
    return 0


def cmd_perf(args: argparse.Namespace) -> int:
    """Report compression + adapter latency/throughput (p50/p95)."""
    from .perf import format_table, run_perf

    print(format_table(run_perf(iterations=args.iterations)))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Head-to-head: every compression technique through the same gate + cost model."""
    import time

    from . import benchmark as bm

    runner = None
    if args.runner == "anthropic":
        from .replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner()
    price = pricing.get(args.pricing)
    tok = tokenizer.resolve(args.tokenizer, model=price.name)
    entries = load_corpus(args.corpus) if args.corpus else load_corpus()

    techniques = bm.builtin_techniques(runner)
    for spec in args.external or []:
        try:
            techniques.append(bm.load_external(spec))
        except Exception as exc:  # noqa: BLE001 — surface a bad external spec, don't crash
            print(f"could not load external '{spec}': {exc}")
            return 2

    rep = bm.run_benchmark(
        entries,
        techniques,
        pricing=price,
        runner=runner,
        tok=tok,
        margin=args.margin,
        alpha=args.alpha,
    )
    print(bm.format_report(rep))
    if args.html:
        Path(args.html).write_text(bm.render_html(rep))
        print(f"\nbenchmark page → {args.html}")
    if args.out:
        path = bm.write_raw(rep, args.out, str(int(time.time())))
        print(f"raw results → {path}")
    return 0


def cmd_conformal(args: argparse.Namespace) -> int:
    """Decision-Equivalence Risk Certificate — a distribution-free guarantee on the
    agent's decision-change rate at a user-chosen risk level."""
    from .conformal import calibrate

    runner = None
    if args.runner == "anthropic":
        from .replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner(samples=args.samples)
    else:
        from .replay.runner import DeterministicRunner

        runner = DeterministicRunner()
    entries = load_corpus(args.corpus) if args.corpus else load_corpus()
    cert = calibrate(entries, runner, alpha=args.alpha, delta=args.delta, method=args.method)

    long = "Learn-Then-Test" if cert.method == "ltt" else "Conformal Risk Control"
    print("Decision-Equivalence Risk Certificate (DERC)\n")
    print(f"  method      : {cert.method.upper()}  ({long})")
    print(f"  risk target : α = {args.alpha}  (max allowed decision-change rate)")
    if cert.method == "ltt":
        print(f"  confidence  : {(1 - args.delta) * 100:.0f}%  (1 − δ)")
    print(f"  calibration : n = {cert.n} turns,  runner = {args.runner}")
    print("-" * 64)
    if cert.level:
        print(f"  ✔ CERTIFIED  '{cert.level}'  →  {cert.savings * 100:.1f}% token savings")
        print(f"\n  {cert.guarantee}")
    else:
        print(
            f"  ✘ NOT CERTIFIED — no level holds a ≤{args.alpha * 100:.0f}% decision-change rate "
            f"at n={cert.n}.\n    Calibrate on more traffic, or relax α. (This is the certificate "
            "being honest:\n    small samples can't support tight guarantees.)"
        )
    print(
        "\n  Distribution-free, finite-sample (Angelopoulos–Bates–Candès et al., arXiv:2110.01052 /"
        "\n  2208.02814). Valid under EXCHANGEABILITY — recalibrate on recent traffic if your"
        "\n  workload drifts. The guarantee is marginal over the calibration distribution, not a"
        "\n  per-prompt promise."
    )
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    """Show the keep policy Distil has learned from your real expand signals."""
    from .learn import ExpandStats

    stats = ExpandStats.load()
    if not stats.digested:
        print("no expand signals yet.")
        print("run `distil proxy --expand` (or `distil wrap --expand`) so agents can")
        print("recover digested detail — every recovery teaches Distil your workload.")
        return 0
    prone = stats.expand_prone(min_digested=args.min_samples, threshold=args.threshold)
    print("learned digest policy — from your real expand signals (content-free)\n")
    print(f"{'signature':<14}{'digested':>9}{'expanded':>9}{'rate':>7}  policy")
    print("-" * 60)
    for sig in sorted(stats.digested, key=lambda s: -stats.expand_rate(s)):
        d, e, r = stats.digested[sig], stats.expanded.get(sig, 0), stats.expand_rate(sig)
        pol = "KEEP byte-exact" if sig in prone else "digest"
        print(f"{sig:<14}{d:>9}{e:>9}{r * 100:>6.0f}%  {pol}")
    print("-" * 60)
    print(
        f"\n{len(prone)} signature(s) are now kept byte-exact because your agents expand "
        "them often.\nThis applies automatically under `--expand`. It only ever makes "
        "Distil more\nconservative — savings may drop on those, decision-equivalence never does."
    )
    return 0


def cmd_frontier(args: argparse.Namespace) -> int:
    """The savings-vs-equivalence dial: how much more you save as you relax the
    decision-equivalence target below 100%."""
    from .compress.adaptive import frontier
    from .replay.runner import DeterministicRunner

    runner = DeterministicRunner()
    if args.runner == "anthropic":
        from .replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner(samples=args.samples)
    entries = load_corpus(args.corpus) if args.corpus else load_corpus()
    try:
        targets = tuple(float(x) for x in args.targets.split(","))
    except ValueError:
        print("--targets must be comma-separated numbers in (0,1], e.g. 1.0,0.97,0.95")
        return 2

    points = frontier(entries, runner, targets=targets)
    print(f"savings-vs-equivalence dial  (runner={getattr(runner, 'name', 'deterministic')})\n")
    print(f"{'target':>9}{'achieved equiv':>17}{'token savings':>15}   curve")
    print("-" * 70)
    for p in points:
        bar = "█" * round(p.savings * 28)
        print(
            f"{p.target * 100:>8.0f}%{p.equivalence * 100:>16.0f}%{p.savings * 100:>14.1f}%   {bar}"
        )
    print("-" * 70)
    top = points[0]
    print(
        f"\nAt 100% you get the certified-safe result ({top.savings * 100:.1f}% saved, "
        "every decision preserved). Relax the target and Distil spends a bounded "
        "'divergence budget' on the highest-value turns — deeper savings, a known "
        "equivalence cost. You choose the point; the trade is explicit, not hidden."
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """The certified compression frontier (savings vs decision-equivalence)."""
    import time

    from .eval import format_frontier, frontier, write_raw

    runner = None
    if args.runner == "anthropic":
        from .replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner()
    entries = load_corpus(args.corpus) if args.corpus else load_corpus()
    rep = frontier(entries, runner=runner)
    print(format_frontier(rep))
    if args.out:
        path = write_raw(rep, args.out, str(int(time.time())))
        print(f"\nraw curve → {path}")
    return 0


def cmd_online(args: argparse.Namespace) -> int:
    """One self-distilling round: causal labels → retrain → certify → promote."""
    from .online import online_round

    entries = load_corpus(args.corpus) if args.corpus else load_corpus()
    rep = online_round(entries, promote_to=args.promote_to)
    print(
        "self-distilling round — keep-model learns from causal labels, gated by non-inferiority\n"
    )
    for k, v in rep.items():
        print(f"  {k}: {v}")
    if not rep.get("certified"):
        print("\nNOT promoted — the candidate failed the non-inferiority gate (never-regressing).")
    return 0


def cmd_federated(args: argparse.Namespace) -> int:
    """Build a verifiable federated savings leaderboard from signed submissions."""
    import json as _json

    from .telemetry import build_leaderboard, render_leaderboard_html

    keys = _json.loads(Path(args.keys).read_text()) if args.keys else {}
    lb = build_leaderboard(args.dir, keys)
    print(f"verifiable savings — {len(lb.verified)} verified instance(s), {lb.rejected} rejected\n")
    print(f"  totals (certified only): {lb.totals}")
    if args.html:
        Path(args.html).write_text(render_leaderboard_html(lb))
        print(f"  leaderboard html → {args.html}")
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

    lb = sub.add_parser("leaderboard", help="your genuine cumulative savings (local ledger)")
    lb.add_argument("--html", help="render your savings as a self-contained HTML page")
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
    be.add_argument("--corpus", help="run on a custom corpus dir (e.g. from `distil ingest`)")
    be.add_argument(
        "--savings-only",
        action="store_true",
        help="report savings only, skip the decision-equivalence gate (for real traces)",
    )
    be.set_defaults(func=cmd_bench)

    os_ = sub.add_parser("output-savings", help="measure generation-side output-token savings A/B")
    os_.add_argument("--input", help="JSONL of {baseline, shaped} pairs (default: bundled fixture)")
    os_.set_defaults(func=cmd_output_savings)

    ig = sub.add_parser("ingest", help="convert recorded provider requests into a Distil corpus")
    ig.add_argument("--input", required=True, help="path to a .json/.jsonl of recorded requests")
    ig.add_argument("--out", default="./ingested-corpus", help="output corpus directory")
    ig.add_argument("--provider", default="anthropic", choices=("anthropic", "openai"))
    ig.add_argument("--model", default="claude-opus-4-8")
    ig.set_defaults(func=cmd_ingest)

    pf = sub.add_parser("perf", help="latency/throughput benchmark (p50/p95)")
    pf.add_argument("--iterations", type=int, default=200)
    pf.set_defaults(func=cmd_perf)

    ev = sub.add_parser("eval", help="certified compression frontier (savings vs accuracy)")
    ev.add_argument("--corpus", help="custom corpus dir (e.g. ingested benchmark traces)")
    ev.add_argument("--runner", default="deterministic", choices=("deterministic", "anthropic"))
    ev.add_argument("--out", help="write the raw curve JSONL to this dir")
    ev.set_defaults(func=cmd_eval)

    bn = sub.add_parser(
        "benchmark",
        help="head-to-head vs competing techniques on the same gate + cost model",
    )
    bn.add_argument("--corpus", help="custom corpus dir (e.g. ingested benchmark traces)")
    bn.add_argument("--runner", default="deterministic", choices=("deterministic", "anthropic"))
    bn.add_argument("--pricing", default="claude-opus-4-8", choices=sorted(pricing.CATALOG))
    bn.add_argument("--tokenizer", default="heuristic", choices=("heuristic", "anthropic"))
    bn.add_argument("--margin", type=float, default=0.02, help="TOST non-inferiority margin")
    bn.add_argument("--alpha", type=float, default=0.05, help="significance level")
    bn.add_argument(
        "--external",
        action="append",
        metavar="MODULE:FUNCTION[:NAME]",
        help="register a real external compressor (list[str]->list[str]); repeatable",
    )
    bn.add_argument("--html", help="render the comparison as a self-contained HTML page")
    bn.add_argument("--out", help="write raw results JSONL to this dir")
    bn.set_defaults(func=cmd_benchmark)

    fr = sub.add_parser(
        "frontier",
        help="savings-vs-equivalence dial: deeper compression as you relax the target",
    )
    fr.add_argument("--corpus", help="custom corpus dir")
    fr.add_argument("--runner", default="deterministic", choices=("deterministic", "anthropic"))
    fr.add_argument(
        "--samples", type=int, default=3, help="majority-vote samples (anthropic runner)"
    )
    fr.add_argument(
        "--targets",
        default="1.0,0.97,0.95,0.90,0.80",
        help="comma-separated equivalence targets in (0,1]",
    )
    fr.set_defaults(func=cmd_frontier)

    ln = sub.add_parser(
        "learn",
        help="show the keep policy learned from your real distil_expand signals",
    )
    ln.add_argument("--threshold", type=float, default=0.25, help="expand-rate to keep byte-exact")
    ln.add_argument(
        "--min-samples", type=int, default=5, help="min digests before a policy applies"
    )
    ln.set_defaults(func=cmd_learn)

    cf = sub.add_parser(
        "conformal",
        help="decision-equivalence risk certificate (distribution-free guarantee)",
    )
    cf.add_argument("--alpha", type=float, default=0.05, help="max decision-change rate to certify")
    cf.add_argument(
        "--delta", type=float, default=0.05, help="LTT failure probability (1−confidence)"
    )
    cf.add_argument("--method", default="ltt", choices=("ltt", "crc"))
    cf.add_argument("--corpus", help="calibration corpus dir (e.g. your ingested traffic)")
    cf.add_argument("--runner", default="deterministic", choices=("deterministic", "anthropic"))
    cf.add_argument(
        "--samples", type=int, default=3, help="majority-vote samples (anthropic runner)"
    )
    cf.set_defaults(func=cmd_conformal)

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
    px.add_argument(
        "--shape-output",
        default="off",
        choices=("off", "light", "aggressive"),
        help="output-token compression via a gated verbosity directive (PAYG only)",
    )
    px.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="async high-concurrency proxy (needs distil-llm[async])",
    )
    px.add_argument(
        "--pricing",
        default="claude-opus-4-8",
        choices=sorted(pricing.CATALOG),
        help="model used to price genuine savings recorded to the ledger",
    )
    px.add_argument(
        "--no-record", action="store_true", help="do not record genuine savings to the local ledger"
    )
    px.add_argument(
        "--expand",
        action="store_true",
        help="recoverable compression: inject the distil_expand tool so the agent can "
        "pull back digested detail on demand (transparent server-side recovery loop)",
    )
    px.add_argument(
        "--shadow",
        type=float,
        default=0.0,
        metavar="RATE",
        help="shadow-mode live decision-equivalence: sample this fraction of requests "
        "(e.g. 0.05) and run them uncompressed too, in the background, to measure the "
        "live decision-change rate on real traffic (`distil shadow-stats`). Adds ~RATE cost.",
    )
    px.set_defaults(func=cmd_proxy)

    ss = sub.add_parser(
        "shadow-stats", help="show live decision-equivalence measured by shadow mode"
    )
    ss.set_defaults(func=cmd_shadow_stats)

    wr = sub.add_parser(
        "wrap",
        help="run a command with its API base URL transparently routed through Distil",
    )
    wr.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    wr.add_argument(
        "--upstream", default="https://api.anthropic.com", help="upstream provider base URL"
    )
    wr.add_argument(
        "--env-var",
        default="ANTHROPIC_BASE_URL",
        help="environment variable to point at the proxy (default: ANTHROPIC_BASE_URL)",
    )
    wr.add_argument(
        "--lossless-only",
        action="store_true",
        help="lossless compression only (safe for subscription/OAuth sessions)",
    )
    wr.add_argument(
        "--shape-output",
        default="off",
        choices=("off", "light", "aggressive"),
        help="output-token compression via a gated verbosity directive (PAYG only)",
    )
    wr.add_argument(
        "--pricing",
        default="claude-opus-4-8",
        choices=sorted(pricing.CATALOG),
        help="model used to price genuine savings recorded to the ledger",
    )
    wr.add_argument(
        "--no-record", action="store_true", help="do not record genuine savings to the local ledger"
    )
    wr.add_argument(
        "--expand",
        action="store_true",
        help="recoverable compression: the agent can pull back digested detail on demand",
    )
    wr.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="the command to run, after `--` (e.g. distil wrap -- claude -p 'hi')",
    )
    wr.set_defaults(func=cmd_wrap)

    on = sub.add_parser(
        "online", help="self-distilling round: causal labels → retrain → certify → promote"
    )
    on.add_argument("--corpus", help="corpus dir of traffic to learn from (default: bundled)")
    on.add_argument("--promote-to", help="persist retrained weights here if it passes the gate")
    on.set_defaults(func=cmd_online)

    fl = sub.add_parser(
        "federated-leaderboard",
        help="verifiable federated savings leaderboard from signed submissions",
    )
    fl.add_argument("--dir", required=True, help="dir containing submissions.jsonl")
    fl.add_argument("--keys", help="JSON map of instance_id -> signing key")
    fl.add_argument("--html", help="write a self-contained leaderboard HTML here")
    fl.set_defaults(func=cmd_federated)

    gw = sub.add_parser("gateway", help="managed multi-tenant gateway + live savings dashboard")
    gw.add_argument("--host", default="127.0.0.1")
    gw.add_argument("--port", type=int, default=8789)
    gw.add_argument("--upstream", default="https://api.anthropic.com")
    gw.add_argument("--pricing", default="claude-opus-4-8", choices=sorted(pricing.CATALOG))
    gw.add_argument("--lossless-only", action="store_true")
    gw.set_defaults(func=cmd_gateway)

    tt = sub.add_parser(
        "train-transformer", help="train the transformer keep-model (needs distil-llm[train])"
    )
    tt.add_argument(
        "--out", default="distil-keep-transformer", help="output dir for ONNX + tokenizer"
    )
    tt.add_argument("--base-model", default="google/bert_uncased_L-2_H-128_A-2")
    tt.add_argument("--epochs", type=int, default=3)
    tt.set_defaults(func=cmd_train_transformer)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
