#!/usr/bin/env python3
"""prove.py — turn decision-equivalence from a slogan into a result.

Runs the credibility experiments from ``docs/PAPER_PLAN.md`` on **real agent
traces** (τ-bench / SWE-bench), graded by a **real model** — i.e. with the
circular ``DECISION:``-marker oracle removed:

  E1  Frontier .......... savings vs. decision-change rate, per ladder level.
  E2  Certification ..... THE proof. Certify at α on a calibration split, then
      coverage            measure the *realized* decision-change rate on a disjoint
                          held-out split, repeated over many random splits. Report
                          empirical P(realized risk ≤ α) — the certificate is sound
                          iff this is ≥ 1−δ. Splits are trajectory-level (no leakage).
  E3  Distribution ...... leave-one-domain-out: calibrate on all domains but one,
      shift               test on the held-out domain. The exchangeability stress
                          test reviewers will demand.

Decisions are cached on disk (per rendered-context hash, runner-namespaced) so the
expensive live-model pass is paid exactly once and the stats are reproducible.

USAGE
  # offline plumbing check (NON-EVIDENTIAL smoke runner, bundled fixtures):
  python benchmarks/prove.py --dataset fixtures --runner smoke

  # the real thing (needs ANTHROPIC_API_KEY + downloaded traces):
  python benchmarks/prove.py --dataset tau --path /data/taubench_runs.json --runner anthropic
  python benchmarks/prove.py --dataset swe --path /data/swe_trajs/ --runner anthropic --samples 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distil.conformal import crc_select, default_ladder, ltt_certify  # noqa: E402
from distil.replay import realtrace  # noqa: E402
from distil.tokenizer import DEFAULT as tok  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / ".cache"


# --------------------------------------------------------------------------- #
# Decision cache — pay the live model once
# --------------------------------------------------------------------------- #


class DecisionCache:
    def __init__(self, runner, namespace: str):
        self.runner = runner
        self.path = CACHE_DIR / f"decisions_{namespace}.json"
        self.store: dict[str, str] = {}
        if self.path.exists():
            self.store = json.loads(self.path.read_text())
        self.hits = self.misses = 0

    @staticmethod
    def _key(blocks) -> str:
        h = hashlib.sha1()
        for b in blocks:
            h.update(f"{b.kind.value}|{b.stability.value}|{b.text}\x00".encode())
        return h.hexdigest()

    def decide(self, blocks) -> str:
        k = self._key(blocks)
        if k in self.store:
            self.hits += 1
            return self.store[k]
        self.misses += 1
        d = self.runner.decide(blocks)
        self.store[k] = d
        if self.misses % 20 == 0:  # checkpoint long live runs so nothing is lost
            self.flush()
        return d

    def flush(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.store))


# --------------------------------------------------------------------------- #
# Build the per-trajectory decision/loss matrix (the one expensive pass)
# --------------------------------------------------------------------------- #


def build_matrix(entries, cache: DecisionCache, ladder, gold) -> dict:
    matrix: dict[str, dict] = {}
    for e in entries:
        tid = e.trajectory.id
        rec = {"domain": e.domain, "success": realtrace.success_label(e), "turns": []}
        for turn in e.trajectory.turns:
            base = cache.decide(turn.blocks)
            base_tok = sum(tok.count(b.text) for b in turn.blocks)
            g = gold.get((tid, turn.index))
            tr = {
                "base": base,
                "base_tok": base_tok,
                "gold": g.fingerprint if g else None,
                "levels": {},
            }
            for name, strat in ladder:
                comp = strat(turn.blocks, turn.index)
                dec = cache.decide(comp)
                tr["levels"][name] = {
                    "loss": 0.0 if dec == base else 1.0,
                    "comp_tok": sum(tok.count(b.text) for b in comp),
                }
            rec["turns"].append(tr)
        matrix[tid] = rec
    return matrix


# --------------------------------------------------------------------------- #
# E1 — frontier
# --------------------------------------------------------------------------- #


def e1_frontier(matrix, ladder) -> list[dict]:
    rows = []
    for name, _ in ladder:
        losses, base_t, comp_t = [], 0, 0
        for rec in matrix.values():
            for tr in rec["turns"]:
                losses.append(tr["levels"][name]["loss"])
                base_t += tr["base_tok"]
                comp_t += tr["levels"][name]["comp_tok"]
        n = len(losses)
        rows.append(
            {
                "level": name,
                "n": n,
                "decision_change": (sum(losses) / n) if n else 0.0,
                "savings": (1.0 - comp_t / base_t) if base_t else 0.0,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# E2 — certification coverage (the proof)
# --------------------------------------------------------------------------- #


def _losses_for(matrix, traj_ids, ladder):
    """level-ordered list of 0/1 loss lists, plus base/comp tokens, over traj_ids."""
    level_losses = [[] for _ in ladder]
    base_t = 0
    comp_t = [0] * len(ladder)
    for tid in traj_ids:
        for tr in matrix[tid]["turns"]:
            base_t += tr["base_tok"]
            for i, (name, _) in enumerate(ladder):
                lv = tr["levels"][name]
                level_losses[i].append(lv["loss"])
                comp_t[i] += lv["comp_tok"]
    return level_losses, base_t, comp_t


def _operating_point(level_losses, base_t, comp_t, *, alpha, delta, method):
    """Pick the certified prefix end, then the highest-SAVINGS level within it
    (matches distil.conformal.calibrate)."""
    if method == "crc":
        idx_end = crc_select(level_losses, alpha=alpha)
    else:
        idx_end, _ = ltt_certify(level_losses, alpha=alpha, delta=delta)
    if idx_end < 0:
        return -1

    def savings(i):
        return (1.0 - comp_t[i] / base_t) if base_t else 0.0

    return max(range(idx_end + 1), key=savings)


def e2_coverage(matrix, ladder, *, alpha, delta, method, reps, seed) -> dict:
    tids = list(matrix.keys())
    rng = random.Random(seed)
    hits = realized = savings = certified_any = 0
    detail = []
    for _rep in range(reps):
        order = tids[:]
        rng.shuffle(order)
        half = max(1, len(order) // 2)
        calib, test = order[:half], order[half:] or order[:half]

        cl, cb, cc = _losses_for(matrix, calib, ladder)
        idx = _operating_point(cl, cb, cc, alpha=alpha, delta=delta, method=method)
        if idx < 0:
            detail.append({"certified": None})
            continue
        certified_any += 1
        # realized risk + savings on the DISJOINT held-out test split, at the certified level
        tl, tb, tc = _losses_for(matrix, test, ladder)
        n_test = len(tl[idx])
        realized_risk = (sum(tl[idx]) / n_test) if n_test else 0.0
        test_savings = (1.0 - tc[idx] / tb) if tb else 0.0
        hit = realized_risk <= alpha
        hits += int(hit)
        realized += realized_risk
        savings += test_savings
        detail.append(
            {
                "certified": ladder[idx][0],
                "realized_risk": realized_risk,
                "savings": test_savings,
                "hit": hit,
            }
        )

    c = certified_any or 1
    return {
        "alpha": alpha,
        "delta": delta,
        "method": method,
        "reps": reps,
        "certified_frac": certified_any / reps,
        "empirical_coverage": hits / c,  # P(realized ≤ α | certified)
        "target_coverage": 1 - delta if method == "ltt" else None,
        "mean_realized_risk": realized / c,
        "mean_test_savings": savings / c,
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# E3 — distribution shift (leave-one-domain-out)
# --------------------------------------------------------------------------- #


def e3_shift(matrix, ladder, *, alpha, delta, method) -> list[dict]:
    by_domain: dict[str, list[str]] = {}
    for tid, rec in matrix.items():
        by_domain.setdefault(rec["domain"], []).append(tid)
    if len(by_domain) < 2:
        return []
    out = []
    for held in by_domain:
        calib = [t for d, ts in by_domain.items() if d != held for t in ts]
        test = by_domain[held]
        cl, cb, cc = _losses_for(matrix, calib, ladder)
        idx = _operating_point(cl, cb, cc, alpha=alpha, delta=delta, method=method)
        if idx < 0:
            out.append({"held_out_domain": held, "certified": None})
            continue
        tl, tb, tc = _losses_for(matrix, test, ladder)
        n = len(tl[idx])
        out.append(
            {
                "held_out_domain": held,
                "certified": ladder[idx][0],
                "realized_risk": (sum(tl[idx]) / n) if n else 0.0,
                "savings": (1.0 - tc[idx] / tb) if tb else 0.0,
                "held_within_alpha": ((sum(tl[idx]) / n) if n else 0.0) <= alpha,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# E4 — downstream task success (does compression preserve the OUTCOME?)
# --------------------------------------------------------------------------- #


def e4_task_success(matrix, ladder, *, seed=0, boot=1000) -> dict | None:
    """Convert per-turn decision-equivalence into a downstream task metric.

    A trajectory keeps its outcome under a compression level iff EVERY decision is
    unchanged (one flip ⇒ the agent diverges and the resolution is no longer
    guaranteed — a conservative, lower-bound reading). So:

      retained success-rate(level) = (# trajectories that originally succeeded AND
                                      stay fully decision-equivalent) / N

    Reported against the uncompressed baseline success-rate, per level, with a
    bootstrap CI over trajectories. Needs trajectories that carry an outcome label
    (τ-bench reward / SWE-bench resolved); returns None if none do.
    """
    labeled = [(tid, rec) for tid, rec in matrix.items() if rec.get("success") is not None]
    if not labeled:
        return None
    n = len(labeled)
    base_success = sum(1 for _, r in labeled if r["success"]) / n

    rng = random.Random(seed)
    rows = []
    for name, _ in ladder:

        def preserved(rec, lvl=name):  # all turns unchanged at this level
            return all(tr["levels"][lvl]["loss"] == 0.0 for tr in rec["turns"])

        retained = [1.0 if (r["success"] and preserved(r)) else 0.0 for _, r in labeled]
        base_t = sum(tr["base_tok"] for _, r in labeled for tr in r["turns"])
        comp_t = sum(tr["levels"][name]["comp_tok"] for _, r in labeled for tr in r["turns"])
        # bootstrap CI over trajectories
        means = []
        for _ in range(boot):
            sample = [retained[rng.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        rows.append(
            {
                "level": name,
                "savings": (1.0 - comp_t / base_t) if base_t else 0.0,
                "retained_success": sum(retained) / n,
                "ci_low": means[int(0.025 * boot)],
                "ci_high": means[int(0.975 * boot)],
                "preserved_frac": sum(1 for _, r in labeled if preserved(r)) / n,
            }
        )
    return {"n": n, "baseline_success": base_success, "levels": rows}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def gold_agreement(matrix) -> tuple[int, int]:
    matched = total = 0
    for rec in matrix.values():
        for tr in rec["turns"]:
            if tr["gold"] is not None:
                total += 1
                matched += int(tr["base"] == tr["gold"])
    return matched, total


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", choices=["tau", "swe", "fixtures"], default="fixtures")
    ap.add_argument("--path", help="trace file/dir (required for tau/swe)")
    ap.add_argument(
        "--runner",
        choices=["smoke", "anthropic", "openai", "claude-cli"],
        default="smoke",
        help="grader: smoke (offline plumbing), anthropic (API key), openai "
        "(local/vLLM/Ollama via --base-url), claude-cli (your Claude Code subscription)",
    )
    ap.add_argument("--samples", type=int, default=1, help="majority-of-k votes per decision")
    ap.add_argument("--model", default="claude-opus-4-8", help="grader model id")
    ap.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="openai runner: OpenAI-compatible endpoint (vLLM/Ollama/LM Studio/OpenAI)",
    )
    ap.add_argument(
        "--api-key-env", default="OPENAI_API_KEY", help="openai runner: env var holding the key"
    )
    ap.add_argument(
        "--json-mode",
        action="store_true",
        help="openai runner: request response_format=json_object",
    )
    ap.add_argument(
        "--cli-bin", default="claude", help="claude-cli runner: path to the claude binary"
    )
    ap.add_argument(
        "--ladder",
        choices=["full", "quick"],
        default="full",
        help="quick = 4 rungs (byte-exact, lossless, truncate@250, truncate@120) to cut paid calls",
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="risk budget (decision-change rate)")
    ap.add_argument("--delta", type=float, default=0.05, help="LTT confidence 1-δ")
    ap.add_argument("--method", choices=["ltt", "crc"], default="ltt")
    ap.add_argument("--reps", type=int, default=200, help="random calib/test splits for E2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap #trajectories (live cost control)")
    ap.add_argument("--report", help="write full JSON report here")
    args = ap.parse_args()

    # ---- load traces -------------------------------------------------------
    if args.dataset == "fixtures":
        fx = Path(__file__).resolve().parent / "fixtures"
        entries = realtrace.load_tau_bench(fx / "tau_bench_sample.json", model=args.model)
        entries += realtrace.load_swe_bench(fx / "swe_bench_sample.json", model=args.model)
    elif args.dataset == "tau":
        entries = realtrace.load_tau_bench(args.path, model=args.model)
    else:
        entries = realtrace.load_swe_bench(args.path, model=args.model)
    if args.limit:
        entries = entries[: args.limit]

    problems = realtrace.validate_real(entries)
    if problems:
        print("STRUCTURAL PROBLEMS:")
        for p in problems[:20]:
            print("  -", p)
    gold = realtrace.gold_actions(entries)
    n_turns = sum(len(e.trajectory.turns) for e in entries)
    print(
        f"loaded {len(entries)} trajectories · {n_turns} decision points · dataset={args.dataset}"
    )

    # ---- runner ------------------------------------------------------------
    if args.runner == "smoke":
        from distil.replay.smoke_runner import SmokeRunner

        runner = SmokeRunner()
        ns = "smoke"
        print(
            "\n" + "!" * 78 + "\n"
            "! SMOKE RUNNER — NON-EVIDENTIAL. This verifies the harness mechanics only.\n"
            "! It does NOT grade real agent behavior. For a publishable result run with a\n"
            "! real model: --runner claude-cli (your subscription) / openai (local) / anthropic.\n"
            + "!"
            * 78
        )
    elif args.runner == "anthropic":
        from distil.replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner(model=args.model, samples=args.samples)
        ns = f"anthropic_{args.model}_s{args.samples}"
    elif args.runner == "openai":
        from distil.replay.openai_runner import OpenAIRunner

        runner = OpenAIRunner(
            args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            samples=args.samples,
            json_mode=args.json_mode,
        )
        ns = f"openai_{args.model.replace('/', '_')}_s{args.samples}"
    else:  # claude-cli
        from distil.replay.claude_cli_runner import ClaudeCliRunner

        cli_model = None if args.model == "claude-opus-4-8" else args.model
        runner = ClaudeCliRunner(bin=args.cli_bin, model=cli_model, samples=args.samples)
        ns = f"claudecli_{(cli_model or 'default').replace('/', '_')}_s{args.samples}"

    evidential = args.runner != "smoke"
    ladder = default_ladder()
    if args.ladder == "quick":
        keep = {"byte-exact", "lossless", "truncate@250", "truncate@120"}
        ladder = [rung for rung in ladder if rung[0] in keep]
    cache = DecisionCache(runner, ns)
    matrix = build_matrix(entries, cache, ladder, gold)
    cache.flush()
    print(f"decisions: {cache.hits} cached / {cache.misses} computed")

    # ---- gold sanity (real runners: does the model reproduce real actions?) ----
    if evidential:
        m, t = gold_agreement(matrix)
        print(
            f"model↔gold next-action agreement (uncompressed): {m}/{t} = {(m / t if t else 0):.1%}"
        )
        print("  (low agreement ⇒ the grader isn't a faithful agent; fix before trusting E1/E2)")

    # ---- E1 ----------------------------------------------------------------
    print("\n=== E1 · FRONTIER (savings vs. decision-change, real grader) ===")
    print(f"{'level':<24}{'savings':>9}{'dec-change':>12}{'n':>7}")
    print("-" * 52)
    f_rows = e1_frontier(matrix, ladder)
    for r in f_rows:
        print(
            f"{r['level']:<24}{r['savings'] * 100:>8.1f}%{r['decision_change'] * 100:>11.1f}%{r['n']:>7}"
        )

    # ---- E2 ----------------------------------------------------------------
    print("\n=== E2 · CERTIFICATION COVERAGE (the proof) ===")
    cov = e2_coverage(
        matrix,
        ladder,
        alpha=args.alpha,
        delta=args.delta,
        method=args.method,
        reps=args.reps,
        seed=args.seed,
    )
    tgt = cov["target_coverage"]
    print(
        f"method={args.method}  α={args.alpha}  δ={args.delta}  splits={args.reps} (trajectory-level, disjoint)"
    )
    print(f"  certified in {cov['certified_frac']:.1%} of splits")
    print(
        f"  empirical coverage  P(realized risk ≤ α) = {cov['empirical_coverage']:.1%}"
        + (f"   (target ≥ {tgt:.0%})" if tgt else "   (CRC: expected-risk control)")
    )
    print(
        f"  mean realized risk on held-out test       = {cov['mean_realized_risk'] * 100:.2f}%  (≤ α={args.alpha * 100:.1f}% ✔)"
        if cov["mean_realized_risk"] <= args.alpha
        else f"  mean realized risk on held-out test       = {cov['mean_realized_risk'] * 100:.2f}%  (> α — UNDERCOVERAGE)"
    )
    print(f"  mean certified token savings (held-out)   = {cov['mean_test_savings'] * 100:.1f}%")
    if tgt and cov["empirical_coverage"] + 1e-9 >= tgt:
        print("  VERDICT: certificate holds out-of-sample ✔  (this is the result the paper needs)")
    elif tgt:
        print("  VERDICT: UNDERCOVERAGE — investigate (too-small n, non-exchangeability, or a bug)")

    # ---- E3 ----------------------------------------------------------------
    e3 = e3_shift(matrix, ladder, alpha=args.alpha, delta=args.delta, method=args.method)
    if e3:
        print("\n=== E3 · DISTRIBUTION SHIFT (leave-one-domain-out) ===")
        print(f"{'held-out domain':<18}{'certified':<22}{'realized':>10}{'savings':>9}{'ok?':>5}")
        print("-" * 64)
        for r in e3:
            if r["certified"] is None:
                print(f"{r['held_out_domain']:<18}{'(none certified)':<22}")
                continue
            print(
                f"{r['held_out_domain']:<18}{r['certified']:<22}{r['realized_risk'] * 100:>9.1f}%"
                f"{r['savings'] * 100:>8.1f}%{'✔' if r['held_within_alpha'] else '✘':>5}"
            )
        print(
            "  (✘ under shift is EXPECTED and is itself a finding — recalibrate per the paper plan)"
        )

    # ---- E4 ----------------------------------------------------------------
    e4 = e4_task_success(matrix, ladder, seed=args.seed)
    if e4:
        print("\n=== E4 · DOWNSTREAM TASK SUCCESS (outcome preserved under compression?) ===")
        print(
            f"labeled trajectories: {e4['n']}  ·  baseline success-rate: {e4['baseline_success'] * 100:.1f}%"
        )
        print(f"{'level':<24}{'savings':>9}{'retained-success (95% CI)':>30}")
        print("-" * 63)
        for r in e4["levels"]:
            ci = f"{r['retained_success'] * 100:.1f}% [{r['ci_low'] * 100:.0f}–{r['ci_high'] * 100:.0f}]"
            print(f"{r['level']:<24}{r['savings'] * 100:>8.1f}%{ci:>30}")
        print(
            "  retained-success = originally-successful AND fully decision-equivalent (a flip\n"
            "  puts the outcome at risk). The safe levels hold the baseline; aggressive ones erode it."
        )

    if args.report:
        Path(args.report).write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "n_trajectories": len(entries),
                    "n_turns": n_turns,
                    "frontier": f_rows,
                    "coverage": cov,
                    "shift": e3,
                    "task_success": e4,
                },
                indent=2,
            )
        )
        print(f"\nreport → {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
