"""Realistic, mixed, decision-DETERMINED corpus for a TRUE live comparison.

Five real-world agent domains, each turn an accumulated context (3-8 KB) of the
content agents actually traffic in:
  * varied JSON record tables (NOT uniform — so it's not a free RLE win),
  * logs with realistic lines + heartbeat runs,
  * stack traces / code / SQL result sets / ticket threads,
  * retrieved RAG chunks that are causally inert (noise), and
  * one load-bearing observation whose fact DETERMINES the single next action.

The decision is determined by context (a real model reproduces it -> byte-exact
live risk ~0), but the determining fact is a needle in a haystack of plausible
noise: lossless structural compaction + causal pruning keep it; blind/aggressive
compression that drops it flips the decision. That is the decision-equivalence
test, on realistic input.
"""

import json
import random
from pathlib import Path

MODEL = "claude-opus-4-8"

SYSTEM = (
    "You are an autonomous operations agent running headless across SRE, payments, "
    "engineering, data, and support surfaces. At each step you read the latest tool "
    "outputs and take exactly ONE next action. Runbooks and policies are "
    "authoritative: when an observation contains a line beginning 'DECISION DIRECTIVE:', "
    "you MUST take precisely that action with precisely that single argument. Return the "
    "action name and its one target argument."
)
TOOLS = (
    "AVAILABLE TOOLS (abbreviated json schema):\n"
    "- rotate_logs(node_id) · restart_service(service_id) · quarantine_host(host_id)\n"
    "- notify_customer(payment_id) · retry_charge(payment_id) · refund_order(order_id)\n"
    "- block_card(card_id) · escalate_case(case_id)\n"
    "- revert_commit(sha) · rerun_pipeline(pipeline_id) · patch_dependency(pkg)\n"
    "- reindex_table(table) · kill_query(query_id) · grant_access(ticket_id)\n"
    "- close_ticket(ticket_id) · apply_credit(account_id)"
)

REGIONS = ["us-east-1", "eu-west-1", "ap-south-1", "us-west-2"]
STATUSES = ["ok", "ok", "degraded", "failed", "retrying"]


def _rng_json_table(rng, n, kind):
    """A VARIED record table (not uniform) — realistic, only partly compressible."""
    rows = [
        {
            "id": rng.randint(1000, 9999),
            "svc": f"{kind}-{rng.randint(0, 40)}",
            "status": rng.choice(STATUSES),
            "lat_ms": rng.randint(2, 1200),
            "region": rng.choice(REGIONS),
            "ok": rng.random() > 0.25,
        }
        for _ in range(n)
    ]
    return json.dumps(rows, indent=2)


def _logs(rng, n):
    lines = [
        f"2026-06-23T1{rng.randint(0, 9)}:{i % 60:02d}:{rng.randint(0, 59):02d}Z "
        f"{rng.choice(['INFO', 'INFO', 'INFO', 'WARN', 'DEBUG'])} "
        f"req id=req-{rng.randint(10000, 99999)} path=/v1/{rng.choice(['charge', 'status', 'sync', 'read'])} "
        f"status={rng.choice([200, 200, 200, 503, 429])} in {rng.randint(1, 300)}ms"
        for i in range(n)
    ]
    lines += ["2026-06-23T11:00:00Z INFO heartbeat ok"] * 6  # a compressible run
    return "\n".join(lines)


def _stacktrace(rng):
    fns = ["handle_request", "validate", "settle", "commit_txn", "flush", "encode"]
    frames = "\n".join(
        f'  File "/srv/app/{rng.choice(["core", "api", "db"])}/{rng.choice(fns)}.py", '
        f"line {rng.randint(20, 900)}, in {rng.choice(fns)}"
        for _ in range(rng.randint(5, 9))
    )
    return f"Traceback (most recent call last):\n{frames}\n{rng.choice(['ValueError', 'KeyError', 'TimeoutError'])}: unexpected state"


NOISE = [
    "RETRIEVED (speculative, similarity={s:.2f}): 'Glossary of internal acronyms' — an "
    "alphabetical list of 200+ team/system acronyms maintained by the platform org. Not "
    "specific to this case.",
    "RETRIEVED (speculative, similarity={s:.2f}): 'On-call onboarding checklist' — generic "
    "rotation handbook covering paging, escalation etiquette, and timezone handoffs. "
    "Pulled on keyword overlap; carries no incident-specific signal.",
    "RETRIEVED (speculative, similarity={s:.2f}): 'Q2 reliability OKRs' — quarterly targets "
    "and a burn-down chart. Background context only; does not bear on this decision.",
]

# (domain, tool, target-prefix, situation that DETERMINES the action) ----------
SCENARIOS = [
    (
        "sre",
        "rotate_logs",
        "NODE",
        "Node {t} at 96% disk; 41GB unrotated logs, logrotate cron disabled. Runbook RB-7: rotate logs, no restart.",
    ),
    (
        "sre",
        "restart_service",
        "SVC",
        "Service {t} leaking 30MB/min since the 14:00 deploy; healthy replicas exhausted. Runbook RB-12: rolling-restart.",
    ),
    (
        "sec",
        "quarantine_host",
        "HOST",
        "Host {t} shows lateral-scan beaconing at high IDS confidence. Runbook SEC-2: quarantine immediately.",
    ),
    (
        "pay",
        "notify_customer",
        "PAY",
        "Payment {t} failed INSUFFICIENT_FUNDS. Policy 4.1: notify the customer; never auto-retry.",
    ),
    (
        "pay",
        "retry_charge",
        "PAY",
        "Payment {t} failed GATEWAY_TIMEOUT (transient). Policy 4.3: retry the charge once.",
    ),
    (
        "pay",
        "block_card",
        "CARD",
        "Card {t}: 7 declined auths in 2 min across 4 geographies — card-testing pattern. Runbook SEC-5: block it.",
    ),
    (
        "pay",
        "refund_order",
        "ORD",
        "Order {t} double-charged via client retry; duplicate settled. Policy 6: refund the duplicate in full.",
    ),
    (
        "eng",
        "revert_commit",
        "SHA",
        "Commit {t} shipped the regression; canary error-rate 8x baseline and rising. Runbook DEP-1: revert it.",
    ),
    (
        "eng",
        "patch_dependency",
        "PKG",
        "Dependency {t} has an actively-exploited RCE (advisory GHSA-high). Policy SEC-9: patch to the fixed version now.",
    ),
    (
        "eng",
        "rerun_pipeline",
        "PIPE",
        "Pipeline {t} failed on a known-flaky network test; infra confirmed healthy. Runbook CI-3: rerun once.",
    ),
    (
        "data",
        "kill_query",
        "QRY",
        "Query {t} has run 47 min holding locks, blocking 12 sessions. Runbook DB-4: kill the runaway query.",
    ),
    (
        "data",
        "reindex_table",
        "TBL",
        "Table {t} index bloat at 71%; planner choosing seq scans, p95 up 9x. Runbook DB-8: reindex.",
    ),
    (
        "support",
        "apply_credit",
        "ACCT",
        "Account {t} was double-billed for the annual plan; finance confirmed the duplicate. Policy CS-2: apply a one-cycle credit.",
    ),
    (
        "support",
        "close_ticket",
        "TICK",
        "Ticket {t} resolved; customer confirmed and no open sub-tasks remain. Policy CS-7: close the ticket.",
    ),
    (
        "support",
        "escalate_case",
        "CASE",
        "Case {t} crossed the $5,000 manual-review threshold. Policy 9: escalate to a human owner.",
    ),
]

PREAMBLE = (
    "diagnostic context (non-authoritative, audit trail only): the agent pulled the "
    "relevant dashboards, cross-checked the last 200 events, and confirmed no correlated "
    "alerts on adjacent services; SLO burn-rate within policy, change-freeze not in effect. "
)


def _blk(bid, kind, stability, text, dr=False):
    return {"id": bid, "kind": kind, "stability": stability, "decision_relevant": dr, "text": text}


def make_trajectory(idx, rng):
    turns = []
    scen = rng.sample(SCENARIOS, 4)
    hist = []
    for ti in range(4):
        domain, tool, pfx, templ = scen[ti]
        target = f"{pfx}-{rng.randint(10000, 99999)}"
        fact = templ.format(t=target)
        # the load-bearing observation: realistic preamble + the determining fact + directive
        obs = f"get_signal() -> {PREAMBLE}\n{fact}\nDECISION DIRECTIVE: {tool}({target})"
        blocks = [
            _blk("system", "system", "stable", SYSTEM, dr=True),
            _blk("tools", "tools", "stable", TOOLS, dr=True),
        ]
        for hi, h in enumerate(hist):
            blocks.append(_blk(f"hist-{hi}", "history", "settling", h, dr=False))
        # realistic, varied, inert tool dumps (volatile, not decision-relevant)
        blocks.append(
            _blk(
                f"tel-{ti}",
                "tool_output",
                "volatile",
                "get_fleet_metrics() -> " + _rng_json_table(rng, rng.randint(14, 22), domain),
                dr=False,
            )
        )
        blocks.append(
            _blk(
                f"log-{ti}",
                "tool_output",
                "volatile",
                "get_logs() -> " + _logs(rng, rng.randint(10, 16)),
                dr=False,
            )
        )
        if domain == "eng":
            blocks.append(
                _blk(
                    f"trace-{ti}",
                    "tool_output",
                    "volatile",
                    "get_trace() -> " + _stacktrace(rng),
                    dr=False,
                )
            )
        # the decision-driving observation (volatile, decision-relevant)
        blocks.append(_blk(f"obs-{ti}", "tool_output", "volatile", obs, dr=True))
        # inert retrieved noise (volatile, prunable)
        blocks.append(
            _blk(
                f"doc-{ti}",
                "retrieved",
                "volatile",
                rng.choice(NOISE).format(s=rng.uniform(0.5, 0.7)),
                dr=False,
            )
        )
        blocks.append(
            _blk(
                f"user-{ti}",
                "user",
                "volatile",
                "Proceed with the single required action.",
                dr=False,
            )
        )
        turns.append({"index": ti, "blocks": blocks})
        hist.append(f"[turn {ti}] {domain}: took {tool} on {target}; resolved.")
    return {"id": f"realworld-{idx}", "model": MODEL, "turns": turns}


def main(n_traj, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260623)
    manifest = {"trajectories": []}
    for i in range(n_traj):
        tj = make_trajectory(i, rng)
        (out / f"{tj['id']}.json").write_text(json.dumps(tj))
        manifest["trajectories"].append(
            {"file": f"{tj['id']}.json", "domain": "realworld", "title": tj["id"]}
        )
    (out / "manifest.json").write_text(json.dumps(manifest))
    print(f"wrote {n_traj} trajectories ({n_traj * 4} turns) to {out}")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    d = sys.argv[2] if len(sys.argv) > 2 else "/tmp/corpus_realworld"
    main(n, d)
