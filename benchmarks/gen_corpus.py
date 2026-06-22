"""Generate a large, varied benchmark corpus spanning the content regimes agents
actually traffic in — so the head-to-head isn't run on any one tool's home turf.

Every generated trajectory satisfies distil.corpus.validate():
  * >=3 turns, known model, byte-stable cacheable prefix;
  * a stable decision + a volatile decision-relevant tool output with a DECISION;
  * at least one causally-inert (decision_relevant=False, no DECISION) noise block.

Content families deliberately include BOTH:
  * structured/repetitive data (JSON record arrays, SQL rows, metrics, logs) —
    where reversible structural compaction and RLE pay off; and
  * diagnostic/error/prose content (k8s incidents, stack traces, transcripts,
    RAG chunks) — which conservative crushers protect and lossy methods mangle.

Deterministic (seeded) so the corpus — and therefore the published numbers — are
reproducible. Run:  python benchmarks/gen_corpus.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "corpus_xl"
MODEL = "claude-opus-4-8"


def _blk(bid, kind, stability, text, decision_relevant=False):
    return {
        "id": bid,
        "kind": kind,
        "stability": stability,
        "decision_relevant": decision_relevant,
        "text": text,
    }


def _json_records(rng, n, kind):
    rows = []
    statuses = ["active", "pending", "failed", "ok", "retrying"]
    for i in range(n):
        rows.append(
            {
                "id": 1000 + i,
                "name": f"{kind}-{i:03d}",
                "status": rng.choice(statuses),
                "latency_ms": rng.randint(2, 900),
                "ok": rng.random() > 0.2,
                "region": rng.choice(["us-east-1", "eu-west-1", "ap-south-1"]),
            }
        )
    return json.dumps(rows, indent=2)


def _logs(rng, n):
    out = []
    for i in range(n):
        out.append(
            f"2026-06-22T10:{i % 60:02d}:{rng.randint(0, 59):02d}Z INFO "
            f"request id=req-{rng.randint(1000, 9999)} handled status=200 in {rng.randint(1, 80)}ms"
        )
    # inject runs of identical heartbeat lines (RLE target)
    out += ["heartbeat ok"] * 8
    return "\n".join(out)


def _sql_rows(rng, n):
    rows = [
        {
            "order_id": 5000 + i,
            "customer": f"cust{rng.randint(1, 40)}",
            "amount": round(rng.uniform(5, 500), 2),
            "currency": "USD",
            "settled": rng.random() > 0.3,
        }
        for i in range(n)
    ]
    return json.dumps(rows)


def _metrics(rng, n):
    return json.dumps(
        [
            {
                "t": 1718000000 + i * 60,
                "cpu": round(rng.uniform(0.1, 0.95), 3),
                "mem": round(rng.uniform(0.2, 0.9), 3),
                "rps": rng.randint(10, 400),
            }
            for i in range(n)
        ]
    )


def _k8s_incident(rng):
    return (
        'kubectl_get("pods","prod","app=checkout") -> '
        + json.dumps(
            {
                "items": [
                    {
                        "metadata": {"name": f"checkout-{rng.randint(1000, 9999)}"},
                        "status": {
                            "phase": "CrashLoopBackOff",
                            "containerStatuses": [
                                {
                                    "restartCount": rng.randint(3, 40),
                                    "lastState": {
                                        "terminated": {"reason": "OOMKilled", "exitCode": 137}
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )
        + "\nERROR back-off restarting failed container; OOMKilled exitCode=137"
    )


def _stacktrace(rng):
    frames = "\n".join(
        f'  File "service/handler_{i}.py", line {rng.randint(10, 400)}, in handle_{i}\n'
        f"    result = process(payload[{i}])"
        for i in range(6)
    )
    return (
        f"Traceback (most recent call last):\n{frames}\nKeyError: 'tenant_id' missing from payload"
    )


def _rag(rng, n):
    chunks = [
        f"RETRIEVED (similarity={round(rng.uniform(0.5, 0.8), 2)}): "
        f"Section {i}: general background prose about the platform that is "
        f"plausibly relevant but not load-bearing for this decision. " * 2
        for i in range(n)
    ]
    return "\n".join(chunks)


def _transcript(rng, n):
    speakers = ["customer", "agent"]
    return "\n".join(
        f"{speakers[i % 2]}: "
        + rng.choice(
            [
                "Thanks for waiting while I check that for you.",
                "My order hasn't arrived and the tracking hasn't moved in days.",
                "I can see the shipment is delayed at the carrier hub.",
                "Could you refund the expedited shipping fee at least?",
            ]
        )
        for i in range(n)
    )


FAMILIES = {
    "api-json": lambda rng, sz: _json_records(rng, sz, "svc"),
    "logs": lambda rng, sz: _logs(rng, sz),
    "sql-rows": lambda rng, sz: _sql_rows(rng, sz),
    "metrics": lambda rng, sz: _metrics(rng, sz),
    "k8s-incident": lambda rng, sz: _k8s_incident(rng),
    "stacktrace": lambda rng, sz: _stacktrace(rng),
    "rag-synthesis": lambda rng, sz: _rag(rng, max(3, sz // 6)),
    "support-chat": lambda rng, sz: _transcript(rng, max(6, sz // 3)),
}

# Per-family canonical decision the agent is converging on across turns.
DECISIONS = {
    "api-json": "scale the failing region's replicas; 3 services report failed status",
    "logs": "no error signature in logs; latency within SLA, continue",
    "sql-rows": "12 unsettled orders found; reconcile against the ledger",
    "metrics": "cpu trending past 0.9; trigger autoscale before saturation",
    "k8s-incident": "checkout pod OOMKilled; raise memory limit and redeploy",
    "stacktrace": "missing tenant_id in payload; add validation at the handler",
    "rag-synthesis": "retrieved context is speculative; answer from primary source only",
    "support-chat": "approve refund of expedited shipping fee for the delayed order",
}


def make_trajectory(family, idx, turns=5, size=None):
    rng = random.Random(f"{family}-{idx}")  # deterministic per trajectory
    size = size if size is not None else 24 + idx * 10  # varied content volume
    tid = f"{family}-{idx}"
    decision = DECISIONS[family]
    sys_text = (
        f"You are an autonomous agent handling a {family} task.\n"
        "DECISION: always re-read the latest tool output before acting"
    )
    tools_text = (
        "AVAILABLE TOOLS (json schema, abbreviated):\n"
        "- query(args): run the domain query\n"
        "- act(plan): take the chosen action\n"
        "DECISION: tools query and act are available"
    )
    out_turns = []
    for t in range(turns):
        gen = FAMILIES[family]
        # Realistic: the decision rationale is BURIED inside a large tool output
        # (it references specific data midway through), not parked at the tail —
        # so naive head/tail truncation drops it, exactly as on real agents.
        body = gen(rng, size)
        mid = len(body) // 2
        primary = (
            f"query() -> result set (turn {t}):\n{body[:mid]}\nDECISION: {decision}\n{body[mid:]}"
        )
        # one PURE structured block (columnar-foldable) + one prose/RAG block.
        noise1 = (
            gen(rng, size)
            if family in ("api-json", "sql-rows", "metrics")
            else (f"query_debug() -> verbose trace (turn {t}):\n{gen(rng, size)}")
        )
        noise2 = _rag(rng, 4) if family != "rag-synthesis" else _logs(rng, 20)
        blocks = [
            _blk("system", "system", "stable", sys_text, True),
            _blk("tools", "tools", "stable", tools_text, True),
            _blk(f"obs-{t}", "tool_output", "volatile", primary, True),
            _blk(f"dbg-{t}", "tool_output", "volatile", noise1, False),
            _blk(f"doc-{t}", "retrieved", "volatile", noise2, False),
        ]
        out_turns.append({"index": t, "blocks": blocks})
    return tid, {
        "id": tid,
        "model": MODEL,
        "_note": f"generated: {family} content, reproducible via gen_corpus.py",
        "turns": out_turns,
    }


def main(per_family=8):
    OUT.mkdir(parents=True, exist_ok=True)
    from distil.trajectory import Trajectory
    from distil.corpus import validate

    manifest = {"trajectories": []}
    bad = 0
    for family in FAMILIES:
        for idx in range(per_family):
            tid, traj = make_trajectory(family, idx)
            problems = validate(Trajectory.from_dict(traj))
            if problems:
                bad += 1
                print(f"INVALID {tid}: {problems}")
                continue
            (OUT / f"{tid}.json").write_text(json.dumps(traj, indent=2))
            manifest["trajectories"].append({"file": f"{tid}.json", "domain": family, "title": tid})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest['trajectories'])} trajectories to {OUT}  ({bad} invalid)")


if __name__ == "__main__":
    main()
