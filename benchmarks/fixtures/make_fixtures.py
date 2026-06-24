#!/usr/bin/env python3
"""Generate τ-bench- and SWE-bench-shaped FIXTURES for the offline smoke harness.

Deliberately small and synthetic, but — unlike the bundled corpus — they contain NO
directive/marker that tells the model what to do. The next action must be INFERRED
from a load-bearing **data record** buried in plausible noise (foldable log runs,
metadata). The record sits PAST the aggressive truncation limits, so:

  * truncation (@250/@120) elides it irrecoverably  → decision flips,
  * the reversible digest folds only the noise and keeps the record (recoverable)
    → decision holds.

This is what lets the offline smoke harness exercise the real frontier/coverage
code path. The actual proof still requires real downloaded traces + ``--runner
anthropic``.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

TAU_SYSTEM = (
    "You are a retail customer-support agent. Resolve each case with exactly one tool call.\n"
    "Policy: refunds only for items returned within 30 days of delivery AND unopened;\n"
    "otherwise offer store credit. Cancel an order only if it has not yet shipped. Escalate fraud."
)
TAU_TOOLS = [
    {"name": "issue_refund", "args": ["order_id"]},
    {"name": "offer_store_credit", "args": ["order_id"]},
    {"name": "cancel_order", "args": ["order_id"]},
    {"name": "escalate_fraud", "args": ["order_id"]},
]


def _tau_noise_pre():
    # sparse log/meta lines (≤2 key=value each) — foldable, ignorable chatter placed
    # BEFORE the record so truncation that keeps the head still misses the record.
    return "\n".join(
        [
            "meta: tenant acme; region us-east-1",
            "log 2026-06-21T08:14:02Z INFO loaded customer profile",
            "log 2026-06-21T08:14:02Z INFO prior_tickets 3; csat 4.6",
            "log 2026-06-21T08:14:03Z INFO loyalty gold; newsletter on",
            "log 2026-06-21T08:14:03Z DEBUG inventory sync ok; carrier UPS",
            "log 2026-06-21T08:14:03Z DEBUG cache warm; shard 7; node b12",
        ]
    )


def _tau_noise_post():
    return "\n".join(
        [
            "note: promotional banner eligible; gift_wrap no",
            "log 2026-06-21T08:14:04Z INFO rendered response template v3",
        ]
    )


def tau_episode(eid, order_id, record, gold_call, user_ask):
    """record = a dense key=value line (the needle), placed in the TAIL after ~8
    noise lines. The reversible digest keeps head+tail (record survives, recoverable);
    aggressive head-truncation (@250/@120) discards the tail and loses it. That is the
    safe-vs-unsafe contrast the frontier must show."""
    obs = f"get_order({order_id}) ->\n{_tau_noise_pre()}\n{_tau_noise_post()}\nRECORD {record}"
    return {
        "id": eid,
        "title": eid,
        "tools": TAU_TOOLS,
        "messages": [
            {"role": "system", "content": TAU_SYSTEM},
            {"role": "user", "content": user_ask},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "get_order", "arguments": {"order_id": order_id}}}
                ],
            },
            {"role": "tool", "content": obs},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": gold_call, "arguments": {"order_id": order_id}}}
                ],
            },
        ],
    }


def build_tau():
    specs = [
        (
            "refund-ok",
            "status=delivered days_since=6 condition=unopened amount=42.00",
            "issue_refund",
        ),
        (
            "credit-opened",
            "status=delivered days_since=11 condition=opened amount=88.50",
            "offer_store_credit",
        ),
        (
            "credit-late",
            "status=delivered days_since=51 condition=unopened amount=19.99",
            "offer_store_credit",
        ),
        ("cancel-ok", "status=processing shipped=false days_since=0 amount=131.00", "cancel_order"),
        (
            "cancel-shipped",
            "status=in_transit shipped=true days_since=2 amount=67.25",
            "offer_store_credit",
        ),
        (
            "fraud",
            "status=delivered ship_country=ru card_mismatch=true velocity=9",
            "escalate_fraud",
        ),
        (
            "refund-ok2",
            "status=delivered days_since=3 condition=unopened amount=215.00",
            "issue_refund",
        ),
        (
            "credit-opened2",
            "status=delivered days_since=20 condition=opened amount=12.00",
            "offer_store_credit",
        ),
        ("cancel-ok2", "status=processing shipped=false days_since=1 amount=9.99", "cancel_order"),
        (
            "fraud2",
            "status=delivered ship_country=ng card_mismatch=true velocity=14",
            "escalate_fraud",
        ),
        (
            "credit-late2",
            "status=delivered days_since=44 condition=unopened amount=55.50",
            "offer_store_credit",
        ),
        (
            "refund-ok3",
            "status=delivered days_since=10 condition=unopened amount=33.10",
            "issue_refund",
        ),
    ]
    eps = [
        tau_episode(
            f"tau-{name}", f"A{1000 + i}", rec, call, f"Please help with order A{1000 + i}."
        )
        for i, (name, rec, call) in enumerate(specs)
    ]
    (HERE / "tau_bench_sample.json").write_text(json.dumps(eps, indent=2))
    return len(eps)


# --------------------------------------------------------------------------- #
# SWE-bench (SWE-agent .traj shaped) — a buried structured failure-record drives
# the next action; truncation that drops it flips the edit target.
# --------------------------------------------------------------------------- #

SWE_PRE = "\n".join(
    [
        "log collecting ... ; rootdir /repo; plugins none",
        "log cachedir .pytest_cache; testpaths tests",
        "log 412 items discovered across 9 modules",
        "log warnings 0; deprecations 0; session start",
        "log loading conftest; fixtures resolved; seed 1337",
    ]
)
SWE_POST = "\n".join(["log teardown ok; coverage skipped", "log session finished in 0.21s"])


def swe_step(action, fail_record):
    obs = (
        f"$ {action}\n{SWE_PRE}\n{SWE_POST}\nFAILREC {fail_record}"
        if fail_record
        else f"$ {action}\n{SWE_PRE}\n{SWE_POST}"
    )
    return [action, obs]


def swe_traj(inst, problem, steps, resolved):
    return {
        "instance_id": inst,
        "problem_statement": problem,
        "system": "You are a software engineer. Inspect the repo and fix the failing test with minimal edits.",
        "trajectory": [{"action": a, "observation": o} for a, o in steps],
        "info": {"resolved": resolved, "exit_status": "submitted"},
    }


def build_swe():
    trajs = [
        swe_traj(
            "proj__off-by-one-1",
            "test_paginate fails: last page omitted when total % size == 0",
            [
                swe_step("ls src/", None),
                swe_step(
                    "python -m pytest -q",
                    "file=src/paginate.py func=pages defect=floor_div expected=4 got=3 line=2",
                ),
                swe_step("edit src/paginate.py", None),
            ],
            True,
        ),
        swe_traj(
            "proj__none-guard-2",
            "AttributeError on empty config",
            [
                swe_step("grep -rn config.get src/", None),
                swe_step(
                    "python -m pytest -q",
                    "file=src/loader.py func=mode defect=none_attr expected=ok got=error line=22",
                ),
                swe_step("edit src/loader.py", None),
            ],
            True,
        ),
        swe_traj(
            "proj__regex-3",
            "URL validator rejects valid https URLs with ports",
            [
                swe_step("open src/validate.py", None),
                swe_step(
                    "python -m pytest -q",
                    "file=src/validate.py func=valid defect=strict_pattern expected=true got=false line=1",
                ),
                swe_step("edit src/validate.py", None),
            ],
            True,
        ),
        swe_traj(
            "proj__timezone-4",
            "Naive timestamps break DST comparison",
            [
                swe_step("grep -rn datetime.now src/", None),
                swe_step(
                    "python -m pytest -q",
                    "file=src/clock.py func=stamp defect=naive_dt expected=aware got=naive line=8",
                ),
                swe_step("edit src/clock.py", None),
            ],
            False,
        ),
        swe_traj(
            "proj__keyerror-5",
            "KeyError when optional header absent",
            [
                swe_step("open src/headers.py", None),
                swe_step(
                    "python -m pytest -q",
                    "file=src/headers.py func=read defect=missing_key expected=none got=keyerror line=14",
                ),
                swe_step("edit src/headers.py", None),
            ],
            True,
        ),
        swe_traj(
            "proj__float-6",
            "Rounding error in currency sum",
            [
                swe_step("open src/money.py", None),
                swe_step(
                    "python -m pytest -q",
                    "file=src/money.py func=total defect=float_round expected=10.00 got=9.99 line=7",
                ),
                swe_step("edit src/money.py", None),
            ],
            True,
        ),
    ]
    (HERE / "swe_bench_sample.json").write_text(json.dumps(trajs, indent=2))
    return len(trajs)


if __name__ == "__main__":
    print(f"tau episodes: {build_tau()}  ·  swe trajectories: {build_swe()}")
    print(f"written to {HERE}")
