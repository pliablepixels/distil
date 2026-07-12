"""Proof tests for the GA audit fixes (findings A–D). Each fails before its fix."""

import argparse
import json


# --- A: bare wrap/proxy on a subscription defaults to lossless-only ---------------


def test_subscription_defaults_to_lossless(monkeypatch):
    from distil import cli

    monkeypatch.setattr("distil.doctor.subscription_mode", lambda: True)
    args = argparse.Namespace(lossless_only=False, verbatim=False, expand=False)
    cli._apply_subscription_safe_default(args)
    assert args.lossless_only is True


def test_explicit_expand_wins_over_subscription_default(monkeypatch):
    from distil import cli

    monkeypatch.setattr("distil.doctor.subscription_mode", lambda: True)
    args = argparse.Namespace(lossless_only=False, verbatim=False, expand=True)
    cli._apply_subscription_safe_default(args)
    assert args.lossless_only is False  # an explicit choice is never overridden


def test_metered_stays_digest(monkeypatch):
    from distil import cli

    monkeypatch.setattr("distil.doctor.subscription_mode", lambda: False)
    args = argparse.Namespace(lossless_only=False, verbatim=False, expand=False)
    cli._apply_subscription_safe_default(args)
    assert args.lossless_only is False  # metered keeps the lossy digest


# --- B: recency exemption honored on the standard tool_result LIST shape ----------


def test_recent_toolresult_list_stays_byte_exact():
    from distil.adapters.anthropic import compress_messages

    body = "\n".join(f"worker {i} finished job {1000 + i} in {10 + i}ms" for i in range(8))
    messages = [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "q", "input": {}}],
        },
        # most-recent turn, tool_result in LIST form (what the real Anthropic SDK sends)
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": body}],
                }
            ],
        },
    ]
    out, _store = compress_messages(messages, verbatim=True)
    txt = out[-1]["content"][0]["content"][0]["text"]  # msg → tool_result block → text block
    assert txt == body, "a recent tool_result in list form must stay byte-exact, not fold"


# --- C: disk restore refuses to clobber a colliding handle ------------------------


def test_record_restore_refuses_clobber(tmp_path, monkeypatch):
    from distil import mcp_server

    monkeypatch.setattr(mcp_server, "_restore_dir", lambda: tmp_path)
    mcp_server.record_restore("deadbeef", "verdict PASS")
    mcp_server.record_restore("deadbeef", "verdict FAIL")  # 32-bit handle collision
    assert mcp_server.load_restore("deadbeef") == "verdict PASS"  # first writer kept


# --- D: fold bails when a key contains the header delimiter -----------------------


def test_fold_bails_on_comma_in_key():
    from distil.compress.structured import fold

    # header joins columns with ',', so a key with ',' would advertise a wrong schema
    assert fold(json.dumps([{"a,b": 1, "c": 2}, {"a,b": 3, "c": 4}, {"a,b": 5, "c": 6}])) is None
    # sanity: the same shape with a clean key still folds
    assert fold(json.dumps([{"ab": 1, "c": 2}, {"ab": 3, "c": 4}, {"ab": 5, "c": 6}])) is not None
