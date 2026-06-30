"""``distil doctor`` — diagnose a distil setup end-to-end.

Answers the questions that otherwise turn into a dead-end: is the proxy machinery
healthy, is traffic being recorded, is outcome-validation running, and (for Claude
Code) is the status line wired and is this a flat-rate subscription. Every check is
cheap, never hangs, and degrades gracefully — a check that errors reports ``fail``
with the reason rather than taking the whole command down.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Status glyphs (ANSI applied by the renderer in cli.py, not here).
OK, WARN, INFO, FAIL = "ok", "warn", "info", "fail"


@dataclass
class Check:
    name: str
    status: str  # OK | WARN | INFO | FAIL
    detail: str
    hint: str = ""


def subscription_mode() -> bool:
    """True when the dollar figures are notional (flat-rate, no per-token bill).

    Explicit ``DISTIL_SUBSCRIPTION`` wins (on/off). Otherwise auto-detect a Claude
    Pro/Max OAuth login (``~/.claude.json`` has ``oauthAccount``) with no API key
    set — the common case where distil's measured dollars don't map to a real bill.
    """
    env = os.environ.get("DISTIL_SUBSCRIPTION", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    if os.environ.get("ANTHROPIC_API_KEY"):
        return False  # a metered key means the dollars are real
    return _claude_oauth_present()


def _claude_oauth_present() -> bool:
    # A content-free presence check: scan for the key string only (never parse or
    # read any token). The file is local and the user's own; we extract nothing.
    p = Path.home() / ".claude.json"
    try:
        return '"oauthAccount"' in p.read_text(encoding="utf-8")
    except OSError:
        return False


def _check_version() -> Check:
    import sys

    from . import __version__

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 11)
    return Check(
        "distil",
        OK if ok else FAIL,
        f"{__version__} (Python {py})",
        "" if ok else "distil needs Python 3.11+",
    )


def _check_ledger() -> Check:
    from . import ledger

    try:
        s = ledger.summary()
    except Exception as exc:  # noqa: BLE001
        return Check("savings ledger", FAIL, f"could not read ledger — {exc}")
    if s.runs == 0:
        return Check(
            "savings ledger",
            INFO,
            "no runs recorded yet",
            "route an agent through distil:  distil wrap -- claude",
        )
    live = "live-proxy" in s.by_trajectory
    saved_tok = (
        f"{s.total_tokens_saved / 1e6:.1f}M"
        if s.total_tokens_saved >= 1e6
        else str(s.total_tokens_saved)
    )
    detail = f"{s.runs} runs, {saved_tok} tokens saved"
    if not subscription_mode():
        detail += f" / ${s.total_dollars_saved:,.2f}"
    detail += " (genuine live traffic)" if live else " (corpus only — no live proxy yet)"
    return Check("savings ledger", OK, detail)


def _check_shadow() -> Check:
    try:
        from .shadow import ShadowLedger

        led = ShadowLedger.load()
    except Exception as exc:  # noqa: BLE001
        return Check("shadow validation", FAIL, f"could not read shadow ledger — {exc}")
    if led.samples == 0:
        return Check(
            "shadow validation",
            WARN,
            "not running — no decision-equivalence samples",
            "start it in one command:  distil wrap --shadow 0.1 -- claude",
        )
    eq = 100 * (1 - led.rate())
    return Check(
        "shadow validation",
        OK,
        f"{eq:.1f}% decision-equivalence over {led.samples} samples",
    )


def _check_proxy_selftest() -> Check:
    """End-to-end proxy round-trip with an in-process fake upstream — no network.

    Proves the proxy machinery (forwarding, compression, response relay) works on
    this machine, which is the thing users can't otherwise verify."""
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from .proxy import build_handler

    class _Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", 0) or 0)
            self.rfile.read(n)
            body = json.dumps(
                {"id": "msg_selftest", "content": [{"type": "text", "text": "ok"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: object) -> None:  # silence
            pass

    up = px = None
    try:
        up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        up_url = f"http://127.0.0.1:{up.server_address[1]}"

        px = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(up_url))
        threading.Thread(target=px.serve_forever, daemon=True).start()
        px_url = f"http://127.0.0.1:{px.server_address[1]}"

        payload = json.dumps(
            {
                "model": "claude-3-5-haiku",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hello from doctor"}],
            }
        ).encode()
        req = urllib.request.Request(
            px_url + "/v1/messages", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            ok = resp.status == 200 and b"msg_selftest" in resp.read()
        if ok:
            return Check("proxy self-test", OK, "request routed through distil and back (local)")
        return Check("proxy self-test", FAIL, "round-trip returned an unexpected response")
    except Exception as exc:  # noqa: BLE001
        return Check("proxy self-test", FAIL, f"round-trip failed — {exc}")
    finally:
        for srv in (px, up):
            if srv is not None:
                srv.shutdown()


def _check_anthropic_extra() -> Check:
    import importlib.util

    has = importlib.util.find_spec("anthropic") is not None
    if has:
        return Check(
            "anthropic extra", OK, "installed (live grading / billing-grade tokenizer available)"
        )
    return Check(
        "anthropic extra",
        INFO,
        "not installed",
        "only needed for --runner/--tokenizer anthropic:  pipx inject distil-llm anthropic",
    )


def _check_api_key() -> Check:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return Check("ANTHROPIC_API_KEY", OK, "set (live Anthropic paths available)")
    return Check(
        "ANTHROPIC_API_KEY",
        INFO,
        "not set",
        "offline compression/cert needs no key; set it only for --runner/--tokenizer anthropic",
    )


def _check_claude_code() -> list[Check]:
    """Claude Code-specific checks: status-line wiring + subscription detection."""
    out: list[Check] = []
    settings = Path.home() / ".claude" / "settings.json"
    try:
        data = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    sl = (data.get("statusLine") or {}).get("command", "") if isinstance(data, dict) else ""
    if "distil" in sl or "statusline.sh" in sl:
        out.append(Check("status line", OK, "wired into ~/.claude/settings.json"))
    else:
        out.append(
            Check(
                "status line",
                INFO,
                "not wired (or uses another script)",
                "see the distil plugin README to add the savings status line",
            )
        )
    if subscription_mode():
        out.append(
            Check(
                "billing mode",
                INFO,
                "flat-rate subscription detected — dollar figures are notional",
                "set DISTIL_SUBSCRIPTION=1 to show tokens-only everywhere",
            )
        )
    return out


def diagnose() -> list[Check]:
    """Run every check; each is isolated so one failure can't abort the rest."""
    checks: list[Check] = []
    for fn in (
        _check_version,
        _check_ledger,
        _check_shadow,
        _check_proxy_selftest,
        _check_anthropic_extra,
        _check_api_key,
    ):
        try:
            checks.append(fn())
        except Exception as exc:  # noqa: BLE001 — a check must never crash doctor
            checks.append(Check(fn.__name__.replace("_check_", ""), FAIL, f"check errored — {exc}"))
    try:
        checks.extend(_check_claude_code())
    except Exception:  # noqa: BLE001
        pass
    return checks
