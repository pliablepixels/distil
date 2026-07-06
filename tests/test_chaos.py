"""Chaos suite — the failure modes that shipped as bugs get pinned in CI.

The 1.11.2→1.11.3 Ctrl+C double-fix proved the release gate was blind to
signal/lifecycle chaos: each fix looked right, shipped, and broke under a
timing the tests never exercised. These tests are the bounded-CI versions of
the ad-hoc verification harnesses used to validate those fixes — they run in
seconds, and they fail if the wrap's survival properties regress.
"""

from __future__ import annotations

import sys

import pytest

from distil import proxy

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals / process groups")


def test_wrap_survives_sigint_hammer(tmp_path):
    """Sustained SIGINT storm — not a 5-press burst, a held-down key.

    The 1.11.2 fix (catch KeyboardInterrupt around proc.wait()) survived single
    presses and small bursts but lost the race when a signal landed inside its
    own except clause. The 1.11.3 fix (no-op SIGINT handler for the parent's
    lifetime) is immune by construction; this pins that property with ~400
    signals over 2s while the child lives, then proves the proxy still answers.
    """
    import os
    import signal
    import subprocess
    import time

    child = tmp_path / "child.py"
    child.write_text(
        "import os, signal, time, urllib.request\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"  # agent swallows Ctrl+C
        "time.sleep(4)\n"  # outlive the 2s hammer
        "r = urllib.request.urlopen(\n"
        "    os.environ['ANTHROPIC_BASE_URL'] + '/distil/health', timeout=2)\n"
        "raise SystemExit(0 if r.status == 200 else 8)\n",
        encoding="utf-8",
    )
    wrap = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "distil.cli",
            "wrap",
            "--no-record",
            "--",
            sys.executable,
            str(child),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,  # own group, like a terminal foreground job
    )
    time.sleep(1.5)  # let the proxy bind and the child start
    pgid = os.getpgid(wrap.pid)
    deadline = time.monotonic() + 2.0
    sent = 0
    while time.monotonic() < deadline:
        os.killpg(pgid, signal.SIGINT)
        sent += 1
        time.sleep(0.005)
    out, _ = wrap.communicate(timeout=30)
    assert wrap.returncode == 0, (
        f"wrap died under {sent} SIGINTs while child lived (exit {wrap.returncode}):\n{out}"
    )
    assert "Connection refused" not in out and "URLError" not in out, out


def test_wrap_proxy_accept_loop_self_heals(monkeypatch, capsys):
    """If the embedded proxy's accept loop crashes, the wrap restarts it —
    the child must never see connection-refused for the rest of the session.

    Simulated by a server whose first serve_forever() dies immediately; the
    child then proves the *restarted* loop answers on the same port.
    """

    class CrashOnce(proxy.QuietHTTPServer):
        crashed = False

        def serve_forever(self, *a, **kw):  # noqa: ANN002, ANN003
            if not CrashOnce.crashed:
                CrashOnce.crashed = True
                raise RuntimeError("simulated accept-loop crash")
            return super().serve_forever(*a, **kw)

    monkeypatch.setattr(proxy, "QuietHTTPServer", CrashOnce)

    child = (
        "import os, time, urllib.request\n"
        "base = os.environ['ANTHROPIC_BASE_URL']\n"
        "for _ in range(20):\n"  # poll: restart may race the first attempt
        "    try:\n"
        "        r = urllib.request.urlopen(base + '/distil/health', timeout=2)\n"
        "        raise SystemExit(0 if r.status == 200 else 8)\n"
        "    except SystemExit:\n"
        "        raise\n"
        "    except Exception:\n"
        "        time.sleep(0.1)\n"
        "raise SystemExit(9)\n"
    )
    code = proxy.wrap_run([sys.executable, "-c", child], record=False)
    assert CrashOnce.crashed  # the crash actually happened
    assert code == 0  # …and the child still reached a live proxy afterwards
