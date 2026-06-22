"""Benchmark adapter for RTK (rtk-ai/rtk) — honest about the layer mismatch.

    pip install rtk-py            # installs the rtk binary  (or: brew install rtk)
    PYTHONPATH=. distil benchmark --external benchmarks.rtk_adapter:compress:RTK

IMPORTANT — read before trusting any RTK number from this seam:

RTK is a *command wrapper*, not a general text compressor. It reduces tokens by
re-running specific dev commands (`git status`, `cargo test`, `ls`, …) and
stripping their known boilerplate. As of writing it exposes **no raw-text /
stdin mode**, so it cannot compress arbitrary agent context (tool-result blocks,
history) the way Distil and Headroom do — they operate at different layers.

This adapter does NOT fake a result. It:
  1. shells out to the real `rtk` binary;
  2. probes for a raw-text/stdin mode (in case your build added one — there is
     an open upstream request for an "input mode");
  3. if none exists, raises a clear error explaining the mismatch instead of
     returning a misleading number.

To compare RTK fairly, benchmark it on trajectories built from REAL dev-command
outputs and run the actual `rtk <command>` — see benchmarks/README.md.
"""

from __future__ import annotations

import shutil
import subprocess

# Candidate raw-text/stdin invocations to probe, best-effort, in order.
_CANDIDATE_MODES = [
    ["rtk", "filter"],
    ["rtk", "compress", "-"],
    ["rtk", "compress"],
    ["rtk", "-"],
]
_PROBE = "line one with some words\nline one with some words\nline two\n"


def _detect_mode() -> list[str] | None:
    for cmd in _CANDIDATE_MODES:
        try:
            r = subprocess.run(cmd, input=_PROBE, capture_output=True, text=True, timeout=10)
        except Exception:
            continue
        if r.returncode == 0 and r.stdout.strip():
            return cmd
    return None


def compress(texts: list[str]) -> list[str]:
    if shutil.which("rtk") is None:
        raise RuntimeError(
            "rtk binary not found on PATH — install with `pip install rtk-py` or `brew install rtk`."
        )
    mode = _detect_mode()
    if mode is None:
        raise RuntimeError(
            "Your installed rtk has no raw-text/stdin mode: it only compresses the output of "
            "specific wrapped commands (git/cargo/ls/…), so it is not comparable on the generic "
            "context-text axis this benchmark uses. This is a layer difference, not a defect — see "
            "benchmarks/README.md for the command-output comparison path. (Refusing to fabricate a "
            "number.)"
        )
    out: list[str] = []
    for t in texts:
        try:
            r = subprocess.run(mode, input=t, capture_output=True, text=True, timeout=30)
            c = r.stdout if (r.returncode == 0 and r.stdout) else t
        except Exception:
            c = t
        out.append(c if len(c) < len(t) else t)  # reject-if-bigger
    return out
