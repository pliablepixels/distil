"""Competitor / structural baselines as ladder strategies, for a head-to-head under
the SAME grader as distil.

Each baseline is a ``(blocks, turn_index) -> blocks`` strategy that compresses the
VOLATILE tail only (the cacheable prefix is left intact, exactly as distil and the
truncation rungs do), so token savings are measured identically for everyone.

Two families:
  * **No-dependency reference baselines** (always available): head-truncation,
    recency-window (keep the most recent text), RECOMP-style extractive (keep the
    most salient lines), and selective-context-style entropy pruning. These are
    faithful reference implementations of those technique families, not the packages.
  * **Real packages** (optional, used iff importable): LLMLingua-2 via the
    ``llmlingua`` package. If it isn't installed, that baseline is skipped with a
    note (``pip install llmlingua``) rather than failing the run.

The headline comparison only needs all methods graded the same way; the certificate
itself still rides distil's risk-ordered ladder.
"""

from __future__ import annotations

import re

from distil.compress.salience import reference_index, salient_tokens
from distil.trajectory import Stability

_WORD = re.compile(r"\S+")


def _map_volatile(blocks, fn):
    """Apply ``fn(text) -> text`` to volatile blocks only; keep the prefix intact."""
    out = []
    for b in blocks:
        if b.stability is Stability.VOLATILE:
            new = fn(b.text)
            out.append(b.copy_with(new) if len(new) < len(b.text) else b)
        else:
            out.append(b)
    return out


# --------------------------------------------------------------------------- #
# No-dependency reference baselines
# --------------------------------------------------------------------------- #


def truncate_head(limit: int):
    """Keep the first ``limit`` chars (the classic prompt-truncation baseline)."""
    return lambda blocks, turn: _map_volatile(blocks, lambda t: t[:limit])


def recency_window(limit: int):
    """Keep only the most recent ``limit`` chars (sliding-window / recency baseline)."""
    return lambda blocks, turn: _map_volatile(blocks, lambda t: t[-limit:])


def recomp_extractive(keep_frac: float = 0.35):
    """RECOMP-style extractive compression: keep the most salient *sentences/lines*,
    drop the rest (lossy, irrecoverable). Selection uses the model-free salience
    signals already in the repo, ranking lines by salient-token count."""

    def strat(blocks, turn):
        ref = reference_index(blocks)

        def fn(text):
            lines = text.splitlines()
            if len(lines) <= 3:
                return text
            scored = sorted(
                range(len(lines)),
                key=lambda i: len(salient_tokens(lines[i], ref_index=ref)),
                reverse=True,
            )
            k = max(1, int(len(lines) * keep_frac))
            keep = sorted(scored[:k])
            return "\n".join(lines[i] for i in keep)

        return _map_volatile(blocks, fn)

    return strat


def selective_context(keep_frac: float = 0.5):
    """Selective-Context-style pruning: drop low-information *tokens* (keep salient
    tokens + a uniform stride of the rest), token-level rather than line-level."""

    def strat(blocks, turn):
        ref = reference_index(blocks)

        def fn(text):
            sal = salient_tokens(text, ref_index=ref)
            words = _WORD.findall(text)
            if len(words) <= 8:
                return text
            stride = max(2, int(1 / max(keep_frac, 1e-3)))
            kept = [w for i, w in enumerate(words) if w in sal or i % stride == 0]
            return " ".join(kept)

        return _map_volatile(blocks, fn)

    return strat


# --------------------------------------------------------------------------- #
# Real packages (optional)
# --------------------------------------------------------------------------- #


def llmlingua2(rate: float = 0.5):
    """LLMLingua-2 via the real ``llmlingua`` package, applied per volatile block —
    the way it deploys. Returns None if the package isn't importable (skipped)."""
    try:
        from llmlingua import PromptCompressor
    except ImportError:
        return None
    comp = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,
    )

    def strat(blocks, turn):
        def fn(text):
            try:
                return comp.compress_prompt(text, rate=rate).get("compressed_prompt", text)
            except Exception:  # noqa: BLE001 — never break the sweep on one block
                return text

        return _map_volatile(blocks, fn)

    return strat


# --------------------------------------------------------------------------- #


def load_baselines(*, include_real: bool = True) -> list[tuple[str, object]]:
    """The baseline rungs for the head-to-head. No-dep baselines always; real
    packages appended iff importable (a note is printed for any that are skipped)."""
    rungs: list[tuple[str, object]] = [
        ("truncate@500", truncate_head(500)),
        ("recency-window@500", recency_window(500)),
        ("recomp-extractive", recomp_extractive()),
        ("selective-context", selective_context()),
    ]
    if include_real:
        ll = llmlingua2()
        if ll is not None:
            rungs.append(("llmlingua-2", ll))
        else:
            print("  [baselines] llmlingua-2 skipped — `pip install llmlingua` to include it")
    return rungs
