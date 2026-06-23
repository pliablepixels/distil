"""Google Gemini ``generateContent`` API runtime adapter — Phase 2 of the roadmap.

Compresses an in-flight Gemini request with no caller code change, mirroring the
Anthropic/OpenAI adapter. Gemini's request shape differs from the Messages API::

    {"contents": [{"role": "user"|"model",
                   "parts": [{"text": ...} | {"functionCall": ...} | {"functionResponse": ...}]}],
     "systemInstruction": {"parts": [{"text": ...}]},
     "tools": [...]}

What we compress (reversibly — the original is kept in the ``RestoreStore`` and is
never sent to the model, so it costs zero tokens):

* ``text`` parts (non-model role) -> Tier-0 lossless (``minify_json`` + ``collapse_runs``).
* ``functionResponse`` parts      -> large string values inside ``response`` are
  digested with the Tier-1 *reversible* digest; the object structure is preserved
  so the request stays valid.

Passed through untouched (the decision-bearing or non-textual parts):
``functionCall``, ``inlineData``, ``fileData``, ``executableCode``, and
**model-authored text** (we never rewrite the model's own words). The
``systemInstruction`` is left byte-exact, matching how the proxy treats the
Anthropic ``system`` field.

Faithful by reuse: the tier logic, the ``RestoreStore``, and the learned
keep-byte-exact policy all come from the Anthropic adapter — only the *shape*
walking is new here, so Gemini gets the exact same compression guarantees.

Not yet wired for Gemini (messages-format-only today, documented gaps):
expand-tool injection, output verbosity shaping, and Gemini context caching.
Shadow-mode live decision-equivalence *does* work for Gemini (see
``distil.shadow.decision_signature``).
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..httpguard import strip_query
from ..tokenizer import DEFAULT as _tokenizer
from .anthropic import (
    RestoreStore,
    _compress_text_content,
    _compress_tool_result_text,
    _keep_tls,
)

# /v1beta/models/{model}:generateContent  — also :streamGenerateContent and the /v1 host.
_GENERATE_RE = re.compile(r"^/v1(?:beta)?/models/[^/:]+:(?:stream)?[Gg]enerateContent$")


def is_gemini_path(path: str) -> bool:
    """True if *path* is a Gemini ``(stream)generateContent`` endpoint."""
    return bool(_GENERATE_RE.match(strip_query(path)))


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def _compress_json_value(val: Any, store: RestoreStore, lossless_only: bool) -> Any:
    """Recursively compress string values inside a ``functionResponse.response``.

    Strings are digested (large) or Tier-0 transformed (small) via the shared
    tool-result path; structure (dicts/lists) is walked but preserved, so the
    object the Gemini API requires stays intact and the request remains valid.
    Returns the *same object* when nothing changed, so callers can use identity
    to detect a no-op. ``lossless_only`` restricts to in-context-lossless Tier-0.
    """
    if isinstance(val, str):
        new = _compress_tool_result_text(val, store, lossless_only)
        return new if new != val else val
    if isinstance(val, dict):
        out: dict[str, Any] = {}
        changed = False
        for k, v in val.items():
            nv = _compress_json_value(v, store, lossless_only)
            out[k] = nv
            if nv is not v:
                changed = True
        return out if changed else val
    if isinstance(val, list):
        out_list: list[Any] = []
        changed = False
        for v in val:
            nv = _compress_json_value(v, store, lossless_only)
            out_list.append(nv)
            if nv is not v:
                changed = True
        return out_list if changed else val
    return val


def _compress_part(part: Any, store: RestoreStore, role: str, lossless_only: bool) -> Any:
    """Compress a single Gemini ``part``; returns the same object when unchanged."""
    if not isinstance(part, dict):
        return part

    text = part.get("text")
    if isinstance(text, str):
        if role == "model":
            return part  # never rewrite the model's own words
        new_text = _compress_text_content(text, store, lossless_only)
        return part if new_text == text else {**part, "text": new_text}

    fr = part.get("functionResponse")
    if isinstance(fr, dict) and "response" in fr:
        resp = fr.get("response")
        new_resp = _compress_json_value(resp, store, lossless_only)
        if new_resp is resp:
            return part
        return {**part, "functionResponse": {**fr, "response": new_resp}}

    # functionCall / inlineData / fileData / executableCode / unknown — untouched.
    return part


def compress_generate_request(
    body: dict[str, Any],
    *,
    lossless_only: bool = False,
    keep: Any = None,
) -> tuple[dict[str, Any], RestoreStore]:
    """Compress a Gemini ``generateContent`` request body (non-mutating).

    Parameters mirror :func:`distil.adapters.anthropic.compress_messages`.
    Returns ``(new_body, store)``; ``new_body`` is a shallow copy with a
    compressed ``contents`` list, ``store`` maps every digest handle back to the
    original text via ``store.expand(handle)``.
    """
    _keep_tls.fn = keep  # learned keep-byte-exact policy for this call (per-thread)
    try:
        store = RestoreStore()
        contents = body.get("contents")
        if not isinstance(contents, list):
            return body, store

        new_contents: list[Any] = []
        changed = False
        for content in contents:
            if not isinstance(content, dict):
                new_contents.append(content)
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                new_contents.append(content)
                continue
            role = content.get("role", "")
            new_parts = [_compress_part(p, store, role, lossless_only) for p in parts]
            if any(np is not p for np, p in zip(new_parts, parts)):
                new_contents.append({**content, "parts": new_parts})
                changed = True
            else:
                new_contents.append(content)

        if not changed:
            return body, store
        return {**body, "contents": new_contents}, store
    finally:
        _keep_tls.fn = None


# ---------------------------------------------------------------------------
# Token accounting (heuristic — same tokeniser as the messages path)
# ---------------------------------------------------------------------------


def _part_tokens(part: Any) -> int:
    if not isinstance(part, dict):
        return 0
    total = 0
    text = part.get("text")
    if isinstance(text, str):
        total += _tokenizer.count(text)
    fr = part.get("functionResponse")
    if isinstance(fr, dict):
        total += _tokenizer.count(json.dumps(fr.get("response"), default=str, sort_keys=True))
    fc = part.get("functionCall")
    if isinstance(fc, dict):
        total += _tokenizer.count(json.dumps(fc.get("args"), default=str, sort_keys=True))
    return total


def count_tokens(body: dict[str, Any]) -> int:
    """Heuristic token count of a Gemini request's ``contents``."""
    total = 0
    contents = body.get("contents")
    if isinstance(contents, list):
        for content in contents:
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if isinstance(parts, list):
                for part in parts:
                    total += _part_tokens(part)
    return total
