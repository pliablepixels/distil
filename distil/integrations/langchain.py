"""LangChain integration — compress message content in-process.

Preferred path is still the proxy (point a LangChain chat model's base URL at
``distil proxy``). This module adds an in-process helper for pipelines that build a
message list and want to compress it before the model call::

    from distil.integrations.langchain import compress_messages

    msgs = compress_messages(state["messages"], verbatim=True)
    resp = model.invoke(msgs)

It is **duck-typed** — it works on LangChain ``BaseMessage`` objects (read via
``.type`` / ``.content``, copied via ``model_copy``/``copy``) *and* on plain
``{"role"/"type", "content"}`` dicts, so it never imports ``langchain``. Tool /
function messages get the reversible Tier-1 digest; human / system messages get
Tier-0 lossless; the model's own words (``ai``/``assistant``) are never rewritten;
non-string content is passed through untouched.
"""

from __future__ import annotations

from typing import Any

from ..adapters.anthropic import (
    RestoreStore,
    _compress_text_content,
    _compress_tool_result_text,
    _keep_tls,
)


def _msg_type(m: Any) -> str:
    if isinstance(m, dict):
        return str(m.get("type") or m.get("role") or "")
    return str(getattr(m, "type", "") or getattr(m, "role", "") or "")


def _content(m: Any) -> Any:
    if isinstance(m, dict):
        return m.get("content")
    return getattr(m, "content", None)


def _with_content(m: Any, new_text: str) -> Any:
    """Return a copy of *m* with its content replaced — across message shapes."""
    if hasattr(m, "model_copy"):  # pydantic v2 (modern LangChain)
        return m.model_copy(update={"content": new_text})
    if hasattr(m, "copy"):  # pydantic v1; dict.copy() takes no args -> TypeError
        try:
            return m.copy(update={"content": new_text})
        except TypeError:
            pass
    if isinstance(m, dict):
        return {**m, "content": new_text}
    return m  # unknown immutable shape — leave it rather than risk corruption


def compress_messages(messages: list[Any], *, verbatim: bool = False) -> list[Any]:
    """Return a new list of messages with compressible text content compressed."""
    _keep_tls.fn = None
    try:
        store = RestoreStore()
        out: list[Any] = []
        for m in messages:
            content = _content(m)
            if not isinstance(content, str):
                out.append(m)
                continue
            t = _msg_type(m).lower()
            if t in ("ai", "assistant"):
                out.append(m)  # never rewrite the model's own words
                continue
            if t in ("tool", "function"):
                new_text = _compress_tool_result_text(content, store, verbatim)
            else:
                new_text = _compress_text_content(content, store, verbatim)
            out.append(m if new_text == content else _with_content(m, new_text))
        return out
    finally:
        _keep_tls.fn = None
