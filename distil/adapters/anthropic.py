"""Anthropic Messages API runtime adapter — Phase 3 of the distil roadmap.

Compresses an in-flight Messages API request with no caller code change:

  client = anthropic.Anthropic(...)
  client = distil.adapters.anthropic.wrap(client)
  # all subsequent client.messages.create(...) calls are transparently compressed

Design decisions
----------------
* No `anthropic` import at module level — the adapter is duck-typed so it works
  even if the `anthropic` SDK is not installed (e.g. in test environments).
* Only `tool_result` blocks with >= 6 lines are digested (Tier 1 / reversible).
  Plain `text` blocks get Tier 0 transforms (minify_json / collapse_runs).
  `tool_use`, `image`, and assistant text blocks are passed through unchanged.
* `RestoreStore` keeps originals keyed by the 8-hex handle that `tier1.digest`
  embeds in its marker lines, so callers can always recover the full content.
* Cache-control placement: marking the last stable system block (or the system
  string itself) as `{"cache_control": {"type": "ephemeral"}}` pins a cacheable
  prefix. Reads of that prefix are billed at ~0.1x vs. a full write, so every
  repeated call after the first amortises the system-prompt tokens cheaply.
  The prefix must be *stable* across turns (same bytes) to get a cache hit —
  hence we mark the *last* system block rather than anything in the volatile
  message history.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from ..compress.tier0 import collapse_runs, minify_json
from ..compress.tier1 import digest as _tier1_digest

# Minimum line count for a tool_result to be digested (matches Tier1Reversible default).
_MIN_LINES = 6

# Thread-local learned "keep byte-exact" predicate, scoped per compress_messages call
# (ThreadingHTTPServer handles requests on separate threads, so this must be per-thread).
import threading as _threading  # noqa: E402

_keep_tls = _threading.local()


def _active_keep(text: str) -> bool:
    fn = getattr(_keep_tls, "fn", None)
    return bool(fn and fn(text))


# ---------------------------------------------------------------------------
# RestoreStore
# ---------------------------------------------------------------------------


class RestoreStore:
    """Maps 8-hex handles -> original text so callers can reverse any digest.

    The store is populated by `compress_messages` and is local — it is never
    sent to the model, so it costs zero tokens.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal (used by compress_messages)
    # ------------------------------------------------------------------

    def _record(self, handle: str, original: str) -> None:
        self._store[handle] = original

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def handles(self) -> frozenset[str]:
        """All handles currently registered in this store."""
        return frozenset(self._store)

    def expand(self, handle: str) -> str:
        """Return the original text for *handle*.

        Raises KeyError if the handle is not in this store.
        """
        return self._store[handle]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _handle(text: str) -> str:
    """8-hex SHA-256 prefix — mirrors tier1._handle exactly."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _apply_tier0(text: str) -> str:
    """Apply lossless Tier-0 transforms: JSON minification then run collapse."""
    mj = minify_json(text)
    if mj is not None:
        text = mj
    return collapse_runs(text)


def _compress_text_content(text: str, store: RestoreStore, lossless_only: bool) -> str:
    """Apply only Tier-0 lossless transforms to a plain text block."""
    return _apply_tier0(text)


def _compress_tool_result_text(text: str, store: RestoreStore, lossless_only: bool = False) -> str:
    """Digest a large tool_result string and record the original in *store*.

    In ``lossless_only`` mode (subscription/OAuth-safe) only *in-context-lossless*
    Tier-0 transforms are applied: the model sees semantically identical content
    (minified JSON, collapsed exact-duplicate runs), never a digest stub. This is
    what makes the mode safe for interactive sessions, where the model must reason
    over the real content rather than recover it via a tool it isn't allowed.
    """
    lines = text.splitlines()
    if len(lines) < _MIN_LINES:
        # Too short to digest — apply lossless transforms only.
        return _apply_tier0(text)

    # Subscription/OAuth-safe: never stub content the model can't recover.
    if lossless_only:
        return _apply_tier0(text)

    # Learned policy: if your agents keep expanding this kind of content, keep it
    # byte-exact (strictly safer — only ever reduces savings, never equivalence).
    if _active_keep(text):
        return _apply_tier0(text)

    digested, changed = _tier1_digest(text)
    if changed:
        h = _handle(text)
        store._record(h, text)
        return digested
    return _apply_tier0(text)


def _compress_content_item(
    item: dict[str, Any], store: RestoreStore, role: str, lossless_only: bool
) -> dict[str, Any]:
    """Return a (possibly new) content block after compression.

    Rules:
    - tool_use / image: pass through unchanged.
    - assistant text: pass through unchanged.
    - user text: Tier-0 lossless transforms.
    - tool_result (any role): digest large string content; recurse into list content.
    """
    btype = item.get("type", "")

    # Never touch tool_use or image blocks.
    if btype in ("tool_use", "image"):
        return item

    if btype == "text":
        if role == "assistant":
            # Never rewrite the assistant's own words.
            return item
        text = item.get("text")
        if not isinstance(text, str):
            return item  # malformed/absent text — pass through untouched
        new_text = _compress_text_content(text, store, lossless_only)
        if new_text == text:
            return item
        return {**item, "text": new_text}

    if btype == "tool_result":
        content = item.get("content")
        if content is None:
            return item

        if isinstance(content, str):
            new_content = _compress_tool_result_text(content, store, lossless_only)
            if new_content == content:
                return item
            return {**item, "content": new_content}

        if isinstance(content, list):
            new_list: list[Any] = []
            changed = False
            for sub in content:
                if (
                    isinstance(sub, dict)
                    and sub.get("type") == "text"
                    and isinstance(sub.get("text"), str)
                ):
                    new_text = _compress_tool_result_text(sub["text"], store, lossless_only)
                    if new_text != sub["text"]:
                        new_list.append({**sub, "text": new_text})
                        changed = True
                    else:
                        new_list.append(sub)
                else:
                    new_list.append(sub)
            if not changed:
                return item
            return {**item, "content": new_list}

    # Unknown block type — leave untouched.
    return item


def _compress_message(
    msg: dict[str, Any], store: RestoreStore, lossless_only: bool
) -> dict[str, Any]:
    """Return a (possibly new) message dict after compressing its content."""
    role = msg.get("role", "")
    content = msg.get("content")

    if isinstance(content, str):
        if role == "assistant":
            return msg
        # OpenAI tool-result messages ({"role":"tool","content":"…"}) get the same
        # decision-aware reversible digest as Anthropic tool_result blocks; other
        # string content gets Tier-0 lossless transforms.
        if role == "tool":
            new_text = _compress_tool_result_text(content, store, lossless_only)
        else:
            new_text = _compress_text_content(content, store, lossless_only)
        if new_text == content:
            return msg
        return {**msg, "content": new_text}

    if isinstance(content, list):
        new_blocks: list[Any] = []
        changed = False
        for item in content:
            if isinstance(item, dict):
                new_item = _compress_content_item(item, store, role, lossless_only)
                new_blocks.append(new_item)
                if new_item is not item:
                    changed = True
            else:
                new_blocks.append(item)
        if not changed:
            return msg
        return {**msg, "content": new_blocks}

    return msg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compress_messages(
    messages: list[dict[str, Any]],
    *,
    lossless_only: bool = False,
    keep: Any = None,
) -> tuple[list[dict[str, Any]], RestoreStore]:
    """Compress an Anthropic Messages API messages list in place (non-mutating).

    Parameters
    ----------
    messages:
        The ``messages`` kwarg value as passed to ``client.messages.create``.
        Each element is ``{"role": ..., "content": str | list[block]}``.
    lossless_only:
        When *True* (subscription/OAuth-safe mode), only *in-context-lossless*
        Tier-0 transforms are applied — the model sees semantically identical
        content, never a Tier-1 digest stub. This is the correct mode when tool
        injection is disallowed (so the agent cannot recover a digest) and a human
        is reading the model's output. When *False* (PAYG), large tool results are
        replaced by reversible Tier-1 digests (recoverable via the RestoreStore /
        the ``distil_expand`` tool) for far higher savings.

    Returns
    -------
    (new_messages, store)
        ``new_messages`` is a new list (input is not mutated).
        ``store`` maps every 8-hex handle embedded in digest markers back to the
        original text; call ``store.expand(handle)`` to recover it.
    """
    _keep_tls.fn = keep  # learned keep-byte-exact policy for this call (per-thread)
    try:
        store = RestoreStore()
        new_messages: list[dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                new_messages.append(msg)  # malformed entry — pass through untouched
                continue
            new_messages.append(_compress_message(msg, store, lossless_only))
        return new_messages, store
    finally:
        _keep_tls.fn = None


def place_cache_control(
    system: list[dict[str, Any]] | str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a kwargs dict that pins the cacheable prefix via cache_control.

    Anthropic's prompt caching works by marking the *boundary* of the stable
    prefix with ``{"cache_control": {"type": "ephemeral"}}``.  A cache *hit*
    costs ~0.1x compared to a full write, so even a single repeated call pays
    back the marking overhead.

    The stable prefix must be byte-identical across calls to get a hit.
    System blocks are a natural boundary — they rarely change between turns —
    so we mark the **last** system block.  Message history is volatile (each
    turn adds a new assistant/user pair), so it is intentionally left outside
    the cached prefix.

    Parameters
    ----------
    system:
        The ``system`` kwarg: either a plain string or a list of content blocks
        (``{"type": "text", "text": ..., ...}``).
    messages:
        The (already compressed) ``messages`` list.

    Returns
    -------
    A dict ready to be spread into ``client.messages.create(**kwargs)``:
    ``{"system": <marked_system>, "messages": messages}``.
    """
    _cc: dict[str, Any] = {"cache_control": {"type": "ephemeral"}}

    if isinstance(system, str):
        # Promote the bare string to a single cacheable block.
        marked_system: list[dict[str, Any]] | str = [{"type": "text", "text": system, **_cc}]
    elif isinstance(system, list) and system:
        # Deep-copy so we do not mutate the caller's list, then mark the last block.
        marked_system = copy.deepcopy(system)
        marked_system[-1].update(_cc)  # type: ignore[union-attr]
    else:
        marked_system = system

    return {"system": marked_system, "messages": messages}


# ---------------------------------------------------------------------------
# Proxy wrapper
# ---------------------------------------------------------------------------


class _MessagesProxy:
    """Thin proxy for ``client.messages`` that compresses before delegating."""

    def __init__(self, real_messages: Any) -> None:
        self._real = real_messages

    def create(self, **kwargs: Any) -> Any:
        # Compress messages if present.
        if "messages" in kwargs:
            compressed, _store = compress_messages(kwargs["messages"])
            kwargs = {**kwargs, "messages": compressed}

        # Apply cache_control if system is present.
        if "system" in kwargs:
            cache_kwargs = place_cache_control(kwargs["system"], kwargs["messages"])
            kwargs = {**kwargs, **cache_kwargs}

        return self._real.create(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _ClientProxy:
    """Thin proxy for an Anthropic client that injects compression transparently."""

    def __init__(self, real_client: Any) -> None:
        self._real = real_client
        self.messages = _MessagesProxy(real_client.messages)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap(client: Any) -> _ClientProxy:
    """Wrap an Anthropic client so every ``messages.create`` call is compressed.

    The wrapper is a pure structural proxy — it imports nothing from the
    ``anthropic`` SDK and works with any duck-typed object that exposes a
    ``messages.create(**kwargs)`` method.

    Example
    -------
    ::

        import anthropic
        import distil.adapters.anthropic as distil_anthropic

        client = distil_anthropic.wrap(anthropic.Anthropic())
        # All subsequent calls are transparently compressed:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello!"}],
        )
    """
    return _ClientProxy(client)
