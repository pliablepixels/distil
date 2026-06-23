"""LiteLLM integration — compress requests in-process, no proxy required.

Two ways to use distil with LiteLLM:

1. **Proxy (zero code):** point LiteLLM at ``distil proxy`` via ``api_base``.
2. **In-process (this module):** compress the messages right before the call::

       from distil.integrations import litellm as distil_litellm

       resp = distil_litellm.completion(
           model="claude-opus-4-8",
           messages=[...],
           distil_lossless_only=True,   # optional; subscription/OAuth-safe
       )

:func:`compress` is the pure, framework-free core (returns new completion kwargs
with the ``messages`` reversibly compressed); :func:`completion` / :func:`acompletion`
are thin wrappers that hand the compressed kwargs to the real ``litellm``.
"""

from __future__ import annotations

from typing import Any

from ..adapters.anthropic import compress_messages


def compress(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``litellm.completion`` kwargs with ``messages`` compressed.

    A ``distil_lossless_only=True`` kwarg (consumed here, not forwarded) selects the
    subscription/OAuth-safe lossless-in-context mode. Non-list ``messages`` are
    returned untouched.
    """
    messages = kwargs.get("messages")
    if not isinstance(messages, list):
        return kwargs
    lossless_only = bool(kwargs.get("distil_lossless_only", False))
    compressed, _store = compress_messages(messages, lossless_only=lossless_only)
    new = {k: v for k, v in kwargs.items() if k != "distil_lossless_only"}
    new["messages"] = compressed
    return new


def completion(**kwargs: Any) -> Any:
    """Drop-in for ``litellm.completion`` that compresses the request first."""
    import litellm  # lazy: optional dependency

    return litellm.completion(**compress(kwargs))


async def acompletion(**kwargs: Any) -> Any:
    """Async drop-in for ``litellm.acompletion`` that compresses the request first."""
    import litellm  # lazy: optional dependency

    return await litellm.acompletion(**compress(kwargs))
