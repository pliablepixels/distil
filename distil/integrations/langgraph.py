"""LangGraph integration — compress graph state before the model node.

LangGraph agents accumulate messages in state and re-send them every step — the
same re-send-the-world cost distil targets. This adds a drop-in ``pre_model_hook``
(LangGraph's seam for transforming state right before the LLM node) and a plain
``compress_state`` helper, both reusing the exact same reversible compression as
the proxy. Duck-typed — it never imports ``langgraph`` or ``langchain``.

    from distil.integrations.langgraph import pre_model_hook

    graph = create_react_agent(model, tools, pre_model_hook=pre_model_hook())
    # or, manually, inside any node:
    from distil.integrations.langgraph import compress_state
    state = compress_state(state, verbatim=True)
"""

from __future__ import annotations

from typing import Any

from .langchain import compress_messages


def compress_state(state: Any, *, verbatim: bool = False, key: str = "messages") -> Any:
    """Return *state* with its message list compressed (non-mutating).

    Works on a dict-like state (``state["messages"]``) or an attribute-style state
    (``state.messages``); anything without a message list is returned untouched.
    """
    if isinstance(state, dict):
        msgs = state.get(key)
        if isinstance(msgs, list):
            return {**state, key: compress_messages(msgs, verbatim=verbatim)}
        return state
    msgs = getattr(state, key, None)
    if isinstance(msgs, list):
        try:
            return state.copy(update={key: compress_messages(msgs, verbatim=verbatim)})
        except (AttributeError, TypeError):
            return state
    return state


def pre_model_hook(*, verbatim: bool = False, key: str = "messages") -> Any:
    """Return a LangGraph ``pre_model_hook`` callable that compresses state messages.

    LangGraph calls the hook with the graph state right before the model node and
    merges the returned dict back in. We return ``{key: <compressed messages>}`` so
    only the message list is updated — every other field is left as-is.
    """

    def _hook(state: Any) -> dict[str, Any]:
        msgs = state.get(key) if isinstance(state, dict) else getattr(state, key, None)
        if isinstance(msgs, list):
            return {key: compress_messages(msgs, verbatim=verbatim)}
        return {}

    return _hook
