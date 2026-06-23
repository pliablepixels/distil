"""In-process LiteLLM + LangChain integration helpers (framework-free core)."""

from __future__ import annotations

from distil.integrations import langchain as dlc
from distil.integrations import litellm as dll

BIG = "\n".join(f"row {i}: value_{i} status=ok detail=lorem" for i in range(40))


# --- LiteLLM --------------------------------------------------------------- #


def test_litellm_compress_compresses_messages():
    kwargs = {"model": "x", "messages": [{"role": "tool", "content": BIG}]}
    out = dll.compress(kwargs)
    assert out["messages"][0]["content"] != BIG
    assert len(out["messages"][0]["content"]) < len(BIG)
    assert out["model"] == "x"


def test_litellm_compress_strips_distil_flag():
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "distil_lossless_only": True}
    out = dll.compress(kwargs)
    assert "distil_lossless_only" not in out


def test_litellm_lossless_only_does_not_digest():
    kwargs = {"messages": [{"role": "tool", "content": BIG}], "distil_lossless_only": True}
    out = dll.compress(kwargs)
    assert "<< +" not in out["messages"][0]["content"]  # no digest stub


def test_litellm_non_list_messages_untouched():
    kwargs = {"messages": "nope"}
    assert dll.compress(kwargs) == kwargs


# --- LangChain (duck-typed) ------------------------------------------------ #


class _Msg:
    """Minimal stand-in for a pydantic-v2 LangChain message."""

    def __init__(self, type_, content):
        self.type = type_
        self.content = content

    def model_copy(self, update):
        return _Msg(self.type, update.get("content", self.content))


def test_langchain_tool_message_digested_dict():
    msgs = [{"type": "tool", "content": BIG}]
    out = dlc.compress_messages(msgs)
    assert out[0]["content"] != BIG and len(out[0]["content"]) < len(BIG)


def test_langchain_ai_message_never_rewritten():
    minifiable = '{"a":   1,    "b": 2}'
    out = dlc.compress_messages([{"type": "ai", "content": minifiable}])
    assert out[0]["content"] == minifiable


def test_langchain_object_messages_via_model_copy():
    out = dlc.compress_messages([_Msg("tool", BIG)])
    assert isinstance(out[0], _Msg)
    assert out[0].content != BIG and len(out[0].content) < len(BIG)


def test_langchain_lossless_only():
    out = dlc.compress_messages([{"type": "tool", "content": BIG}], lossless_only=True)
    assert "<< +" not in out[0]["content"]


def test_langchain_non_string_content_untouched():
    blocks = [{"type": "text", "text": "x"}]
    out = dlc.compress_messages([{"type": "human", "content": blocks}])
    assert out[0]["content"] is blocks
