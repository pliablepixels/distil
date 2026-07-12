"""The expand loop must intercept a distil_expand call on ANY recoverable handle,
not only handles created by the current request.

Regression for #25: a streamed turn that digested nothing new but referenced a
handle stub persisted from an earlier turn took the streaming pass-through with
no expand tool injected, so the model's distil_expand tool_use escaped to the
client as "No such tool available". The gate must key on "is a recoverable handle
in the outgoing conversation", not "did THIS request create a handle".
See distil/proxy.py:_expand_should_intercept.
"""

from distil.proxy import _expand_should_intercept, _has_recoverable_stub


class _Store:
    def __init__(self, handles):
        self.handles = handles


def _msg(text):
    return {"role": "user", "content": [{"type": "tool_result", "content": text}]}


STUB = "ran suite\n<< +60 lines, handle=965d42f6 >>\n  Tests 1955 passed"


def test_intercepts_on_persisted_stub_with_no_new_handles():
    # this request digested nothing (store.handles empty) but an OLD stub is in context
    body = {"messages": [_msg(STUB)]}
    assert _expand_should_intercept(True, _Store(set()), body) is True


def test_intercepts_on_fresh_handles():
    body = {"messages": [_msg("no stub here")]}
    assert _expand_should_intercept(True, _Store({"abc12345"}), body) is True


def test_no_intercept_when_no_handle_anywhere():
    body = {"messages": [_msg("plain output, nothing folded")]}
    assert _expand_should_intercept(True, _Store(set()), body) is False


def test_no_intercept_when_expand_off():
    body = {"messages": [_msg(STUB)]}
    assert _expand_should_intercept(False, _Store({"abc12345"}), body) is False


def test_stub_detection_handles_nested_and_string_content():
    assert _has_recoverable_stub({"messages": [{"role": "user", "content": STUB}]}) is True
    assert _has_recoverable_stub({"messages": [_msg(STUB)]}) is True
    assert _has_recoverable_stub({"messages": [_msg("clean")]}) is False
    assert _has_recoverable_stub({}) is False


if __name__ == "__main__":
    test_intercepts_on_persisted_stub_with_no_new_handles()
    test_intercepts_on_fresh_handles()
    test_no_intercept_when_no_handle_anywhere()
    test_no_intercept_when_expand_off()
    test_stub_detection_handles_nested_and_string_content()
    print("ok — expand gate intercepts on any recoverable handle (#25)")
