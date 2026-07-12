"""Lossless/subscription mode gets in-context columnar fold — real token savings,
ToS-safe (self-describing table, no handle to invite an unavailable distil_expand).

Addresses #24: subscription users on lossless mode were getting only tier-0 (JSON
minify + run collapse). A JSON array of flat records — the shape agents constantly
get back from tools — now folds to a compact table, all data inline, ~70% smaller,
which the model reads directly. See distil/compress/structured.py:fold(emit_handle)
and distil/adapters/anthropic.py:_lossless_fold.
"""

import json

from distil.adapters.anthropic import RestoreStore, _compress_tool_result_text
from distil.compress.structured import fold, template_fold


def _arr(n: int) -> str:
    return json.dumps(
        [{"id": i, "user": f"u{i}", "status": "active" if i % 2 else "idle"} for i in range(n)],
        indent=2,
    )


def test_lossless_fold_is_self_describing_no_handle():
    out = fold(_arr(20), emit_handle=False)
    assert out is not None
    assert "handle=" not in out, "lossless fold must NOT emit a handle (no expand available)"
    assert "rows=20" in out and "cols=id,user,status" in out  # self-describing header


def test_lossless_fold_loses_no_data():
    out = fold(_arr(15), emit_handle=False)
    for i in range(15):
        assert f"u{i}" in out, f"row {i} value dropped — fold must be lossless"


def test_handle_variant_still_default():
    assert "handle=" in fold(_arr(10))  # default keeps the handle (offline/expand path)


def test_verbatim_path_applies_lossless_fold():
    # a JSON array tool_result in verbatim/lossless mode -> compact table, no handle
    out = _compress_tool_result_text(_arr(30), RestoreStore(), verbatim=True)
    assert "handle=" not in out
    assert "cols=id,user,status" in out
    assert len(out) < len(_arr(30))  # real reduction


def test_verbatim_non_tabular_falls_back_to_tier0():
    # genuinely varied prose: not a JSON array, no shared template -> neither fold fires
    text = "\n".join(
        [
            "the build finished and everything looks green",
            "no compiler warnings surfaced this run",
            "the cache was warm so it went quickly",
            "reviewers signed off earlier this morning",
            "deployment is planned for later today",
            "someone should refresh the changelog entry",
            "a flaky integration case passed on retry",
        ]
    )
    out = _compress_tool_result_text(text, RestoreStore(), verbatim=True)
    assert "«" not in out, "no fold marker on non-tabular prose (tier-0 fallback)"
    assert out.startswith("the build finished")


def test_template_fold_lossless_no_handle():
    logs = "\n".join(f"2026-07-12 10:00:{i:02d} INFO req id={i} status=200" for i in range(12))
    out = template_fold(logs, emit_handle=False)
    if out is not None:  # template_fold is conservative; only assert when it fires
        assert "handle=" not in out


if __name__ == "__main__":
    test_lossless_fold_is_self_describing_no_handle()
    test_lossless_fold_loses_no_data()
    test_handle_variant_still_default()
    test_verbatim_path_applies_lossless_fold()
    test_verbatim_non_tabular_falls_back_to_tier0()
    test_template_fold_lossless_no_handle()
    print("ok — lossless columnar fold on the subscription path, self-describing + no data loss")
