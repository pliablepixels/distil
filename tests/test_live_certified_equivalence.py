"""Live (adapter) vs certified (strategy) equivalence — make drift visible (F9).

The empirical decision-equivalence certificate is computed over the block-level
``distil`` strategy in :mod:`distil.compress.strategies`, but live serving compresses
message dicts through :func:`distil.adapters.anthropic.compress_messages`. These are
two *different* code paths, so a certificate proven on one is only meaningful for the
other insofar as they make the same keep/digest decisions and produce recoverable
handles with the same identity. This test pins that relationship so any divergence
becomes a visible, reviewed change rather than silent drift between "what we certify"
and "what we ship".

Both paths anchor recovery on the same handle: ``sha256(original_text)[:8]``. The test
asserts the two agree on that anchor wherever they both digest, and documents the ONE
intentional difference below.

Documented intentional delta (reviewed, not a bug):
  * The live adapter keeps the last ``_RECENCY_KEEP_TURNS`` user/tool turns byte-exact
    (an agent must see its freshest tool output verbatim to choose its next action, and
    the in-context path may not be able to expand a stub there). The certified strategy
    has no recency carve-out — it digests every volatile tool-output block. So the set
    of blocks the live path digests equals the certified set MINUS the recency tail.

If either the digest algorithm or the recency rule changes, this test breaks — which is
the point: convergence (or a deliberate new delta) must be re-reviewed here.
"""

from __future__ import annotations

import hashlib
import re

from distil.adapters.anthropic import _RECENCY_KEEP_TURNS, compress_messages
from distil.compress.strategies import REGISTRY
from distil.trajectory import Block, Kind, Stability

_HANDLE = re.compile(r"handle=([0-9a-f]{8})")


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _tool_output(name: str) -> str:
    # 12 keyword-free filler lines: well above the 6-line digest threshold, not JSON
    # (so it never folds) and not templated (so template-mining never fires) — the
    # plain reversible-digest path, with a droppable middle guaranteed.
    return "\n".join([f"{name}: detail line {i} of routine output" for i in range(12)])


def _tool_result_message(text: str) -> dict:
    return {"role": "user", "content": [{"type": "tool_result", "content": text}]}


def _handles_in(text: str) -> set[str]:
    return set(_HANDLE.findall(text))


def test_live_and_certified_agree_except_recency_tail() -> None:
    originals = [_tool_output(n) for n in ("alpha", "beta", "gamma", "delta")]
    assert len(originals) > _RECENCY_KEEP_TURNS  # need older blocks AND a recency tail

    # --- certified path: the block-level `distil` strategy the certificate is proven on
    blocks = [
        Block(id=f"b{i}", kind=Kind.TOOL_OUTPUT, text=t, stability=Stability.VOLATILE)
        for i, t in enumerate(originals)
    ]
    certified_out = REGISTRY["distil"](blocks, 0)
    cert_by_orig = {originals[i]: b.text for i, b in enumerate(certified_out)}
    cert_digested = {o for o, t in cert_by_orig.items() if t != o}

    # --- live path: the message-dict adapter used in real serving
    messages = [_tool_result_message(t) for t in originals]
    new_messages, store = compress_messages(messages)
    live_by_orig = {originals[i]: m["content"][0]["content"] for i, m in enumerate(new_messages)}
    live_digested = {o for o, t in live_by_orig.items() if t != o}

    # 1) The certified strategy digests every volatile tool-output block.
    assert cert_digested == set(originals)

    # 2) The live adapter digests the same blocks EXCEPT the recency-exempt tail, which
    #    it keeps byte-exact. This is the sole documented, reviewed divergence.
    recency_tail = set(originals[-_RECENCY_KEEP_TURNS:])
    assert live_digested == cert_digested - recency_tail
    for o in recency_tail:
        assert live_by_orig[o] == o  # verbatim-recency semantics: byte-exact

    # 3) Wherever BOTH paths digest, they emit the SAME recovery handle == sha256(orig)[:8],
    #    and the live store resolves it back to the byte-exact original. Same anchor ->
    #    a certificate about the digest's recoverability transfers to the live path.
    for o in live_digested:
        want = _sha8(o)
        assert _handles_in(live_by_orig[o]) == {want}
        assert _handles_in(cert_by_orig[o]) == {want}
        assert store.expand(want) == o
