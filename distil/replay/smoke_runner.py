"""SmokeRunner — a PLUMBING stand-in. NOT a grader for any published claim.

It exists for one reason: to exercise the proof harness (frontier + certification
coverage) offline, with no API key and no downloaded dataset, so the mechanics can
be unit-tested. It is deliberately NOT the ``DeterministicRunner`` (which keys on
planted ``DECISION:`` markers and is the circularity we are removing).

How it "decides": it models an agent that reads the **load-bearing data record** in
the fresh observation and ignores chatter. Concretely it extracts ``key=value``
fields only from *dense* record lines (≥3 fields), ignoring sparse log/meta lines,
and returns a hash of that field set. Fold markers from the reversible digest
(``handle=…``) are stripped first, because that content is *recoverable*. So:

  * byte-exact / lossless (reversible) transforms keep the record line intact (they
    only minify whitespace/JSON and fold *recoverable* noise) → same fields → no
    decision change. This is faithful: a real model with ``distil_expand`` recovers
    folded detail, so lossless must not flip.
  * truncation that elides the record drops its fields → different hash → change.
    This is faithful too: truncation is irrecoverable.

This mimics the *direction* a real model moves (lose the load-bearing fact → flip)
WITHOUT planting a directive, which is enough to verify the harness distinguishes
safe from unsafe compression. It is NOT evidence about real agents. For evidence,
run with ``--runner anthropic`` on real τ-/SWE-bench traces. The harness prints a
banner saying so.
"""

from __future__ import annotations

import hashlib
import re

from ..trajectory import Block, Stability

_FIELD = re.compile(r"\b([A-Za-z_][\w.]*)=([\w.:/+-]+)")
_FOLD = re.compile(r"<<[^>]*handle=[^>]*>>")  # reversible-digest placeholder (recoverable)


class SmokeRunner:
    name = "smoke"
    evidential = False  # the harness reads this to print a loud non-evidence banner

    def decide(self, blocks: list[Block]) -> str:
        facts: set[str] = set()
        for b in blocks:
            # the decision turns on the *fresh* observation, like a real agent step
            if b.stability is not Stability.VOLATILE:
                continue
            text = _FOLD.sub("", b.text)  # folded noise is recoverable → don't penalize
            for line in text.splitlines():
                fields = _FIELD.findall(line)
                if len(fields) >= 3:  # a data record, not a sparse log/meta line
                    facts.update(f"{k.lower()}={v.lower()}" for k, v in fields)
        if not facts:
            return "<no-record>"
        return "act:" + hashlib.sha1("|".join(sorted(facts)).encode()).hexdigest()[:12]
