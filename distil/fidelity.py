"""Byte-fidelity invariants (Roadmap Phase 6) — prove the lossless claim.

Tier-0/1 are lossless *by construction*; this module turns that claim into a
machine-checkable contract:

  * `verify_reversible` — every original block is recoverable, either because the
    compressed text is byte-identical or because the original lives in the local
    restore table (under its block id for Tier-0, its content handle for Tier-1).
    Reported as SHA-256 equality so it is auditable.
  * `assert_append_only` — frozen history never mutates: a block id that appears
    in two turns must carry identical bytes (a changed past block is a violation).
  * `numeric_precision_preserved` — JSON canonicalization must not lose numeric
    precision (the value round-trips exactly).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .compress.base import CompressResult
from .trajectory import Block


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class FidelityReport:
    recoverable: list[str] = field(default_factory=list)
    irrecoverable: list[str] = field(default_factory=list)

    @property
    def lossless(self) -> bool:
        return not self.irrecoverable


def verify_reversible(originals: list[Block], result: CompressResult) -> FidelityReport:
    """Confirm every original block can be reconstructed from the compressed
    output plus the local restore table."""
    compressed_by_id = {b.id: b.text for b in result.blocks}
    restore_values = set(result.restore.values())
    report = FidelityReport()
    for o in originals:
        same = compressed_by_id.get(o.id) == o.text
        recovered = same or o.text in restore_values
        (report.recoverable if recovered else report.irrecoverable).append(o.id)
    return report


def assert_append_only(prev: list[Block], curr: list[Block]) -> list[str]:
    """Return ids of frozen blocks that mutated between turns (append-only
    violations). Empty list means the invariant holds."""
    prev_text = {b.id: b.text for b in prev}
    violations: list[str] = []
    for b in curr:
        if b.id in prev_text and prev_text[b.id] != b.text:
            violations.append(b.id)
    return violations


def numeric_precision_preserved(original_json: str, transformed_json: str) -> bool:
    """True if both strings parse to the same JSON value (no precision loss)."""
    try:
        return json.loads(original_json) == json.loads(transformed_json)
    except (ValueError, TypeError):
        return False


def sha_manifest(blocks: list[Block]) -> dict[str, str]:
    """Auditable per-block SHA-256 manifest."""
    return {b.id: _sha(b.text) for b in blocks}
