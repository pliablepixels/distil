"""Shared compressor types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..trajectory import Block


@dataclass
class CompressResult:
    """Compressed blocks plus a local restore table.

    `restore` maps a handle/marker -> the original text. It is stored LOCALLY in
    the runtime and is NOT sent to the model, so it costs no tokens. Its presence
    is what makes Tier-0/1 lossless *in effect*: any dropped detail is one local
    lookup (or one tool call) away.
    """

    blocks: list[Block]
    restore: dict[str, str] = field(default_factory=dict)


class Compressor(Protocol):
    tier: int
    name: str

    def compress(self, blocks: list[Block]) -> CompressResult: ...
