"""Agent-transcript adapter contract.

distil is agent-generic: the wrap proxies any base-url-honoring tool, and the
dissect report must stay useful for all of them. Correlating a wrap session
with the *agent's own conversation log* is therefore an adapter concern — one
small module per agent, all normalizing into the model below.

To add support for a new agent (codex, gemini, aider, ...):

1. Create ``distil/transcripts/<agent>.py`` with a class implementing
   :class:`TranscriptAdapter` — two methods, no base class required.
2. Normalize into :class:`Transcript`: human turns, tool calls, tool results.
   Timestamps are epoch seconds; ``text`` fields carry the agent's original
   content (read at render time only — distil never copies it anywhere).
3. Register it in ``distil/transcripts/__init__.py`` under the executable
   name the wrap manifest records (``tool``), e.g. ``"codex"``.

That's the whole surface: discovery (find candidate log files for a time
window) and load (parse one). Everything downstream — fold naming, unused-tool
detection, re-fetch analysis, per-turn cost — is adapter-independent and comes
for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class UserTurn:
    """One real human prompt (not tool feedback, not a subagent's input)."""

    index: int
    ts: float
    text: str  # trimmed to a label-sized excerpt by the adapter


@dataclass
class ToolCall:
    """The agent invoking a tool."""

    ts: float
    name: str
    call_id: str = ""
    turn: int = 0  # index of the human turn this happened under


@dataclass
class ToolResult:
    """A tool's output entering the conversation (what distil may later fold)."""

    ts: float
    text: str
    tool: str = ""  # resolved tool name ("" when the agent's log doesn't say)
    turn: int = 0


@dataclass
class Transcript:
    """A wrap-session-shaped view of one agent conversation log."""

    agent: str  # adapter name, e.g. "claude"
    path: Path
    label: str = ""  # human title if the agent records one
    cwd: str = ""
    started: float = 0.0
    ended: float = 0.0
    turns: list[UserTurn] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


class TranscriptAdapter(Protocol):
    """What an agent integration must provide. Keep both methods tolerant:
    a malformed line is skipped, never raised — correlation is best-effort."""

    name: str  # must equal the wrap manifest's "tool" (executable basename)

    def discover(self, window: tuple[float, float], cwd: str | None) -> list[Path]:
        """Candidate transcript files whose activity overlaps *window*,
        best match first. ``cwd`` is the wrap session's working directory
        when known — use it to narrow the search."""
        ...

    def load(self, path: Path) -> Transcript:
        """Parse one transcript file into the normalized model."""
        ...
