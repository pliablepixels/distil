"""The trajectory corpus — load and validate the multi-domain test set.

The corpus is the asset that makes certification meaningful: a strategy is only
trustworthy if it stays non-inferior across many real agent shapes, not one. Each
trajectory is listed in `manifest.json` with its domain; `validate()` enforces
the structural invariants every trajectory must satisfy so the behavioural gate
(`distil bench`) measures what it claims to.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import pricing
from .trajectory import Stability, Trajectory


def _default_corpus_dir() -> Path:
    """Resolve the corpus directory: $DISTIL_CORPUS, then the packaged sibling,
    then ./corpus. The env override makes single-file (zipapp) distributables and
    custom corpora work without a repo checkout."""
    env = os.environ.get("DISTIL_CORPUS")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    for candidate in (here / "_corpus", here.parent / "corpus"):  # installed wheel, then repo
        if (candidate / "manifest.json").exists():
            return candidate
    return Path.cwd() / "corpus"


CORPUS_DIR = _default_corpus_dir()
_DECISION = "DECISION:"


@dataclass
class CorpusEntry:
    file: str
    domain: str
    title: str
    trajectory: Trajectory


def load_corpus(corpus_dir: Path | str = CORPUS_DIR) -> list[CorpusEntry]:
    corpus_dir = Path(corpus_dir)
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    entries: list[CorpusEntry] = []
    for e in manifest["trajectories"]:
        traj = Trajectory.load(corpus_dir / e["file"])
        entries.append(CorpusEntry(e["file"], e["domain"], e.get("title", traj.id), traj))
    return entries


def validate(traj: Trajectory) -> list[str]:
    """Return a list of structural problems; empty means the trajectory is well-formed.

    Invariants (these are what make the cache, ablation, and certification signals
    real, not artifacts of a hand-tuned example):
      1. >=3 turns, on a known model.
      2. Volatile blocks come after all non-volatile blocks each turn (cacheable prefix).
      3. STABLE blocks are byte-identical across every turn they appear in.
      4. At least one STABLE block carries a decision (so ablation keeps the prefix).
      5. At least one VOLATILE, decision-relevant tool output carries a decision.
      6. At least one VOLATILE, non-decision-relevant block carries no decision
         (causally-inert noise that ablation can prune).
    """
    problems: list[str] = []

    if len(traj.turns) < 3:
        problems.append(f"{traj.id}: needs >=3 turns, has {len(traj.turns)}")
    if traj.model not in pricing.CATALOG:
        problems.append(f"{traj.id}: unknown model {traj.model!r}")

    stable_text: dict[str, str] = {}
    has_stable_decision = False
    has_volatile_decision = False
    has_prunable_noise = False

    for turn in traj.turns:
        last_nonvolatile = -1
        first_volatile = len(turn.blocks)
        for i, b in enumerate(turn.blocks):
            if b.stability is Stability.VOLATILE:
                first_volatile = min(first_volatile, i)
            else:
                last_nonvolatile = max(last_nonvolatile, i)
            if b.stability is Stability.STABLE:
                if b.id in stable_text and stable_text[b.id] != b.text:
                    problems.append(
                        f"{traj.id} turn {turn.index}: STABLE block {b.id!r} changed "
                        "across turns (breaks the cache)"
                    )
                stable_text[b.id] = b.text
                if _DECISION in b.text:
                    has_stable_decision = True
            if b.stability is Stability.VOLATILE:
                if b.decision_relevant and _DECISION in b.text:
                    has_volatile_decision = True
                if not b.decision_relevant and _DECISION not in b.text:
                    has_prunable_noise = True
        if first_volatile < last_nonvolatile:
            problems.append(
                f"{traj.id} turn {turn.index}: a volatile block precedes a stable/settling "
                "block (prefix not cacheable)"
            )

    if not has_stable_decision:
        problems.append(f"{traj.id}: no STABLE block carries a {_DECISION} marker")
    if not has_volatile_decision:
        problems.append(f"{traj.id}: no decision-relevant volatile tool output carries a decision")
    if not has_prunable_noise:
        problems.append(f"{traj.id}: no causally-inert noise block (nothing for ablation to prune)")

    return problems
