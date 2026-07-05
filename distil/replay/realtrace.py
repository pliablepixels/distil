"""Adapters that load **real agent traces** into Distil's trajectory model.

This is the module that breaks the circularity flagged in ``docs/PAPER_PLAN.md``.
The bundled corpus plants ``DECISION:`` markers that the offline
``DeterministicRunner`` keys on, so "decision-equivalence" there is a tautology.
These adapters instead ingest **τ-bench** and **SWE-bench** trajectories, where:

  * nothing in the context tells the model what to do (no directive/marker), and
  * the decision is the agent's *actual next action* — a tool call (τ-bench) or
    an edit/command (SWE-bench) — which the model must INFER from context.

Graded with ``AnthropicRunner`` (a real model), decision-equivalence becomes a
genuine measurement, not a string-preservation check. The adapters return plain
``CorpusEntry`` objects, so the existing ``conformal.calibrate`` /
``certify`` machinery consumes them unchanged.

Each entry's trajectory carries, per decision point, the **gold action** recorded
in the trace — exposed via :func:`gold_actions` for downstream metrics (model↔gold
agreement, task success) without ever leaking the answer into the model's context.

Native formats accepted (both are common public shapes; parsers are defensive):

  τ-bench   : a JSON list of episodes, each ``{"messages"|"traj": [...], "reward": x}``
              where messages are ``{"role": system|user|assistant|tool, "content": str,
              "tool_calls": [{"function": {"name","arguments"}}]}``.
  SWE-bench : a SWE-agent ``.traj`` ``{"trajectory": [{"action","observation",
              "thought"?}], "info": {"exit_status"?, "resolved"?}}`` plus an optional
              top-level ``problem_statement`` / ``instance_id`` / ``repo``.

A normalized fixture of each (no planted answers) ships under
``benchmarks/fixtures/`` so the harness is exercisable offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..corpus import CorpusEntry
from ..trajectory import Block, Kind, Stability, Trajectory, Turn


@dataclass
class GoldDecision:
    """The action the agent actually took at a decision point, from the trace.

    Used for downstream metrics only — never injected into model context.
    ``fingerprint`` is the canonical ``{action,target}`` JSON the live runner also
    emits, so model↔gold agreement is a direct string compare.
    """

    trajectory_id: str
    turn_index: int
    action: str
    target: str

    @property
    def fingerprint(self) -> str:
        # same canonical form the grader's parse_fingerprint emits, so model↔gold
        # agreement compares like with like (paraphrase of the same tool ≠ mismatch)
        from .prompts import canonical

        return canonical(self.action, self.target)


# in-memory side table: (trajectory_id, turn_index) -> GoldDecision
_GOLD: dict[tuple[str, int], GoldDecision] = {}
# in-memory side table: trajectory_id -> task succeeded? (τ-bench reward / SWE resolved)
_SUCCESS: dict[str, bool] = {}


def success_label(entry: CorpusEntry) -> bool | None:
    """Did this trajectory's task succeed (τ-bench reward>0 / SWE-bench resolved)?
    ``None`` if the trace carried no outcome. Used for the downstream task-success
    metric — never injected into model context."""
    return _SUCCESS.get(entry.trajectory.id)


def gold_actions(entries: list[CorpusEntry]) -> dict[tuple[str, int], GoldDecision]:
    """Return the recorded gold decisions for the given entries (loaded by the
    adapters). Keyed by (trajectory_id, turn_index)."""
    keys = {(e.trajectory.id, t.index) for e in entries for t in e.trajectory.turns}
    return {k: v for k, v in _GOLD.items() if k in keys}


def _register_gold(traj_id: str, turn_index: int, action: str, target: str) -> None:
    _GOLD[(traj_id, turn_index)] = GoldDecision(traj_id, turn_index, action or "", target or "")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _norm_args(arguments) -> str:
    """A tool call's arguments → a single canonical 'target' string (first value,
    or the whole arg blob). Mirrors the {action,target} fingerprint the runner uses."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return arguments.strip()
    if isinstance(arguments, dict) and arguments:
        # primary argument = the first value; stable across reorderings via sorted keys
        first_key = sorted(arguments)[0]
        return str(arguments[first_key])
    return json.dumps(arguments, sort_keys=True) if arguments else ""


def _structural_problems(traj: Trajectory) -> list[str]:
    """Light structural check for REAL traces (no DECISION-marker requirement —
    that requirement is exactly the circularity we are removing)."""
    problems: list[str] = []
    if len(traj.turns) < 1:
        problems.append(f"{traj.id}: no decision points")
    for turn in traj.turns:
        last_nonvol = -1
        first_vol = len(turn.blocks)
        for i, b in enumerate(turn.blocks):
            if b.stability is Stability.VOLATILE:
                first_vol = min(first_vol, i)
            else:
                last_nonvol = max(last_nonvol, i)
        if first_vol < last_nonvol:
            problems.append(f"{traj.id} turn {turn.index}: volatile block precedes a cacheable one")
        if not any(b.stability is Stability.VOLATILE for b in turn.blocks):
            problems.append(
                f"{traj.id} turn {turn.index}: no volatile block (nothing fresh to decide on)"
            )
    return problems


def validate_real(entries: list[CorpusEntry]) -> list[str]:
    out: list[str] = []
    for e in entries:
        out += _structural_problems(e.trajectory)
    return out


# --------------------------------------------------------------------------- #
# τ-bench
# --------------------------------------------------------------------------- #


def _tau_messages(episode: dict) -> list[dict]:
    return episode.get("messages") or episode.get("traj") or episode.get("trajectory") or []


def load_tau_bench(path: str | Path, *, model: str = "claude-opus-4-8") -> list[CorpusEntry]:
    """Load τ-bench episodes into trajectories.

    Each assistant message that issues a tool call is a decision point: the context
    is everything before it (system + tools as a STABLE prefix, prior exchange as
    SETTLING history, the most recent tool/user output as the VOLATILE tail), and
    the gold decision is that tool call's ``{name, primary-arg}``. No marker, no
    directive — the model must read the observation to choose the call.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    episodes = raw if isinstance(raw, list) else raw.get("episodes") or raw.get("results") or [raw]
    entries: list[CorpusEntry] = []

    for ei, ep in enumerate(episodes):
        ep_id = str(ep.get("id") or ep.get("task_id") or ep.get("instance_id") or f"tau-{ei}")
        msgs = _tau_messages(ep)
        if not msgs:
            continue

        reward = ep.get("reward", ep.get("success"))
        if reward is not None:
            _SUCCESS[ep_id] = (
                float(reward) > 0 if isinstance(reward, (int, float)) else bool(reward)
            )

        system_text = ""
        tools_text = ""
        for m in msgs:
            if m.get("role") == "system" and not system_text:
                system_text = m.get("content") or ""
        if "tools" in ep:
            tools_text = json.dumps(ep["tools"], indent=2)

        turns = []
        history: list[str] = []  # settled exchange text, byte-stable once written
        pending_obs: list[str] = []  # fresh tool/user outputs since the last decision

        decision_no = 0
        for m in msgs:
            role = m.get("role")
            content = m.get("content") or ""
            calls = m.get("tool_calls") or []
            if role == "system":
                continue
            if role in ("user", "tool"):
                if content:
                    pending_obs.append(f"[{role}] {content}")
                continue
            if role == "assistant":
                if calls:
                    fn = calls[0].get("function", calls[0])
                    name = fn.get("name", "")
                    target = _norm_args(fn.get("arguments", {}))
                    blocks: list[Block] = []
                    if system_text:
                        blocks.append(
                            Block(f"{ep_id}:system", Kind.SYSTEM, system_text, Stability.STABLE)
                        )
                    if tools_text:
                        blocks.append(
                            Block(f"{ep_id}:tools", Kind.TOOLS, tools_text, Stability.STABLE)
                        )
                    if history:
                        blocks.append(
                            Block(
                                f"{ep_id}:hist@{decision_no}",
                                Kind.HISTORY,
                                "\n\n".join(history),
                                Stability.SETTLING,
                            )
                        )
                    obs = "\n\n".join(pending_obs) if pending_obs else content or "(continue)"
                    blocks.append(
                        Block(
                            f"{ep_id}:obs@{decision_no}",
                            Kind.TOOL_OUTPUT,
                            obs,
                            Stability.VOLATILE,
                            True,
                        )
                    )
                    turns.append(Turn(decision_no, blocks))
                    _register_gold(ep_id, decision_no, name, target)
                    # settle this exchange into history for subsequent turns
                    if pending_obs:
                        history.extend(pending_obs)
                        pending_obs = []
                    history.append(f"[assistant] called {name}({target})")
                    decision_no += 1
                elif content:
                    pending_obs.append(f"[assistant] {content}")

        if turns:
            traj = Trajectory(id=ep_id, model=model, turns=turns)
            entries.append(CorpusEntry(f"tau::{ep_id}", "tau-bench", ep.get("title", ep_id), traj))
    return entries


# --------------------------------------------------------------------------- #
# SWE-bench (SWE-agent .traj)
# --------------------------------------------------------------------------- #


def _swe_action_fingerprint(action: str) -> tuple[str, str]:
    """A shell/edit action string → (verb, primary-target). E.g.
    'edit 12:14 src/foo.py' → ('edit', 'src/foo.py'); 'python -m pytest' → ('python','-m')."""
    action = (action or "").strip()
    if not action:
        return ("", "")
    parts = action.split()
    verb = parts[0]
    target = next(
        (p for p in parts[1:] if "/" in p or "." in p), parts[1] if len(parts) > 1 else ""
    )
    return (verb, target)


def load_swe_bench(path: str | Path, *, model: str = "claude-opus-4-8") -> list[CorpusEntry]:
    """Load SWE-agent ``.traj`` trajectories (single file or a directory of them).

    Each step is a decision point: the context is the problem statement + setup
    (STABLE), prior steps (SETTLING), and the latest observation (VOLATILE); the
    gold decision is the step's ``action`` (verb + primary file/target). Resolution
    status (``info.resolved`` / ``exit_status``) is carried for the downstream
    task-success metric.
    """
    p = Path(path)
    files = sorted(p.glob("*.traj")) + sorted(p.glob("*.json")) if p.is_dir() else [p]
    entries: list[CorpusEntry] = []

    # a single file may hold one trajectory (dict) or many (list) — normalize to a list
    raws: list[tuple[dict, str]] = []
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(doc, list):
            raws += [(d, f.stem) for d in doc]
        else:
            raws.append((doc, f.stem))

    for raw, stem in raws:
        inst = str(raw.get("instance_id") or raw.get("id") or stem)
        problem = raw.get("problem_statement") or raw.get("issue") or ""
        setup = raw.get("system") or raw.get("setup") or ""
        steps = raw.get("trajectory") or raw.get("steps") or []
        info = raw.get("info") or {}
        resolved = bool(
            info.get("resolved", info.get("exit_status") == "submitted" and info.get("submission"))
        )

        stable: list[Block] = []
        if setup:
            stable.append(Block(f"{inst}:system", Kind.SYSTEM, setup, Stability.STABLE))
        if problem:
            stable.append(
                Block(f"{inst}:problem", Kind.SYSTEM, f"ISSUE:\n{problem}", Stability.STABLE)
            )

        turns = []
        history: list[str] = []
        for si, step in enumerate(steps):
            action = step.get("action") or ""
            obs = step.get("observation") or ""
            verb, target = _swe_action_fingerprint(action)
            blocks = [b.copy_with(b.text) for b in stable]
            if history:
                blocks.append(
                    Block(
                        f"{inst}:hist@{si}", Kind.HISTORY, "\n\n".join(history), Stability.SETTLING
                    )
                )
            blocks.append(
                Block(
                    f"{inst}:obs@{si}",
                    Kind.TOOL_OUTPUT,
                    obs or "(no observation)",
                    Stability.VOLATILE,
                    True,
                )
            )
            turns.append(Turn(si, blocks))
            _register_gold(inst, si, verb, target)
            if obs:
                history.append(f"[observation@{si}] {obs[:2000]}")
            history.append(f"[action@{si}] {action}")

        if turns:
            traj = Trajectory(id=inst, model=model, turns=turns)
            title = f"{inst} ({'resolved' if resolved else 'unresolved'})"
            entries.append(CorpusEntry(f"swe::{inst}", "swe-bench", title, traj))
            _GOLD[(inst, -1)] = GoldDecision(
                inst, -1, "RESOLVED" if resolved else "UNRESOLVED", inst
            )
            _SUCCESS[inst] = resolved
    return entries


def resolved_status(entry: CorpusEntry) -> bool | None:
    """SWE-bench only: did this trajectory resolve the issue (from the trace)? None
    if unknown / not a SWE entry."""
    g = _GOLD.get((entry.trajectory.id, -1))
    return None if g is None else (g.action == "RESOLVED")
