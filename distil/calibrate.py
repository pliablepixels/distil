"""Auto-calibration of the relevance-gate operating point to agent capability.

E11 is the reason this module exists. The relevance gate's non-inferiority to full context
is real but the *operating point* is capability-dependent: the working-set size that is
non-inferior on a weak agent (claude-haiku-4-5: keep 6) costs **-31 pp** on a strong one
(DeepSeek-V3), which needs keep 12. Left as a hand-tuned constant, that is a silent-loss
hazard — point distil at a new model and you may ship a lossy operating point without
noticing. This module removes the hazard.

It is the operating-point analogue of the Decision-Equivalence Risk Certificate
(:mod:`distil.conformal`): the certificate selects the most aggressive *compression level*
whose decision-change rate is provably controlled; this selects the most aggressive
*working-set size* whose **task-success loss** is provably controlled, using the same paired
non-inferiority test the papers report (:func:`distil.certify.stats.mcnemar_noninferiority`).

Two production guarantees:

* **Most-aggressive-under-a-budget.** Among candidate operating points, pick the one that
  digests the most periphery while remaining statistically non-inferior to full context at a
  pre-registered ``margin``.
* **Fail-safe.** If *no* candidate is certified non-inferior, calibration refuses to compress
  and falls back to full context. The default is fail-closed: silence never ships loss.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from distil.certify.stats import Z_95, mcnemar_noninferiority


@dataclass(frozen=True)
class OperatingPoint:
    """One candidate operating point and its paired outcome against the baseline (full).

    ``gate_recent`` is the working-set size; **smaller is more aggressive** (digests more
    periphery). ``losses``/``gains`` are the McNemar discordant counts vs. the baseline:
    ``losses`` = baseline solved & candidate did not, ``gains`` = candidate solved & baseline
    did not. ``n`` is the number of paired instances.
    """

    name: str
    gate_recent: int
    losses: int
    gains: int
    n: int


@dataclass(frozen=True)
class LevelVerdict:
    name: str
    gate_recent: int
    n: int
    delta: float
    ci95_low: float
    ci95_high: float
    noninferior: bool


@dataclass(frozen=True)
class CalibrationCertificate:
    """The selected operating point (or fail-safe) plus the full audited candidate table."""

    selected: str | None
    selected_gate_recent: int | None
    margin: float
    fail_safe: bool
    levels: tuple[LevelVerdict, ...]
    rationale: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["levels"] = [asdict(v) for v in self.levels]
        return d


def paired_discordant(
    baseline: Mapping[str, bool], candidate: Mapping[str, bool]
) -> tuple[int, int, int]:
    """McNemar discordant counts over the instances scored by **both** conditions.

    Returns ``(losses, gains, n)``: ``losses`` = baseline solved & candidate did not,
    ``gains`` = candidate solved & baseline did not, ``n`` = paired instances. Instances
    missing from either side are dropped (you can only compare what both scored).
    """
    ids = set(baseline) & set(candidate)
    losses = sum(1 for i in ids if baseline[i] and not candidate[i])
    gains = sum(1 for i in ids if candidate[i] and not baseline[i])
    return losses, gains, len(ids)


def calibrate_operating_point(
    points: Sequence[OperatingPoint], *, margin: float = 0.05, z: float = Z_95
) -> CalibrationCertificate:
    """Select the most aggressive operating point still non-inferior to full; else fail safe.

    Candidates are evaluated most-aggressive-first (ascending ``gate_recent``). The first one
    whose paired task-success loss is non-inferior to the baseline at ``margin`` is selected —
    that is the maximum compression the agent's capability supports. If none qualifies, the
    certificate is ``fail_safe`` with ``selected=None`` and the caller must keep full context.
    """
    ordered = sorted(points, key=lambda p: p.gate_recent)
    verdicts: list[LevelVerdict] = []
    selected: LevelVerdict | None = None
    for p in ordered:
        r = mcnemar_noninferiority(p.losses, p.gains, p.n, margin, z)
        v = LevelVerdict(
            name=p.name,
            gate_recent=p.gate_recent,
            n=r.n,
            delta=round(r.delta, 4),
            ci95_low=round(r.ci95_low, 4),
            ci95_high=round(r.ci95_high, 4),
            noninferior=r.noninferior,
        )
        verdicts.append(v)
        if selected is None and v.noninferior:
            selected = v

    if selected is None:
        rationale = (
            f"No candidate operating point is non-inferior to full context at margin "
            f"{margin:.0%}. Failing safe to full context (no compression) — the operating "
            f"point must be widened or recalibrated on more data before the gate can ship."
        )
        return CalibrationCertificate(None, None, margin, True, tuple(verdicts), rationale)

    rationale = (
        f"Selected {selected.name} (gate_recent={selected.gate_recent}): the most aggressive "
        f"operating point with task-success non-inferior to full at margin {margin:.0%} "
        f"(delta {selected.delta:+.1%}, 95% CI lower {selected.ci95_low:+.1%} > "
        f"-{margin:.0%}, n={selected.n})."
    )
    return CalibrationCertificate(
        selected.name, selected.gate_recent, margin, False, tuple(verdicts), rationale
    )


# --------------------------------------------------------------------------- #
# Loaders for real harness output (swebench-style score JSON: per_instance map)
# --------------------------------------------------------------------------- #


def resolved_map(score_json: Mapping) -> dict[str, bool]:
    """Extract ``{instance_id: resolved}`` from a swebench-style score JSON.

    Accepts either ``{"per_instance": {id: {"resolved": bool}}}`` (the long-horizon harness
    format) or a flat ``{id: bool}`` map.
    """
    per = score_json.get("per_instance", score_json)
    out: dict[str, bool] = {}
    for iid, v in per.items():
        out[iid] = bool(v["resolved"] if isinstance(v, Mapping) else v)
    return out


def calibrate_from_scores(
    baseline_path: str | Path,
    candidates: Sequence[tuple[str, str | Path, int]],
    *,
    margin: float = 0.05,
) -> CalibrationCertificate:
    """Calibrate directly from score JSON files.

    Args:
        baseline_path: path to the full-context score JSON.
        candidates: ``(name, score_path, gate_recent)`` per candidate operating point.
        margin: tolerated absolute pass-rate drop (proportion).
    """
    base = resolved_map(json.loads(Path(baseline_path).read_text()))
    points: list[OperatingPoint] = []
    for name, path, gate_recent in candidates:
        cand = resolved_map(json.loads(Path(path).read_text()))
        losses, gains, n = paired_discordant(base, cand)
        points.append(OperatingPoint(name, gate_recent, losses, gains, n))
    return calibrate_operating_point(points, margin=margin)
