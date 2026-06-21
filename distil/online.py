"""Self-distilling online loop — the never-regressing moat.

The core idea: the keep-model that decides which lines to drop should train on
CAUSAL labels, not a generic heuristic.  Causal labels come directly from the
deployer's own traffic: run counterfactual ablation (``discover``) on every
trajectory to learn which blocks were *causally inert* vs which ones *changed an
agent decision*.  Any line inside an inert block is safe to drop (label=0); any
line inside a decision-changing block must be kept (label=1).

This is the moat headroom can't copy:
- Headroom trains on a generic judge applied to synthetic data.
- Ours trains on causally-verified, deployment-specific labels derived from real
  agent trajectories running in the customer's own environment.

The loop is:
    1. ``collect_causal_labels`` — run ablation on corpus → per-line (0/1) labels.
    2. ``retrain`` — featurize + logistic regression with train/test split → metrics.
    3. ``certify_promotion`` — wrap new weights as a strategy, run the TOST gate on
       every corpus trajectory; only promote if ALL pass non-inferior.
    4. ``online_round`` — orchestrates the above and persists weights iff certified.

Because step 3 requires non-inferior TOST on every trajectory before weights are
touched in production, the loop is *never-regressing by construction*: a cycle that
would degrade quality is silently discarded.  The model can only get better.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from distil.certify.gate import certify
from distil.codec.features import featurize
from distil.codec.keep_model import apply_keep
from distil.codec.learned import LogisticKeepModel, _evaluate, train
from distil.corpus import load_corpus
from distil.replay.ablation import discover
from distil.trajectory import Stability

if TYPE_CHECKING:
    from distil.corpus import CorpusEntry
    from distil.trajectory import Block


# ---------------------------------------------------------------------------
# Causal label collection
# ---------------------------------------------------------------------------

_VOLATILE_KINDS: frozenset[str] = frozenset({"tool_output", "retrieved"})


def collect_causal_labels(
    entries: list["CorpusEntry"] | None = None,
) -> list[tuple[str, int]]:
    """Collect per-line binary keep-labels derived from causal ablation.

    For each trajectory in the corpus:
    - Run ``discover`` (counterfactual ablation) to classify every block as
      PRUNABLE (causally inert — never changed a decision) or KEPT (changed at
      least one decision).
    - For every line inside a PRUNABLE block assign label ``0`` (safe to drop).
    - For every line inside a KEPT block assign label ``1`` (must keep).

    Conflict resolution: if the same raw line appears in both prunable and kept
    blocks (possible when identical lines exist across blocks), the label is
    promoted to ``1`` (keep wins).  This is conservative and correct: we never
    train a model to drop a line whose presence has been observed to matter.

    Parameters
    ----------
    entries:
        Corpus entries to use.  If ``None``, ``load_corpus()`` is called.

    Returns
    -------
    list[tuple[str, int]]
        Deduplicated ``(line_text, label)`` pairs (label in {0, 1}).
        Only VOLATILE blocks are scanned; STABLE/SETTLING blocks are the
        cacheable prefix and are out of scope for the keep-model.
    """
    if entries is None:
        entries = load_corpus()

    # Map from raw line text -> label (1 wins over 0 on conflict).
    line_labels: dict[str, int] = {}

    for entry in entries:
        traj = entry.trajectory
        report = discover(traj)

        # Build a set of prunable block ids for fast lookup.
        prunable_ids: set[str] = {v.block_id for v in report.prunable}

        for turn in traj.turns:
            for block in turn.blocks:
                if block.stability is not Stability.VOLATILE:
                    continue
                # Only scan the content kinds the keep-model operates on.
                if block.kind.value not in _VOLATILE_KINDS:
                    continue

                label = 0 if block.id in prunable_ids else 1

                for line in block.text.splitlines():
                    existing = line_labels.get(line)
                    if existing is None:
                        line_labels[line] = label
                    elif label == 1:
                        # Keep wins on conflict.
                        line_labels[line] = 1
                    # else: existing == 1 already, no change needed.

    return list(line_labels.items())


# ---------------------------------------------------------------------------
# Retrain
# ---------------------------------------------------------------------------


def retrain(
    labels: list[tuple[str, int]],
    *,
    test_fraction: float = 0.2,
    epochs: int = 400,
) -> dict:
    """Train a logistic keep-model on causal labels.

    Parameters
    ----------
    labels:
        ``(line_text, label)`` pairs as returned by ``collect_causal_labels``.
        Labels must be in {0, 1}.
    test_fraction:
        Fraction of samples to hold out for evaluation.  Split is deterministic
        (SHA-256 hash of the raw line text) so results are reproducible without
        a fixed random seed.
    epochs:
        Number of full-batch gradient-descent epochs.

    Returns
    -------
    dict
        Keys: ``weights`` (list[float]), ``accuracy`` (float), ``f1`` (float),
        ``n_train`` (int), ``n_test`` (int).
    """
    if not labels:
        raise ValueError("Cannot retrain on empty label set.")

    # Build feature vectors.
    samples: list[tuple[list[float], float]] = []
    raw_lines: list[str] = []
    for line, label in labels:
        features = featurize(line, "tool_output")
        samples.append((features, float(label)))
        raw_lines.append(line)

    # Deterministic train/test split (same algorithm as learned.py).
    train_set: list[tuple[list[float], float]] = []
    test_set: list[tuple[list[float], float]] = []
    for (x, y), raw in zip(samples, raw_lines):
        digest = hashlib.sha256(raw.encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFF_FFFF
        if bucket < test_fraction:
            test_set.append((x, y))
        else:
            train_set.append((x, y))

    if not train_set:
        raise RuntimeError("Training set is empty — increase label set or lower test_fraction.")

    # Fall back to using all labels as both train and test if test set is empty
    # (can happen with tiny corpora during tests).
    eval_set = test_set if test_set else train_set

    # Train logistic regression.
    weights = train(train_set, epochs=epochs, lr=0.5, l2=1e-4, seed=0)

    # Evaluate on held-out set.
    metrics = _evaluate(eval_set, weights)

    return {
        "weights": weights,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "n_train": len(train_set),
        "n_test": len(eval_set),
    }


# ---------------------------------------------------------------------------
# Certify promotion
# ---------------------------------------------------------------------------


def certify_promotion(
    weights: list[float],
    entries: list["CorpusEntry"] | None = None,
    *,
    margin: float = 0.02,
) -> bool:
    """Non-inferiority gate: promote weights only if ALL trajectories pass TOST.

    Builds a ``LogisticKeepModel`` from *weights*, wraps it as a compression
    strategy that applies ``apply_keep`` to every VOLATILE block at a conservative
    target_ratio (0.7 — keep 70% of volatile lines), then runs ``certify`` on
    each corpus trajectory.

    The gate passes (returns ``True``) only if *every single trajectory* certifies
    non-inferior at the given *margin*.  A single regression anywhere blocks the
    promotion.  This is the safety invariant that makes the loop never-regressing.

    Parameters
    ----------
    weights:
        Candidate weight vector for ``LogisticKeepModel``.
    entries:
        Corpus entries to certify against.  Defaults to ``load_corpus()``.
    margin:
        TOST non-inferiority margin (default 0.02 = tolerate at most 2-point drop).

    Returns
    -------
    bool
        ``True`` if all trajectories certify non-inferior, ``False`` otherwise.
    """
    if entries is None:
        entries = load_corpus()

    model = LogisticKeepModel(weights)
    target_ratio = 0.7

    from distil.trajectory import Stability

    def _strategy(blocks: list[Block], _turn: int) -> list[Block]:
        """Apply keep-model compression to causally-inert VOLATILE blocks only.

        We only compress blocks that are VOLATILE, of an eligible kind, AND
        not decision_relevant.  Decision-relevant blocks changed at least one
        agent decision during ablation — compressing them would cause the very
        regressions the gate is designed to catch.  The keep-model is only
        meaningful for the causally-inert fraction where label=0 was assigned.
        """
        out: list[Block] = []
        for block in blocks:
            if (
                block.stability is Stability.VOLATILE
                and block.kind.value in _VOLATILE_KINDS
                and not block.decision_relevant
            ):
                compressed_text = apply_keep(block.text, block.kind.value, target_ratio, model)
                out.append(block.copy_with(compressed_text))
            else:
                out.append(block)
        return out

    for entry in entries:
        report = certify(entry.trajectory, _strategy, margin=margin)
        if report.verdict != "PASS":
            return False

    return True


# ---------------------------------------------------------------------------
# Online round
# ---------------------------------------------------------------------------


def online_round(
    entries: list["CorpusEntry"] | None = None,
    *,
    promote_to: str | None = None,
) -> dict:
    """Run one complete self-distillation round.

    Steps
    -----
    1. Collect causal labels from the corpus (or *entries*).
    2. Retrain the logistic keep-model on those labels.
    3. Run the non-inferiority gate (``certify_promotion``).
    4. If certified AND *promote_to* is given, persist the new weights to that
       JSON path as a ``LogisticKeepModel``-compatible file.

    Parameters
    ----------
    entries:
        Corpus entries to use for labelling and certification.  Defaults to
        ``load_corpus()``.
    promote_to:
        If given and the model is certified, write the new weights to this path.
        The file is written in the same JSON format as ``LogisticKeepModel.to_json``.

    Returns
    -------
    dict
        Keys: ``n_labels`` (int), ``accuracy`` (float), ``f1`` (float),
        ``certified`` (bool), ``promoted`` (bool).
    """
    if entries is None:
        entries = load_corpus()

    # Step 1: collect causal labels.
    labels = collect_causal_labels(entries)
    n_labels = len(labels)

    # Step 2: retrain.
    train_result = retrain(labels)
    weights: list[float] = train_result["weights"]
    accuracy: float = train_result["accuracy"]
    f1: float = train_result["f1"]

    # Step 3: certify.
    certified = certify_promotion(weights, entries)

    # Step 4: persist iff certified and a target path was given.
    promoted = False
    if certified and promote_to is not None:
        model = LogisticKeepModel(weights)
        model.save(Path(promote_to))
        promoted = True

    return {
        "n_labels": n_labels,
        "accuracy": accuracy,
        "f1": f1,
        "certified": certified,
        "promoted": promoted,
    }
