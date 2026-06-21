"""transformer — ONNX-backed transformer keep model.

Implements the ``KeepModel`` Protocol using a fine-tuned token-classification
transformer exported to ONNX.  The ONNX runtime and tokenizer are lazy-imported
so the stdlib core of distil runs with zero deps; this file only costs anything
when you actually call ``from_pretrained``.

The production checkpoint is NOT included in this repo.  It must be trained on
YOUR own traces using ``distil.codec.train_transformer.train_transformer`` (or
the ``distil-train`` CLI entry point added in pyproject).  The zero-dep logistic
model (``distil.codec.learned``) is the default until you deploy your checkpoint.

Injection-seam design
---------------------
``TransformerKeepModel.__init__`` accepts any ``session`` object whose ``.run()``
matches the ``onnxruntime.InferenceSession`` signature, and any ``encode``
callable — making the class fully testable with lightweight fakes in stdlib-only
unit tests.  Heavy deps are required only in ``from_pretrained``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------


@runtime_checkable
class _OrtSession(Protocol):
    """Structural type for anything that behaves like an ORT InferenceSession."""

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]: ...


# ---------------------------------------------------------------------------
# Pure-Python softmax (tolerates list or array inputs; no numpy required)
# ---------------------------------------------------------------------------


def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over a 1-D sequence of logits."""
    if not logits:
        return []
    max_l = max(logits)
    exps = [math.exp(v - max_l) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

_TOP_K = 3  # mean of top-k token keep-probs for line aggregation


def _aggregate_token_probs(keep_probs: list[float]) -> float:
    """Return a robust line-level keep score from per-token keep probabilities.

    Strategy: mean of the top-k token keep-probs (k=``_TOP_K``), falling back
    to plain mean when fewer than k real tokens are available.  This is more
    robust than plain mean because most lines contain a mix of content and
    punctuation tokens; the highest-salience tokens carry the signal.
    """
    if not keep_probs:
        return 0.0
    k = min(_TOP_K, len(keep_probs))
    top_k = sorted(keep_probs, reverse=True)[:k]
    return sum(top_k) / len(top_k)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TransformerKeepModel:
    """ONNX-backed transformer keep model implementing the ``KeepModel`` Protocol.

    Parameters
    ----------
    session:
        Anything with a ``.run(None, input_feed)`` interface — typically an
        ``onnxruntime.InferenceSession`` loaded from a fine-tuned checkpoint.
        See ``from_pretrained`` for the canonical loader.
    encode:
        Callable ``str -> dict[str, list[int]]`` that tokenizes a line and
        returns at minimum ``input_ids`` and ``attention_mask`` as plain lists
        of ints.  Typically wraps an ``AutoTokenizer`` from 🤗 Transformers.
    keep_label_index:
        Which output-logit index corresponds to the "keep" class.  Default 1
        (standard two-label classification: 0=drop, 1=keep).
    floor_decisions:
        When ``True`` (default), any line containing ``"DECISION:"`` is given
        a minimum score of 1.0 regardless of what the model predicts.  This
        matches the never-drop guarantee in ``SalienceKeepModel`` and ensures
        the transformer model cannot accidentally discard explicit agent intent.

    Notes
    -----
    The production checkpoint is trained by the deployer on their own traces
    (see ``distil.codec.train_transformer``).  This class does not ship a
    pretrained checkpoint — it is the inference adapter only.
    """

    def __init__(
        self,
        session: Any,
        encode: Callable[[str], dict[str, list[int]]],
        *,
        keep_label_index: int = 1,
        floor_decisions: bool = True,
        input_names: set[str] | None = None,
    ) -> None:
        self._session = session
        self._encode = encode
        self._keep_label_index = keep_label_index
        self._floor_decisions = floor_decisions
        # When set, feeds are restricted to these names — the exported ONNX graph
        # may declare fewer inputs (e.g. no token_type_ids) than the tokenizer emits.
        self._input_names = input_names

    # ------------------------------------------------------------------
    # KeepModel Protocol
    # ------------------------------------------------------------------

    def score(self, line: str, kind: str) -> float:  # noqa: ARG002
        """Return a salience score in [0, 1] via the transformer model.

        Steps
        -----
        1. Tokenize the line.
        2. Run the ONNX session to obtain per-token logits shaped [1, seq, num_labels].
        3. Apply softmax over the label axis to get per-token class probabilities.
        4. Collect the keep-probability for each real (non-pad) token (attention_mask == 1).
        5. Aggregate via mean of top-k keep-probs (k=3).
        6. If ``floor_decisions`` and ``"DECISION:"`` is in the line, return max(score, 1.0).

        The *kind* parameter is accepted for interface compatibility; the current
        implementation does not use it (the transformer sees the raw text).
        """
        if not line.strip():
            return 0.0

        # --- tokenize ---
        encoded = self._encode(line)
        input_ids: list[int] = encoded["input_ids"]
        attention_mask: list[int] = encoded.get("attention_mask", [1] * len(input_ids))

        # Build feeds; token_type_ids are optional (BERT needs them, DistilBERT doesn't).
        feeds: dict[str, Any] = {
            "input_ids": [input_ids],  # shape [1, seq]
            "attention_mask": [attention_mask],  # shape [1, seq]
        }
        if "token_type_ids" in encoded:
            feeds["token_type_ids"] = [encoded["token_type_ids"]]

        # Restrict to the graph's declared inputs (the export may omit token_type_ids).
        if self._input_names is not None:
            feeds = {k: v for k, v in feeds.items() if k in self._input_names}

        # --- run ONNX session ---
        outputs = self._session.run(None, feeds)
        # outputs[0] is logits: shape [1, seq, num_labels] — may be a nested list or numpy array.
        logits_3d = outputs[0]

        # --- extract per-token keep probabilities (real tokens only) ---
        keep_probs: list[float] = []
        seq_len = len(attention_mask)
        for tok_idx in range(seq_len):
            if not attention_mask[tok_idx]:
                continue  # skip padding
            # Support both nested lists and numpy-array-style indexing.
            try:
                tok_logits = logits_3d[0][tok_idx]
            except (IndexError, TypeError):
                continue
            # Convert to plain Python floats (works for numpy scalars too).
            logit_list = [float(v) for v in tok_logits]
            probs = _softmax(logit_list)
            keep_idx = self._keep_label_index
            if keep_idx < len(probs):
                keep_probs.append(probs[keep_idx])

        raw_score = _aggregate_token_probs(keep_probs)

        # --- never-drop floor ---
        if self._floor_decisions and "DECISION:" in line:
            return max(raw_score, 1.0)

        return float(raw_score)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        onnx_path: str,
        tokenizer_dir: str,
        **kw: Any,
    ) -> "TransformerKeepModel":
        """Load an ONNX checkpoint + tokenizer and return a ready instance.

        Parameters
        ----------
        onnx_path:
            Path to the ``.onnx`` model file produced by
            ``distil.codec.train_transformer.train_transformer``.
        tokenizer_dir:
            Directory holding the saved tokenizer (``tokenizer.json``, vocab,
            etc.) as written by ``tokenizer.save_pretrained(out_dir)``.
        **kw:
            Extra keyword arguments forwarded to ``TransformerKeepModel.__init__``
            (e.g. ``keep_label_index``, ``floor_decisions``).

        Raises
        ------
        ImportError
            If ``onnxruntime`` or ``transformers`` is not installed.  Install
            with ``pip install 'distil-llm[onnx]'``.
        """
        try:
            import onnxruntime as ort  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for TransformerKeepModel.from_pretrained.\n"
                "Install it with:  pip install 'distil-llm[onnx]'\n"
                "  or:             pip install onnxruntime"
            ) from exc

        try:
            from transformers import AutoTokenizer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "transformers is required for TransformerKeepModel.from_pretrained.\n"
                "Install it with:  pip install 'distil-llm[onnx]'\n"
                "  or:             pip install transformers"
            ) from exc

        session = ort.InferenceSession(onnx_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        kw.setdefault("input_names", {i.name for i in session.get_inputs()})

        def encode(line: str) -> dict[str, list[int]]:
            enc = tokenizer(
                line,
                return_tensors=None,  # plain Python lists
                truncation=True,
                padding=False,
            )
            # AutoTokenizer returns BatchEncoding with list values when return_tensors=None
            return {k: list(v) for k, v in enc.items()}

        return cls(session, encode, **kw)
