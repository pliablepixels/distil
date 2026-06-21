"""train_transformer — fine-tune a token-classification transformer for line keep/drop.

THIS PRODUCES YOUR CHECKPOINT FROM YOUR TRACES.

The model trained here is specific to the text distribution in YOUR corpus.  It
generalises the deterministic ``SalienceKeepModel`` rules to unseen text via a
lightweight transformer (default: ``google/bert_uncased_L-2_H-128_A-2``).

**The zero-dep logistic model (``distil.codec.learned``) is the default for all
distil users.**  Run this script only when you have a corpus of real traces and
want to upgrade to the transformer keep model.

Training labels
---------------
Labels are derived from ``build_dataset()`` in ``distil.codec.learned``: every
line in a VOLATILE block is assigned a binary label — 1 (keep) if
``SalienceKeepModel`` scores it >= 0.6, else 0 (drop).  Each *token* of the line
receives the same label as its containing line.  Pad/special tokens are masked
with ``-100`` (ignored by PyTorch CrossEntropyLoss).

Output layout (``out_dir``)
---------------------------
::

    out_dir/
        model.onnx          — exported ONNX graph (feed: input_ids, attention_mask)
        tokenizer.json      — fast tokenizer for TransformerKeepModel.from_pretrained
        tokenizer_config.json
        vocab.txt           — (BERT) or equivalent
        metrics.json        — held-out accuracy / F1 from the training run

Optional heavy dependencies
---------------------------
``torch`` and ``transformers`` are NOT installed by default.  Install them with::

    pip install 'distil-llm[train]'
    # or:
    pip install torch transformers

This module may be imported freely — heavy imports are deferred inside functions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_transformer(
    out_dir: str,
    *,
    base_model: str = "google/bert_uncased_L-2_H-128_A-2",
    epochs: int = 3,
    lr: float = 5e-5,
    max_len: int = 64,
) -> dict[str, Any]:
    """Fine-tune a token-classification transformer and export to ONNX.

    Parameters
    ----------
    out_dir:
        Directory to write the ONNX model, tokenizer, and metrics JSON.
        Created if it does not exist.
    base_model:
        HuggingFace model ID to fine-tune.  Default ``google/bert_uncased_L-2_H-128_A-2``
        (~4 MB) keeps training fast without a GPU; swap for a larger encoder
        (e.g. ``distilbert-base-uncased``) for higher accuracy.
    epochs:
        Number of full passes over the training set.
    lr:
        AdamW learning rate.
    max_len:
        Maximum token sequence length; lines are truncated/padded to this.

    Returns
    -------
    dict
        Keys: ``train_size``, ``test_size``, ``accuracy``, ``f1``,
        ``onnx_path``, ``tokenizer_dir``.

    Raises
    ------
    ImportError
        If ``torch`` or ``transformers`` is not installed.
        Install with ``pip install 'distil-llm[train]'``.
    RuntimeError
        If the corpus yields an empty dataset.

    Notes
    -----
    The produced checkpoint is trained on YOUR traces — do not distribute it as
    a generic pretrained model.  For inference, use
    ``TransformerKeepModel.from_pretrained(onnx_path, out_dir)``.
    """
    # --- lazy imports (heavy deps) ---
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for train_transformer.\n"
            "Install it with:  pip install 'distil-llm[train]'\n"
            "  or:             pip install torch transformers"
        ) from exc

    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for train_transformer.\n"
            "Install it with:  pip install 'distil-llm[train]'\n"
            "  or:             pip install torch transformers"
        ) from exc

    # --- corpus labels ---
    from distil.codec.learned import build_dataset

    _, raw_lines_all = build_dataset()
    samples_lf, raw_lines = build_dataset()  # (features, label) pairs — we need raw_lines
    # We only need the raw lines and their labels; discard the feature vectors.
    labeled: list[tuple[str, int]] = [
        (line, int(label)) for (_, label), line in zip(samples_lf, raw_lines)
    ]

    if not labeled:
        raise RuntimeError(
            "Corpus produced an empty dataset.  Ensure ~/.distil/corpus/ contains trajectory files."
        )

    # --- deterministic train/test split (80/20 by hash) ---
    import hashlib

    train_set: list[tuple[str, int]] = []
    test_set: list[tuple[str, int]] = []
    for line, label in labeled:
        digest = hashlib.sha256(line.encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFF_FFFF
        if bucket < 0.2:
            test_set.append((line, label))
        else:
            train_set.append((line, label))

    if not train_set:
        raise RuntimeError("Training set is empty after split.")
    if not test_set:
        raise RuntimeError("Test set is empty after split.")

    # --- tokenizer + model ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    except (ValueError, OSError):
        # Some tiny model repos (e.g. google/bert_uncased_L-2_H-128_A-2) ship no tokenizer
        # files; fall back to the standard BERT WordPiece vocab, which is
        # compatible with the bert-tiny/mini/small family (same 30522 vocab).
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(base_model, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # type: ignore[attr-defined]

    # --- tokenize helper ---
    def tokenize_and_label(
        pairs: list[tuple[str, int]],
    ) -> list[dict[str, torch.Tensor]]:
        """Return a list of single-example dicts ready for the model."""
        examples: list[dict[str, torch.Tensor]] = []
        for text, line_label in pairs:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            attn = enc["attention_mask"][0].tolist()
            # Assign line label to real tokens; mask pad + special tokens with -100.
            # Special tokens are positions where attention_mask==1 but word_ids is None.
            word_ids = enc.word_ids(batch_index=0)  # None for special / padding tokens
            labels_seq = []
            for wid, am in zip(word_ids, attn):
                if wid is None or am == 0:
                    labels_seq.append(-100)
                else:
                    labels_seq.append(line_label)
            examples.append(
                {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": torch.tensor(labels_seq, dtype=torch.long),
                }
            )
        return examples

    train_examples = tokenize_and_label(train_set)
    test_examples = tokenize_and_label(test_set)

    # --- training loop ---
    rng = random.Random(42)
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_examples)
        total_loss = 0.0
        for ex in train_examples:
            optimizer.zero_grad()
            out = model(
                input_ids=ex["input_ids"].unsqueeze(0).to(device),
                attention_mask=ex["attention_mask"].unsqueeze(0).to(device),
                labels=ex["labels"].unsqueeze(0).to(device),
            )
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
        avg = total_loss / len(train_examples) if train_examples else 0.0
        print(f"Epoch {epoch + 1}/{epochs}  train_loss={avg:.4f}")

    # --- evaluation ---
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for ex in test_examples:
            logits = model(
                input_ids=ex["input_ids"].unsqueeze(0).to(device),
                attention_mask=ex["attention_mask"].unsqueeze(0).to(device),
            ).logits[0]  # [seq, 2]
            preds = logits.argmax(dim=-1).tolist()
            true_labels = ex["labels"].tolist()
            for pred, true in zip(preds, true_labels):
                if true == -100:
                    continue
                if pred == 1 and true == 1:
                    tp += 1
                elif pred == 1 and true == 0:
                    fp += 1
                elif pred == 0 and true == 1:
                    fn += 1
                else:
                    tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- export to ONNX ---
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_path = str(out_path / "model.onnx")

    model.eval()
    dummy_ids = torch.zeros(1, max_len, dtype=torch.long).to(device)
    dummy_mask = torch.ones(1, max_len, dtype=torch.long).to(device)

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=14,
    )

    # --- save tokenizer ---
    tokenizer.save_pretrained(str(out_path))

    metrics: dict[str, Any] = {
        "train_size": len(train_set),
        "test_size": len(test_set),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "onnx_path": onnx_path,
        "tokenizer_dir": str(out_path),
    }
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(
        f"Exported ONNX to {onnx_path}\n"
        f"Accuracy={accuracy:.4f}  F1={f1:.4f}\n"
        f"Load with: TransformerKeepModel.from_pretrained({onnx_path!r}, {str(out_path)!r})"
    )
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI wrapper: ``python -m distil.codec.train_transformer --out <dir>``."""
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a transformer token-classifier for line keep/drop and export to ONNX. "
            "Requires 'distil-llm[train]' extras (torch + transformers)."
        )
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="Output directory for model.onnx, tokenizer, and metrics.json.",
    )
    parser.add_argument(
        "--base-model",
        default="google/bert_uncased_L-2_H-128_A-2",
        metavar="MODEL_ID",
        help="HuggingFace model ID to fine-tune (default: google/bert_uncased_L-2_H-128_A-2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        metavar="LR",
        help="AdamW learning rate (default: 5e-5).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=64,
        metavar="L",
        help="Maximum token sequence length (default: 64).",
    )
    args = parser.parse_args(argv)
    result = train_transformer(
        args.out,
        base_model=args.base_model,
        epochs=args.epochs,
        lr=args.lr,
        max_len=args.max_len,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
