"""distil.codec — model-agnostic heuristic codec for per-content-type line selection.

This package implements the *keep-model seam*: a pluggable interface through which a
deterministic salience heuristic (``SalienceKeepModel``) can be swapped for a learned
classifier without changing any call-site code.

Drop-in upgrade path
--------------------
A trained ModernBERT-style token-keep classifier implements the same ``KeepModel.score``
Protocol (``score(line, kind) -> float``) and is passed as the ``model`` argument to
``apply_keep``.  The framework here is model-agnostic; no trained model exists in this
repo — the heuristic default ships instead so the seam is honest and exercised from day
one.
"""

from __future__ import annotations

from .keep_model import KeepModel, SalienceKeepModel, apply_keep

__all__ = ["KeepModel", "SalienceKeepModel", "apply_keep"]
