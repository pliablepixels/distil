"""features — fixed-length feature vectors for the learned token-keep classifier.

Each feature is normalized to [0, 1] or is a binary indicator, so the logistic
regression weights are on a comparable scale without additional preprocessing.
"""

from __future__ import annotations

import re

from distil.codec.keep_model import ERROR_KEYWORDS, DEBUG_KEYWORDS, NEVER_DROP_MARKER

# ---------------------------------------------------------------------------
# Pre-compiled regexes (module-level for speed)
# ---------------------------------------------------------------------------

_RE_KEY_VALUE = re.compile(r"^\s*[\w\-\.]+\s*:\s*\S")
_RE_TABLE_ROW = re.compile(r"\|")
_RE_JSON_START = re.compile(r"^\s*[\[\{]")
_RE_DIGITS = re.compile(r"\d+(?:\.\d+)?")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Ordered list of feature names, parallel to the vector returned by ``featurize``.
FEATURE_NAMES: list[str] = [
    "bias",  # constant 1.0 — intercept learned via weight
    "has_decision_marker",  # DECISION: present → always keep
    "has_error_keyword",  # any ERROR_KEYWORDS word present (case-insensitive)
    "looks_structured",  # JSON start, key:value, or pipe-table
    "digit_density",  # fraction of chars that are digits, clamped to [0,1]
    "length_norm",  # min(len / 120, 1.0)
    "has_debug_keyword",  # any DEBUG_KEYWORDS word (penalises keep)
    "uppercase_ratio",  # fraction of alpha chars that are uppercase
    "is_blank",  # 1.0 if line is empty/whitespace — always drop
]


def featurize(line: str, kind: str) -> list[float]:  # noqa: ARG001
    """Return a fixed-length feature vector for *line*.

    The *kind* parameter is accepted for API compatibility with future
    per-content-type features but is not used in the current feature set.

    Returns
    -------
    list[float]
        A vector of length ``len(FEATURE_NAMES)``, each entry in [0, 1].
    """
    stripped = line.strip()
    n = len(stripped)

    # --- bias ---
    bias = 1.0

    # --- is_blank ---
    is_blank = 1.0 if n == 0 else 0.0

    if n == 0:
        # All other features are zero for blank lines.
        return [bias, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, is_blank]

    lower = stripped.lower()

    # --- has_decision_marker ---
    has_decision_marker = 1.0 if NEVER_DROP_MARKER in line else 0.0

    # --- has_error_keyword ---
    has_error_keyword = 1.0 if any(kw in lower for kw in ERROR_KEYWORDS) else 0.0

    # --- looks_structured ---
    looks_structured = (
        1.0
        if (
            _RE_JSON_START.match(stripped)
            or _RE_KEY_VALUE.match(stripped)
            or _RE_TABLE_ROW.search(stripped)
        )
        else 0.0
    )

    # --- digit_density ---
    digit_count = sum(1 for ch in stripped if ch.isdigit())
    digit_density = min(digit_count / n, 1.0)

    # --- length_norm ---
    length_norm = min(n / 120.0, 1.0)

    # --- has_debug_keyword ---
    has_debug_keyword = 1.0 if any(kw in lower for kw in DEBUG_KEYWORDS) else 0.0

    # --- uppercase_ratio ---
    alpha_chars = [ch for ch in stripped if ch.isalpha()]
    if alpha_chars:
        upper_count = sum(1 for ch in alpha_chars if ch.isupper())
        uppercase_ratio = upper_count / len(alpha_chars)
    else:
        uppercase_ratio = 0.0

    return [
        bias,
        has_decision_marker,
        has_error_keyword,
        looks_structured,
        digit_density,
        length_norm,
        has_debug_keyword,
        uppercase_ratio,
        is_blank,
    ]
