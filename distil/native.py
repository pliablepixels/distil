"""Unified interface to distil-core functions, with pure-Python fallback.

Tries to import the compiled Rust extension ``distil_core``.  If the wheel
has been built and installed (via ``maturin develop`` or the sdist build),
the fast native implementations are used and ``BACKEND = "rust"``.

If the extension module is absent (source-only install, CI without Rust,
etc.) we fall back to the pure-Python implementations from
``distil.compress.tier0`` and ``distil.tokenizer`` and set
``BACKEND = "python"``.

Public API
----------
BACKEND : str
    Either ``"rust"`` or ``"python"``.

minify_json(text: str) -> str | None
    Re-encode JSON with no incidental whitespace.  Returns None for non-JSON.

collapse_runs(text: str) -> str
    Run-length-encode consecutive identical lines.

count_tokens(text: str, subword_factor: float = 1.33) -> int
    Estimate token count using the heuristic segmenter.
"""

from __future__ import annotations

try:
    import distil_core as _core

    BACKEND: str = "rust"

    def minify_json(text: str) -> str | None:
        """Re-encode JSON with no incidental whitespace (Rust implementation)."""
        return _core.minify_json(text)

    def collapse_runs(text: str) -> str:
        """Run-length-encode consecutive identical lines (Rust implementation)."""
        return _core.collapse_runs(text)

    def count_tokens(text: str, subword_factor: float = 1.33) -> int:
        """Estimate token count via heuristic segmenter (Rust implementation)."""
        return _core.count_tokens(text, subword_factor)

except ImportError:
    BACKEND = "python"

    from distil.compress.tier0 import minify_json as _py_minify_json
    from distil.compress.tier0 import collapse_runs as _py_collapse_runs
    from distil.tokenizer import HeuristicTokenizer as _HeuristicTokenizer

    def minify_json(text: str) -> str | None:  # type: ignore[misc]
        """Re-encode JSON with no incidental whitespace (Python fallback)."""
        return _py_minify_json(text)

    def collapse_runs(text: str) -> str:  # type: ignore[misc]
        """Run-length-encode consecutive identical lines (Python fallback)."""
        return _py_collapse_runs(text)

    def count_tokens(text: str, subword_factor: float = 1.33) -> int:  # type: ignore[misc]
        """Estimate token count via heuristic segmenter (Python fallback)."""
        return _HeuristicTokenizer(subword_factor).count(text)
