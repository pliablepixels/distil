"""Content-aware skeleton digest — a structure-preserving, reversible compressor.

The reversible tier's job is to make a *peripheral* context block small while keeping it
**navigable** (the agent can see what's there and recover the part it needs). Plain
head-truncation fails at both: it shows an arbitrary first-N chars of a file (you cannot
tell which functions exist) and it *drops the tail of a traceback* — exactly where the
exception and failing assertion live.

This module produces a skeleton instead:

* **Python source** (the SWE-bench regime): keep every ``import``, every class/def
  **signature** (with decorators and the first docstring line), and elide function
  *bodies* to ``...``. The agent sees the full structure — which symbols exist, where —
  and can ``distil_expand`` the one block it needs. Reconstructed via :mod:`ast`; falls
  back cleanly on syntax errors / partial files.
* **Tracebacks & test output**: keep the head *and the tail* (the exception, the
  ``file:line``, the failing assertion), collapsing the quiet middle.
* **Anything else**: head+tail window rather than head-only.

Every transform is deterministic, stdlib-only (no model, no network — auditable and
safe to run on untrusted context), and **lossy only at the surface**: the caller keeps
the original behind a content handle, so the block is fully recoverable (the reversible
tier's contract). This is the digest the certified relevance-gate digests periphery with.
"""

from __future__ import annotations

import ast

# Markers an agent (or a human) can grep for; kept short to not eat the savings.
_ELIDED = "..."  # body placeholder, emitted at the body's indentation


def _docstring_first_line(node: ast.AST) -> str | None:
    """First physical line of a node's docstring, if it has one (kept as a hint)."""
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        doc = first.value.value.strip().splitlines()
        if doc:
            return doc[0].strip()
    return None


def _function_body_ranges(tree: ast.AST) -> list[tuple[int, int, int, str | None]]:
    """For each function whose enclosing scope is *not* another function, return
    ``(body_first_line, end_line, col_offset, docstring_first_line)`` — the line span to
    elide, the indentation to place ``...`` at, and a docstring hint to keep. Methods
    (in a class) are included; closures (in a function) are not — eliding the outer body
    already removes them.
    """
    ranges: list[tuple[int, int, int, str | None]] = []

    def visit(node: ast.AST, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            is_func = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_func and not in_function:
                body = child.body
                doc = _docstring_first_line(child)
                # Body starts after the signature (and after the docstring, if kept).
                first_stmt = body[1] if (doc and len(body) > 1) else body[0]
                start = first_stmt.lineno
                end = child.end_lineno or start
                if end >= start:  # only elide if there is a body to elide
                    ranges.append((start, end, first_stmt.col_offset, doc))
                # Descend with in_function=True so closures inside are not double-counted.
                visit(child, True)
            else:
                visit(child, is_func or in_function)

    visit(tree, False)
    return ranges


def code_skeleton(text: str) -> str | None:
    """Python skeleton: signatures + imports kept, bodies elided to ``...``.

    Returns ``None`` when the text is not parseable Python (caller should fall back), or
    when the skeleton would not actually be smaller than the original.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None

    ranges = _function_body_ranges(tree)
    if not ranges:
        return None  # no function bodies to elide — skeleton wouldn't help

    lines = text.splitlines()
    # Map each body's first line -> (end, indent, doc) for one-pass emission.
    elide_start = {start: (end, col, doc) for (start, end, col, doc) in ranges}
    out: list[str] = []
    i = 1  # 1-based line numbers (ast convention)
    n = len(lines)
    while i <= n:
        if i in elide_start:
            end, col, doc = elide_start[i]
            pad = " " * col
            if doc:
                out.append(f'{pad}"""{doc}"""')
            out.append(f"{pad}{_ELIDED}")
            i = end + 1  # skip the elided body
        else:
            out.append(lines[i - 1])
            i += 1
    skeleton = "\n".join(out)
    return skeleton if len(skeleton) < len(text) else None


def text_window(text: str, *, head: int = 400, tail: int = 200) -> str:
    """Head+tail window for non-code blocks. Unlike head-only truncation this keeps the
    *end* — where tracebacks put the exception and tests put the failing assertion."""
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [{omitted} chars elided] ...\n{text[-tail:]}"


def smart_digest(text: str, *, head: int = 400, tail: int = 200) -> str:
    """Best available structure-preserving digest of one context block.

    Code → skeleton (signatures kept, bodies elided); otherwise a head+tail window.
    Deterministic and lossy *only at the surface* — callers keep the original behind a
    content handle for recovery.
    """
    sk = code_skeleton(text)
    if sk is not None:
        return sk
    return text_window(text, head=head, tail=tail)
