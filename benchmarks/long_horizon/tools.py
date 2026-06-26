"""Tool schemas (Anthropic tool-use format) and executors for the long-horizon agent.

Each executor operates on a worktree ``Path`` and returns a plain string result
(tool outputs are the large, accumulating peripheral context the compression proxy
must handle). Outputs are capped to keep individual turns realistic without
exhausting the model's context window prematurely.

Tool surface:
* list_dir(path=".")   — directory listing
* read_file(path)      — full file text (primary source of peripheral context)
* search(pattern)      — ripgrep / grep -rn, capped
* edit_file(path, old_str, new_str) — exact-match replace; errors loudly if not unique
* run_tests(path=None) — pytest on the repo or a sub-path, capped
* finish(reason)       — signals the agent is done (terminates the loop)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Cap on any single tool output — large enough to be realistic, small enough that a
# handful of read_file calls don't blow past the 200k context limit by turn 5.
MAX_OUTPUT_CHARS = 8_000

# Tool schemas in Anthropic tool-use format.
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "list_dir",
        "description": (
            "List the files and directories inside a path within the repository. "
            "Use '.' for the repo root. Returns a newline-separated listing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within the repository (default '.')",
                }
            },
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file within the repository, returned with 1-based line numbers. "
            "For large files, pass start_line/end_line to read a specific window (find "
            "the line first with search, then read around it). Without a range you get "
            "the file head and a note of the total line count — then re-read the window "
            "you need so you can construct an exact edit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path to read"},
                "start_line": {
                    "type": "integer",
                    "description": "1-based first line to read (optional)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "1-based last line to read, inclusive (optional)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "search",
        "description": (
            "Search for a pattern (regex) in the repository files using ripgrep or grep. "
            "Returns matching lines with file:line prefixes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                }
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Edit a file by replacing an exact unique string with new text. old_str "
            "must appear EXACTLY ONCE in the file — use enough surrounding context to "
            "make it unique. Use the RAW file text, WITHOUT the 'NNN\\t' line-number "
            "prefixes that read_file shows for display. Returns confirmation or an error."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path to edit"},
                "old_str": {
                    "type": "string",
                    "description": "Exact string to replace (must appear exactly once)",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string",
                },
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "run_tests",
        "description": (
            "Run pytest on the repository or a specific path. "
            "Returns the test output (capped). Use this to verify your changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional sub-path or test file to run (default: repo root)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "finish",
        "description": (
            "Signal that you have completed the task. "
            "Call this when you have made the necessary changes and verified them, "
            "or when you are confident you cannot solve the issue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of what was done or why stopping",
                }
            },
            "required": ["reason"],
        },
    },
]

# Fast lookup by name.
TOOL_SCHEMA_BY_NAME: dict[str, dict] = {t["name"]: t for t in TOOL_SCHEMAS}


def _cap(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate long output with a clear notice so the agent knows there is more."""
    if len(text) <= limit:
        return text
    kept = text[:limit]
    return kept + f"\n... [truncated: {len(text) - limit} more chars not shown]"


def execute_tool(name: str, inputs: dict, worktree: Path) -> str:
    """Dispatch a tool call and return its string result.

    All paths are resolved relative to ``worktree``; any attempt to escape via ``..``
    is blocked. Unknown tool names return an error string (the proxy may inject its own
    tools such as ``distil_expand`` — those are consumed transparently upstream and will
    never arrive here, but we handle them gracefully anyway).
    """
    if name == "list_dir":
        return _list_dir(inputs.get("path", "."), worktree)
    if name == "read_file":
        return _read_file(
            inputs.get("path", ""), worktree, inputs.get("start_line"), inputs.get("end_line")
        )
    if name == "search":
        return _search(inputs.get("pattern", ""), worktree)
    if name == "edit_file":
        return _edit_file(
            inputs.get("path", ""),
            inputs.get("old_str", ""),
            inputs.get("new_str", ""),
            worktree,
        )
    if name == "run_tests":
        return _run_tests(inputs.get("path"), worktree)
    if name == "finish":
        # finish is handled at the loop level; the executor just confirms.
        return f"Task finished: {inputs.get('reason', '(no reason given)')}"
    # Unknown tools (e.g. the proxy's distil_expand if it ever leaks through) get an
    # informative error rather than a crash.
    return f"(unknown tool '{name}' — no executor registered)"


# --------------------------------------------------------------------------- #
# Executors
# --------------------------------------------------------------------------- #


_LINENO_PREFIX = re.compile(r"^\s*\d+\t")


def _strip_line_numbers(text: str) -> str:
    """Remove read_file's ``"   NNN\\t"`` display prefixes if a model pasted them into an
    edit. Only strips when EVERY non-empty line carries the prefix (so we never corrupt
    legitimate tab-indented code)."""
    lines = text.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if nonempty and all(_LINENO_PREFIX.match(ln) for ln in nonempty):
        return "\n".join(_LINENO_PREFIX.sub("", ln) for ln in lines)
    return text


def _resolve(rel: str, worktree: Path) -> Path:
    """Resolve ``rel`` inside ``worktree``, blocking path traversal."""
    target = (worktree / rel).resolve()
    if not str(target).startswith(str(worktree.resolve())):
        raise ValueError(f"path '{rel}' escapes worktree")
    return target


def _list_dir(rel: str, worktree: Path) -> str:
    try:
        target = _resolve(rel or ".", worktree)
    except ValueError as e:
        return f"ERROR: {e}"
    if not target.exists():
        return f"ERROR: path not found: {rel}"
    if target.is_file():
        return f"ERROR: '{rel}' is a file, not a directory"
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines = []
    for e in entries:
        suffix = "/" if e.is_dir() else ""
        lines.append(f"{e.name}{suffix}")
    return _cap("\n".join(lines) or "(empty directory)")


def _coerce_line_window(
    start_line: object, end_line: object, total: int
) -> tuple[int | None, int | None]:
    """Robustly parse a line window from possibly-messy model arguments.

    Models sometimes pack a whole range into one field (``start_line="1, 50"`` or
    ``"1-50"``) or send stringified ints. We extract integers rather than calling
    ``int()`` directly, which would crash the whole tool call (and the instance) on
    inputs like ``"1, 50"``. Returns ``(start, end)`` or ``(None, None)`` if no integer
    is present at all.
    """

    def _ints(v: object) -> list[int]:
        if v is None:
            return []
        if isinstance(v, bool):  # avoid True/False sneaking through as 1/0
            return []
        if isinstance(v, int):
            return [v]
        return [int(x) for x in re.findall(r"\d+", str(v))]  # line nums are non-negative

    s_ints, e_ints = _ints(start_line), _ints(end_line)
    if len(s_ints) >= 2 and not e_ints:
        return s_ints[0], s_ints[1]  # range packed into start_line, e.g. "1, 50"
    nums = s_ints + e_ints
    if not nums:
        return None, None
    return nums[0], (nums[1] if len(nums) > 1 else total)


def _read_file(
    rel: str, worktree: Path, start_line: int | None = None, end_line: int | None = None
) -> str:
    if not rel:
        return "ERROR: path is required"
    try:
        target = _resolve(rel, worktree)
    except ValueError as e:
        return f"ERROR: {e}"
    if not target.exists():
        return f"ERROR: file not found: {rel}"
    if target.is_dir():
        return f"ERROR: '{rel}' is a directory, use list_dir"
    try:
        lines = target.read_text(errors="replace").splitlines()
    except OSError as e:
        return f"ERROR reading {rel}: {e}"

    total = len(lines)
    # Explicit window: return exactly those lines, numbered (how an agent reads around a
    # search hit to construct an exact edit).
    if start_line is not None or end_line is not None:
        s_raw, e_raw = _coerce_line_window(start_line, end_line, total)
        if s_raw is None:
            return f"ERROR: could not parse line window from start_line={start_line!r} end_line={end_line!r}"
        s = max(1, s_raw)
        e = min(total, e_raw)
        if s > total:
            return f"ERROR: start_line {s} > file length {total}"
        body = "\n".join(f"{i + 1:>6}\t{lines[i]}" for i in range(s - 1, e))
        return _cap(f"{rel} (lines {s}-{e} of {total}):\n{body}")
    # No window: numbered head + a note of total length so the agent knows to page in.
    numbered = "\n".join(f"{i + 1:>6}\t{ln}" for i, ln in enumerate(lines))
    out = _cap(f"{rel} ({total} lines; pass start_line/end_line to read a window):\n{numbered}")
    return out


def _search(pattern: str, worktree: Path) -> str:
    if not pattern:
        return "ERROR: pattern is required"
    # prefer ripgrep, fall back to grep
    import shutil

    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--no-heading", "-n", "--color=never", pattern, "."]
    else:
        cmd = ["grep", "-rn", "--color=never", pattern, "."]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(worktree),
            capture_output=True,
            text=True,
            errors="replace",  # grep can emit bytes from binary files; never crash on decode
            timeout=30,
        )
        output = result.stdout or result.stderr or "(no matches)"
    except subprocess.TimeoutExpired:
        output = "(search timed out after 30s)"
    except OSError as e:
        output = f"ERROR: {e}"
    return _cap(output)


def _edit_file(rel: str, old_str: str, new_str: str, worktree: Path) -> str:
    if not rel:
        return "ERROR: path is required"
    if not old_str:
        return "ERROR: old_str is required"
    try:
        target = _resolve(rel, worktree)
    except ValueError as e:
        return f"ERROR: {e}"
    if not target.exists():
        return f"ERROR: file not found: {rel}"
    if target.is_dir():
        return f"ERROR: '{rel}' is a directory"
    try:
        original = target.read_text(errors="replace")
    except OSError as e:
        return f"ERROR reading {rel}: {e}"
    count = original.count(old_str)
    if count == 0:
        # Defensive: if the model pasted read_file's "NNN\t" line-number prefixes into
        # old_str/new_str, strip them and retry once before giving up.
        stripped_old = _strip_line_numbers(old_str)
        if stripped_old != old_str and original.count(stripped_old) >= 1:
            old_str, new_str = stripped_old, _strip_line_numbers(new_str)
            count = original.count(old_str)
        else:
            return f"ERROR: old_str not found in {rel}"
    if count > 1:
        return (
            f"ERROR: old_str appears {count} times in {rel} — provide more surrounding "
            "context to make it unique"
        )
    updated = original.replace(old_str, new_str, 1)
    try:
        target.write_text(updated)
    except OSError as e:
        return f"ERROR writing {rel}: {e}"
    return f"OK: replaced 1 occurrence in {rel}"


def _run_tests(rel: str | None, worktree: Path) -> str:
    cmd = ["python", "-m", "pytest", "-x", "-q", "--tb=short", "--no-header"]
    if rel:
        try:
            target = _resolve(rel, worktree)
        except ValueError as e:
            return f"ERROR: {e}"
        cmd.append(str(target))
    try:
        result = subprocess.run(
            cmd,
            cwd=str(worktree),
            capture_output=True,
            text=True,
            errors="replace",  # test output can contain non-UTF8 bytes; never crash on decode
            timeout=120,
        )
        output = result.stdout + ("\n" + result.stderr if result.stderr else "")
    except subprocess.TimeoutExpired:
        output = "(pytest timed out after 120s)"
    except OSError as e:
        output = f"ERROR: {e}"
    return _cap(output)
