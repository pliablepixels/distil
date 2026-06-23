"""Request-path safety helpers shared by the proxy, async proxy, and gateway.

These servers forward to a single configured upstream and (for the gateway) are
meant to be exposed. The helpers here defend the forwarding path against hostile
but well-formed input: upstream-host injection (SSRF), malformed/oversized bodies,
and credential-leaking path tricks. Kept dependency-free and side-effect-free so
every server can apply them identically.
"""

from __future__ import annotations

# Default maximum request body. Agent contexts are large but bounded; anything
# past this is almost certainly abuse, and reading it would be a memory-DoS.
MAX_BODY_BYTES = 8 * 1024 * 1024  # 8 MiB


def safe_forward_path(target: str) -> str | None:
    """Validate a request target before concatenating it onto the upstream base URL.

    Returns the (unchanged) target if it is a safe origin-form path, else ``None``.
    Blocks the host-injection / credential-leak vectors that raw ``base + path``
    string concat enables: ``@`` userinfo (``base@evil.com``), protocol-relative
    ``//evil.com``, scheme injection, ``..`` traversal, and control characters.
    Only the path portion (before ``?``) is constrained; the query string is
    forwarded as-is since it cannot change the upstream host.
    """
    if not isinstance(target, str) or not target:
        return None
    if any(ord(c) < 0x20 for c in target) or "\\" in target:
        return None
    path = target.split("?", 1)[0].split("#", 1)[0]
    if not path.startswith("/") or path.startswith("//"):
        return None
    if "@" in path or "://" in path:
        return None
    if any(seg == ".." for seg in path.split("/")):
        return None
    return target


def strip_query(target: str) -> str:
    """The path without query/fragment — for matching compressible routes."""
    return target.split("?", 1)[0].split("#", 1)[0]


def parse_content_length(raw: object, *, max_bytes: int = MAX_BODY_BYTES) -> int | None:
    """Defensively parse a ``Content-Length`` header value.

    Returns a non-negative byte count, or ``None`` if the header is missing,
    non-numeric, negative, or exceeds ``max_bytes`` (caller should reject with
    400/413). Prevents ``int()`` crashes, negative-length read hangs, and
    unbounded-body memory exhaustion.
    """
    if raw is None:
        return 0
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    if n < 0 or n > max_bytes:
        return None
    return n
