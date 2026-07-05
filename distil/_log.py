"""Package logger — the debug escape hatch for fail-open paths.

Compression must never break a request, so every compression/learning/shadow
error is swallowed. Silent-by-default is correct for a transparent proxy, but
an operator needs a way to see what is being swallowed: set ``DISTIL_DEBUG=1``
(or ``DISTIL_LOG_LEVEL=DEBUG|INFO|...``) and swallowed exceptions are written
to stderr with tracebacks. Handlers attach to the ``distil`` logger only —
never the root logger — so embedding applications are unaffected.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("distil")

_level = os.environ.get("DISTIL_LOG_LEVEL", "").upper() or (
    "DEBUG" if os.environ.get("DISTIL_DEBUG") else ""
)
if _level and not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("distil %(levelname)s: %(message)s"))
    log.addHandler(_handler)
    log.setLevel(_level)
