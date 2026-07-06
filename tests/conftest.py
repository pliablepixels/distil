"""Test env hygiene: no test may touch the developer's real ~/.distil.

Developers run this suite from terminals that are themselves under
`distil wrap` (dogfooding), so the inherited env carries a LIVE
DISTIL_SESSION and the default DISTIL_HOME. Without isolation, wrap/proxy
tests would write ledger rows and session markers into the real store —
monkeypatch mutates os.environ, so subprocess-spawning tests inherit the
sandbox too.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _distil_home_sandbox(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path_factory.mktemp("distil-home")))
    monkeypatch.delenv("DISTIL_SESSION", raising=False)
