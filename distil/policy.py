"""Auth-mode gating (Roadmap Phase 4) — a safety boundary, not an optimization.

The hard-won lesson from this space: aggressive, lossy, tool-injecting
compression is fine on a metered pay-as-you-go key, but applying it to a
subscription / OAuth session can alter a metered conversation in ways that
violate provider terms (injected retrieval tools the user never authorized,
rewritten history). So the *mode* gates which strategies are even allowed:

  * PAYG          — full toolbox, including lossy strategies and tool injection.
  * SUBSCRIPTION  — lossless-only. No lossy compression, no tool injection.

This is a tightening boundary: a project can never loosen it.
"""

from __future__ import annotations

from enum import Enum


class AuthMode(str, Enum):
    PAYG = "payg"
    SUBSCRIPTION = "subscription"  # also covers OAuth / first-party app sessions


class PolicyError(RuntimeError):
    """Raised when a strategy is not permitted under the active auth mode."""


# Lossless strategies are always permitted; lossy ones only on PAYG.
_LOSSLESS = {"none", "distil"}
_LOSSY = {"naive", "aggressive"}


def allowed_strategies(mode: AuthMode) -> set[str]:
    if mode is AuthMode.PAYG:
        return _LOSSLESS | _LOSSY
    return set(_LOSSLESS)


def may_compress_lossy(mode: AuthMode) -> bool:
    return mode is AuthMode.PAYG


def may_inject_tools(mode: AuthMode) -> bool:
    """Injecting a retrieval/expand tool into the request is only safe on PAYG."""
    return mode is AuthMode.PAYG


def guard(mode: AuthMode, strategy: str) -> None:
    """Raise PolicyError if `strategy` is not permitted under `mode`."""
    if strategy not in allowed_strategies(mode):
        raise PolicyError(
            f"strategy {strategy!r} is not permitted under auth mode {mode.value!r} "
            f"(lossless-only); allowed: {sorted(allowed_strategies(mode))}"
        )
