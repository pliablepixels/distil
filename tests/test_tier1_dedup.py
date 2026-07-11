"""Near-identical error/warn repeats fold instead of flooding the digest.

Second half of the power-user finding (see test_tier1_verdict.py): a run that
logs the same ERROR on purpose in a loop produced hundreds of kept lines that
differ only in an index — noise flooding the digest while adding no signal.
See distil/compress/tier1.py:_shape.
"""

from distil.compress.tier1 import _shape, digest


def _log(noise: list[str]) -> str:
    lines = ["RUN v1.6.0 /repo", "", "stdout | suite"]
    lines += [f"rendering component {i}" for i in range(20)]
    lines += noise
    lines += ["", "  Tests  1955 passed (1955)", " Duration  3.21s"]
    return "\n".join(lines)


def test_repeated_error_shape_folds():
    noise = [f"ERROR [Crypto] Decryption failed for payload {i}" for i in range(200)]
    d, changed = digest(_log(noise))
    assert changed
    kept = [ln for ln in d.splitlines() if "Decryption failed" in ln]
    assert len(kept) == 2, f"expected first 2 of shape, got {len(kept)}"
    assert "handle=" in d, "suppressed repeats must fold behind a handle"
    assert "1955 passed (1955)" in d, "verdict untouched by dedup"


def test_distinct_errors_all_kept():
    noise = [
        "ERROR [Crypto] Decryption failed for payload 7",
        "ERROR [Net] connection refused to gateway",
        "WARN  [Auth] token near expiry",
        "Traceback (most recent call last):",
    ]
    d, _ = digest(_log(noise))
    for ln in noise:
        assert ln in d, f"distinct error dropped: {ln!r}"


def test_summary_and_decision_never_deduped():
    noise = ["DECISION: retry with backoff"] * 5 + ["  3 passed, 1 failed"] * 5
    d, _ = digest(_log(noise))
    assert d.count("DECISION: retry with backoff") == 5
    assert d.count("3 passed, 1 failed") == 5


def test_shape_normalizes_numerics():
    a = _shape("ERROR [Crypto] Decryption failed for payload 17 at 0xDEADBEEF")
    b = _shape("error  [crypto]  decryption failed for payload 399 at 0x1f")
    assert a == b


if __name__ == "__main__":
    test_repeated_error_shape_folds()
    test_distinct_errors_all_kept()
    test_summary_and_decision_never_deduped()
    test_shape_normalizes_numerics()
    print("ok — noise dedup holds, answers exempt")
