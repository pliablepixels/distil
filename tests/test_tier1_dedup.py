"""Near-identical error/warn repeats fold instead of flooding the digest,
with the dedup budget routed by the log's own outcome.

Second half of the power-user finding (see test_tier1_verdict.py): a run that
logs the same ERROR on purpose in a loop produced hundreds of kept lines that
differ only in an index — noise flooding the digest while adding no signal.
On a GREEN run those errors did not fail anything, so one sample per shape is
enough; on a red/unknown run they may BE the answer, so two are kept.
See distil/compress/tier1.py:_shape and _outcome.
"""

from distil.compress.tier1 import _outcome, _shape, digest


def _log(noise: list[str], verdict: str = "  Tests  1955 passed (1955)") -> str:
    lines = ["RUN v1.6.0 /repo", "", "stdout | suite"]
    lines += [f"rendering component {i}" for i in range(20)]
    lines += noise
    lines += ["", verdict, " Duration  3.21s"]
    return "\n".join(lines)


NOISE = [f"ERROR [Crypto] Decryption failed for payload {i}" for i in range(200)]


def _kept(d: str) -> list[str]:
    return [ln for ln in d.splitlines() if "Decryption failed" in ln]


def test_green_run_keeps_one_sample_per_shape():
    d, changed = digest(_log(NOISE))
    assert changed
    assert len(_kept(d)) == 1, "green run: errors are noise, one sample is the signal"
    assert "handle=" in d, "suppressed repeats must fold behind a handle"
    assert "1955 passed (1955)" in d, "verdict untouched by dedup"


def test_red_run_keeps_cautious_two():
    d, _ = digest(_log(NOISE, verdict="  Tests  3 failed | 1952 passed (1955)"))
    assert len(_kept(d)) == 2, "red run: errors may be the answer, keep the cautious 2"
    assert "3 failed" in d


def test_unknown_outcome_keeps_cautious_two():
    d, _ = digest(_log(NOISE, verdict="done."))
    assert len(_kept(d)) == 2, "no verdict -> cautious default"


def test_explicit_max_repeats_overrides_routing():
    d, _ = digest(_log(NOISE), max_repeats=5)
    assert len(_kept(d)) == 5


def test_outcome_detection():
    assert _outcome(["  Tests  1955 passed (1955)"]) == "green"
    assert _outcome(["test result: ok. 42 passed; 0 failed; 0 ignored"]) == "green"
    assert _outcome(["Tests: 2 failed, 1953 passed, 1955 total"]) == "red"
    assert _outcome(["--- FAIL: TestFoo (0.00s)", "FAIL"]) == "red"
    assert _outcome(["BUILD FAILED in 3s"]) == "red"
    assert _outcome(["process exited with exit code 1"]) == "red"
    assert _outcome(["just some prose", "no verdicts here"]) == "unknown"


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
    test_green_run_keeps_one_sample_per_shape()
    test_red_run_keeps_cautious_two()
    test_unknown_outcome_keeps_cautious_two()
    test_explicit_max_repeats_overrides_routing()
    test_outcome_detection()
    test_distinct_errors_all_kept()
    test_summary_and_decision_never_deduped()
    test_shape_normalizes_numerics()
    print("ok — outcome-routed dedup holds, answers exempt")
