"""The result/verdict line of command output must survive digestion.

Regression for a content-agnostic miss a power-user comparison surfaced
(moonlit-piroshki-fbaa04.netlify.app, tested on ZoneMinder/zmNinjaNg): a
*passing* test run's summary ("1955 passed", which carries no error word) was
folded into a retrieval handle while the ERROR/WARN stdout the run emitted on
purpose was kept — distil "compressed away the answer and kept the noise".
See distil/compress/tier1.py:_SUMMARY_RE.
"""

from distil.compress.tier1 import digest, _must_keep


def _log(summary: str) -> str:
    """A passing-run shape: head + neutral filler + on-purpose noise + verdict near tail."""
    lines = ["RUN v1.6.0 /repo", "", "stdout | crypto.test.ts"]
    lines += [
        f"rendering sample component {i} to virtual dom" for i in range(30)
    ]  # neutral -> folds
    lines += [f"ERROR [Crypto] Decryption failed attempt {i}" for i in range(20)]  # kept (noise)
    lines += [f"WARN  [Auth] token near expiry {i}" for i in range(20)]  # kept (noise)
    lines += ["", summary, " Duration  3.21s"]  # verdict NOT in tail
    return "\n".join(lines)


def test_passing_verdict_survives_digest():
    for summary in [
        "  Tests  1955 passed (1955)",  # vitest, all green
        "Tests:       1955 passed, 1955 total",  # jest
        "===== 1955 passed in 12.34s =====",  # pytest
        "test result: ok. 42 passed; 0 failed; 0 ignored",  # cargo
        "  1955 passing",  # mocha
    ]:
        d, changed = digest(_log(summary))
        assert changed, "log should compress"
        assert "handle=" in d, "neutral middle must still fold — real compression preserved"
        assert summary.strip() in d, f"verdict folded away:\n{summary!r}\nGOT:\n{d}"


def test_go_and_build_verdicts_kept():
    for line in [
        "ok  \tgithub.com/x/y\t0.02s",
        "PASS",
        "--- FAIL: TestFoo (0.00s)",
        "BUILD SUCCESSFUL in 3s",
        "process exited with exit code 1",
    ]:
        assert _must_keep(line), f"result line dropped: {line!r}"


def test_no_false_keep_on_prose():
    # ordinary code/log lines must NOT trip the result net, or compression regresses
    for line in [
        "const passed = true;",
        "// this test is important",
        "logger.info('user logged in ok')",
        "return ok(result)",
        "the request passed through the gateway",
    ]:
        assert not _must_keep(line), f"false keep hurts compression: {line!r}"


if __name__ == "__main__":
    test_passing_verdict_survives_digest()
    test_go_and_build_verdicts_kept()
    test_no_false_keep_on_prose()
    print("ok — verdict-preservation holds across vitest/jest/pytest/cargo/mocha/go")
