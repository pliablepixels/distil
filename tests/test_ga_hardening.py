"""GA hardening regression tests — request-path safety and crash-resistance.

Locks in the fixes from the pre-GA security + correctness audit so they can't
silently regress.
"""

from __future__ import annotations

from distil.adapters.anthropic import compress_messages
from distil.compress.salience import protect
from distil.httpguard import MAX_BODY_BYTES, parse_content_length, safe_forward_path, strip_query
from distil.trajectory import Block, Kind, Stability


# --- httpguard: SSRF / upstream-host injection -------------------------------
def test_safe_forward_path_accepts_normal_paths():
    assert safe_forward_path("/v1/messages") == "/v1/messages"
    assert safe_forward_path("/v1/messages?beta=tools-2024") == "/v1/messages?beta=tools-2024"


def test_safe_forward_path_blocks_host_injection():
    for bad in [
        "@evil.com/",
        "//evil.com/x",
        "/../../etc",
        "http://evil.com",
        "/v1\r\nHost: evil",
        "/v1\x00",
        "\\evil",
        "",
        "/a/../../b",
    ]:
        assert safe_forward_path(bad) is None, bad


def test_parse_content_length_defensive():
    assert parse_content_length("123") == 123
    assert parse_content_length(None) == 0
    assert parse_content_length("abc") is None  # non-numeric: don't crash
    assert parse_content_length("-5") is None  # negative: no read-hang
    assert parse_content_length(str(MAX_BODY_BYTES + 1)) is None  # oversized: reject
    assert parse_content_length(str(MAX_BODY_BYTES)) == MAX_BODY_BYTES


def test_strip_query():
    assert strip_query("/v1/messages?x=1#f") == "/v1/messages"


# --- adapter: must never crash on malformed-but-valid JSON (C1) --------------
def test_compress_messages_survives_malformed_blocks():
    hostile = [
        [{"role": "user", "content": [{"type": "text"}]}],  # text block, no 'text'
        [{"role": "user", "content": [{"type": "tool_result", "content": [{"type": "text"}]}]}],
        ["just a string"],  # message is not a dict
        [None],
        [{"role": "user", "content": [{"type": "text", "text": 123}]}],  # non-string text
        [
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": [{"type": "text", "text": None}]}],
            }
        ],
    ]
    for msgs in hostile:
        out, _ = compress_messages(msgs)  # must not raise
        assert isinstance(out, list)


# --- gateway: tenant sanitization + dashboard escaping (H2/H3) ---------------
def test_gateway_tenant_sanitized_and_dashboard_escaped():
    from distil import gateway

    assert gateway.tenant_of({"x-distil-tenant": "acme-prod"}) == "acme-prod"
    # injection / overlong labels are rejected (fall through), never trusted as-is
    assert gateway.tenant_of({"x-distil-tenant": "<script>alert(1)</script>"}) == "default"
    assert gateway.tenant_of({"x-distil-tenant": "x" * 999, "x-api-key": "k"}).startswith("anon-")
    snap = {
        "tenants": [
            {
                "tenant": "<script>x</script>",
                "requests": 1,
                "tokens_saved": 2,
                "dollars_saved": 0.1,
                "pct_saved": 5.0,
            }
        ],
        "totals": {"requests": 1, "tokens_saved": 2, "dollars_saved": 0.1, "pct_saved": 5.0},
    }
    page = gateway._dashboard_html(snap)
    assert "<script>x</script>" not in page
    assert "&lt;script&gt;" in page


# --- salience: short block must keep the needle (BLOCKING bug regression) ----
def test_protect_short_block_falls_back_to_original_not_compressed():
    # patched form would exceed the tiny original -> must fall back to ORIGINAL
    # (which keeps the salient id), never to the compressed (id-less) block.
    block = Block("obs", Kind.TOOL_OUTPUT, "id: PAY-12345", Stability.VOLATILE, True)

    def zero(blocks, turn):
        return [b.copy_with("") for b in blocks]

    out = protect(zero)([block], 0)[0]
    assert "PAY-12345" in out.text  # the needle survived
