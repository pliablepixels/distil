"""Tests for distil.retrieval — BM25-filtered partial retrieval."""

from __future__ import annotations

import pytest

from distil.retrieval import BM25Index, expand_handle, expand_handle_from_store

# ---------------------------------------------------------------------------
# A realistic multi-line document: a JSON-structured service log with a
# variety of topics so we can verify that rare/specific terms outrank
# common ones.
# ---------------------------------------------------------------------------
SAMPLE_LOG = """\
2024-01-15T08:00:01Z INFO  service=gateway request_id=abc123 method=GET path=/healthz
2024-01-15T08:00:02Z INFO  service=gateway request_id=abc124 method=POST path=/api/ingest
2024-01-15T08:00:03Z WARN  service=gateway request_id=abc124 latency_ms=1502 threshold_ms=1000
2024-01-15T08:00:04Z ERROR service=database connection_pool=exhausted wait_queue=47 active=100
2024-01-15T08:00:05Z INFO  service=gateway request_id=abc125 method=GET path=/api/v1/users
2024-01-15T08:00:06Z INFO  service=gateway request_id=abc126 method=DELETE path=/api/v1/users/99
2024-01-15T08:00:07Z ERROR service=database deadlock_detected=true table=orders txn_id=tx9912
2024-01-15T08:00:08Z INFO  service=cache hit_rate=0.91 evictions=3 max_memory_mb=512
2024-01-15T08:00:09Z WARN  service=gateway request_id=abc127 retry=3 upstream=database
2024-01-15T08:00:10Z INFO  service=billing invoice_id=INV-5050 amount_usd=299.00 status=paid
2024-01-15T08:00:11Z INFO  service=billing invoice_id=INV-5051 amount_usd=49.00 status=pending
2024-01-15T08:00:12Z ERROR service=auth token_expired=true user_id=u42 path=/api/v1/admin
2024-01-15T08:00:13Z INFO  service=gateway request_id=abc128 method=GET path=/api/v1/products
2024-01-15T08:00:14Z WARN  service=cache evictions=120 pressure=high max_memory_mb=512
2024-01-15T08:00:15Z INFO  service=gateway request_id=abc129 method=PUT path=/api/v1/users/7
"""


def _lines(text: str) -> list[str]:
    return text.splitlines()


# ---------------------------------------------------------------------------
# BM25Index — unit tests
# ---------------------------------------------------------------------------


class TestBM25Index:
    def test_specific_query_ranks_matching_line_first(self) -> None:
        """A query containing 'deadlock_detected' must surface the deadlock log line first.

        The tokenizer preserves underscores (``\\w+`` matches word chars incl. ``_``), so the
        field name ``deadlock_detected`` in the log line is a single token, and the same token
        in the query matches it exactly.
        """
        lines = _lines(SAMPLE_LOG)
        index = BM25Index(lines)
        results = index.search("deadlock_detected", k=3)
        assert results, "Expected at least one result"
        top_idx, top_score = results[0]
        assert top_score > 0
        assert "deadlock_detected" in lines[top_idx].lower()

    def test_rare_term_outranks_common_term(self) -> None:
        """'deadlock_detected' appears in one doc; 'service' appears on every line.

        The tokenizer preserves underscores so ``deadlock_detected`` is a single token with
        df=1 giving it a high IDF, while ``service`` has df=N giving it a near-zero IDF.
        A query for the rare token should score its line higher than the best score from a
        query for the ubiquitous token.
        """
        lines = _lines(SAMPLE_LOG)
        index = BM25Index(lines)

        deadlock_results = index.search("deadlock_detected", k=1)
        service_results = index.search("service", k=1)

        assert deadlock_results, "deadlock_detected search returned nothing"
        assert service_results, "service search returned nothing"

        # The top deadlock_detected score should be higher than the top service score —
        # rarity drives IDF which boosts the specific term.
        top_deadlock_score = deadlock_results[0][1]
        top_service_score = service_results[0][1]
        assert top_deadlock_score > top_service_score, (
            f"Expected rare-term score ({top_deadlock_score:.4f}) > "
            f"common-term score ({top_service_score:.4f})"
        )

    def test_idf_is_positive_for_present_terms(self) -> None:
        lines = _lines(SAMPLE_LOG)
        index = BM25Index(lines)
        assert index._idf("deadlock") > 0
        assert index._idf("nonexistent_xyz") > 0  # still defined (df=0)

    def test_idf_of_rare_term_exceeds_common_term(self) -> None:
        lines = _lines(SAMPLE_LOG)
        index = BM25Index(lines)
        # "deadlock" appears in 1 doc; "service" appears in all docs
        assert index._idf("deadlock") > index._idf("service")

    def test_search_returns_at_most_k_results(self) -> None:
        lines = _lines(SAMPLE_LOG)
        index = BM25Index(lines)
        results = index.search("error database connection", k=3)
        assert len(results) <= 3

    def test_search_results_are_sorted_descending(self) -> None:
        lines = _lines(SAMPLE_LOG)
        index = BM25Index(lines)
        results = index.search("error database", k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_match_returns_empty(self) -> None:
        lines = _lines(SAMPLE_LOG)
        index = BM25Index(lines)
        results = index.search("zzznonexistentterm999", k=5)
        assert results == []

    def test_single_doc_corpus(self) -> None:
        index = BM25Index(["hello world"])
        results = index.search("hello", k=1)
        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][1] > 0

    def test_returns_doc_index_not_position(self) -> None:
        """Indices in results must be valid positions into docs."""
        lines = _lines(SAMPLE_LOG)
        index = BM25Index(lines)
        results = index.search("invoice billing", k=5)
        for idx, _ in results:
            assert 0 <= idx < len(lines)


# ---------------------------------------------------------------------------
# expand_handle
# ---------------------------------------------------------------------------


class TestExpandHandle:
    def test_returns_only_relevant_lines(self) -> None:
        """Query 'deadlock_detected database' should pull the deadlock line and exclude
        health-check and billing lines.

        ``deadlock_detected`` is the actual token in the log (underscores are word chars),
        so the query must use the compound form to match it.
        """
        result = expand_handle(SAMPLE_LOG, "deadlock_detected database", k=3)
        result_lines = result.splitlines()
        assert len(result_lines) <= 3
        # The deadlock line must be present
        assert any("deadlock_detected" in ln for ln in result_lines), (
            "Expected the deadlock line to appear in results"
        )
        # Billing lines about invoices should not appear
        assert not any("invoice" in ln for ln in result_lines), (
            "Invoice lines are irrelevant and should not appear"
        )

    def test_original_order_preserved(self) -> None:
        """Lines in the output must appear in the same relative order they do
        in the original document, even when scored differently."""
        # Use two distinct topics that appear at different positions in the log
        result = expand_handle(SAMPLE_LOG, "error database connection deadlock_detected", k=4)
        result_lines = result.splitlines()

        original_lines = _lines(SAMPLE_LOG)
        # For every consecutive pair in the result, their indices in the
        # original must be strictly increasing.
        prev_idx = -1
        for ln in result_lines:
            idx = original_lines.index(ln)
            assert idx > prev_idx, (
                f"Line '{ln}' appears at index {idx} but previous was {prev_idx} — "
                "order not preserved"
            )
            prev_idx = idx

    def test_excludes_irrelevant_lines(self) -> None:
        """A very specific query should not dredge up completely unrelated content."""
        result = expand_handle(SAMPLE_LOG, "token expired auth admin", k=2)
        result_lines = result.splitlines()
        assert len(result_lines) <= 2
        # The auth/token line must be present
        assert any("token_expired" in ln for ln in result_lines)
        # Healthz lines are irrelevant
        assert not any("healthz" in ln for ln in result_lines)

    def test_k_limits_number_of_lines(self) -> None:
        result = expand_handle(SAMPLE_LOG, "service gateway", k=2)
        assert len(result.splitlines()) <= 2

    def test_empty_document_returns_empty(self) -> None:
        assert expand_handle("", "anything", k=5) == ""

    def test_single_line_document(self) -> None:
        result = expand_handle("only one line here", "one line", k=5)
        assert result == "only one line here"

    def test_billing_query_avoids_error_lines(self) -> None:
        """Query for billing/invoice should surface billing lines, not error lines."""
        result = expand_handle(SAMPLE_LOG, "invoice billing amount paid pending", k=3)
        result_lines = result.splitlines()
        # At least one billing line
        assert any("billing" in ln for ln in result_lines)
        # No database deadlock lines
        assert not any("deadlock_detected" in ln for ln in result_lines)


# ---------------------------------------------------------------------------
# expand_handle_from_store
# ---------------------------------------------------------------------------


class _SimpleStore:
    """Minimal duck-typed store with an expand() method."""

    def __init__(self, data: dict[str, str]) -> None:
        self._data = data

    def expand(self, handle: str) -> str:
        return self._data[handle]


class TestExpandHandleFromStore:
    def test_looks_up_handle_and_retrieves(self) -> None:
        store = _SimpleStore({"h1": SAMPLE_LOG})
        result = expand_handle_from_store("h1", store, "deadlock_detected database", k=3)
        assert result
        result_lines = result.splitlines()
        assert any("deadlock_detected" in ln for ln in result_lines)

    def test_delegates_to_expand_handle_consistently(self) -> None:
        """expand_handle_from_store should produce the same result as calling
        expand_handle directly with the same text."""
        store = _SimpleStore({"key": SAMPLE_LOG})
        via_store = expand_handle_from_store("key", store, "cache evictions memory", k=4)
        direct = expand_handle(SAMPLE_LOG, "cache evictions memory", k=4)
        assert via_store == direct

    def test_missing_handle_raises(self) -> None:
        store = _SimpleStore({})
        with pytest.raises(KeyError):
            expand_handle_from_store("missing", store, "query", k=5)
