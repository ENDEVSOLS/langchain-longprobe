"""
Unit tests for langchain-longprobe.

These tests use mock retrievers and golden sets to verify that the
langchain-longprobe wrapper correctly delegates to the core longprobe
library without requiring a live vector store.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import yaml
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_golden_file(tmpdir: str, questions: list[dict] | None = None) -> str:
    """Write a minimal goldens.yaml and return the path."""
    if questions is None:
        questions = [
            {
                "id": "q1",
                "question": "What is the refund policy?",
                "match_mode": "id",
                "required_chunks": ["chunk_refund_01", "chunk_refund_02"],
                "top_k": 5,
            },
            {
                "id": "q2",
                "question": "What are the payment terms?",
                "match_mode": "text",
                "required_chunks": ["net 30 days from invoice"],
                "top_k": 5,
            },
        ]
    data = {"name": "test-golden-set", "version": "1.0", "questions": questions}
    path = os.path.join(tmpdir, "goldens.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


class MockRetriever(BaseRetriever):
    """A mock LangChain retriever that returns fixed documents."""

    docs: list[Document] = []

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.docs


def _make_retriever(docs: list[Document] | None = None) -> MockRetriever:
    """Create a MockRetriever with the given documents."""
    if docs is None:
        docs = [
            Document(
                page_content="Full refund within 30 days of purchase.",
                metadata={"chunk_id": "chunk_refund_01", "source": "policy.pdf"},
            ),
            Document(
                page_content="Partial refund after 30 days, pro-rated.",
                metadata={"chunk_id": "chunk_refund_02", "source": "policy.pdf"},
            ),
            Document(
                page_content="Payment is due net 30 days from invoice date.",
                metadata={"chunk_id": "chunk_payment_01", "source": "terms.pdf"},
            ),
        ]
    return MockRetriever(docs=docs)


# ---------------------------------------------------------------------------
# Tests: RetrievalProbe
# ---------------------------------------------------------------------------


class TestRetrievalProbe:
    """Tests for the main RetrievalProbe class."""

    def test_from_retriever_factory(self, tmp_path: Any) -> None:
        """Test that from_retriever() creates a valid probe."""
        from langchain_longprobe import RetrievalProbe

        goldens_path = _create_golden_file(str(tmp_path))
        retriever = _make_retriever()

        probe = RetrievalProbe.from_retriever(
            retriever=retriever,
            goldens_path=goldens_path,
            recall_threshold=0.5,
        )

        assert probe is not None
        assert probe.retriever is retriever
        assert probe.last_report is None

    def test_run_returns_report(self, tmp_path: Any) -> None:
        """Test that run() returns a ProbeReport with valid data."""
        from langchain_longprobe import RetrievalProbe

        goldens_path = _create_golden_file(str(tmp_path))
        retriever = _make_retriever()

        probe = RetrievalProbe.from_retriever(
            retriever=retriever,
            goldens_path=goldens_path,
            recall_threshold=0.5,
        )

        report = probe.run()

        assert report is not None
        assert 0.0 <= report.overall_recall <= 1.0
        assert 0.0 <= report.pass_rate <= 1.0
        assert len(report.results) == 2  # two golden questions
        assert probe.last_report is report

    def test_perfect_recall(self, tmp_path: Any) -> None:
        """Test that perfect retrieval yields recall = 1.0."""
        from langchain_longprobe import RetrievalProbe

        goldens_path = _create_golden_file(str(tmp_path))
        retriever = _make_retriever()

        probe = RetrievalProbe.from_retriever(
            retriever=retriever,
            goldens_path=goldens_path,
            recall_threshold=0.5,
        )

        report = probe.run()
        # q1 requires chunk_refund_01 and chunk_refund_02 (both present)
        # q2 requires "net 30 days from invoice" (text match in chunk_payment_01)
        assert report.overall_recall == 1.0

    def test_missing_chunks_detected(self, tmp_path: Any) -> None:
        """Test that missing chunks are correctly identified."""
        from langchain_longprobe import RetrievalProbe

        goldens_path = _create_golden_file(str(tmp_path))
        # Only return one of the two required chunks for q1
        docs = [
            Document(
                page_content="Full refund within 30 days.",
                metadata={"chunk_id": "chunk_refund_01"},
            ),
            Document(
                page_content="Payment is due net 30 days from invoice date.",
                metadata={"chunk_id": "chunk_payment_01"},
            ),
        ]
        retriever = _make_retriever(docs)

        probe = RetrievalProbe.from_retriever(
            retriever=retriever,
            goldens_path=goldens_path,
            recall_threshold=0.9,
        )

        report = probe.run()
        missing = probe.get_missing_chunks()

        # q1 should be missing chunk_refund_02
        assert "q1" in missing
        assert "chunk_refund_02" in missing["q1"]

    def test_get_missing_chunks_raises_without_run(self, tmp_path: Any) -> None:
        """Test that get_missing_chunks raises if run() hasn't been called."""
        from langchain_longprobe import RetrievalProbe

        goldens_path = _create_golden_file(str(tmp_path))
        retriever = _make_retriever()

        probe = RetrievalProbe.from_retriever(
            retriever=retriever,
            goldens_path=goldens_path,
        )

        with pytest.raises(RuntimeError, match="No report available"):
            probe.get_missing_chunks()

    def test_baseline_save_and_diff(self, tmp_path: Any) -> None:
        """Test baseline save and diff workflow."""
        from langchain_longprobe import RetrievalProbe

        goldens_path = _create_golden_file(str(tmp_path))
        retriever = _make_retriever()

        # Create a config with baseline in tmp dir
        config_path = os.path.join(str(tmp_path), "longprobe.yaml")
        config_data = {
            "scoring": {"recall_threshold": 0.5, "fail_on_regression": False},
            "baseline": {
                "db_path": os.path.join(str(tmp_path), ".longprobe", "baselines.db"),
                "auto_compare": False,
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        probe = RetrievalProbe.from_retriever(
            retriever=retriever,
            goldens_path=goldens_path,
            config_path=config_path,
            recall_threshold=0.5,
        )

        # Run and save baseline
        report1 = probe.run()
        probe.save_baseline("v1.0")

        # Run again and diff
        report2 = probe.run()
        diff = probe.diff("v1.0")

        assert isinstance(diff, dict)
        assert "regressions" in diff or "improvements" in diff or "unchanged" in diff


# ---------------------------------------------------------------------------
# Tests: LongProbeCallbackHandler
# ---------------------------------------------------------------------------


class TestLongProbeCallbackHandler:
    """Tests for the LangChain callback handler."""

    def test_callback_records_retrieval(self) -> None:
        """Test that on_retriever_end records documents."""
        from langchain_longprobe import LongProbeCallbackHandler

        handler = LongProbeCallbackHandler()

        docs = [
            Document(page_content="Test content", metadata={"chunk_id": "c1"}),
            Document(page_content="More content", metadata={"source": "test.pdf"}),
        ]

        handler.on_retriever_end(
            documents=docs,
            run_id=uuid4(),
        )

        assert len(handler.retrieval_log) == 1
        entry = handler.retrieval_log[0]
        assert entry["num_documents"] == 2
        assert "c1" in entry["document_ids"]

    def test_callback_handles_error(self) -> None:
        """Test that on_retriever_error doesn't crash."""
        from langchain_longprobe import LongProbeCallbackHandler

        handler = LongProbeCallbackHandler()

        # Should not raise
        handler.on_retriever_error(
            error=RuntimeError("Connection timeout"),
            run_id=uuid4(),
        )

    def test_callback_initial_state(self) -> None:
        """Test that callback starts with empty state."""
        from langchain_longprobe import LongProbeCallbackHandler

        handler = LongProbeCallbackHandler(
            goldens_path="custom.yaml",
            recall_threshold=0.9,
        )

        assert handler.last_report is None
        assert handler.retrieval_log == []
        assert handler.goldens_path == "custom.yaml"
        assert handler.recall_threshold == 0.9


# ---------------------------------------------------------------------------
# Tests: ProbedRetriever
# ---------------------------------------------------------------------------


class TestProbedRetriever:
    """Tests for the ProbedRetriever wrapper."""

    def test_probed_retriever_forwards_calls(self, tmp_path: Any) -> None:
        """Test that ProbedRetriever correctly forwards to the wrapped retriever."""
        from langchain_longprobe import ProbedRetriever

        inner_docs = [
            Document(page_content="Result A", metadata={"chunk_id": "a"}),
            Document(page_content="Result B", metadata={"chunk_id": "b"}),
        ]
        inner = _make_retriever(inner_docs)

        probed = ProbedRetriever(
            retriever=inner,
            goldens_path=_create_golden_file(str(tmp_path)),
            check_on_invoke=False,
        )

        docs = probed.invoke("test query")

        assert len(docs) == 2
        assert docs[0].page_content == "Result A"
        assert docs[1].page_content == "Result B"

    def test_probed_retriever_check(self, tmp_path: Any) -> None:
        """Test that check() runs a full regression check."""
        from langchain_longprobe import ProbedRetriever

        goldens_path = _create_golden_file(str(tmp_path))
        inner = _make_retriever()

        probed = ProbedRetriever(
            retriever=inner,
            goldens_path=goldens_path,
        )

        report = probed.check()

        assert report is not None
        assert 0.0 <= report.overall_recall <= 1.0
        assert probed.last_report is report

    def test_probed_retriever_invocation_count(self, tmp_path: Any) -> None:
        """Test that invocation_count tracks calls."""
        from langchain_longprobe import ProbedRetriever

        goldens_path = _create_golden_file(str(tmp_path))
        inner = _make_retriever()

        probed = ProbedRetriever(
            retriever=inner,
            goldens_path=goldens_path,
        )

        assert probed.invocation_count == 0
        probed.invoke("query 1")
        probed.invoke("query 2")
        assert probed.invocation_count == 2


# ---------------------------------------------------------------------------
# Tests: RetrievalRegressionRunnable
# ---------------------------------------------------------------------------


class TestRetrievalRegressionRunnable:
    """Tests for the LCEL Runnable."""

    def test_runnable_invoke(self, tmp_path: Any) -> None:
        """Test that the runnable returns a valid result dict."""
        from langchain_longprobe import RetrievalRegressionRunnable

        goldens_path = _create_golden_file(str(tmp_path))
        retriever = _make_retriever()

        runnable = RetrievalRegressionRunnable(
            retriever=retriever,
            goldens_path=goldens_path,
            recall_threshold=0.5,
        )

        result = runnable.invoke({"check": True})

        assert isinstance(result, dict)
        assert "overall_recall" in result
        assert "pass_rate" in result
        assert "regression_detected" in result
        assert "num_questions" in result
        assert "missing_chunks" in result
        assert "timestamp" in result

    def test_runnable_with_top_k(self, tmp_path: Any) -> None:
        """Test that top_k override is respected."""
        from langchain_longprobe import RetrievalRegressionRunnable

        goldens_path = _create_golden_file(str(tmp_path))
        retriever = _make_retriever()

        runnable = RetrievalRegressionRunnable(
            retriever=retriever,
            goldens_path=goldens_path,
        )

        result = runnable.invoke({"top_k": 3})
        assert isinstance(result, dict)
        assert "overall_recall" in result

    def test_runnable_save_baseline(self, tmp_path: Any) -> None:
        """Test that the runnable can save a baseline."""
        from langchain_longprobe import RetrievalRegressionRunnable

        goldens_path = _create_golden_file(str(tmp_path))
        retriever = _make_retriever()

        config_path = os.path.join(str(tmp_path), "longprobe.yaml")
        config_data = {
            "scoring": {"recall_threshold": 0.5},
            "baseline": {
                "db_path": os.path.join(str(tmp_path), ".longprobe", "baselines.db"),
                "auto_compare": False,
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        runnable = RetrievalRegressionRunnable(
            retriever=retriever,
            goldens_path=goldens_path,
            config_path=config_path,
        )

        result = runnable.invoke({"save_baseline": "v1.0"})
        assert result.get("baseline_saved") == "v1.0"


# ---------------------------------------------------------------------------
# Tests: Import & version
# ---------------------------------------------------------------------------


class TestPackageMeta:
    """Tests for package-level metadata."""

    def test_version(self) -> None:
        """Test that version is accessible."""
        import langchain_longprobe

        assert langchain_longprobe.__version__ == "0.1.0"

    def test_all_exports(self) -> None:
        """Test that all public exports are importable."""
        from langchain_longprobe import (
            LongProbeCallbackHandler,
            ProbedRetriever,
            RetrievalProbe,
            RetrievalRegressionRunnable,
        )

        assert LongProbeCallbackHandler is not None
        assert ProbedRetriever is not None
        assert RetrievalProbe is not None
        assert RetrievalRegressionRunnable is not None
