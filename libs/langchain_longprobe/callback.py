"""
LangChain Callback Handler for LongProbe.

Integrates LongProbe regression checks into LangChain's callback system,
allowing automatic regression detection during retriever invocations.

Usage::

    from langchain_longprobe import LongProbeCallbackHandler

    handler = LongProbeCallbackHandler(
        goldens_path="goldens.yaml",
        recall_threshold=0.85,
    )

    # Attach to any retriever call
    docs = retriever.invoke("query", config={"callbacks": [handler]})

    # Check results after the chain finishes
    if handler.last_report:
        print(f"Recall: {handler.last_report.overall_recall:.2%}")
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class LongProbeCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that runs LongProbe checks on retrieval.

    Listens for retriever events and records the retrieved documents.
    After the chain finishes, you can inspect ``last_report`` for recall
    scores and regression data.

    This handler does **not** block the retrieval pipeline. It records
    results passively for post-run analysis.

    Args:
        goldens_path: Path to the golden questions YAML file.
        config_path: Path to the LongProbe configuration file.
        recall_threshold: Minimum recall for a question to pass.
        fail_on_regression: If True, logs a critical warning on regression.
    """

    name: str = "LongProbeCallbackHandler"

    def __init__(
        self,
        goldens_path: str = "goldens.yaml",
        config_path: str = "longprobe.yaml",
        recall_threshold: float = 0.85,
        fail_on_regression: bool = False,
    ) -> None:
        super().__init__()
        self.goldens_path = goldens_path
        self.config_path = config_path
        self.recall_threshold = recall_threshold
        self.fail_on_regression = fail_on_regression

        self._retrieval_log: list[dict[str, Any]] = []
        self._last_report: Any = None

    @property
    def last_report(self) -> Any:
        """The most recent ProbeReport, or None."""
        return self._last_report

    @property
    def retrieval_log(self) -> list[dict[str, Any]]:
        """All retrieval events recorded during the chain run."""
        return list(self._retrieval_log)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record that a retrieval has started."""
        logger.debug("LongProbe: retriever start for query=%r", query)

    def on_retriever_end(
        self,
        documents: list[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record retrieved documents for later analysis."""
        entry = {
            "run_id": str(run_id),
            "num_documents": len(documents),
            "document_ids": [
                doc.metadata.get("chunk_id", doc.metadata.get("source", f"doc_{i}"))
                for i, doc in enumerate(documents)
            ],
            "document_previews": [
                doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
                for doc in documents
            ],
        }
        self._retrieval_log.append(entry)

        logger.info(
            "LongProbe: retriever returned %d documents (run=%s)",
            len(documents),
            run_id,
        )

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log retriever errors."""
        logger.error("LongProbe: retriever error in run=%s: %s", run_id, error)

    def run_probe(self, retriever: Any) -> Any:
        """Manually trigger a full LongProbe regression check.

        This is useful after a chain run to get a complete picture of
        retrieval quality.

        Args:
            retriever: The LangChain retriever to probe.

        Returns:
            A ``ProbeReport`` from LongProbe.
        """
        from langchain_longprobe.probe import RetrievalProbe

        probe = RetrievalProbe(
            retriever=retriever,
            goldens_path=self.goldens_path,
            config_path=self.config_path,
            recall_threshold=self.recall_threshold,
        )
        report = probe.run()
        self._last_report = report

        if self.fail_on_regression and report.regression_detected:
            logger.critical(
                "LongProbe REGRESSION DETECTED: recall=%.4f (delta=%.4f). "
                "Lost chunks in %d question(s).",
                report.overall_recall,
                report.recall_delta or 0.0,
                sum(1 for r in report.results if r.missing_chunks),
            )

        return report
