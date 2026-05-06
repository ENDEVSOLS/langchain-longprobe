"""
Core integration: RetrievalProbe — the main entry point for LangChain users.

Wraps any LangChain BaseRetriever into LongProbe's regression testing harness,
providing a native LangChain-first API while delegating all heavy lifting to
the ``longprobe`` core library.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.retrievers import BaseRetriever
from longprobe import LongProbe
from longprobe.adapters.langchain import LangChainRetrieverAdapter
from longprobe.core.scorer import ProbeReport

logger = logging.getLogger(__name__)


class RetrievalProbe:
    """LangChain-native wrapper around LongProbe for RAG regression testing.

    This is the recommended entry point for LangChain users. It accepts any
    ``BaseRetriever`` (or duck-typed retriever with ``.invoke()``) and runs
    LongProbe's sub-second regression checks against a golden question set.

    Usage::

        from langchain_longprobe import RetrievalProbe
        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma(persist_directory="./db")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        probe = RetrievalProbe.from_retriever(
            retriever=retriever,
            goldens_path="goldens.yaml",
        )

        # Run regression check
        report = probe.run()
        print(f"Recall: {report.overall_recall:.2%}")

        # Save baseline for future comparisons
        probe.save_baseline("v1.0")

        # After code changes, compare against baseline
        report2 = probe.run()
        diff = probe.diff("v1.0")
        print(f"Regressions: {len(diff['regressions'])}")

    Args:
        retriever: Any LangChain-compatible retriever.
        goldens_path: Path to the golden questions YAML file.
        config_path: Path to the LongProbe configuration YAML file.
        recall_threshold: Minimum recall for a question to pass.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        goldens_path: str = "goldens.yaml",
        config_path: str = "longprobe.yaml",
        recall_threshold: float | None = None,
    ) -> None:
        self._retriever = retriever
        self._adapter = LangChainRetrieverAdapter(retriever)
        self._probe = LongProbe(
            adapter=self._adapter,
            goldens_path=goldens_path,
            config_path=config_path,
            recall_threshold=recall_threshold,
        )
        self._last_report: ProbeReport | None = None

    @classmethod
    def from_retriever(
        cls,
        retriever: BaseRetriever,
        goldens_path: str = "goldens.yaml",
        config_path: str = "longprobe.yaml",
        recall_threshold: float | None = None,
    ) -> RetrievalProbe:
        """Create a RetrievalProbe from any LangChain retriever.

        This is the preferred factory method for LangChain users.

        Args:
            retriever: Any LangChain ``BaseRetriever`` or compatible object.
            goldens_path: Path to the golden questions YAML file.
            config_path: Path to the LongProbe config file.
            recall_threshold: Minimum recall for a question to pass.

        Returns:
            A configured ``RetrievalProbe`` instance ready to run checks.
        """
        return cls(
            retriever=retriever,
            goldens_path=goldens_path,
            config_path=config_path,
            recall_threshold=recall_threshold,
        )

    def run(self, top_k: int | None = None) -> ProbeReport:
        """Run the regression probe against the golden set.

        Args:
            top_k: Override the per-question top_k value.

        Returns:
            A ``ProbeReport`` with recall scores, missing chunks, and
            regression detection results.
        """
        report = self._probe.run(top_k_override=top_k)
        self._last_report = report
        logger.info(
            "LongProbe check complete: recall=%.4f pass_rate=%.4f regression=%s",
            report.overall_recall,
            report.pass_rate,
            report.regression_detected,
        )
        return report

    def save_baseline(self, label: str = "latest") -> None:
        """Save the last run as a baseline for future comparisons.

        Args:
            label: A label for this baseline (e.g. ``"v1.0"``, ``"pre-refactor"``).

        Raises:
            RuntimeError: If no report has been generated yet.
        """
        self._probe.save_baseline(label=label)

    def diff(self, baseline_label: str = "latest") -> dict[str, Any]:
        """Compare the last run against a saved baseline.

        Args:
            baseline_label: The label of the baseline to compare against.

        Returns:
            A dict with ``"regressions"``, ``"improvements"``, and
            ``"unchanged"`` keys.

        Raises:
            RuntimeError: If no report has been generated yet.
            ValueError: If the baseline label is not found.
        """
        return self._probe.diff(baseline_label=baseline_label)

    def get_missing_chunks(self) -> dict[str, list[str]]:
        """Return a mapping of question_id → missing chunks for the last run.

        Returns:
            Dict mapping question IDs to their lists of missing chunks.

        Raises:
            RuntimeError: If no report has been generated yet.
        """
        return self._probe.get_missing_chunks()

    @property
    def last_report(self) -> ProbeReport | None:
        """The most recent ``ProbeReport``, or ``None`` if no run has been executed."""
        return self._last_report

    @property
    def retriever(self) -> BaseRetriever:
        """The underlying LangChain retriever being probed."""
        return self._retriever

    def assert_no_regression(self, baseline_label: str = "latest") -> None:
        """Run the probe and assert no regressions against the given baseline.

        Convenience method for pytest integration::

            def test_no_regression(probe):
                probe.assert_no_regression("v1.0")

        Args:
            baseline_label: The baseline to compare against.

        Raises:
            AssertionError: If regressions are detected with a detailed message.
        """
        self.run()
        diff = self.diff(baseline_label=baseline_label)

        regressions = diff.get("regressions", [])
        if regressions:
            details = []
            for reg in regressions:
                if isinstance(reg, dict):
                    qid = reg.get("question_id", "?")
                    lost = reg.get("newly_lost_chunks", [])
                    b_recall = reg.get("baseline_recall", 0)
                    c_recall = reg.get("current_recall", 0)
                else:
                    qid = getattr(reg, "question_id", "?")
                    lost = getattr(reg, "newly_lost_chunks", [])
                    b_recall = getattr(reg, "baseline_recall", 0)
                    c_recall = getattr(reg, "current_recall", 0)

                details.append(
                    f"  {qid}: recall {b_recall:.4f} → {c_recall:.4f}, lost chunks: {lost}"
                )

            msg = f"LongProbe detected {len(regressions)} regression(s):\n" + "\n".join(details)
            raise AssertionError(msg)
