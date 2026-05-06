"""
ProbedRetriever — a retriever wrapper that automatically runs LongProbe checks.

Wraps any LangChain BaseRetriever to intercept retrieval calls and optionally
run regression checks in the background, making LongProbe transparent to the
user's existing chain.

Usage::

    from langchain_longprobe import ProbedRetriever

    probed = ProbedRetriever(
        retriever=your_retriever,
        goldens_path="goldens.yaml",
        check_on_invoke=False,  # only check manually
    )

    # Use as a drop-in replacement
    docs = probed.invoke("What is the termination clause?")

    # Manually trigger a regression check
    report = probed.check()
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from longprobe import LongProbe
from longprobe.adapters.langchain import LangChainRetrieverAdapter
from longprobe.core.scorer import ProbeReport

logger = logging.getLogger(__name__)


class ProbedRetriever(BaseRetriever):
    """A LangChain retriever that wraps another retriever and adds LongProbe
    regression testing capabilities.

    This is a **drop-in replacement** for any LangChain retriever. It forwards
    all retrieval calls to the underlying retriever while optionally running
    LongProbe regression checks.

    Use this when you want LongProbe integrated directly into your retrieval
    chain without any additional test files.

    Args:
        retriever: The underlying LangChain retriever to wrap.
        goldens_path: Path to the golden questions YAML file.
        config_path: Path to the LongProbe configuration file.
        recall_threshold: Minimum recall for a question to pass.
        check_on_invoke: Whether to run checks on every invoke (default: False).
    """

    retriever: BaseRetriever
    goldens_path: str = "goldens.yaml"
    config_path: str = "longprobe.yaml"
    recall_threshold: float = 0.85
    check_on_invoke: bool = False

    # Private attributes (not part of the Pydantic model)
    _probe: LongProbe | None = None
    _last_report: ProbeReport | None = None
    _invocation_count: int = 0

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Forward retrieval to the wrapped retriever.

        If ``check_on_invoke`` is True, a full LongProbe regression check
        runs after the retrieval (note: this adds latency).
        """
        docs = self.retriever.invoke(query)
        self._invocation_count += 1

        if self.check_on_invoke:
            try:
                self.check()
            except Exception as e:
                logger.warning("LongProbe check failed: %s", e)

        return docs

    def check(self, top_k: int | None = None) -> ProbeReport:
        """Run a full LongProbe regression check against the golden set.

        Args:
            top_k: Override the per-question top_k value.

        Returns:
            A ``ProbeReport`` with recall scores and regression data.
        """
        if self._probe is None:
            adapter = LangChainRetrieverAdapter(self.retriever)
            self._probe = LongProbe(
                adapter=adapter,
                goldens_path=self.goldens_path,
                config_path=self.config_path,
                recall_threshold=self.recall_threshold,
            )

        report = self._probe.run(top_k_override=top_k)
        self._last_report = report
        return report

    def save_baseline(self, label: str = "latest") -> None:
        """Save the last check result as a baseline.

        Args:
            label: A human-readable label for this baseline.
        """
        if self._probe is None:
            raise RuntimeError("Run check() first before saving a baseline.")
        self._probe.save_baseline(label=label)

    def diff(self, baseline_label: str = "latest") -> dict[str, Any]:
        """Compare the last check against a saved baseline.

        Args:
            baseline_label: The baseline label to compare against.

        Returns:
            A dict with regressions, improvements, and unchanged data.
        """
        if self._probe is None:
            raise RuntimeError("Run check() first before comparing.")
        return self._probe.diff(baseline_label=baseline_label)

    @property
    def last_report(self) -> ProbeReport | None:
        """The most recent ProbeReport from a check() call."""
        return self._last_report

    @property
    def invocation_count(self) -> int:
        """Number of times this retriever has been invoked."""
        return self._invocation_count
