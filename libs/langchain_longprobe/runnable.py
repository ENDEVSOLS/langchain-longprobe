"""
RetrievalRegressionRunnable — a LangChain Runnable for regression testing.

Integrates LongProbe into LCEL (LangChain Expression Language) chains
as a composable step.

Usage::

    from langchain_longprobe import RetrievalRegressionRunnable

    regression_check = RetrievalRegressionRunnable(
        retriever=your_retriever,
        goldens_path="goldens.yaml",
    )

    # Use in an LCEL chain
    chain = retriever | regression_check | llm

    # Or invoke standalone
    report = regression_check.invoke({"check": True})
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig, RunnableSerializable
from longprobe import LongProbe
from longprobe.adapters.langchain import LangChainRetrieverAdapter

logger = logging.getLogger(__name__)


class RetrievalRegressionRunnable(RunnableSerializable[dict[str, Any], dict[str, Any]]):
    """A LangChain Runnable that executes LongProbe regression checks.

    Designed to be composed into LCEL chains or used standalone. When invoked,
    it runs the full LongProbe check against the configured golden set and
    returns the results as a dictionary.

    This is the most "LangChain-native" way to use LongProbe — as a composable
    step in your chain pipeline.

    Args:
        retriever: The LangChain retriever to test.
        goldens_path: Path to the golden questions YAML file.
        config_path: Path to the LongProbe configuration file.
        recall_threshold: Minimum recall for a question to pass.
        fail_on_regression: If True, raises an error on regression.
    """

    retriever: Any
    goldens_path: str = "goldens.yaml"
    config_path: str = "longprobe.yaml"
    recall_threshold: float = 0.85
    fail_on_regression: bool = False

    _probe: LongProbe | None = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def lc_namespace(self) -> list[str]:
        """LangChain namespace for serialization."""
        return ["langchain_longprobe", "runnable"]

    def _ensure_probe(self) -> LongProbe:
        """Lazily initialize the LongProbe instance."""
        if self._probe is None:
            adapter = LangChainRetrieverAdapter(self.retriever)
            self._probe = LongProbe(
                adapter=adapter,
                goldens_path=self.goldens_path,
                config_path=self.config_path,
                recall_threshold=self.recall_threshold,
            )
        return self._probe

    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a LongProbe regression check.

        Args:
            input: A dict. Accepted keys:
                - ``"top_k"`` (int, optional): Override the top_k value.
                - ``"baseline_label"`` (str, optional): Compare against
                  this baseline label.
                - ``"save_baseline"`` (str, optional): Save the report as
                  a baseline with this label.
            config: LangChain runnable config (callbacks, tags, etc.).

        Returns:
            A dict containing:
                - ``"overall_recall"`` (float)
                - ``"pass_rate"`` (float)
                - ``"regression_detected"`` (bool)
                - ``"num_questions"`` (int)
                - ``"missing_chunks"`` (dict)
                - ``"regressions"`` (list, if baseline comparison was done)

        Raises:
            RuntimeError: If ``fail_on_regression`` is True and a regression
                is detected.
        """
        probe = self._ensure_probe()
        top_k = input.get("top_k")

        report = probe.run(top_k_override=top_k)

        result: dict[str, Any] = {
            "overall_recall": report.overall_recall,
            "pass_rate": report.pass_rate,
            "regression_detected": report.regression_detected,
            "num_questions": len(report.results),
            "missing_chunks": probe.get_missing_chunks(),
            "timestamp": report.timestamp,
        }

        # Save baseline if requested
        save_label = input.get("save_baseline")
        if save_label:
            probe.save_baseline(label=save_label)
            result["baseline_saved"] = save_label

        # Compare against baseline if requested
        baseline_label = input.get("baseline_label")
        if baseline_label:
            try:
                diff = probe.diff(baseline_label=baseline_label)
                result["diff"] = diff
                regressions = diff.get("regressions", [])
                result["regressions"] = regressions

                if self.fail_on_regression and regressions:
                    raise RuntimeError(
                        f"LongProbe detected {len(regressions)} regression(s) "
                        f"against baseline '{baseline_label}'."
                    )
            except ValueError as e:
                result["diff_error"] = str(e)

        logger.info(
            "LongProbe Runnable: recall=%.4f pass_rate=%.4f regression=%s",
            report.overall_recall,
            report.pass_rate,
            report.regression_detected,
        )

        return result
