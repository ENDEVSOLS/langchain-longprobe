"""
langchain-longprobe — LangChain integration for LongProbe.

Sub-second RAG retrieval regression testing with chunk-level diffing.
The first regression testing integration for LangChain — not an evaluation
framework, but a test runner that catches lost chunks in milliseconds.

Developed by EnDevSols — https://endevsols.com

Quick start::

    from langchain_longprobe import RetrievalProbe

    probe = RetrievalProbe.from_retriever(
        retriever=your_langchain_retriever,
        goldens_path="goldens.yaml",
    )
    report = probe.run()
    assert not report.regression_detected
"""

from langchain_longprobe.callback import LongProbeCallbackHandler
from langchain_longprobe.probe import RetrievalProbe
from langchain_longprobe.retriever import ProbedRetriever
from langchain_longprobe.runnable import RetrievalRegressionRunnable

__version__ = "0.1.0"

__all__ = [
    "LongProbeCallbackHandler",
    "ProbedRetriever",
    "RetrievalProbe",
    "RetrievalRegressionRunnable",
]
