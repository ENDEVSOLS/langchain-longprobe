<div align="center">

# 🔬 langchain-longprobe

**Sub-second RAG retrieval regression testing for LangChain**

[![PyPI version](https://badge.fury.io/py/langchain-longprobe.svg)](https://badge.fury.io/py/langchain-longprobe)
[![Python Versions](https://img.shields.io/pypi/pyversions/langchain-longprobe.svg)](https://pypi.org/project/langchain-longprobe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LongProbe Core](https://img.shields.io/badge/Powered%20by-LongProbe-blue)](https://github.com/ENDEVSOLS/LongProbe)

*The first retrieval regression testing integration for LangChain.*
*Not another evaluation framework — a test runner that catches lost chunks in milliseconds.*

[Quick Start](#quick-start) • [API Reference](#api-reference) • [Pytest Integration](#pytest-integration) • [Examples](#examples)

</div>

---

## Why langchain-longprobe?

Every RAG developer faces the same problem: you upgrade LangChain, swap a vector store, or tweak a chunking strategy — and your retrieval **silently degrades**. Existing evaluation tools (Ragas, DeepEval) tell you the LLM answer got worse, but they can't tell you *which specific chunks were lost*.

**langchain-longprobe** bridges this gap:

| Feature | LangChain Evaluators / Ragas | langchain-longprobe |
| :--- | :--- | :--- |
| **Focus** | LLM response quality | Retrieval stability |
| **Speed** | 10s–60s (LLM-as-judge) | **Sub-second** (chunk match) |
| **Feedback** | Pass/Fail score | **Visual diff** of lost/gained chunks |
| **Workflow** | Batch analysis | `pytest` / CI integration |
| **Detects** | Bad answers | **Why** retrieval failed |

## Installation

```bash
pip install langchain-longprobe
```

This installs both `langchain-longprobe` and the core [`longprobe`](https://pypi.org/project/longprobe/) library.

## Quick Start

### 1. Define Golden Questions

Create a `goldens.yaml` with your expected retrieval results:

```yaml
name: "my-rag-golden-set"
version: "1.0"

questions:
  - id: "q1"
    question: "What is the refund policy?"
    match_mode: "id"
    required_chunks:
      - "chunk_refund_01"
      - "chunk_refund_02"
    top_k: 5

  - id: "q2"
    question: "What are the payment terms?"
    match_mode: "text"
    required_chunks:
      - "net 30 days from invoice"
    top_k: 5
```

### 2. Probe Your Retriever

```python
from langchain_longprobe import RetrievalProbe
from langchain_community.vectorstores import Chroma

# Your existing LangChain setup
vectorstore = Chroma(persist_directory="./db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create a probe
probe = RetrievalProbe.from_retriever(
    retriever=retriever,
    goldens_path="goldens.yaml",
)

# Run regression check (sub-second)
report = probe.run()
print(f"Recall: {report.overall_recall:.2%}")
print(f"Pass Rate: {report.pass_rate:.2%}")

# See exactly what's missing
for qid, chunks in probe.get_missing_chunks().items():
    print(f"  {qid}: lost {chunks}")
```

### 3. Track Regressions Over Time

```python
# Save a baseline after your first successful run
probe.run()
probe.save_baseline("v1.0")

# After code changes, compare against baseline
probe.run()
diff = probe.diff("v1.0")
print(f"Regressions: {len(diff['regressions'])}")
print(f"Improvements: {len(diff['improvements'])}")
```

## API Reference

### `RetrievalProbe` — Main Entry Point

The recommended way to use langchain-longprobe. Wraps any LangChain `BaseRetriever`.

```python
from langchain_longprobe import RetrievalProbe

probe = RetrievalProbe.from_retriever(
    retriever=your_retriever,
    goldens_path="goldens.yaml",
    recall_threshold=0.85,
)

report = probe.run()
probe.save_baseline("v1.0")
diff = probe.diff("v1.0")
missing = probe.get_missing_chunks()
```

### `ProbedRetriever` — Drop-in Retriever Wrapper

A LangChain `BaseRetriever` that wraps your retriever and adds regression testing.
Use as a **drop-in replacement** in your existing chains.

```python
from langchain_longprobe import ProbedRetriever

probed = ProbedRetriever(
    retriever=your_retriever,
    goldens_path="goldens.yaml",
    check_on_invoke=False,  # set True for automatic checks
)

# Use exactly like a normal retriever
docs = probed.invoke("What is the refund policy?")

# Manually trigger a regression check
report = probed.check()
probed.save_baseline("v1.0")
```

### `LongProbeCallbackHandler` — Passive Monitoring

Attach to any LangChain chain to passively record retrieval events.

```python
from langchain_longprobe import LongProbeCallbackHandler

handler = LongProbeCallbackHandler(
    goldens_path="goldens.yaml",
    recall_threshold=0.85,
    fail_on_regression=True,
)

# Attach to retriever calls
docs = retriever.invoke("query", config={"callbacks": [handler]})

# Inspect results
print(handler.retrieval_log)

# Run a full check
report = handler.run_probe(retriever)
```

### `RetrievalRegressionRunnable` — LCEL Integration

Use LongProbe as a composable step in LangChain Expression Language chains.

```python
from langchain_longprobe import RetrievalRegressionRunnable

runnable = RetrievalRegressionRunnable(
    retriever=your_retriever,
    goldens_path="goldens.yaml",
    fail_on_regression=True,
)

# Invoke with options
result = runnable.invoke({
    "top_k": 10,
    "save_baseline": "v2.0",
    "baseline_label": "v1.0",  # compare against this
})

print(result["overall_recall"])
print(result["missing_chunks"])
```

## Pytest Integration

### conftest.py

```python
import pytest
from langchain_longprobe import RetrievalProbe

@pytest.fixture
def probe(my_retriever):
    return RetrievalProbe.from_retriever(
        retriever=my_retriever,
        goldens_path="goldens.yaml",
        recall_threshold=0.85,
    )
```

### Writing Tests

```python
def test_retrieval_recall(probe):
    """Ensure retrieval recall stays above threshold."""
    report = probe.run()
    assert report.overall_recall >= 0.85, (
        f"Recall dropped to {report.overall_recall:.2f}. "
        f"Missing: {probe.get_missing_chunks()}"
    )

def test_no_regression(probe):
    """Ensure no chunks were lost vs. baseline."""
    probe.assert_no_regression("v1.0")
```

### Command Line

```bash
pytest --langchain-longprobe-goldens goldens.yaml --langchain-longprobe-threshold 0.85
```

## GitHub Actions

```yaml
name: RAG Regression Check

on: [push, pull_request]

jobs:
  rag-probe:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install langchain-longprobe
      - name: Run regression check
        run: pytest tests/test_rag_regression.py -v
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Examples

### Basic Regression Check

```python
from langchain_longprobe import RetrievalProbe
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever()

probe = RetrievalProbe.from_retriever(retriever, goldens_path="goldens.yaml")
report = probe.run()

if report.regression_detected:
    print("⚠️  Regression detected!")
    for qid, chunks in probe.get_missing_chunks().items():
        print(f"  Question {qid}: lost chunks {chunks}")
else:
    print("✅ All chunks present")
```

### CI/CD Pipeline Integration

```python
# tests/test_rag_regression.py
import pytest
from langchain_longprobe import RetrievalProbe

@pytest.fixture(scope="session")
def probe():
    from langchain_community.vectorstores import Chroma
    retriever = Chroma(persist_directory="./db").as_retriever()
    return RetrievalProbe.from_retriever(
        retriever=retriever,
        goldens_path="goldens.yaml",
        recall_threshold=0.85,
    )

def test_recall_above_threshold(probe):
    report = probe.run()
    assert report.overall_recall >= 0.85

def test_no_regressions_vs_baseline(probe):
    probe.assert_no_regression("production")

def test_critical_questions_pass(probe):
    report = probe.run()
    for result in report.results:
        if "critical" in result.question_id:
            assert result.passed, f"Critical question {result.question_id} failed"
```

## Part of the Long Suite

langchain-longprobe is the official LangChain integration for [LongProbe](https://github.com/ENDEVSOLS/LongProbe), part of the [EnDevSols Long Suite](https://endevsols.com/open-source) of RAG tools:

- **[LongParser](https://github.com/ENDEVSOLS/LongParser)** — Document ingestion and chunking
- **[LongTrainer](https://github.com/ENDEVSOLS/Long-Trainer)** — RAG chatbot framework
- **[LongTracer](https://github.com/ENDEVSOLS/LongTracer)** — Hallucination detection
- **[LongProbe](https://github.com/ENDEVSOLS/LongProbe)** — Retrieval regression testing (core library)
- **[langchain-longprobe](https://github.com/ENDEVSOLS/langchain-longprobe)** — LangChain integration ← You are here

## Contributing

We welcome contributions! Please see the [LongProbe Contributing Guide](https://github.com/ENDEVSOLS/LongProbe/blob/main/CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Developed by [EnDevSols](https://endevsols.com)** • [GitHub](https://github.com/ENDEVSOLS) • [LongProbe Core](https://github.com/ENDEVSOLS/LongProbe)

</div>