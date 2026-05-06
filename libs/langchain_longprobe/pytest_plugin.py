"""
Pytest plugin for langchain-longprobe.

Provides fixtures and options specifically for testing LangChain retrievers
with LongProbe regression checks.

Usage in conftest.py::

    from langchain_longprobe import RetrievalProbe

    @pytest.fixture
    def lc_probe(my_retriever):
        return RetrievalProbe.from_retriever(
            retriever=my_retriever,
            goldens_path="goldens.yaml",
        )

Usage in tests::

    def test_retrieval_regression(lc_probe):
        report = lc_probe.run()
        assert report.overall_recall >= 0.85
        assert not report.regression_detected

Command line::

    pytest --langchain-longprobe-goldens goldens.yaml
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.fixtures import FixtureRequest


# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------


def pytest_addoption(parser: Parser) -> None:
    """Register langchain-longprobe-specific pytest command line options."""
    group = parser.getgroup("langchain-longprobe", "LangChain LongProbe regression testing")
    group.addoption(
        "--langchain-longprobe-goldens",
        action="store",
        dest="lc_longprobe_goldens",
        default="goldens.yaml",
        help="Path to LongProbe golden questions YAML file (default: goldens.yaml)",
    )
    group.addoption(
        "--langchain-longprobe-threshold",
        action="store",
        dest="lc_longprobe_threshold",
        type=float,
        default=None,
        help="Minimum overall recall to pass. Fail tests if recall drops below this.",
    )


# ---------------------------------------------------------------------------
# Configuration hook
# ---------------------------------------------------------------------------


@pytest.hookimpl(trylast=True)
def pytest_configure(config: Config) -> None:
    """Store langchain-longprobe options on the config object."""
    config._lc_longprobe_goldens = config.getoption(  # type: ignore[attr-defined]
        "lc_longprobe_goldens", "goldens.yaml"
    )
    config._lc_longprobe_threshold = config.getoption(  # type: ignore[attr-defined]
        "lc_longprobe_threshold", None
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def langchain_longprobe_goldens_path(request: FixtureRequest) -> str:
    """Fixture that returns the path to the golden questions file."""
    return request.config._lc_longprobe_goldens  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def langchain_longprobe_threshold(request: FixtureRequest) -> float | None:
    """Fixture that returns the configured fail threshold (or None)."""
    return request.config._lc_longprobe_threshold  # type: ignore[attr-defined]
