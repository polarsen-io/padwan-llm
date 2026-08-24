import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from dotenv import load_dotenv
from niquests.exceptions import HTTPError

from tests.otel_coverage import OTEL_COVERAGE_KEY, report

# xai-sdk is not installed on 3.15 (grpcio lacks 3.15 support)
collect_ignore = ["test_grok.py"] if sys.version_info >= (3, 15) else []


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--env-file", default=".env", help="Path to .env file for e2e tests"
    )
    parser.addoption(
        "--otel",
        action="store_true",
        help="Instrument the e2e suite and check GenAI semconv coverage",
    )
    parser.addoption(
        "--langfuse",
        action="store_true",
        help="Instrument through the Langfuse adapter (implies --otel)",
    )


def pytest_configure(config: pytest.Config) -> None:
    load_dotenv(config.getoption("--env-file"))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Report --otel span coverage and fail the run on attributes the docs never name."""
    coverage = session.config.stash.get(OTEL_COVERAGE_KEY, None)
    if coverage is None:
        return
    if reporter := session.config.pluginmanager.get_plugin("terminalreporter"):
        reporter.write_sep("=", "otel semconv coverage")
        for line in report(coverage):
            reporter.write_line(line)
    if coverage.undocumented:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(autouse=True)
def _clear_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the unified-gateway env vars out of tests that don't opt in."""
    monkeypatch.delenv("PADWAN_BASE_URL", raising=False)
    monkeypatch.delenv("PADWAN_API_KEY", raising=False)


@pytest.fixture
def make_resp():
    def _make(status_code: int, json_data: dict, headers: dict | None = None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data
        resp.headers = headers or {}
        if status_code >= 400:
            resp.raise_for_status.side_effect = HTTPError(response=resp)
        else:
            resp.raise_for_status.return_value = resp
        return resp

    return _make


@pytest.fixture
def make_sse_event():
    def _make(data: str, event_id: str = "", retry: int | None = None):
        ev = MagicMock()
        ev.id = event_id
        ev.retry = retry
        ev.data = data
        if data:
            ev.json = MagicMock(return_value=json.loads(data))
        else:
            ev.json = MagicMock(side_effect=ValueError("empty SSE data"))
        return ev

    return _make


@pytest.fixture
def make_sse_resp():
    def _make(events: list, status: int = 200):
        resp = AsyncMock()
        resp.status_code = status
        resp.headers = {"content-type": "text/event-stream"}
        resp.raise_for_status = MagicMock(return_value=resp)
        ext = MagicMock()
        ext.closed = False
        payloads = list(events) + [None]
        idx = {"i": 0}

        async def _next(**_kw):
            if idx["i"] < len(payloads):
                val = payloads[idx["i"]]
                idx["i"] += 1
                if val is None:
                    ext.closed = True
                return val
            ext.closed = True
            return None

        ext.next_payload = _next

        async def _close():
            ext.closed = True

        ext.close = _close
        resp.extension = ext
        raw = MagicMock()
        raw.close = AsyncMock()
        raw.release_conn = MagicMock()
        resp.raw = raw
        return resp

    return _make
