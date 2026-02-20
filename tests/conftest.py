from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from niquests.exceptions import HTTPError


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--env-file", default=".env", help="Path to .env file for e2e tests"
    )


def pytest_configure(config: pytest.Config) -> None:
    load_dotenv(config.getoption("--env-file"))


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
