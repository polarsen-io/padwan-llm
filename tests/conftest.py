from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from niquests.exceptions import HTTPError


@pytest.fixture
def make_resp():
    def _make(status_code: int, json_data: dict, headers: dict | None = None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json = AsyncMock(return_value=json_data)
        resp.headers = headers or {}
        if status_code >= 400:
            resp.raise_for_status.side_effect = HTTPError(response=resp)
        else:
            resp.raise_for_status.return_value = None
        return resp

    return _make
