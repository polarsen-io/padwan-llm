from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import MappingProxyType
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import niquests
import pytest
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError
from urllib3.util.retry import Retry

from padwan_llm import TYPESAFE_MODELS, TypeSafeClient
from padwan_llm.errors import LLMError, TooManyRequestsError
from padwan_llm.typesafe import NoulQuestion, ScoreAnswer
from padwan_llm.typesafe.client import _check_resp

RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "billing": {"type": "noul", "noul": 0.98},
        "tone": {
            "type": "choice",
            "choice": "calm",
            "confidence": 0.8,
            "probabilities": {"calm": 0.9, "angry": 0.1},
        },
        "urgency": {
            "type": "score",
            "score": 1.7,
            "confidence": 0.7,
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "probabilities": {"0": 0.0, "1": 0.3, "2": 0.7},
        },
    },
    "usage": {"input_tokens": 120, "output_tokens": 12},
}

QUESTIONS = {
    "billing": {"type": "noul", "instructions": "Is this about billing?"},
    "tone": {
        "type": "choice",
        "instructions": "What is the tone?",
        "criteria": {"calm": None, "angry": None},
    },
    "urgency": {
        "type": "score",
        "instructions": {"question": "How urgent is this?"},
        "criteria": ["low", "medium", "high"],
    },
}


@pytest.mark.parametrize(
    "model, expected",
    [
        pytest.param(None, "jev-latest", id="default"),
        pytest.param("jev-1.13.0", "jev-1.13.0", id="versioned-override"),
    ],
)
async def test_system_one(model, expected, make_resp):
    client = TypeSafeClient(api_key="secret")
    session = AsyncMock()
    session.post.return_value = make_resp(200, RESPONSE)
    client._session = session

    result = await client.system_one(
        {"ticket": "charged twice", "attempt": 2}, QUESTIONS, model=model
    )

    assert result == RESPONSE
    urgency = result["answers"]["urgency"]
    assert isinstance(urgency, dict) and urgency["type"] == "score"
    score = cast("ScoreAnswer", urgency)
    assert score["probabilities"] == {
        "0": 0.0,
        "1": 0.3,
        "2": 0.7,
    }
    session.post.assert_awaited_once_with(
        "systemone",
        json={
            "state": {"ticket": "charged twice", "attempt": 2},
            "model": expected,
            "questions": QUESTIONS,
        },
    )


async def test_system_one_normalizes_abstract_json_inputs(make_resp) -> None:
    client = TypeSafeClient(api_key="secret")
    session = AsyncMock()
    session.post.return_value = make_resp(200, RESPONSE)
    client._session = session
    state = MappingProxyType({"items": (MappingProxyType({"values": range(3)}),)})
    questions = cast(
        "dict[str, NoulQuestion]",
        {
            "billing": {
                "type": "noul",
                "instructions": MappingProxyType({"parts": ("Is", "billing")}),
            }
        },
    )

    await client.system_one(state, questions)

    body = session.post.await_args.kwargs["json"]
    assert body["state"] == {"items": [{"values": [0, 1, 2]}]}
    assert body["questions"]["billing"]["instructions"] == {"parts": ["Is", "billing"]}
    niquests.Request(
        "POST", "https://api.typesafe.ai/v1/systemone", json=body
    ).prepare()


@pytest.mark.parametrize(
    "state, questions, ctx",
    [
        pytest.param("ticket", QUESTIONS, nullcontext(), id="valid"),
        pytest.param(
            None, QUESTIONS, pytest.raises(LLMError, match="State"), id="null-state"
        ),
        pytest.param(
            "ticket",
            {},
            pytest.raises(LLMError, match="At least one"),
            id="empty-questions",
        ),
        pytest.param(
            "ticket",
            {"rank": {"type": "score", "criteria": []}},
            pytest.raises(LLMError, match="nonempty criteria"),
            id="empty-score",
        ),
        pytest.param(
            "ticket",
            {"route": {"type": "unknown"}},
            pytest.raises(LLMError, match="unknown type"),
            id="unknown-question",
        ),
        pytest.param(
            "ticket",
            {"route": {"type": "choice", "criteria": {"ok": object()}}},
            pytest.raises(LLMError, match="invalid criteria"),
            id="non-json-criteria",
        ),
    ],
)
async def test_request_validation(state, questions, ctx, make_resp):
    client = TypeSafeClient(api_key="test")
    session = AsyncMock()
    session.post.return_value = make_resp(200, RESPONSE)
    client._session = session
    with ctx:
        result = await client.system_one(state, questions)
        assert result == RESPONSE


@pytest.mark.parametrize(
    "payload, ctx",
    [
        pytest.param(RESPONSE, nullcontext(), id="valid"),
        pytest.param(
            {}, pytest.raises(LLMError, match="Malformed"), id="missing-fields"
        ),
        pytest.param(
            {**RESPONSE, "usage": {"input_tokens": True, "output_tokens": 1}},
            pytest.raises(LLMError, match="Malformed"),
            id="boolean-token-count",
        ),
        pytest.param(
            {**RESPONSE, "answers": {"answer": {"type": "future"}}},
            pytest.raises(LLMError, match="Malformed"),
            id="unknown-answer",
        ),
        pytest.param(
            {**RESPONSE, "answers": {"answer": {"type": "noul", "noul": "yes"}}},
            pytest.raises(LLMError, match="Malformed"),
            id="wrong-answer-value",
        ),
    ],
)
def test_response_validation(payload, ctx, make_resp):
    with ctx:
        assert _check_resp(make_resp(200, payload)) == RESPONSE


@pytest.mark.parametrize(
    "status, payload, message",
    [
        pytest.param(
            401, {"error": {"message": "bad key"}}, "401 bad key", id="nested-error"
        ),
        pytest.param(422, {"detail": "bad request"}, "422 bad request", id="detail"),
        pytest.param(
            422,
            {"detail": [{"loc": ["body", "questions"], "msg": "Field required"}]},
            "422 Field required",
            id="validation-detail",
        ),
        pytest.param(500, {}, "500", id="empty-error"),
    ],
)
def test_provider_errors(status, payload, message, make_resp):
    with pytest.raises(LLMError, match=message) as caught:
        _check_resp(make_resp(status, payload))
    assert caught.value.provider == "typesafe"
    assert caught.value.body == payload


@pytest.mark.parametrize(
    "retry_after, expected",
    [
        pytest.param("7", 7, id="seconds"),
        pytest.param(
            "Wed, 21 Oct 2999 07:28:00 GMT",
            Retry().retry_after_max,
            id="http-date-capped",
        ),
        pytest.param("invalid", 60, id="malformed-fallback"),
        pytest.param(None, 60, id="missing-fallback"),
    ],
)
def test_rate_limit_error(retry_after, expected, make_resp):
    headers = {} if retry_after is None else {"retry-after": retry_after}
    response = make_resp(429, {"error": {"message": "slow down"}}, headers)
    with pytest.raises(TooManyRequestsError) as caught:
        _check_resp(response)
    assert caught.value.retry_delay == expected
    assert caught.value.message == "slow down"
    assert caught.value.response is response


async def test_context_lifecycle() -> None:
    session = MagicMock()
    session.headers = {}
    session.close = AsyncMock()
    with patch(
        "padwan_llm.typesafe.client.niquests.AsyncSession", return_value=session
    ) as factory:
        client = TypeSafeClient(api_key="secret")
        async with client as opened:
            assert opened is client
            assert client.session is session
            assert session.headers["Authorization"] == "Bearer secret"
        assert client._session is None
        session.close.assert_awaited_once()
    assert factory.call_args.kwargs["timeout"] == 60


@pytest.mark.parametrize(
    "api_key, env_key, ctx",
    [
        pytest.param("explicit", None, nullcontext(), id="explicit"),
        pytest.param(None, "environment", nullcontext(), id="environment"),
        pytest.param(
            None, None, pytest.raises(LLMError, match="TYPESAFE_API_KEY"), id="missing"
        ),
    ],
)
def test_api_key_resolution(api_key, env_key, ctx, monkeypatch):
    if env_key is None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TYPESAFE_API_KEY", env_key)
    with ctx:
        client = TypeSafeClient(api_key=api_key)
        assert "explicit" not in repr(client)
        assert "environment" not in repr(client)


def test_retry_policy_exhausts_after_two_retries() -> None:
    retry = TypeSafeClient(api_key="test")._retry
    error = ConnectTimeoutError(None, "/systemone", "timed out")

    retry = retry.increment(method="POST", url="/systemone", error=error)
    retry = retry.increment(method="POST", url="/systemone", error=error)

    with pytest.raises(MaxRetryError):
        retry.increment(method="POST", url="/systemone", error=error)


async def test_niquests_retries_twice_then_maps_rate_limit() -> None:
    class RateLimitHandler(BaseHTTPRequestHandler):
        attempts = 0

        def do_POST(self) -> None:
            type(self).attempts += 1
            self.send_response(429)
            self.send_header("content-type", "application/json")
            self.send_header("retry-after", "0")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"slow down"}}')

        def log_message(self, format: str, *args: object) -> None:
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), RateLimitHandler)
    except PermissionError:
        pytest.skip("loopback sockets are unavailable in this sandbox")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = cast("tuple[str, int]", server.server_address)
        client = TypeSafeClient(api_key="test", base_url=f"http://{host}:{port}/")
        with pytest.raises(TooManyRequestsError) as caught:
            async with client:
                await client.system_one("ticket", QUESTIONS)
        assert caught.value.message == "slow down"
        assert RateLimitHandler.attempts == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_public_models() -> None:
    assert TYPESAFE_MODELS == {"jev-latest", "jev-preview"}
