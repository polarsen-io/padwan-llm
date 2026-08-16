import json
from dataclasses import dataclass

import niquests
import pytest

from padwan_llm import OpenAIClient
from padwan_llm.anthropic.client import _check_resp as anthropic_check_resp
from padwan_llm.errors import LLMError, TooManyRequestsError
from padwan_llm.gemini.client import _check_resp as gemini_check_resp
from padwan_llm.openai.client import _check_resp as openai_check_resp


@dataclass
class FakeModel:
    name: str
    value: int


def _fake_decoder(raw: bytes) -> FakeModel:
    data = json.loads(raw)
    return FakeModel(name=data["name"], value=data["value"])


def _make_real_resp(status_code: int, body: bytes) -> niquests.Response:
    resp = niquests.Response()
    resp.status_code = status_code
    resp._content = body
    return resp


@pytest.fixture(
    params=[
        pytest.param(openai_check_resp, id="openai"),
        pytest.param(gemini_check_resp, id="gemini"),
        pytest.param(anthropic_check_resp, id="anthropic"),
    ]
)
def check_resp(request):
    return request.param


def test_no_decoder_returns_json(check_resp):
    resp = _make_real_resp(200, b'{"key": "value"}')
    assert check_resp(resp) == {"key": "value"}


def test_decoder_returns_typed_object(check_resp):
    resp = _make_real_resp(200, b'{"name": "test", "value": 42}')
    assert check_resp(resp, decoder=_fake_decoder) == FakeModel(name="test", value=42)


def test_decoder_receives_raw_bytes(check_resp):
    received: list[bytes] = []

    def spy_decoder(raw: bytes) -> dict:
        received.append(raw)
        return json.loads(raw)

    resp = _make_real_resp(200, b'{"x": 1}')
    assert check_resp(resp, decoder=spy_decoder) == {"x": 1}
    assert received == [b'{"x": 1}']


def test_decoder_validation_error_propagates(check_resp):
    def decoder(raw: bytes) -> FakeModel:
        raise ValueError("value must be an int")

    resp = _make_real_resp(200, b'{"value": "nope"}')
    with pytest.raises(ValueError, match="value must be an int"):
        check_resp(resp, decoder=decoder)


def test_decoder_empty_body_raises(check_resp):
    resp = _make_real_resp(200, b"")
    with pytest.raises(LLMError, match="Empty response body"):
        check_resp(resp, decoder=_fake_decoder)


@pytest.mark.parametrize(
    "check_fn, status, json_data, headers, exc_type",
    [
        pytest.param(
            openai_check_resp,
            429,
            {},
            {"retry-after": "10"},
            TooManyRequestsError,
            id="openai-429",
        ),
        pytest.param(
            openai_check_resp,
            500,
            {"error": "internal"},
            None,
            LLMError,
            id="openai-500",
        ),
        pytest.param(
            gemini_check_resp,
            500,
            {"error": {"message": "fail"}},
            None,
            LLMError,
            id="gemini-500",
        ),
        pytest.param(
            anthropic_check_resp,
            500,
            {"error": {"message": "fail"}},
            None,
            LLMError,
            id="anthropic-500",
        ),
    ],
)
def test_http_errors_raise_with_decoder(
    check_fn, status, json_data, headers, exc_type, make_resp
):
    resp = make_resp(status, json_data, headers)
    with pytest.raises(exc_type):
        check_fn(resp, decoder=_fake_decoder)


async def test_client_json_encoder_forwarded_to_session():
    def encoder(obj) -> bytes:
        return b"{}"

    async with OpenAIClient(api_key="k", json_encoder=encoder) as client:
        assert client.session.json_encoder is encoder
