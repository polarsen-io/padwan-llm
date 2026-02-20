from contextlib import nullcontext

import pytest

from padwan_llm.client import LLMClient
from padwan_llm.conversation import ConversationState, Message
from padwan_llm.errors import LLMError
from padwan_llm.gemini.client import GeminiClient, is_gemini_model
from padwan_llm.grok.client import GrokClient, is_grok_model
from padwan_llm.mistral.client import MistralClient, is_mistral_model
from padwan_llm.models import UsageToken
from padwan_llm.openai.client import OpenAIClient, is_openai_model


# ConversationState


class TestConversationState:
    @pytest.mark.parametrize(
        "system, expected_messages",
        [
            pytest.param(None, [], id="no-system"),
            pytest.param(
                "You are helpful.",
                [Message(role="system", content="You are helpful.")],
                id="with-system",
            ),
        ],
    )
    def test_init(self, system: str | None, expected_messages: list[Message]):
        state = ConversationState(system=system)
        assert state.system == system
        assert state.messages == expected_messages
        assert state.last_usage is None
        assert state.total_usage == {"total": 0, "input": 0, "output": 0}

    @pytest.mark.parametrize(
        "method, role",
        [
            pytest.param("add_user_message", "user", id="user"),
            pytest.param("add_assistant_message", "assistant", id="assistant"),
        ],
    )
    def test_add_message(self, method: str, role: str):
        state = ConversationState()
        msg = getattr(state, method)("Hello")
        assert msg["role"] == role
        assert state.messages == [msg]

    def test_message_ordering_with_system(self):
        state = ConversationState(system="Be concise.")
        state.add_user_message("Hello")
        state.add_assistant_message("Hi")
        assert [m["role"] for m in state.messages] == ["system", "user", "assistant"]

    @pytest.mark.parametrize(
        "usages, expected_total, expected_last",
        [
            pytest.param(
                [
                    UsageToken(total=100, input=80, output=20),
                    UsageToken(total=50, input=30, output=20),
                ],
                UsageToken(total=150, input=110, output=40),
                UsageToken(total=50, input=30, output=20),
                id="multiple",
            ),
            pytest.param(
                [
                    UsageToken(total=100, input=80, output=20, cached=10),
                    UsageToken(total=50, input=30, output=20, cached=5),
                ],
                UsageToken(total=150, input=110, output=40, cached=15),
                UsageToken(total=50, input=30, output=20, cached=5),
                id="cached-accumulation",
            ),
        ],
    )
    def test_accumulate_usage(
        self,
        usages: list[UsageToken],
        expected_total: UsageToken,
        expected_last: UsageToken,
    ):
        state = ConversationState()
        for u in usages:
            state.accumulate_usage(u)
        assert state.total_usage == expected_total
        assert state.last_usage == expected_last

    @pytest.mark.parametrize(
        "system, expected_after_clear",
        [
            pytest.param(None, [], id="no-system"),
            pytest.param(
                "Be concise.",
                [Message(role="system", content="Be concise.")],
                id="with-system",
            ),
        ],
    )
    def test_clear(self, system: str | None, expected_after_clear: list[Message]):
        state = ConversationState(system=system)
        state.add_user_message("Hello")
        state.clear()
        assert state.messages == expected_after_clear


# Errors


@pytest.mark.parametrize(
    "cause",
    [
        pytest.param(None, id="no-cause"),
        pytest.param(ValueError("bad"), id="with-cause"),
    ],
)
def test_llm_error(cause: Exception | None):
    err = LLMError("openai", "failed", cause=cause)
    assert str(err) == "[openai] failed"
    assert err.cause is cause


# Model detection (edge cases not covered by routing)


@pytest.mark.parametrize(
    "func, model, expected",
    [
        pytest.param(is_openai_model, "o3", True, id="openai-o3"),
        pytest.param(is_openai_model, "gpt-4.1", True, id="openai-gpt4.1"),
        pytest.param(is_openai_model, "unknown", False, id="openai-miss"),
        pytest.param(is_gemini_model, "gemini-custom", True, id="gemini-prefix"),
        pytest.param(is_gemini_model, "gpt-4o", False, id="gemini-miss"),
        pytest.param(is_mistral_model, "codestral-latest", True, id="mistral-set"),
        pytest.param(is_mistral_model, "mistral-custom", True, id="mistral-prefix"),
        pytest.param(is_mistral_model, "gpt-4o", False, id="mistral-miss"),
        pytest.param(is_grok_model, None, False, id="grok-none"),
    ],
)
def test_is_model(func, model, expected: bool):
    assert func(model) is expected


# LLMClient factory


@pytest.mark.parametrize(
    "model, expected_type, ctx",
    [
        pytest.param("gpt-4o", OpenAIClient, nullcontext(), id="openai"),
        pytest.param("gemini-2.0-flash", GeminiClient, nullcontext(), id="gemini"),
        pytest.param(
            "mistral-large-latest", MistralClient, nullcontext(), id="mistral"
        ),
        pytest.param("grok-3", GrokClient, nullcontext(), id="grok"),
        pytest.param(
            "unknown-xyz",
            None,
            pytest.raises(ValueError, match="Unknown model"),
            id="unknown",
        ),
    ],
)
def test_llm_client_routing(model: str, expected_type: type | None, ctx):
    with ctx:
        client = LLMClient(model, api_key="fake-key")
        assert isinstance(client, expected_type)  # type: ignore[arg-type]


def test_llm_client_passes_params():
    client = LLMClient("gpt-4o", temperature=0.8, timeout=120, api_key="k")
    assert client.temperature == 0.8
    assert client.timeout == 120


# Client init


@pytest.mark.parametrize(
    "cls, env_var, kwargs",
    [
        pytest.param(OpenAIClient, "OPENAI_API_KEY", {}, id="openai"),
        pytest.param(GeminiClient, "GEMINI_API_KEY", {}, id="gemini"),
        pytest.param(MistralClient, "MISTRAL_API_KEY", {}, id="mistral"),
        pytest.param(GrokClient, "GROK_API_KEY", {}, id="grok"),
        pytest.param(
            OpenAIClient,
            None,
            {"base_url": "http://localhost:8080/v1/"},
            id="openai-no-key",
        ),
    ],
)
def test_missing_api_key(
    cls: type, env_var: str | None, kwargs: dict, monkeypatch: pytest.MonkeyPatch
):
    if env_var:
        monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(LLMError, match="not set|required"):
        cls(**kwargs)


async def test_session_lifecycle():
    client = OpenAIClient(api_key="k")
    with pytest.raises(LLMError, match="not initialized"):
        _ = client.session
    async with client as c:
        assert c._session is not None
    assert c._session is None
