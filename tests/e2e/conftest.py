from pathlib import Path

import pytest

from padwan_llm.conversation import Message
from padwan_llm.models import ToolDefinition


def _skip_no_key(env_var: str) -> pytest.MarkDecorator:
    """Lazy skipif — the string condition is evaluated at test time, not import time."""
    return pytest.mark.skipif(
        f"not __import__('os').environ.get('{env_var}')",
        reason=f"{env_var} not set",
    )


skip_no_gemini = _skip_no_key("GEMINI_API_KEY")
skip_no_openai = _skip_no_key("OPENAI_API_KEY")
skip_no_mistral = _skip_no_key("MISTRAL_API_KEY")
skip_no_grok = _skip_no_key("GROK_API_KEY")

pytestmark = pytest.mark.e2e

PROMPT = [Message(role="user", content="Reply with only the word 'hello'.")]

AUDIO_FIXTURE = Path(__file__).parent.parent / "fixtures" / "audio.wav"

TOOL_PROMPT = [Message(role="user", content="What is the weather in Paris?")]

WEATHER_TOOL: ToolDefinition = {
    "name": "get_weather",
    "description": "Get the current weather for a given city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "The city name"}},
        "required": ["city"],
    },
}
