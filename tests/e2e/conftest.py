import os
from pathlib import Path

import pytest

from padwan_llm.conversation import Message


def _has_key(env_var: str) -> bool:
    return bool(os.environ.get(env_var))


skip_no_gemini = pytest.mark.skipif(
    not _has_key("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)
skip_no_openai = pytest.mark.skipif(
    not _has_key("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
skip_no_mistral = pytest.mark.skipif(
    not _has_key("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not set"
)
skip_no_grok = pytest.mark.skipif(
    not _has_key("GROK_API_KEY"), reason="GROK_API_KEY not set"
)

pytestmark = pytest.mark.e2e

PROMPT = [Message(role="user", content="Reply with only the word 'hello'.")]

AUDIO_FIXTURE = Path(__file__).parent.parent / "fixtures" / "merci_greenway.wav"
