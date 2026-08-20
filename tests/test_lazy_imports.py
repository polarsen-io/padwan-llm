import json
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    sys.version_info < (3, 15), reason="lazy imports require Python 3.15"
)
@pytest.mark.parametrize(
    ("attribute", "provider"),
    [
        pytest.param("AnthropicClient", "anthropic", id="anthropic"),
        pytest.param("GeminiClient", "gemini", id="gemini"),
        pytest.param("GrokClient", "grok", id="grok"),
        pytest.param("MistralClient", "mistral", id="mistral"),
        pytest.param("OpenAIClient", "openai", id="openai"),
    ],
)
def test_root_import_defers_provider_modules(attribute: str, provider: str):
    code = """
import json
import sys

import padwan_llm

providers = (
    "padwan_llm.anthropic",
    "padwan_llm.gemini",
    "padwan_llm.grok",
    "padwan_llm.mistral",
    "padwan_llm.openai",
)
before = [name for name in providers if name in sys.modules]
getattr(padwan_llm, sys.argv[1])
after = [name for name in providers if name in sys.modules]
print(json.dumps({"before": before, "after": after}))
"""

    output = subprocess.check_output([sys.executable, "-c", code, attribute], text=True)

    result = json.loads(output)
    expected = f"padwan_llm.{provider}"
    assert result["before"] == []
    assert expected in result["after"]
    assert set(result["after"]) <= {expected, "padwan_llm.openai"}
