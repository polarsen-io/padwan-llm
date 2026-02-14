from __future__ import annotations

import dataclasses
import os
from typing import ClassVar, Literal, get_args

from .._base import LLMError, Provider
from ..openai.client import OpenAIClient

GrokModel = Literal[
    "grok-3",
    "grok-3-mini-fast",
    "grok-3-mini",
]

__all__ = ("GrokClient", "GROK_MODELS", "GROK_ENDPOINT", "GrokModel", "is_grok_model")


def is_grok_model(model_name: str | None) -> bool:
    """Check if the model name looks like a Grok model based on known prefixes."""
    if model_name is None:
        return False
    return model_name.startswith("grok-")


GROK_MODELS: set[str] = set(get_args(GrokModel))

GROK_ENDPOINT = "https://api.x.ai/v1/"


@dataclasses.dataclass
class GrokClient(OpenAIClient):
    """Grok API client."""

    provider: ClassVar[Provider] = "grok"
    model: str | None = "grok-3"
    base_url: str = GROK_ENDPOINT

    def _get_default_api_key(self) -> str:
        api_key = os.environ.get("GROK_API_KEY")
        if not api_key:
            raise LLMError(self.provider, "GROK_API_KEY not set")
        return api_key
