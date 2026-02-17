from __future__ import annotations

import dataclasses
import mimetypes
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, get_args

from .._base import LLMError, Provider
from ..openai.client import OpenAIClient, _check_resp

if TYPE_CHECKING:
    from .types import EmbeddingRequest, EmbeddingResponse, TranscriptionResponse

MistralModel = Literal[
    "mistral-large-latest",
    "pixtral-large-latest",
    "mistral-medium-latest",
    "mistral-moderation-latest",
    "ministral-3b-latest",
    "ministral-8b-latest",
    "open-mistral-nemo",
    "mistral-small-latest",
    "devstral-small-latest",
    "mistral-saba-latest",
    "codestral-latest",
    "mistral-ocr-latest",
    "magistral-small-latest",
]

MistralEmbeddingModel = Literal["mistral-embed"]
MistralAudioModel = Literal["voxtral-mini-latest"]

__all__ = (
    "MistralClient",
    "MISTRAL_MODELS",
    "MISTRAL_ENDPOINT",
    "MistralModel",
    "MistralEmbeddingModel",
    "MistralAudioModel",
    "is_mistral_model",
)


_MISTRAL_PREFIXES = (
    "mistral-",
    "pixtral-",
    "ministral-",
    "open-mistral-",
    "devstral-",
    "codestral-",
    "magistral-",
    "voxtral-",
)


def is_mistral_model(model_name: str | None) -> bool:
    """Check if the model name looks like a Mistral model based on known prefixes."""
    if model_name is None:
        return False
    return model_name.startswith(_MISTRAL_PREFIXES)


MISTRAL_MODELS: set[str] = set(get_args(MistralModel))

MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/"


@dataclasses.dataclass
class MistralClient(OpenAIClient):
    """Mistral API client."""

    provider: ClassVar[Provider] = "mistral"
    model: str | None = "mistral-large-latest"
    base_url: str = MISTRAL_ENDPOINT

    def _get_default_api_key(self) -> str:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise LLMError(self.provider, "MISTRAL_API_KEY not set")
        return api_key

    async def fetch_embeddings(
        self,
        input: str | list[str],
        model: str = "mistral-embed",
    ) -> list[list[float]]:
        """Fetch embeddings from Mistral.

        Args:
            input: Text or list of texts to embed.
            model: Embedding model to use.

        Returns:
            List of embedding vectors (one per input text).
        """
        body: EmbeddingRequest = {"model": model, "input": input}
        resp = await self.session.post("/embeddings", json=body)
        data: EmbeddingResponse = _check_resp(resp)
        return [item["embedding"] for item in data["data"]]

    async def transcribe(
        self,
        *,
        file: str | Path | bytes | None = None,
        file_id: str | None = None,
        file_url: str | None = None,
        model: str = "voxtral-mini-latest",
        language: str | None = None,
        temperature: float | None = None,
        diarize: bool = False,
        timestamp_granularities: list[Literal["segment", "word"]] | None = None,
    ) -> TranscriptionResponse:
        """Transcribe audio using Mistral's audio API.

        Exactly one of ``file``, ``file_id``, or ``file_url`` must be provided.
        ``file`` accepts a filesystem path (str/Path) or raw bytes for multipart upload.
        """
        sources = sum(x is not None for x in (file, file_id, file_url))
        if sources != 1:
            raise LLMError(
                self.provider,
                "Provide exactly one of file, file_id, or file_url",
            )

        if file is not None:
            if isinstance(file, bytes):
                file_tuple = ("audio", file, "application/octet-stream")
            else:
                path = Path(file)
                mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                file_tuple = (path.name, path.read_bytes(), mime)

            data: list[tuple[str, str]] = [("model", model)]
            if language is not None:
                data.append(("language", language))
            if temperature is not None:
                data.append(("temperature", str(temperature)))
            if diarize:
                data.append(("diarize", "true"))
            if timestamp_granularities is not None:
                for g in timestamp_granularities:
                    data.append(("timestamp_granularities[]", g))

            resp = await self.session.post(
                "/audio/transcriptions",
                files=[("file", file_tuple)],
                data=data,
            )
        else:
            body: dict[str, Any] = {"model": model}
            if file_id is not None:
                body["file_id"] = file_id
            if file_url is not None:
                body["file_url"] = file_url
            if language is not None:
                body["language"] = language
            if temperature is not None:
                body["temperature"] = temperature
            if diarize:
                body["diarize"] = True
            if timestamp_granularities is not None:
                body["timestamp_granularities"] = timestamp_granularities

            resp = await self.session.post("/audio/transcriptions", json=body)

        result: TranscriptionResponse = _check_resp(resp)
        return result
