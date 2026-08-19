import dataclasses
from typing import ClassVar, Literal, get_args

from .._base import Provider, env_api_key
from ..errors import LLMError
from ..openai.client import _check_resp, _check_resp_status, _OpenAIBase
from .batch import GrokBatchJob, GrokBatchRequest, GrokBatchResult
from .types import (
    AddBatchRequestsBody,
    BatchRequestItem,
    BatchRequestPayload,
    BatchResponse,
    ChatGetCompletion,
    CreateBatchBody,
    ListBatchesResponse,
    ListBatchResultsResponse,
)

GrokModel = Literal[
    "grok-3",
    "grok-3-fast",
    "grok-3-fast-latest",
    "grok-3-latest",
    "grok-3-mini",
    "grok-3-mini-fast",
    "grok-3-mini-fast-high",
    "grok-3-mini-fast-latest",
    "grok-3-mini-high",
    "grok-3-mini-latest",
    "grok-4",
    "grok-4-fast",
    "grok-4-fast-reasoning",
    "grok-4-fast-non-reasoning",
    "grok-4-fast-non-reasoning-latest",
    "grok-4-fast-reasoning-latest",
    "grok-4-latest",
    "grok-4-1-fast",
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-reasoning-latest",
    "grok-4-1-fast-non-reasoning",
    "grok-4-1-fast-non-reasoning-latest",
    "grok-4.5",
    "grok-4.5-latest",
    "grok-4.6",
    "grok-4.3",
    "grok-4.3-latest",
    "grok-4.20",
    "grok-4.20-reasoning",
    "grok-4.20-reasoning-latest",
    "grok-4.20-non-reasoning",
    "grok-4.20-non-reasoning-latest",
    "grok-build-0.1",
    "grok-build-latest",
    "grok-code-fast",
    "grok-code-fast-1",
    "grok-latest",
]

__all__ = ("GROK_ENDPOINT", "GROK_MODELS", "GrokClient", "GrokModel", "is_grok_model")


def is_grok_model(model_name: str | None) -> bool:
    """Check if the model name looks like a Grok model based on known prefixes."""
    if model_name is None:
        return False
    return model_name.startswith("grok-")


GROK_MODELS: set[str] = set(get_args(GrokModel))

GROK_ENDPOINT = "https://api.x.ai/v1/"


@dataclasses.dataclass
class _GrokAuth:
    """Grok provider identity, shared by the chat and realtime clients."""

    provider: ClassVar[Provider] = "grok"

    def _get_default_api_key(self) -> str:
        return env_api_key(self.provider, "GROK_API_KEY")


@dataclasses.dataclass
class GrokClient(_GrokAuth, _OpenAIBase):
    """Grok API client with xAI-native batch support.

    https://docs.x.ai/developers/rest-api-reference/inference/batches
    """

    model: str | None = "grok-3"
    base_url: str = GROK_ENDPOINT

    async def create_batch(
        self,
        requests: list[GrokBatchRequest],
        name: str | None = None,
        model: str | None = None,
    ) -> GrokBatchJob:
        """Create a batch and submit requests to it.

        Creates an empty batch via POST /batches, then adds all requests
        in a single call. Returns the batch with updated state.

        https://docs.x.ai/developers/rest-api-reference/inference/batches#create-a-new-batch
        """
        payload = CreateBatchBody()
        if name:
            payload["name"] = name
        resp = await self.session.post("/batches", json=payload)
        data: BatchResponse = _check_resp(resp)
        job = GrokBatchJob.load(data)

        if requests:
            await self.add_batch_requests(job.batch_id, requests, model)
            job = await self.get_batch(job.batch_id)

        return job

    async def add_batch_requests(
        self,
        batch_id: str,
        requests: list[GrokBatchRequest],
        model: str | None = None,
    ) -> None:
        """Add requests to an existing batch.

        https://docs.x.ai/developers/rest-api-reference/inference/batches#add-requests-to-a-batch
        """
        _model = model or self.model
        if not _model:
            raise LLMError(self.provider, "No model specified")
        items: list[BatchRequestItem] = []
        for idx, req in enumerate(requests):
            items.append(
                BatchRequestItem(
                    batch_request_id=req.custom_id or f"request-{idx}",
                    batch_request=BatchRequestPayload(
                        chat_get_completion=ChatGetCompletion(
                            model=_model, messages=req.body.get("messages", [])
                        )
                    ),
                )
            )
        body = AddBatchRequestsBody(batch_requests=items)
        resp = await self.session.post(f"/batches/{batch_id}/requests", json=body)
        _check_resp_status(resp)

    async def get_batch(self, batch_id: str) -> GrokBatchJob:
        """Get the current status of a batch.

        https://docs.x.ai/developers/rest-api-reference/inference/batches#get-batch-details
        """
        resp = await self.session.get(f"/batches/{batch_id}")
        data: BatchResponse = _check_resp(resp)
        return GrokBatchJob.load(data)

    async def list_batches(
        self,
        limit: int = 20,
        pagination_token: str | None = None,
    ) -> tuple[list[GrokBatchJob], str | None]:
        """List batches with cursor-based pagination.

        Returns (jobs, next_token). Pass `next_token` as `pagination_token`
        to fetch the next page.

        https://docs.x.ai/developers/rest-api-reference/inference/batches#list-all-batches
        """
        params: dict[str, str] = {"page_size": str(limit)}
        if pagination_token:
            params["pagination_token"] = pagination_token
        resp = await self.session.get("/batches", params=params)
        data: ListBatchesResponse = _check_resp(resp)
        jobs = [GrokBatchJob.load(item) for item in data.get("batches", [])]
        next_token: str | None = data.get("pagination_token")
        return jobs, next_token

    async def cancel_batch(self, batch_id: str) -> GrokBatchJob:
        """Cancel a batch.

        https://docs.x.ai/developers/rest-api-reference/inference/batches#cancel-a-batch
        """
        resp = await self.session.post(f"/batches/{batch_id}:cancel")
        data: BatchResponse = _check_resp(resp)
        return GrokBatchJob.load(data)

    async def get_batch_results(
        self,
        batch_id: str,
        limit: int = 100,
        pagination_token: str | None = None,
    ) -> tuple[list[GrokBatchResult], str | None]:
        """Fetch results for a completed batch.

        Returns (results, next_token). Failed entries have ``error_message`` set.

        https://docs.x.ai/developers/rest-api-reference/inference/batches#get-batch-results
        """
        params: dict[str, str] = {"page_size": str(limit)}
        if pagination_token:
            params["pagination_token"] = pagination_token
        resp = await self.session.get(f"/batches/{batch_id}/results", params=params)
        data: ListBatchResultsResponse = _check_resp(resp)
        results = [
            GrokBatchResult.from_response(item) for item in data.get("results", [])
        ]
        next_token: str | None = data.get("pagination_token")
        return results, next_token
