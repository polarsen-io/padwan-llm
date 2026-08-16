from typing import Any, NotRequired, TypedDict

# POST /v1/batches


class CreateBatchBody(TypedDict, total=False):
    """Request body for POST /v1/batches.

    https://docs.x.ai/developers/rest-api-reference/inference/batches#create-a-new-batch
    """

    name: NotRequired[str]


# POST /v1/batches/{batch_id}/requests


class ChatGetCompletion(TypedDict):
    """Inner completion request wrapped in a batch request item."""

    model: str
    messages: Any


class BatchRequestPayload(TypedDict):
    """The ``batch_request`` wrapper containing the completion type."""

    chat_get_completion: ChatGetCompletion


class BatchRequestItem(TypedDict):
    """A single item in the ``batch_requests`` array."""

    batch_request_id: str
    batch_request: BatchRequestPayload


class AddBatchRequestsBody(TypedDict):
    """Request body for POST /v1/batches/{batch_id}/requests.

    https://docs.x.ai/developers/rest-api-reference/inference/batches#add-requests-to-a-batch
    """

    batch_requests: list[BatchRequestItem]


# GET /v1/batches/{batch_id}


class BatchState(TypedDict, total=False):
    """Counter-based state tracking for a batch."""

    num_requests: int
    num_pending: int
    num_success: int
    num_error: int
    num_cancelled: int


class BatchResponse(TypedDict):
    """Response from GET /v1/batches/{batch_id} and other batch endpoints.

    https://docs.x.ai/developers/rest-api-reference/inference/batches#get-batch-details
    """

    batch_id: str
    name: NotRequired[str]
    state: NotRequired[BatchState]
    create_time: NotRequired[str]
    expire_time: NotRequired[str]


# GET /v1/batches


class ListBatchesResponse(TypedDict, total=False):
    """Response from GET /v1/batches.

    https://docs.x.ai/developers/rest-api-reference/inference/batches#list-all-batches
    """

    batches: list[BatchResponse]
    pagination_token: NotRequired[str]


# GET /v1/batches/{batch_id}/results


class BatchResultDataResponse(TypedDict, total=False):
    """Nested completion response inside a batch result."""

    chat_get_completion: dict[str, Any]


class BatchResultPayload(TypedDict, total=False):
    """The ``batch_result`` wrapper containing the response and optional error."""

    response: BatchResultDataResponse
    error: dict[str, str] | str


class BatchResultItem(TypedDict):
    """A single item in the batch results list.

    https://docs.x.ai/developers/rest-api-reference/inference/batches#get-batch-results
    """

    batch_request_id: str
    batch_result: BatchResultPayload


class ListBatchResultsResponse(TypedDict, total=False):
    """Response from GET /v1/batches/{batch_id}/results.

    https://docs.x.ai/developers/rest-api-reference/inference/batches#get-batch-results
    """

    results: list[BatchResultItem]
    pagination_token: NotRequired[str]
