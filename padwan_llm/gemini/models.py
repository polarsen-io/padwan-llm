from typing import Any, Literal, NotRequired, TypedDict

__all__ = (
    "CompletionBody",
    "Content",
    "Part",
    "StreamBody",
    "SystemInstruction",
    "GenerationConfig",
    "ThinkingConfig",
    # Batch types
    "BatchState",
    "BatchStats",
    "BatchJobMetadata",
    "BatchJobResponse",
    "InlinedResponse",
    "BatchDestination",
    "BatchRequestPayload",
)


# Batch job states (normalized from BATCH_STATE_* to JOB_STATE_*)
BatchState = Literal[
    "JOB_STATE_UNSPECIFIED",
    "JOB_STATE_PENDING",
    "JOB_STATE_QUEUED",
    "JOB_STATE_RUNNING",
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLING",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
]


class BatchStats(TypedDict, total=False):
    """Batch statistics from Gemini API."""

    pendingRequestCount: str
    requestCount: str
    successfulRequestCount: str
    failedRequestCount: str


class BatchJobMetadata(TypedDict, total=False):
    """Metadata for a batch job."""

    state: str
    displayName: str
    model: str
    createTime: str
    updateTime: str
    startTime: str
    endTime: str
    batchStats: BatchStats


class BatchJobResponse(TypedDict, total=False):
    """Response from batch job creation/get endpoints."""

    name: str
    metadata: BatchJobMetadata
    error: dict


class InlinedResponse(TypedDict):
    """Single inlined response in batch results."""

    key: str
    response: dict  # GenerateContentResponseDict


class BatchDestination(TypedDict, total=False):
    """Destination for batch results."""

    file_name: str
    inlined_responses: list[InlinedResponse]


class BatchRequestPayload(TypedDict, total=False):
    """Payload for a single request in a batch."""

    contents: list[Content]
    generationConfig: NotRequired[GenerationConfig]
    systemInstruction: NotRequired[SystemInstruction]


class Part(TypedDict):
    text: str


class Content(TypedDict):
    role: Literal["user", "model"]
    parts: list[Part]


class SystemInstruction(TypedDict):
    parts: list[Part]


class ThinkingConfig(TypedDict, total=False):
    includeThoughts: bool
    thinkingBudget: int
    thinkingLevel: Literal[
        "THINKING_LEVEL_UNSPECIFIED",
        "LOW",
        "MEDIUM",
        "HIGH",
    ]


class GenerationConfig(TypedDict, total=False):
    responseMimeType: Literal["application/json"]
    responseSchema: dict[str, Any]
    temperature: float
    thinkingConfig: ThinkingConfig


class StreamBody(TypedDict):
    contents: list[Content]
    temperature: NotRequired[float]
    systemInstruction: NotRequired[SystemInstruction]


class CompletionBody(TypedDict):
    contents: list[Content]
    systemInstruction: NotRequired[SystemInstruction]
    generationConfig: NotRequired[GenerationConfig]
