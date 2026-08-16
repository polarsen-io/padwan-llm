from typing import cast, get_type_hints

import pytest
from google.protobuf.descriptor import Descriptor
from xai_sdk.proto.v6.batch_pb2 import (
    Batch,
    CreateBatchRequest,
)
from xai_sdk.proto.v6.batch_pb2 import (
    BatchState as SdkBatchState,
)
from xai_sdk.proto.v6.batch_pb2 import (
    ListBatchesResponse as SdkListBatchesResponse,
)
from xai_sdk.proto.v6.batch_pb2 import (
    ListBatchResultsResponse as SdkListBatchResultsResponse,
)
from xai_sdk.proto.v6.chat_pb2 import GetCompletionsRequest

from padwan_llm.grok.batch import GrokBatchJob, GrokBatchResult
from padwan_llm.grok.types import (
    BatchResponse,
    BatchResultItem,
    BatchState,
    ChatGetCompletion,
    CreateBatchBody,
    ListBatchesResponse,
    ListBatchResultsResponse,
)


class TestGrokBatchJobLoad:
    @pytest.mark.parametrize(
        "data, expected",
        [
            pytest.param(
                {
                    "batch_id": "batch-abc",
                    "name": "my-batch",
                    "state": {
                        "num_requests": 10,
                        "num_pending": 3,
                        "num_success": 5,
                        "num_error": 2,
                        "num_cancelled": 0,
                    },
                    "create_time": "2026-02-19",
                    "expire_time": "2026-03-21",
                },
                GrokBatchJob(
                    batch_id="batch-abc",
                    name="my-batch",
                    num_requests=10,
                    num_pending=3,
                    num_success=5,
                    num_error=2,
                    num_cancelled=0,
                    create_time="2026-02-19",
                    expire_time="2026-03-21",
                ),
                id="full",
            ),
            pytest.param(
                {"batch_id": "batch-empty"},
                GrokBatchJob(batch_id="batch-empty"),
                id="minimal",
            ),
        ],
    )
    def test_load(self, data: dict, expected: GrokBatchJob):
        assert GrokBatchJob.load(cast(BatchResponse, data)) == expected


class TestGrokBatchJobProperties:
    @pytest.mark.parametrize(
        "num_requests, num_pending, expected_terminal, expected_succeeded",
        [
            pytest.param(10, 0, True, True, id="all-success"),
            pytest.param(10, 3, False, False, id="still-pending"),
            pytest.param(0, 0, False, False, id="empty-batch"),
        ],
    )
    def test_is_terminal_and_succeeded(
        self,
        num_requests: int,
        num_pending: int,
        expected_terminal: bool,
        expected_succeeded: bool,
    ):
        job = GrokBatchJob(
            batch_id="b",
            num_requests=num_requests,
            num_pending=num_pending,
            num_success=num_requests - num_pending,
        )
        assert job.is_terminal is expected_terminal
        assert job.succeeded is expected_succeeded

    @pytest.mark.parametrize(
        "num_error, num_cancelled, expected",
        [
            pytest.param(1, 0, False, id="has-errors"),
            pytest.param(0, 1, False, id="has-cancelled"),
            pytest.param(0, 0, True, id="all-success"),
        ],
    )
    def test_succeeded_with_failures(
        self, num_error: int, num_cancelled: int, expected: bool
    ):
        job = GrokBatchJob(
            batch_id="b",
            num_requests=10,
            num_pending=0,
            num_success=10 - num_error - num_cancelled,
            num_error=num_error,
            num_cancelled=num_cancelled,
        )
        assert job.succeeded is expected


class TestGrokBatchResult:
    @pytest.mark.parametrize(
        "data, expected",
        [
            pytest.param(
                {
                    "batch_request_id": "req-1",
                    "batch_result": {
                        "response": {
                            "chat_get_completion": {
                                "choices": [
                                    {
                                        "message": {"content": "Hi!"},
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 170,
                                    "completion_tokens": 5,
                                    "total_tokens": 175,
                                },
                            }
                        }
                    },
                },
                GrokBatchResult(
                    custom_id="req-1",
                    content="Hi!",
                    input_tokens=170,
                    output_tokens=5,
                    total_tokens=175,
                    finish_reason="stop",
                ),
                id="full-response",
            ),
            pytest.param(
                {
                    "batch_request_id": "req-2",
                    "batch_result": {"response": {}},
                },
                GrokBatchResult(custom_id="req-2", content=""),
                id="empty-response",
            ),
            pytest.param(
                {
                    "batch_request_id": "req-3",
                    "batch_result": {
                        "error": {"message": "Model overloaded"},
                        "response": {},
                    },
                },
                GrokBatchResult(
                    custom_id="req-3",
                    content="",
                    error_message="Model overloaded",
                ),
                id="error-response",
            ),
            pytest.param(
                {
                    "batch_request_id": "req-4",
                    "batch_result": {
                        "response": {
                            "chat_get_completion": {
                                "choices": [
                                    {
                                        "message": {"content": None},
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {},
                            }
                        }
                    },
                },
                GrokBatchResult(custom_id="req-4", content="", finish_reason="stop"),
                id="null-content",
            ),
        ],
    )
    def test_from_response(self, data: dict, expected: GrokBatchResult):
        assert GrokBatchResult.from_response(cast(BatchResultItem, data)) == expected


def _proto_fields(descriptor: Descriptor) -> set[str]:
    """Extract field names from a protobuf message Descriptor."""
    return {f.name for f in descriptor.fields}


@pytest.mark.parametrize(
    "local_type, sdk_descriptor",
    [
        pytest.param(BatchState, SdkBatchState.DESCRIPTOR, id="BatchState"),
        pytest.param(BatchResponse, Batch.DESCRIPTOR, id="BatchResponse-Batch"),
        pytest.param(
            CreateBatchBody, CreateBatchRequest.DESCRIPTOR, id="CreateBatchBody"
        ),
        pytest.param(
            ListBatchesResponse,
            SdkListBatchesResponse.DESCRIPTOR,
            id="ListBatchesResponse",
        ),
        pytest.param(
            ListBatchResultsResponse,
            SdkListBatchResultsResponse.DESCRIPTOR,
            id="ListBatchResultsResponse",
        ),
        pytest.param(
            ChatGetCompletion,
            GetCompletionsRequest.DESCRIPTOR,
            id="ChatGetCompletion",
        ),
    ],
)
def test_sdk_compat(local_type: type, sdk_descriptor: Descriptor):
    """Verify every field in our local types maps to a field in the xai-sdk protobuf."""
    sdk_keys = _proto_fields(sdk_descriptor)
    for key in get_type_hints(local_type):
        assert key in sdk_keys, (
            f"{local_type.__name__}.{key} not in {sdk_descriptor.name}"
        )
