import dataclasses
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import field
from functools import partial
from types import TracebackType
from typing import Any, ClassVar, Literal, Self, cast, get_args

import niquests
from urllib3.exceptions import InvalidHeader
from urllib3.util.retry import Retry

from ..errors import LLMError, Provider, TooManyRequestsError
from .models import (
    ChoiceQuestion,
    JSONContent,
    JSONValue,
    NoulCriteria,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)

TypeSafeModel = Literal["jev-latest", "jev-preview"]

TYPESAFE_MODELS: set[str] = set(get_args(TypeSafeModel))
TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/"

__all__ = (
    "TYPESAFE_ENDPOINT",
    "TYPESAFE_MODELS",
    "TypeSafeClient",
    "TypeSafeModel",
)


def _normalize_json_value(value: object) -> JSONValue | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_value(item) for item in value]
    raise ValueError("Value is not JSON-compatible")


def _normalize_json_content(value: object) -> JSONContent:
    if not isinstance(value, (str, Mapping, Sequence)) or isinstance(
        value, (bytes, bytearray)
    ):
        raise ValueError("Content must be a string, mapping, or sequence")
    return cast("JSONContent", _normalize_json_value(value))


def _is_json_content(value: object) -> bool:
    try:
        _normalize_json_content(value)
    except ValueError:
        return False
    return True


def _normalize_questions(questions: Mapping[str, Question]) -> dict[str, Question]:
    if not questions:
        raise LLMError("typesafe", "At least one question is required")
    normalized: dict[str, Question] = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not isinstance(question, Mapping):
            raise LLMError("typesafe", "Question names and values must be mappings")
        instructions = question.get("instructions")
        try:
            normalized_instructions = (
                None if instructions is None else _normalize_json_content(instructions)
            )
        except ValueError as error:
            raise LLMError(
                "typesafe", f"Question {name!r} has invalid instructions"
            ) from error
        question_type = question.get("type")
        criteria = question.get("criteria")
        if question_type == "noul":
            if criteria is not None and (
                not isinstance(criteria, Mapping)
                or any(key not in {"true", "false"} for key in criteria)
            ):
                raise LLMError("typesafe", f"Question {name!r} has invalid criteria")
            normalized_question: NoulQuestion = {"type": "noul"}
            if "instructions" in question:
                normalized_question["instructions"] = normalized_instructions
            if "criteria" in question:
                if criteria is None:
                    normalized_question["criteria"] = None
                else:
                    try:
                        normalized_criteria = {
                            key: (
                                None if item is None else _normalize_json_content(item)
                            )
                            for key, item in criteria.items()
                        }
                    except ValueError as error:
                        raise LLMError(
                            "typesafe", f"Question {name!r} has invalid criteria"
                        ) from error
                    normalized_question["criteria"] = cast(
                        "NoulCriteria", normalized_criteria
                    )
            normalized[name] = normalized_question
        elif question_type == "choice":
            if not isinstance(criteria, Mapping) or not all(
                isinstance(key, str) for key in criteria
            ):
                raise LLMError("typesafe", f"Question {name!r} has invalid criteria")
            try:
                choice_criteria = {
                    key: None if item is None else _normalize_json_content(item)
                    for key, item in criteria.items()
                }
            except ValueError as error:
                raise LLMError(
                    "typesafe", f"Question {name!r} has invalid criteria"
                ) from error
            choice_question: ChoiceQuestion = {
                "type": "choice",
                "criteria": choice_criteria,
            }
            if "instructions" in question:
                choice_question["instructions"] = normalized_instructions
            normalized[name] = choice_question
        elif question_type == "score":
            if (
                not isinstance(criteria, Sequence)
                or isinstance(criteria, (str, bytes, bytearray))
                or not criteria
            ):
                raise LLMError(
                    "typesafe", f"Question {name!r} requires nonempty criteria"
                )
            try:
                score_criteria = [_normalize_json_content(item) for item in criteria]
            except ValueError as error:
                raise LLMError(
                    "typesafe", f"Question {name!r} requires nonempty criteria"
                ) from error
            score_question: ScoreQuestion = {
                "type": "score",
                "criteria": score_criteria,
            }
            if "instructions" in question:
                score_question["instructions"] = normalized_instructions
            normalized[name] = score_question
        else:
            raise LLMError(
                "typesafe", f"Question {name!r} has unknown type {question_type!r}"
            )
    return normalized


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number_map(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and _number(item) for key, item in value.items()
    )


def _valid_answer(answer: object) -> bool:
    if not isinstance(answer, dict):
        return False
    answer_type = answer.get("type")
    if answer_type == "noul":
        return _number(answer.get("noul"))
    if answer_type == "choice":
        return (
            isinstance(answer.get("choice"), str)
            and _number(answer.get("confidence"))
            and _number_map(answer.get("probabilities"))
        )
    if answer_type == "score":
        legend = answer.get("legend")
        return (
            _number(answer.get("score"))
            and _number(answer.get("confidence"))
            and isinstance(legend, dict)
            and all(
                isinstance(key, str) and _is_json_content(item)
                for key, item in legend.items()
            )
            and _number_map(answer.get("probabilities"))
        )
    return False


def _validate_response(data: object) -> SystemOneResponse:
    if not isinstance(data, dict):
        raise LLMError("typesafe", "Malformed response body")
    usage = data.get("usage")
    answers = data.get("answers")
    if (
        not isinstance(data.get("model"), str)
        or not isinstance(usage, dict)
        or not isinstance(usage.get("input_tokens"), int)
        or isinstance(usage.get("input_tokens"), bool)
        or not isinstance(usage.get("output_tokens"), int)
        or isinstance(usage.get("output_tokens"), bool)
        or not isinstance(answers, dict)
        or not answers
        or not all(
            isinstance(name, str) and _valid_answer(answer)
            for name, answer in answers.items()
        )
    ):
        raise LLMError("typesafe", "Malformed response body")
    return cast("SystemOneResponse", data)


def _error_message(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    for value in (data.get("error"), data.get("message"), data.get("detail")):
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("message"), str):
            return value["message"]
        if isinstance(value, list):
            messages = [
                item["msg"]
                for item in value
                if isinstance(item, dict) and isinstance(item.get("msg"), str)
            ]
            if messages:
                return "; ".join(messages)
    return ""


def _check_resp(resp: niquests.Response) -> SystemOneResponse:
    try:
        resp.raise_for_status()
    except niquests.exceptions.HTTPError as error:
        try:
            data: Any = resp.json()
        except Exception:
            data = None
        body = data if isinstance(data, dict) else None
        message = _error_message(data)
        if resp.status_code == 429:
            raw_retry_after = resp.headers.get("retry-after")
            try:
                retry_delay = (
                    60
                    if raw_retry_after is None
                    else math.ceil(Retry().parse_retry_after(raw_retry_after))
                )
            except (InvalidHeader, OverflowError, ValueError):
                retry_delay = 60
            raise TooManyRequestsError(
                retry_delay=retry_delay,
                message=message or None,
                response=resp,
            ) from error
        raise LLMError(
            "typesafe", f"{resp.status_code} {message}".rstrip(), cause=error, body=body
        ) from error
    try:
        data = resp.json()
    except Exception as error:
        raise LLMError("typesafe", "Malformed response body", cause=error) from error
    return _validate_response(data)


@dataclasses.dataclass
class TypeSafeClient:
    """Async client for TypeSafe's System One API."""

    provider: ClassVar[Provider] = "typesafe"
    model: str = "jev-latest"
    timeout: float = 60
    api_key: str | None = field(default=None, repr=False)
    base_url: str = TYPESAFE_ENDPOINT
    _retry: Retry = field(
        default_factory=partial(
            Retry,
            total=2,
            connect=2,
            read=2,
            status=2,
            backoff_factor=0.5,
            status_forcelist=[408, 429, *range(500, 600)],
            allowed_methods=["POST"],
            respect_retry_after_header=True,
            raise_on_status=False,
        ),
        repr=False,
    )
    _api_key: str = field(init=False, repr=False)
    _session: niquests.AsyncSession | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._api_key = self.api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self._api_key:
            raise LLMError(self.provider, "TYPESAFE_API_KEY not set")

    @property
    def session(self) -> niquests.AsyncSession:
        """Return the active session."""
        if self._session is None:
            raise LLMError(
                self.provider, "Client not initialized. Use async context manager."
            )
        return self._session

    async def __aenter__(self) -> Self:
        if self._session is not None:
            raise RuntimeError("TypeSafeClient is already open")
        session = niquests.AsyncSession(
            timeout=self.timeout, retries=self._retry, base_url=self.base_url
        )
        session.headers["Authorization"] = f"Bearer {self._api_key}"
        session.headers["Accept"] = "application/json"
        self._session = session
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._session is not None:
            try:
                await self._session.close()
            finally:
                self._session = None

    async def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, NoulQuestion | ChoiceQuestion | ScoreQuestion],
        *,
        model: str | None = None,
    ) -> SystemOneResponse:
        """Evaluate typed questions against shared state."""
        try:
            normalized_state = _normalize_json_content(state)
        except ValueError as error:
            raise LLMError(
                self.provider, "State must be a string, mapping, or sequence"
            ) from error
        normalized_questions = _normalize_questions(questions)
        body = SystemOneRequest(
            state=normalized_state,
            model=self.model if model is None else model,
            questions=normalized_questions,
        )
        resp = await self.session.post("systemone", json=body)
        return _check_resp(resp)
