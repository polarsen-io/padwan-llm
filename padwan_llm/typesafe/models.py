from collections.abc import Mapping, Sequence
from typing import Literal, NotRequired, TypedDict

type JSONValue = (
    str
    | int
    | float
    | bool
    | Sequence[JSONValue | None]
    | Mapping[str, JSONValue | None]
)
type JSONContent = str | Mapping[str, JSONValue | None] | Sequence[JSONValue | None]


class NoulCriteria(TypedDict, total=False):
    true: JSONContent | None
    false: JSONContent | None


class NoulQuestion(TypedDict):
    type: Literal["noul"]
    instructions: NotRequired[JSONContent | None]
    criteria: NotRequired[NoulCriteria | None]


class ChoiceQuestion(TypedDict):
    type: Literal["choice"]
    criteria: Mapping[str, JSONContent | None]
    instructions: NotRequired[JSONContent | None]


class ScoreQuestion(TypedDict):
    type: Literal["score"]
    criteria: Sequence[JSONContent]
    instructions: NotRequired[JSONContent | None]


type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


class NoulAnswer(TypedDict):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(TypedDict):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, JSONContent]
    probabilities: dict[str, float]


type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class TypeSafeUsage(TypedDict):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(TypedDict):
    model: str
    answers: dict[str, Answer]
    usage: TypeSafeUsage


class SystemOneRequest(TypedDict):
    state: JSONContent
    model: str
    questions: Mapping[str, Question]


__all__ = (
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "JSONContent",
    "JSONValue",
    "NoulAnswer",
    "NoulCriteria",
    "NoulQuestion",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "SystemOneRequest",
    "SystemOneResponse",
    "TypeSafeUsage",
)
