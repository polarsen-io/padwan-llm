from typing import cast

import pytest

from padwan_llm import TypeSafeClient
from padwan_llm.typesafe import (
    ChoiceAnswer,
    JSONContent,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
)

from .conftest import _skip_no_key

pytestmark = [pytest.mark.e2e, _skip_no_key("TYPESAFE_API_KEY")]


QUESTIONS = {
    "billing": {"type": "noul", "instructions": "Is this about billing?"},
    "tone": {
        "type": "choice",
        "instructions": "What is the tone?",
        "criteria": {"calm": None, "frustrated": None, "angry": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["can wait", "this week", "today"],
    },
}


def _check_answers(response: SystemOneResponse) -> None:
    assert response["model"]
    assert response["usage"]["input_tokens"] >= 0
    assert response["usage"]["output_tokens"] >= 0

    billing = cast("NoulAnswer", response["answers"]["billing"])
    tone = cast("ChoiceAnswer", response["answers"]["tone"])
    urgency = cast("ScoreAnswer", response["answers"]["urgency"])
    assert billing["type"] == "noul"
    assert 0 <= billing["noul"] <= 1
    assert tone["type"] == "choice"
    assert tone["choice"] in tone["probabilities"]
    assert 0 <= tone["confidence"] <= 1
    assert all(0 <= probability <= 1 for probability in tone["probabilities"].values())
    assert sum(tone["probabilities"].values()) == pytest.approx(1, abs=0.01)
    assert urgency["type"] == "score"
    assert 0 <= urgency["score"] <= 2
    assert 0 <= urgency["confidence"] <= 1
    assert urgency["legend"].keys() == urgency["probabilities"].keys()
    assert all(
        0 <= probability <= 1 for probability in urgency["probabilities"].values()
    )
    assert sum(urgency["probabilities"].values()) == pytest.approx(1, abs=0.01)


@pytest.mark.parametrize(
    "state, model",
    [
        pytest.param(
            "I was charged twice and need help today.", None, id="text-default"
        ),
        pytest.param(
            {"ticket": "The export crashes", "customer_tier": "business"},
            "jev-preview",
            id="structured-override",
        ),
    ],
)
async def test_system_one(state: JSONContent, model: str | None) -> None:
    async with TypeSafeClient() as client:
        response = await client.system_one(state, QUESTIONS, model=model)
    _check_answers(response)
