import warnings

import pytest

from padwan_llm import ModelDeprecationWarning
from padwan_llm._deprecation import _warned, warn_if_deprecated
from padwan_llm.mistral._deprecations import DEPRECATED
from padwan_llm.mistral.client import MistralClient

_DEPRECATIONS = {"mistral-moderation-latest": "2026-06-30T12:00:00Z"}


@pytest.fixture(autouse=True)
def _reset_warned():
    # The once-per-process guard is module-global; isolate it between tests.
    _warned.clear()
    yield
    _warned.clear()


def _matched(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if issubclass(w.category, ModelDeprecationWarning)]


@pytest.mark.parametrize(
    "model, warns",
    [
        pytest.param("mistral-moderation-latest", True, id="deprecated-warns"),
        pytest.param("mistral-large-latest", False, id="active-silent"),
        pytest.param(None, False, id="none-silent"),
    ],
)
def test_warn_if_deprecated(model: str | None, warns: bool):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_deprecated("mistral", model, _DEPRECATIONS)
    matched = _matched(caught)
    assert bool(matched) is warns
    if warns:
        assert model is not None
        message = str(matched[0].message)
        assert model in message
        assert "retirement" in message


def test_warns_once_per_process():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            warn_if_deprecated("mistral", "mistral-moderation-latest", _DEPRECATIONS)
    assert len(_matched(caught)) == 1


def test_mistral_client_wired_to_generated_map():
    # The runtime path must read the generated map, not a stale copy.
    assert MistralClient._deprecations is DEPRECATED


@pytest.mark.parametrize(
    "model, warns",
    [
        pytest.param("mistral-moderation-latest", True, id="deprecated"),
        pytest.param("mistral-large-latest", False, id="active"),
    ],
)
def test_client_construction(monkeypatch: pytest.MonkeyPatch, model: str, warns: bool):
    monkeypatch.setattr(MistralClient, "_deprecations", _DEPRECATIONS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MistralClient(model=model, api_key="test")
    assert bool(_matched(caught)) is warns


def test_warning_is_suppressible(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(MistralClient, "_deprecations", _DEPRECATIONS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=ModelDeprecationWarning)
        MistralClient(model="mistral-moderation-latest", api_key="test")
    assert _matched(caught) == []
