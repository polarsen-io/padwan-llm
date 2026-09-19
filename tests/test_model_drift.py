import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="module")
def model_drift() -> Iterator[ModuleType]:
    path = Path(__file__).resolve().parents[1] / "bin/drift/check_model_drift.py"
    spec = importlib.util.spec_from_file_location("padwan_model_drift", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load model drift script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize(
    "has_key, payload, failure, expected",
    [
        pytest.param(
            True,
            {"models": [{"name": "jev-latest"}, {"name": "jev-next"}]},
            None,
            (
                "**Available but not tracked**",
                "`jev-next`",
                "**Tracked but absent",
                "`jev-preview`",
            ),
            id="additions-and-absences",
        ),
        pytest.param(
            True,
            {"models": [{"name": "jev-latest"}, {"name": "jev-preview"}]},
            None,
            ("No live drift",),
            id="unchanged",
        ),
        pytest.param(
            False,
            None,
            None,
            ("Skipped", "TYPESAFE_API_KEY is not configured"),
            id="missing-key",
        ),
        pytest.param(
            True,
            None,
            "HTTP 401 Unauthorized",
            ("Failed: HTTP 401 Unauthorized",),
            id="request-failure",
        ),
        pytest.param(
            True,
            {"data": []},
            None,
            ("Failed: response has no models list",),
            id="malformed-list",
        ),
        pytest.param(
            True,
            {"models": [{"name": "jev-latest"}, {"name": 123}]},
            None,
            ("Failed: response contains an invalid model name",),
            id="malformed-name",
        ),
    ],
)
def test_typesafe_drift_report(
    model_drift: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    has_key: bool,
    payload: object,
    failure: str | None,
    expected: tuple[str, ...],
) -> None:
    for provider in ("OPENAI", "GEMINI", "MISTRAL", "GROK", "ANTHROPIC", "TYPESAFE"):
        monkeypatch.delenv(f"{provider}_API_KEY", raising=False)
    if has_key:
        monkeypatch.setenv("TYPESAFE_API_KEY", "test-typesafe-key")
    fetch = Mock(
        return_value=payload,
        side_effect=model_drift.FetchError(failure) if failure else None,
    )
    monkeypatch.setattr(model_drift, "_json_get", fetch)

    model_drift.check()

    report = capsys.readouterr().out.split("## TypeSafe (JEV)", 1)[1]
    for text in expected:
        assert text in report
    assert model_drift.TYPESAFE_MODELS == {"jev-latest", "jev-preview"}
    if has_key:
        fetch.assert_called_once_with(
            "https://api.typesafe.ai/v1/models",
            headers={"Authorization": "Bearer test-typesafe-key"},
        )
    else:
        fetch.assert_not_called()
