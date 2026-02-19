from collections.abc import Coroutine
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from padwan_llm.__main__ import _run, main


def _close_coroutine(coro: Coroutine) -> None:  # type: ignore[type-arg]
    """Close a coroutine to avoid 'was never awaited' warnings."""
    coro.close()


async def _fake_stream(*_chunks: str):
    for c in _chunks:
        yield c


async def test_run(capsys: pytest.CaptureFixture[str]):
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream_chat.return_value = _fake_stream("Hello", " world")

    with patch("padwan_llm.__main__.LLMClient", return_value=mock_client):
        await _run("gpt-4o-mini", "hi")

    captured = capsys.readouterr()
    assert "Hello world" in captured.out


@pytest.mark.parametrize(
    "side_effect, exit_code",
    [
        pytest.param(None, 0, id="success"),
        pytest.param(RuntimeError("boom"), 1, id="error"),
    ],
)
def test_main(
    side_effect: Exception | None,
    exit_code: int,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.argv", ["padwan-llm", "hello"])
    with patch(
        "padwan_llm.__main__.asyncio.run",
        side_effect=side_effect or _close_coroutine,
    ) as mock_run:
        if exit_code:
            with pytest.raises(SystemExit, match=str(exit_code)):
                main()
        else:
            main()
        mock_run.assert_called_once()
