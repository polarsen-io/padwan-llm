import base64

import pytest

from padwan_llm.content import (
    ContentImagePart,
    content_parts,
    image_part,
    text_file_part,
    text_part,
)
from padwan_llm.conversation import ConversationState
from padwan_llm.gemini.client import _content_to_gemini_parts
from padwan_llm.vision import supports_vision


def test_text_part():
    assert text_part("hello") == {"type": "text", "text": "hello"}


@pytest.mark.parametrize(
    "name, mime_arg, expected_mime",
    [
        pytest.param("shot.png", None, "image/png", id="guess-png"),
        pytest.param("pic.jpg", None, "image/jpeg", id="guess-jpeg"),
        pytest.param("blob.unknownext", None, "image/png", id="fallback"),
        pytest.param("shot.png", "image/webp", "image/webp", id="explicit-override"),
    ],
)
def test_image_part(tmp_path, name, mime_arg, expected_mime):
    raw = b"\x89PNG\r\n\x1a\n binary bytes"
    path = tmp_path / name
    path.write_bytes(raw)

    part = image_part(path, mime=mime_arg)

    assert part["type"] == "image_url"
    prefix, _, data = part["image_url"]["url"].partition(",")
    assert prefix == f"data:{expected_mime};base64"
    assert base64.b64decode(data) == raw


def test_text_file_part(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Title\nbody")

    assert text_file_part(path) == {
        "type": "text",
        "text": "--- notes.md ---\n# Title\nbody",
    }


def test_content_parts_inference(tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG fake")
    notes = tmp_path / "notes.md"
    notes.write_text("body")
    ready: ContentImagePart = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }

    parts = content_parts("mentions shot.png", img, notes, ready)

    assert parts[0] == {"type": "text", "text": "mentions shot.png"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[2] == {"type": "text", "text": "--- notes.md ---\nbody"}
    assert parts[3] is ready


def test_add_user_message_accepts_content_parts():
    state = ConversationState()
    parts = [
        text_part("look"),
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]

    msg = state.add_user_message(parts)

    assert msg["role"] == "user"
    assert msg["content"] == parts
    assert state.messages[-1] is msg


@pytest.mark.parametrize(
    "model, expected",
    [
        pytest.param("gpt-4o", True, id="openai-4o"),
        pytest.param("gpt-4o-mini", True, id="openai-4o-mini"),
        pytest.param("gpt-4.1", True, id="openai-4.1"),
        pytest.param("o3", True, id="openai-o3"),
        pytest.param("o4-mini", True, id="openai-o4-mini"),
        pytest.param("gpt-4", False, id="openai-base-gpt4"),
        pytest.param("o1-mini", False, id="openai-o1-mini"),
        pytest.param("o3-mini", False, id="openai-o3-mini"),
        pytest.param("codex-mini-latest", False, id="openai-codex"),
        pytest.param("gpt-oss-120b", False, id="openai-gpt-oss"),
        pytest.param("gpt-oss-20b", False, id="openai-gpt-oss-small"),
        pytest.param("gpt-3.5-turbo", False, id="openai-gpt-3.5"),
        pytest.param("gemini-2.5-flash", True, id="gemini"),
        pytest.param("pixtral-large-latest", True, id="mistral-pixtral"),
        pytest.param("mistral-large-latest", False, id="mistral-text"),
        pytest.param("ministral-8b-latest", False, id="mistral-ministral"),
        pytest.param("grok-4", True, id="grok-default"),
        pytest.param("some-local-model", True, id="unknown-attempt"),
        pytest.param(None, False, id="none"),
        pytest.param("", False, id="empty"),
    ],
)
def test_supports_vision(model, expected):
    assert supports_vision(model) is expected


def test_content_to_gemini_parts_text():
    assert _content_to_gemini_parts("hi") == [{"text": "hi"}]


def test_content_to_gemini_parts_multimodal():
    content = [
        text_part("what is this?"),
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
    ]

    parts = _content_to_gemini_parts(content)

    assert parts == [
        {"text": "what is this?"},
        {"inlineData": {"mimeType": "image/jpeg", "data": "QUJD"}},
    ]
