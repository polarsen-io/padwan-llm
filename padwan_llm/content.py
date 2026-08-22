import base64
import mimetypes
from pathlib import Path
from typing import Literal, TypedDict

__all__ = (
    "AudioFormat",
    "ContentAudioPart",
    "ContentImagePart",
    "ContentPart",
    "ContentTextPart",
    "ImageUrl",
    "InputAudio",
    "audio_part",
    "content_parts",
    "image_part",
    "text_file_part",
    "text_part",
)


class ContentTextPart(TypedDict):
    """A text segment of a multimodal user message (OpenAI content-part shape)."""

    type: Literal["text"]
    text: str


class ImageUrl(TypedDict):
    """The `image_url` payload of an image content part."""

    url: str


class ContentImagePart(TypedDict):
    """An image segment carrying a `data:` URL (OpenAI content-part shape)."""

    type: Literal["image_url"]
    image_url: ImageUrl


AudioFormat = Literal["wav", "mp3", "flac", "ogg", "aac", "aiff", "m4a"]


class InputAudio(TypedDict):
    """The `input_audio` payload of an audio content part."""

    data: str
    format: AudioFormat


class ContentAudioPart(TypedDict):
    """An audio segment carrying base64 data (OpenAI content-part shape)."""

    type: Literal["input_audio"]
    input_audio: InputAudio


ContentPart = ContentTextPart | ContentImagePart | ContentAudioPart

# Audio MIME types mapped to the wire formats the content-part shape allows.
# Provider support varies (see supports_audio); this is the builder-side union.
_AUDIO_FORMATS: dict[str, AudioFormat] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/ogg": "ogg",
    "audio/aac": "aac",
    "audio/aiff": "aiff",
    "audio/x-aiff": "aiff",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
}


def text_part(text: str) -> ContentTextPart:
    """Wrap plain text as a text content part."""
    return {"type": "text", "text": text}


def image_part(path: str | Path, *, mime: str | None = None) -> ContentImagePart:
    """Read an image file into a base64 `data:` URL image content part.

    The MIME type is guessed from the file name when not given, falling back to
    image/png so providers still receive a usable data URL.
    """
    path = Path(path)
    resolved = mime or mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{resolved};base64,{data}"}}


def audio_part(path: str | Path, *, fmt: AudioFormat | None = None) -> ContentAudioPart:
    """Read an audio file into a base64 audio content part.

    The format is guessed from the file extension when not given; an
    unrecognised extension raises ``ValueError``. Provider support varies by
    format; check with ``supports_audio(model, fmt)``.
    """
    path = Path(path)
    if fmt is None:
        mime = mimetypes.guess_type(path.name)[0]
        if mime is None or (fmt := _AUDIO_FORMATS.get(mime)) is None:
            supported = "/".join(dict.fromkeys(_AUDIO_FORMATS.values()))
            raise ValueError(
                f"Unsupported audio format for {path.name!r}: expected "
                f"{supported} (pass fmt= to override)"
            )
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}


def text_file_part(path: str | Path, *, encoding: str = "utf-8") -> ContentTextPart:
    """Read a text file and wrap its contents in a labelled text content part."""
    path = Path(path)
    return {"type": "text", "text": f"--- {path.name} ---\n{path.read_text(encoding)}"}


def content_parts(*items: str | Path | ContentPart) -> list[ContentPart]:
    """Build content parts with type inference.

    Plain strings become text parts (never treated as paths), so message text
    that mentions a filename is safe. ``Path`` items are read from disk: an
    image MIME type (by extension) yields an image part, an audio MIME type an
    audio part, anything else is inlined as a labelled text file part.
    Ready-made part dicts pass through.
    """
    parts: list[ContentPart] = []
    for item in items:
        if isinstance(item, Path):
            mime = mimetypes.guess_type(item.name)[0]
            if mime and mime.startswith("image/"):
                parts.append(image_part(item, mime=mime))
            elif mime and mime.startswith("audio/"):
                # audio_part raises a clear ValueError on unknown formats,
                # instead of falling through to a binary UTF-8 decode error.
                parts.append(audio_part(item))
            else:
                parts.append(text_file_part(item))
        elif isinstance(item, str):
            parts.append(text_part(item))
        else:
            parts.append(item)
    return parts
