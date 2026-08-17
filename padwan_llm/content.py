import base64
import mimetypes
from pathlib import Path
from typing import Literal, TypedDict

__all__ = (
    "ContentImagePart",
    "ContentPart",
    "ContentTextPart",
    "ImageUrl",
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


ContentPart = ContentTextPart | ContentImagePart


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


def text_file_part(path: str | Path, *, encoding: str = "utf-8") -> ContentTextPart:
    """Read a text file and wrap its contents in a labelled text content part."""
    path = Path(path)
    return {"type": "text", "text": f"--- {path.name} ---\n{path.read_text(encoding)}"}


def content_parts(*items: str | Path | ContentPart) -> list[ContentPart]:
    """Build content parts with type inference.

    Plain strings become text parts — never treated as paths, so message text
    that mentions a filename is safe. ``Path`` items are read from disk: an
    image MIME type (by extension) yields an image part, anything else is
    inlined as a labelled text file part. Ready-made part dicts pass through.
    """
    parts: list[ContentPart] = []
    for item in items:
        if isinstance(item, Path):
            mime = mimetypes.guess_type(item.name)[0]
            if mime and mime.startswith("image/"):
                parts.append(image_part(item, mime=mime))
            else:
                parts.append(text_file_part(item))
        elif isinstance(item, str):
            parts.append(text_part(item))
        else:
            parts.append(item)
    return parts
