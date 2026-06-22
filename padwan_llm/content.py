import base64
import mimetypes
from pathlib import Path
from typing import Literal, TypedDict

__all__ = (
    "ContentImagePart",
    "ContentPart",
    "ContentTextPart",
    "ImageUrl",
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
