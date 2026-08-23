from ..content import AudioFormat

__all__ = ("supports_audio",)


def supports_audio(model: str, fmt: AudioFormat | None = None) -> bool:
    """Whether this Grok model accepts audio input: no current chat model does."""
    return False
