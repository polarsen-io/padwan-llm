from ..content import AudioFormat

__all__ = ("supports_audio",)


def supports_audio(model: str, fmt: AudioFormat | None = None) -> bool:
    """Whether this Claude model accepts audio input: the Messages API has none."""
    return False
