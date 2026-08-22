__all__ = ("supports_audio",)


def supports_audio(model: str) -> bool:
    """Whether this Grok model accepts audio input: no current chat model does."""
    return False
