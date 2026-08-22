__all__ = ("supports_audio",)


def supports_audio(model: str) -> bool:
    """Whether this Gemini model accepts audio input: all current chat models do."""
    return True
