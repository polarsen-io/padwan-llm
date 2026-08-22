__all__ = ("supports_audio",)


def supports_audio(model: str) -> bool:
    """Whether this Claude model accepts audio input: the Messages API has none."""
    return False
