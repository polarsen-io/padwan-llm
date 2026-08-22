__all__ = ("supports_audio",)


def supports_audio(model: str) -> bool:
    """Whether this Mistral model accepts audio input: only the voxtral line does."""
    return model.startswith("voxtral-")
