__all__ = ("supports_audio",)


def supports_audio(model: str) -> bool:
    """Whether this OpenAI model accepts audio input: only the audio-tier chat models
    (gpt-audio and the *-audio-preview variants) do."""
    return "audio" in model
