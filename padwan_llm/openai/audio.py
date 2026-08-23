from ..content import AudioFormat

__all__ = ("AUDIO_FORMATS", "supports_audio")

# The chat completions input_audio schema only admits wav and mp3.
AUDIO_FORMATS: frozenset[AudioFormat] = frozenset({"wav", "mp3"})


def supports_audio(model: str, fmt: AudioFormat | None = None) -> bool:
    """Whether this OpenAI model accepts audio input (in `fmt` when given): only
    the audio-tier chat models (gpt-audio and the *-audio-preview variants) do."""
    if "audio" not in model:
        return False
    return fmt is None or fmt in AUDIO_FORMATS
