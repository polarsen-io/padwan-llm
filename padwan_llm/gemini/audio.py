from ..content import AudioFormat

__all__ = ("AUDIO_FORMATS", "supports_audio")

# Gemini accepts every format the content-part shape allows.
AUDIO_FORMATS: frozenset[AudioFormat] = frozenset(
    {"wav", "mp3", "flac", "ogg", "aac", "aiff", "m4a"}
)


def supports_audio(model: str, fmt: AudioFormat | None = None) -> bool:
    """Whether this Gemini model accepts audio input (in `fmt` when given): all
    current chat models accept every supported format."""
    return fmt is None or fmt in AUDIO_FORMATS
