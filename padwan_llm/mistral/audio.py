from ..content import AudioFormat

__all__ = ("AUDIO_FORMATS", "supports_audio")

# Verified against the chat API: voxtral sniffs and accepts these; m4a/aac are
# rejected with invalid_request_audio.
AUDIO_FORMATS: frozenset[AudioFormat] = frozenset({"wav", "mp3", "flac", "ogg"})


def supports_audio(model: str, fmt: AudioFormat | None = None) -> bool:
    """Whether this Mistral model accepts audio input (in `fmt` when given):
    only the voxtral line does."""
    if not model.startswith("voxtral-"):
        return False
    return fmt is None or fmt in AUDIO_FORMATS
