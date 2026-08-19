__all__ = ("supports_vision",)

# Known OpenAI models that do NOT accept image input. Everything else that looks
# like an OpenAI model is assumed multimodal (gpt-4o/4.1/5 families, gpt-4-turbo,
# o1, o3, o4-mini all take images).
_TEXT_ONLY = frozenset(
    {"gpt-4", "o1-mini", "o1-preview", "o3-mini", "codex-mini-latest"}
)
# Text-only families, matched by prefix because they ship many size/date
# variants: open-weight gpt-oss (gpt-oss-20b/120b) and legacy gpt-3.5.
_TEXT_ONLY_PREFIXES = ("gpt-oss", "gpt-3.5", "gpt-35")


def supports_vision(model: str) -> bool:
    """Whether this OpenAI model accepts image input; curated, not authoritative."""
    if model.startswith(_TEXT_ONLY_PREFIXES):
        return False
    return model not in _TEXT_ONLY
