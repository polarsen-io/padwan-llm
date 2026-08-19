__all__ = ("supports_vision",)


def supports_vision(model: str) -> bool:
    """Whether this Grok model accepts image input: all current chat models do."""
    return True
