__all__ = ("supports_vision",)


def supports_vision(model: str) -> bool:
    """Whether this Mistral model accepts image input: only the pixtral line does."""
    return model.startswith("pixtral-")
