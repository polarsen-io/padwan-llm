from typing import NotRequired, TypedDict

__all__ = ("UsageToken",)


class UsageToken(TypedDict):
    """Token usage information for LLM interactions."""

    total: int
    input: int
    output: int
    cached: NotRequired[int]
