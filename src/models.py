from typing import NotRequired, TypedDict

__all__ = ("UsageToken",)


class UsageToken(TypedDict):
    total: int
    input: int
    output: int
    cached: NotRequired[int]
