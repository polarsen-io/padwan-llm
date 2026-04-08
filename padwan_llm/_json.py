from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from json import loads as loads
else:
    try:
        from orjson import loads
    except ImportError:
        from json import loads

__all__ = ("loads",)
