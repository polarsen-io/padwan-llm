from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from json import dumps, loads
else:
    try:
        import orjson as _orjson

        loads = _orjson.loads

        def dumps(obj: Any) -> str:
            return _orjson.dumps(obj).decode()
    except ImportError:
        from json import dumps, loads

__all__ = ("dumps", "loads")
