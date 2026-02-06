from __future__ import annotations

import argparse
import asyncio
import sys

from .client import LLMClient
from .conversation import Message

_CLI_COMMANDS = frozenset(("chat", "batch", "models", "info"))


async def _run(model: str, prompt: str) -> None:
    client = LLMClient(model=model)
    messages: list[Message] = [Message(role="user", content=prompt)]
    async with client:
        async for chunk in client.stream_chat(messages):
            sys.stdout.write(chunk)
            sys.stdout.flush()
    print()


def _has_subcommand() -> bool:
    return any(arg in _CLI_COMMANDS for arg in sys.argv[1:] if not arg.startswith("-"))


def main() -> None:
    from .cli import cli as _cli, main as cli_main

    if _cli is not None:
        cli_main()
        return

    if _has_subcommand():
        print("CLI mode requires additional dependencies.")
        print("Install with: pip install polarsen-llm[cli]  or  uv sync --group cli")
        raise SystemExit(1)

    # Fallback to simple one-shot mode when CLI is not available
    parser = argparse.ArgumentParser(
        prog="polarsen-llm",
        description="One-shot LLM query (CLI mode not installed)",
    )
    parser.add_argument("prompt", help="Prompt to send")
    parser.add_argument("-m", "--model", default="gpt-4o-mini", help="Model to use")
    args = parser.parse_args()

    try:
        asyncio.run(_run(args.model, args.prompt))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
