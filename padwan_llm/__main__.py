from __future__ import annotations

import argparse
import asyncio
import sys

from .client import LLMClient
from .conversation import Message


async def _run(model: str, prompt: str) -> None:
    client = LLMClient(model=model)
    messages: list[Message] = [Message(role="user", content=prompt)]
    async with client:
        async for chunk in client.stream_chat(messages):
            sys.stdout.write(chunk)
            sys.stdout.flush()
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="padwan-llm",
        description="One-shot LLM query",
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
