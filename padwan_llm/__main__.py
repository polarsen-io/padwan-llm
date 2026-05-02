import argparse
import asyncio
import json
import sys
from typing import Any

from ._base import OnThought
from .client import LLMClient
from .conversation import Message


def _make_thought_streamer() -> OnThought:
    def on_thought(chunk: str) -> None:
        sys.stderr.write(chunk)
        sys.stderr.flush()

    return on_thought


async def _run(
    model: str,
    prompt: str,
    base_url: str | None = None,
    extra_params: dict[str, Any] | None = None,
    stream_thinking: bool = False,
    debug: bool = False,
    pretty: bool = False,
) -> None:
    on_thought: OnThought | None = _make_thought_streamer() if stream_thinking else None
    client = LLMClient(model=model, base_url=base_url, on_thought=on_thought)
    messages: list[Message] = [Message(role="user", content=prompt)]
    async with client:
        if debug:
            body: dict[str, Any] = {
                "model": model,
                "messages": list(messages),
                "temperature": client.temperature,
                "stream_options": {"include_usage": True},
            }
            if extra_params:
                body.update(extra_params)
            async for raw in client.stream(body):
                sys.stderr.write(json.dumps(raw, indent=2 if pretty else None) + "\n")
                sys.stderr.flush()
                if choices := raw.get("choices"):
                    delta = choices[0].get("delta") or {}
                    if text := delta.get("content"):
                        sys.stdout.write(text)
                        sys.stdout.flush()
        else:
            stream = client.stream_chat(messages, extra_params=extra_params)
            async for chunk in stream:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            if reason := getattr(stream, "finish_reason", None):
                if reason != "stop":
                    sys.stderr.write(f"\n[finish_reason: {reason}]\n")
                    sys.stderr.flush()
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="padwan-llm",
        description="One-shot LLM query",
    )
    parser.add_argument("prompt", help="Prompt to send")
    parser.add_argument("-m", "--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument(
        "--base-url", default=None, help="Custom base URL for the LLM provider"
    )
    parser.add_argument(
        "--extra-params",
        default=None,
        metavar="JSON",
        help="Extra JSON parameters merged into the request body (e.g. '{\"temperature\": 0}')",
    )
    parser.add_argument(
        "--stream-thinking",
        action="store_true",
        default=False,
        help="Stream model reasoning/thinking tokens to stderr",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print raw SSE chunks as JSON to stderr",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=False,
        help="Pretty-print --debug JSON output (implies --debug)",
    )
    args = parser.parse_args()

    extra_params: dict[str, Any] | None = None
    if args.extra_params is not None:
        try:
            extra_params = json.loads(args.extra_params)
        except json.JSONDecodeError as e:
            print(f"Error: --extra-params is not valid JSON: {e}", file=sys.stderr)
            raise SystemExit(1)
        if not isinstance(extra_params, dict):
            print("Error: --extra-params must be a JSON object", file=sys.stderr)
            raise SystemExit(1)

    try:
        asyncio.run(
            _run(
                args.model,
                args.prompt,
                args.base_url,
                extra_params,
                args.stream_thinking,
                args.debug or args.pretty,
                args.pretty,
            )
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
