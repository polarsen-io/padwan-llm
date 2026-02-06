from __future__ import annotations

import sys

from piou import Cli, Option

from rich.console import Console
from rich.table import Table

from .. import (
    GEMINI_MODELS,
    GROK_MODELS,
    MISTRAL_MODELS,
    OPENAI_MODELS,
    LLMClient,
)
from ..conversation import Message
from ..errors import Provider

from .batch import batch_group
from .chat import chat_group

console = Console()

CUSTOM_CSS = """
Rule.chat-mode {
    color: #87ceeb;
}
"""

cli = Cli(description="Polarsen LLM - Test CLI for the unified LLM client library")

ALL_MODELS = sorted([*OPENAI_MODELS, *GEMINI_MODELS, *MISTRAL_MODELS, *GROK_MODELS])

cli.add_command_group(batch_group)
cli.add_command_group(chat_group)


@cli.command("models", help="List available models")
def list_models(
    provider: Provider | None = Option(
        None, "-p", "--provider", help="Filter by provider"
    ),
) -> None:
    """List all available models, optionally filtered by provider."""
    provider_models: dict[str, list[str]] = {}
    if not provider or provider == "openai":
        provider_models["OpenAI"] = list(OPENAI_MODELS)
    if not provider or provider == "gemini":
        provider_models["Gemini"] = list(GEMINI_MODELS)
    if not provider or provider == "mistral":
        provider_models["Mistral"] = list(MISTRAL_MODELS)
    if not provider or provider == "grok":
        provider_models["Grok"] = list(GROK_MODELS)

    if not provider_models:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        return

    table = Table(title="Available Models")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="green")

    for prov, models in provider_models.items():
        for i, model in enumerate(sorted(models)):
            table.add_row(prov if i == 0 else "", model)

    console.print(table)


@cli.command("info", help="Show library information")
def info() -> None:
    """Show library information."""
    table = Table(title="Polarsen LLM Library")
    table.add_column("Provider", style="cyan")
    table.add_column("Models", style="green", justify="right")

    table.add_row("OpenAI", str(len(OPENAI_MODELS)))
    table.add_row("Gemini", str(len(GEMINI_MODELS)))
    table.add_row("Mistral", str(len(MISTRAL_MODELS)))
    table.add_row("Grok", str(len(GROK_MODELS)))

    total = (
        len(OPENAI_MODELS) + len(GEMINI_MODELS) + len(MISTRAL_MODELS) + len(GROK_MODELS)
    )
    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")

    console.print(table)


@cli.main(help="One-shot LLM query")
async def oneshot(
    prompt: str = Option(..., help="Prompt to send"),
    model: str = Option("gpt-4o-mini", "-m", "--model", help="Model to use"),
    stream: bool = Option(False, "--stream", "-s", help="Stream output as it arrives"),
) -> None:
    """Send a single prompt and print the response."""
    client = LLMClient(model=model)
    messages: list[Message] = [Message(role="user", content=prompt)]
    async with client:
        if stream:
            async for chunk in client.stream_chat(messages):
                sys.stdout.write(chunk)
                sys.stdout.flush()
        else:
            text, _ = await client.complete_chat(messages)
            print(text)
    print()


def main() -> None:
    """Entry point for the CLI."""
    cli.run(css=CUSTOM_CSS)


def main_tui() -> None:
    """Entry point for the TUI CLI."""
    cli.tui = True
    main()


if __name__ == "__main__":
    main()
