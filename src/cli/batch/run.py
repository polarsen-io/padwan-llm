from __future__ import annotations

import asyncio
import pprint
from pathlib import Path

from piou import CommandGroup, Option
from piou.tui import TuiContext, TuiOption

from ...gemini import BatchRequest, BatchResult, GeminiClient
from ..widgets import BatchProgressWidget, BatchResultWidget
from .format import console, format_job, format_job_table
from .utils import load_prompts_from_file, save_results_to_file

batch_group = CommandGroup(name="batch", help="Gemini batch operations")


@batch_group.command("create", help="Create a batch job with prompts")
async def batch_create(
    prompts: list[str] | None = Option(
        None, "-p", "--prompt", help="Inline prompts (multiple)"
    ),
    file: str | None = Option(
        None, "-f", "--file", help="Read prompts from JSON/text file"
    ),
    output: str | None = Option(None, "-o", "--output", help="Output file for results"),
    model: str = Option("gemini-2.0-flash", "-m", "--model", help="Model to use"),
    name: str = Option("cli-batch", "-n", "--name", help="Display name for the batch"),
) -> None:
    """Create a batch job with inline prompts or from a file.

    Prompts can be specified inline with -p (multiple times) or loaded from
    a file with -f. The file can be JSON (array of strings or objects with
    'prompt' and 'key' fields) or plain text (one prompt per line).
    """
    requests: list[BatchRequest] = []

    if file:
        requests = load_prompts_from_file(file)
    elif prompts:
        requests = [
            BatchRequest(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                key=f"prompt-{i}",
            )
            for i, prompt in enumerate(prompts)
        ]

    if not requests:
        console.print(
            "[red]No prompts provided. Use -p or -f to specify prompts.[/red]"
        )
        return

    async with GeminiClient(model=model) as client:  # pyright: ignore[reportArgumentType]
        console.print(f"Creating batch with {len(requests)} requests...")
        job = await client.create_batch(requests, display_name=name)
        console.print("[green]Batch created:[/green]")
        format_job(job)

        if output:
            console.print(f"Results will be saved to: {output}")


@batch_group.command("status", help="Get status of a batch job")
async def batch_status(
    job_name: str = Option(..., "-j", "--job", help="Batch job name"),
    show_results: bool = Option(
        False, "-r", "--results", help="Show results if completed"
    ),
    ctx: TuiContext = TuiOption(),
) -> None:
    """Get the current status of a batch job with optional results display."""
    async with GeminiClient() as client:
        job = await client.get_batch(job_name)
        console.print("[bold]Batch status:[/bold]")
        format_job(job)

        if show_results and job.succeeded and job.inlined_responses:
            console.print("\n[bold]Results:[/bold]")
            results = [
                BatchResult.from_inlined_response(r) for r in job.inlined_responses
            ]
            ctx.mount_widget(BatchResultWidget(results))


@batch_group.command("list", help="List batch jobs")
async def batch_list(
    limit: int = Option(10, "-l", "--limit", help="Maximum jobs to show"),
    ctx: TuiContext = TuiOption(),
) -> None:
    """List recent batch jobs with their status."""
    async with GeminiClient() as client:
        jobs, next_token = await client.list_batches(page_size=limit)
        pprint.pprint(jobs)

        if not jobs:
            console.print("No batch jobs found.")
            return

        table = format_job_table(jobs)
        console.print(table)

        if next_token:
            console.print(
                f"\n[dim]More jobs available (next_token: {next_token})[/dim]"
            )


@batch_group.command("cancel", help="Cancel a batch job")
async def batch_cancel(
    job_name: str = Option(..., "-j", "--job", help="Batch job name to cancel"),
) -> None:
    """Cancel a pending or running batch job."""
    async with GeminiClient() as client:
        console.print(f"Cancelling batch job: {job_name}")
        await client.cancel_batch(job_name)
        console.print("[green]Batch job cancelled.[/green]")


@batch_group.command("poll", help="Poll a batch job until completion")
async def batch_poll(
    job_name: str = Option(..., "-j", "--job", help="Batch job name to poll"),
    interval: int = Option(5, "-i", "--interval", help="Poll interval in seconds"),
    timeout: int | None = Option(
        None, "-t", "--timeout", help="Max wait time in seconds"
    ),
    show_results: bool = Option(
        True, "-r", "--results", help="Show results on completion"
    ),
    output: str | None = Option(None, "-o", "--output", help="Save results to file"),
    ctx: TuiContext = TuiOption(),
) -> None:
    """Poll a batch job until it reaches a terminal state.

    Displays a progress widget showing the job state and stats.
    Can optionally save results to a file when complete.
    """
    async with GeminiClient() as client:
        console.print(f"Polling batch job: {job_name} (every {interval}s)")
        job = await client.get_batch(job_name)

        widget = BatchProgressWidget(job)
        ctx.mount_widget(widget)

        elapsed = 0
        while not job.is_terminal:
            if timeout and elapsed >= timeout:
                console.print(f"[yellow]Timeout reached after {elapsed}s[/yellow]")
                break

            await asyncio.sleep(interval)
            elapsed += interval
            job = await client.get_batch(job_name)
            widget.update_job(job)

        console.print("\n[bold]Batch job completed:[/bold]")
        format_job(job)

        if job.succeeded and job.inlined_responses:
            results = [
                BatchResult.from_inlined_response(r) for r in job.inlined_responses
            ]

            if output:
                fmt = Path(output).suffix.lstrip(".") or "json"
                if fmt not in ("json", "csv", "txt"):
                    fmt = "json"
                save_results_to_file(results, output, fmt)
                console.print(f"[green]Results saved to: {output}[/green]")

            if show_results:
                console.print("\n[bold]Results:[/bold]")
                ctx.mount_widget(BatchResultWidget(results))


@batch_group.command("retry", help="Retry failed requests from a batch")
async def batch_retry(
    job_name: str = Option(..., "-j", "--job", help="Original job name"),
) -> None:
    """Create a new batch with only the failed requests from an original job.

    Extracts failed request keys from the original job and creates a new
    batch job to retry just those requests.
    """
    async with GeminiClient() as client:
        original = await client.get_batch(job_name)

        if not original.is_terminal:
            console.print("[red]Original job has not completed yet.[/red]")
            return

        if not original.inlined_responses:
            console.print("[red]No responses found in original job.[/red]")
            return

        # Find failed requests by checking for missing or error responses
        failed_keys: list[str] = []
        all_keys: set[str] = set()

        for resp in original.inlined_responses:
            key = resp.get("key", "")
            all_keys.add(key)
            response = resp.get("response", {})
            # Check if response has an error or no candidates
            if response.get("error") or not response.get("candidates"):
                failed_keys.append(key)

        if not failed_keys:
            console.print("[green]No failed requests to retry.[/green]")
            return

        console.print(
            f"Found {len(failed_keys)} failed requests out of {len(all_keys)}"
        )

        # For now, we can't easily recreate the original requests without
        # storing them. This command would need the original prompts.
        console.print(
            "[yellow]Note: Retry requires original prompts to be available.[/yellow]"
        )
        console.print("Failed keys: " + ", ".join(failed_keys))


@batch_group.command("export", help="Export batch results to a file")
async def batch_export(
    job_name: str = Option(..., "-j", "--job", help="Batch job name"),
    output: str = Option(..., "-o", "--output", help="Output file path"),
    fmt: str = Option("json", "-f", "--format", help="Format: json, csv, txt"),
) -> None:
    """Export completed batch results to a file.

    Supports JSON (structured with all fields), CSV (tabular), and TXT
    (key-content pairs) formats.
    """
    if fmt not in ("json", "csv", "txt"):
        console.print(f"[red]Unknown format: {fmt}. Use json, csv, or txt.[/red]")
        return

    async with GeminiClient() as client:
        job = await client.get_batch(job_name)

        if not job.succeeded:
            console.print(f"[red]Job not successful. State: {job.state}[/red]")
            return

        if not job.inlined_responses:
            console.print("[red]No results found in job.[/red]")
            return

        results = [BatchResult.from_inlined_response(r) for r in job.inlined_responses]
        save_results_to_file(results, output, fmt)

        console.print(f"[green]Exported {len(results)} results to: {output}[/green]")
