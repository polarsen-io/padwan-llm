from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from src.gemini import BatchJob

console = Console()


def format_job(job: BatchJob) -> None:
    """Pretty print a batch job with colored state and key fields."""
    from ..widgets import STATE_COLORS

    state_color = STATE_COLORS.get(job.state, "white")
    console.print(f"  Name:    {job.name}")
    console.print(f"  State:   [{state_color}]{job.state}[/{state_color}]")
    if job.display_name:
        console.print(f"  Display: {job.display_name}")
    if job.model:
        console.print(f"  Model:   {job.model}")
    if job.create_time:
        console.print(f"  Created: {job.create_time}")
    if job.stats:
        stats = job.stats
        succeeded = stats.get("successfulRequestCount", 0)
        total = stats.get("requestCount", 0)
        console.print(f"  Stats:   {succeeded}/{total} succeeded")
    if job.error:
        console.print(f"  [red]Error:   {job.error}[/red]")


def format_job_table(jobs: list[BatchJob]) -> Table:
    """Create a rich table displaying batch jobs with state colors and stats."""
    from ..widgets import STATE_COLORS

    table = Table(title="Batch Jobs")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("State", style="bold")
    table.add_column("Display Name")
    table.add_column("Model")
    table.add_column("Stats", justify="right")

    for job in jobs:
        state_color = STATE_COLORS.get(job.state, "white")
        stats_str = ""
        if job.stats:
            succeeded = job.stats.get("successfulRequestCount", 0)
            total = job.stats.get("requestCount", 0)
            stats_str = f"{succeeded}/{total}"

        table.add_row(
            job.name.split("/")[-1],  # Just the job ID
            f"[{state_color}]{job.state.replace('JOB_STATE_', '')}[/{state_color}]",
            job.display_name or "",
            job.model or "",
            stats_str,
        )

    return table
