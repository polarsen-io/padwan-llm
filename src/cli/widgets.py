from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from src.gemini import BatchJob, BatchResult

# Colors for chat messages
USER_COLOR = "#87ceeb"
ASSISTANT_COLOR = "#98fb98"

# Colors for batch job states
STATE_COLORS = {
    "JOB_STATE_PENDING": "#ffa500",
    "JOB_STATE_QUEUED": "#ffff00",
    "JOB_STATE_RUNNING": "#00bfff",
    "JOB_STATE_SUCCEEDED": "#98fb98",
    "JOB_STATE_FAILED": "#ff6b6b",
    "JOB_STATE_CANCELLED": "#808080",
    "JOB_STATE_EXPIRED": "#808080",
}


class UserMessage(Static):
    """A styled widget for displaying user messages."""

    def __init__(self, content: str, **kwargs) -> None:
        text = Text(content, style=USER_COLOR)
        super().__init__(text, **kwargs)


class StreamingMessage(Static):
    """A widget that displays streaming text content with assistant styling.

    Mount this widget then call append() to add text chunks as they arrive.
    """

    def __init__(self, initial: str = "", **kwargs) -> None:
        super().__init__(initial, **kwargs)
        self._content = initial
        self._update_styled()

    def _update_styled(self) -> None:
        """Update display with styled text."""
        self.update(Text(self._content, style=ASSISTANT_COLOR))

    def append(self, text: str) -> None:
        """Append text to the message and refresh display."""
        self._content += text
        self._update_styled()
        if (parent := self.parent) is not None:
            self.call_after_refresh(parent.scroll_end, animate=False)


class BatchProgressWidget(Static):
    """A widget that displays batch job progress with state coloring.

    Shows job name, current state (with color), progress bar showing
    succeeded/total requests, and elapsed time. Call update() to refresh
    with new job data.
    """

    def __init__(self, job: BatchJob, **kwargs) -> None:
        super().__init__(**kwargs)
        self._job = job
        self._state_history: list[str] = [job.state]
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Update the display with current job state."""
        job = self._job
        state_color = STATE_COLORS.get(job.state, "white")

        # Build progress info
        succeeded = 0
        total = 0
        if job.stats:
            succeeded = cast(int, job.stats.get("successfulRequestCount", 0))
            total = cast(int, job.stats.get("requestCount", 0))

        # Create progress bar
        bar_width = 30
        if total > 0:
            filled = int((succeeded / total) * bar_width)
        else:
            filled = 0
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

        # Build display text
        lines = [
            f"Job: {job.name.split('/')[-1]}",
            f"State: [{state_color}]{job.state}[/{state_color}]",
            f"Progress: [{bar}] {succeeded}/{total}",
        ]

        if job.display_name:
            lines.insert(1, f"Name: {job.display_name}")

        # Show state history if there were transitions
        if len(self._state_history) > 1:
            history = " -> ".join(
                s.replace("JOB_STATE_", "") for s in self._state_history
            )
            lines.append(f"History: {history}")

        text = Text("\n".join(lines))
        self.update(text)

    def update_job(self, job: BatchJob) -> None:
        """Update with new job data and refresh display."""
        if job.state != self._job.state:
            self._state_history.append(job.state)
        self._job = job
        self._refresh_display()


class BatchResultWidget(Static):
    """A widget that displays batch results in a styled format.

    Shows each result with its key, token counts, and content preview.
    Useful for viewing results in TUI mode.
    """

    def __init__(self, results: list[BatchResult], **kwargs) -> None:
        super().__init__(**kwargs)
        self._results = results
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Update the display with results."""
        lines: list[str] = []

        for result in self._results:
            # Header with key and token info
            lines.append(
                f"[cyan][{result.key}][/cyan] "
                f"[dim]({result.input_tokens} in, {result.output_tokens} out, "
                f"{result.total_tokens} total tokens)[/dim]"
            )

            # Content preview (truncated)
            preview = result.content[:300].replace("\n", " ")
            if len(result.content) > 300:
                preview += "..."
            lines.append(f"  {preview}")
            lines.append("")

        text = Text.from_markup("\n".join(lines))
        self.update(text)
