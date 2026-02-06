from __future__ import annotations

try:
    from .run import cli, main, main_tui
except ImportError:

    def main() -> None:
        """Entry point when piou is not installed."""
        print("CLI mode requires additional dependencies.")
        print("Install with: pip install polarsen-llm[cli]  or  uv sync --group cli")
        raise SystemExit(1)

    main_tui = main
    cli = None

__all__ = ("cli", "main", "main_tui")
