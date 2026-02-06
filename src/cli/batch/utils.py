from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

from ...gemini import BatchRequest, BatchResult

if TYPE_CHECKING:
    from typing import Any


def load_prompts_from_file(file_path: str) -> list[BatchRequest]:
    """Load prompts from a JSON array or newline-delimited text file.

    For JSON files, accepts either a simple string array or an array of objects
    with 'prompt' and optional 'key' fields. For text files, each line becomes
    a separate prompt.
    """
    path = Path(file_path)
    content = path.read_text()

    if path.suffix.lower() == ".json":
        data: list[str | dict[str, Any]] = json.loads(content)
        requests: list[BatchRequest] = []
        for i, item in enumerate(data):
            if isinstance(item, str):
                requests.append(
                    BatchRequest(
                        contents=[{"role": "user", "parts": [{"text": item}]}],
                        key=f"prompt-{i}",
                    )
                )
            else:
                prompt = item.get("prompt", "")
                key = item.get("key", f"prompt-{i}")
                requests.append(
                    BatchRequest(
                        contents=[{"role": "user", "parts": [{"text": prompt}]}],
                        key=key,
                    )
                )
        return requests

    # Text file: one prompt per line
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return [
        BatchRequest(
            contents=[{"role": "user", "parts": [{"text": line}]}],
            key=f"prompt-{i}",
        )
        for i, line in enumerate(lines)
    ]


def save_results_to_file(results: list[BatchResult], path: str, fmt: str) -> None:
    """Export batch results in the specified format (json, csv, or txt)."""
    output = Path(path)

    if fmt == "json":
        data = [
            {
                "key": r.key,
                "content": r.content,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_tokens,
            }
            for r in results
        ]
        output.write_text(json.dumps(data, indent=2))

    elif fmt == "csv":
        with output.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["key", "content", "input_tokens", "output_tokens", "total_tokens"]
            )
            for r in results:
                writer.writerow(
                    [r.key, r.content, r.input_tokens, r.output_tokens, r.total_tokens]
                )

    else:  # txt
        lines = [f"[{r.key}]\n{r.content}\n" for r in results]
        output.write_text("\n".join(lines))
