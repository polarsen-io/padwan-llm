#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openai>=2.36.0",
#     "piou>=0.34.1",
#     "padwan-llm",
# ]
#
# [tool.uv.sources]
# padwan-llm = { path = "../../", editable = false }
# ///
"""Detect drift between provider model lists and the project's model Literals.

OpenAI uses ``openai.types.ChatModel`` from the installed SDK as the deterministic
source of truth, because it ships a static Literal. Gemini, Mistral, and xAI do
not expose equivalent SDK Literals, so this script optionally calls their live
model-list endpoints when API keys are present in the environment.

Always exits 0. The workflow decides whether to open a PR based on ``git status``
and the generated report.
"""

import os
import re
import typing
from dataclasses import dataclass, field
from typing import Any

import niquests
from openai.types import ChatModel
from piou import Cli, MaybePath, Option

from padwan_llm.gemini.client import GEMINI_MODELS
from padwan_llm.grok.client import GROK_MODELS
from padwan_llm.mistral.client import (
    MISTRAL_MODELS,
    MistralAudioModel,
    MistralEmbeddingModel,
)
from padwan_llm.openai.client import OPENAI_MODELS, _OPENAI_PREFIXES

# Trailing date or dated-preview stamps used by upstream to expose pinned model
# versions. The project prefers stable aliases such as "*-latest" where possible.
_DATE_SUFFIX = re.compile(
    r"-(?:\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|\d{4,6})(?:-preview)?$"
)
_VERSION_SEGMENT = re.compile(r"-\d{4,6}(?:-|$)")
_INTERNAL_SEGMENT = re.compile(r"-[a-z]\d")

# Capability families and deprecated lines not tracked in OpenAIModel.
# TODO: drop entries here as their capability Literal lands (e.g. OpenAIEmbeddingModel).
_OPENAI_BLACKLIST = (
    "embedding",
    "whisper",
    "tts",
    "dall-e",
    "davinci",
    "babbage",
    "moderation",
    "realtime",
    "transcribe",
    "audio",
    "search",
    "image",
    "computer-use",
    "vision",
    "3.5-turbo",
    "-32k",
    "chatgpt-",
    "chat-latest",
    "turbo-preview",
)

# TODO: drop entries here as their capability Literal lands (e.g. GeminiEmbeddingModel, GeminiImageModel).
_GEMINI_BLACKLIST = (
    "embedding",
    "aqa",
    "imagen",
    "veo",
    "-tts",
    "-image-",
    "image-generation",
    "customtools",
    "robotics",
)
_GEMINI_OLD_PREFIXES = ("gemini-1.", "gemini-2.0-")

_MISTRAL_BLACKLIST = ("mistral-vibe-cli",)

_GROK_BLACKLIST = (
    "-beta",
    "experimental",
    "-gv2",
    "multi-agent",
)


@dataclass
class Diff:
    added: set[str]
    removed: set[str]

    @property
    def has_drift(self) -> bool:
        return bool(self.added or self.removed)


@dataclass
class RemoteModels:
    models: set[str]
    skipped: str | None = None
    error: str | None = None
    # model id/alias -> retirement date, when the provider exposes one.
    deprecations: dict[str, str] = field(default_factory=dict)


@dataclass
class LiveCheck:
    provider: str
    source: str
    target: str
    docs_url: str
    diff: Diff | None = None
    skipped: str | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()
    # tracked model -> retirement date, for models still served but scheduled.
    deprecations: dict[str, str] = field(default_factory=dict)


class FetchError(Exception):
    """Raised when a provider model-list request fails."""


def _json_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> Any:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "padwan-llm-model-drift/1.0",
    }
    if headers:
        request_headers.update(headers)
    try:
        return (
            niquests.get(
                url, headers=request_headers, params=params, timeout=20
            ).raise_for_status()
        ).json()
    except niquests.exceptions.HTTPError as e:
        resp = e.response
        if resp is None:
            raise FetchError(str(e)) from e
        raise FetchError(f"HTTP {resp.status_code} {resp.reason}") from e
    except niquests.exceptions.JSONDecodeError as e:
        raise FetchError("response was not valid JSON") from e
    except niquests.exceptions.RequestException as e:
        raise FetchError(str(e)) from e


def _items(data: Any, key: str) -> list[Any]:
    if isinstance(data, dict):
        value = data.get(key)
        if isinstance(value, list):
            return value
    if isinstance(data, list):
        return data
    return []


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _is_openai_chat_alias(model_id: str) -> bool:
    if not model_id.startswith(_OPENAI_PREFIXES):
        return False
    if _DATE_SUFFIX.search(model_id):
        return False
    return not any(b in model_id for b in _OPENAI_BLACKLIST)


def _is_gemini_text_model(model_id: str) -> bool:
    if not model_id.startswith("gemini-"):
        return False
    if _DATE_SUFFIX.search(model_id):
        return False
    if re.search(r"-\d{3}$", model_id):
        return False
    if model_id.endswith("-image"):
        return False
    if model_id.startswith(_GEMINI_OLD_PREFIXES):
        return False
    return not any(b in model_id for b in _GEMINI_BLACKLIST)


def _is_public_mistral_model(model_id: str) -> bool:
    if model_id.startswith("ft:"):
        return False
    if _DATE_SUFFIX.search(model_id):
        return False
    if _INTERNAL_SEGMENT.search(model_id):
        return False
    return not any(b in model_id for b in _MISTRAL_BLACKLIST)


def _is_grok_public_model(model_id: str) -> bool:
    if not model_id.startswith("grok-"):
        return False
    if _DATE_SUFFIX.search(model_id):
        return False
    if _VERSION_SEGMENT.search(model_id):
        return False
    return not any(b in model_id for b in _GROK_BLACKLIST)


def _openai_sdk_aliases() -> set[str]:
    """Return stable aliases from the SDK's ChatModel after capability filtering."""
    keep: set[str] = set()
    for mid in typing.get_args(ChatModel):
        if isinstance(mid, str) and _is_openai_chat_alias(mid):
            keep.add(mid)
    return keep


def _openai_live_aliases() -> RemoteModels:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return RemoteModels(set(), skipped="OPENAI_API_KEY is not configured")
    try:
        data = _json_get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    except FetchError as e:
        return RemoteModels(set(), error=str(e))

    keep: set[str] = set()
    for item in _items(data, "data"):
        if not isinstance(item, dict):
            continue
        model_id = _string(item.get("id"))
        if model_id is not None and _is_openai_chat_alias(model_id):
            keep.add(model_id)
    return RemoteModels(keep)


def _gemini_live_models() -> RemoteModels:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return RemoteModels(set(), skipped="GEMINI_API_KEY is not configured")

    keep: set[str] = set()
    page_token: str | None = None
    try:
        while True:
            params = {"key": api_key, "pageSize": "1000"}
            if page_token:
                params["pageToken"] = page_token
            data = _json_get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params=params,
            )
            for item in _items(data, "models"):
                if not isinstance(item, dict):
                    continue
                supported = _strings(item.get("supportedGenerationMethods"))
                if "generateContent" not in supported:
                    continue

                names = [
                    _string(item.get("name")),
                    _string(item.get("baseModelId")),
                ]
                for name in names:
                    if name is None:
                        continue
                    model_id = name.removeprefix("models/")
                    if _is_gemini_text_model(model_id):
                        keep.add(model_id)

            raw_token = data.get("nextPageToken") if isinstance(data, dict) else None
            page_token = _string(raw_token)
            if page_token is None:
                break
    except FetchError as e:
        return RemoteModels(set(), error=str(e))

    return RemoteModels(keep)


def _mistral_live_models() -> RemoteModels:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return RemoteModels(set(), skipped="MISTRAL_API_KEY is not configured")
    try:
        data = _json_get(
            "https://api.mistral.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    except FetchError as e:
        return RemoteModels(set(), error=str(e))

    keep: set[str] = set()
    deprecations: dict[str, str] = {}
    for item in _items(data, "data"):
        if not isinstance(item, dict):
            continue
        if item.get("archived") is True:
            continue
        if item.get("TYPE") == "fine-tuned":
            continue

        # Still-served models carry a retirement date here once announced.
        deprecation = _string(item.get("deprecation"))
        candidates = [_string(item.get("id")), *_strings(item.get("aliases"))]
        for model_id in candidates:
            if model_id is not None and _is_public_mistral_model(model_id):
                keep.add(model_id)
                if deprecation is not None:
                    deprecations[model_id] = deprecation
    return RemoteModels(keep, deprecations=deprecations)


def _grok_live_models() -> RemoteModels:
    api_key = os.environ.get("GROK_API_KEY")
    if not api_key:
        return RemoteModels(set(), skipped="GROK_API_KEY is not configured")
    try:
        data = _json_get(
            "https://api.x.ai/v1/language-models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    except FetchError as e:
        return RemoteModels(set(), error=str(e))

    keep: set[str] = set()
    for item in _items(data, "models"):
        if not isinstance(item, dict):
            continue
        candidates = [_string(item.get("id")), *_strings(item.get("aliases"))]
        for model_id in candidates:
            if model_id is not None and _is_grok_public_model(model_id):
                keep.add(model_id)
    return RemoteModels(keep)


def _diff(live: set[str], known: set[str]) -> Diff:
    return Diff(added=live - known, removed=known - live)


def _live_check(
    provider: str,
    source: str,
    target: str,
    docs_url: str,
    remote: RemoteModels,
    known: set[str],
    notes: tuple[str, ...] = (),
) -> LiveCheck:
    if remote.skipped is not None:
        return LiveCheck(
            provider=provider,
            source=source,
            target=target,
            docs_url=docs_url,
            skipped=remote.skipped,
            notes=notes,
        )
    if remote.error is not None:
        return LiveCheck(
            provider=provider,
            source=source,
            target=target,
            docs_url=docs_url,
            error=remote.error,
            notes=notes,
        )
    return LiveCheck(
        provider=provider,
        source=source,
        target=target,
        docs_url=docs_url,
        diff=_diff(remote.models, known),
        notes=notes,
        deprecations={m: d for m, d in remote.deprecations.items() if m in known},
    )


def _render_diff(
    lines: list[str],
    diff: Diff,
    *,
    add_label: str,
    remove_label: str,
) -> None:
    if diff.added:
        lines.append(add_label)
        lines.extend(f"- `{model}`" for model in sorted(diff.added))
        lines.append("")
    if diff.removed:
        lines.append(remove_label)
        lines.extend(f"- `{model}`" for model in sorted(diff.removed))
        lines.append("")


def _render_openai(lines: list[str], sdk_diff: Diff, live: RemoteModels) -> None:
    lines.append("## OpenAI")
    lines.append("")
    lines.append("Source: `openai.types.ChatModel` from the installed OpenAI SDK.")
    lines.append("Target: `padwan_llm/openai/client.py::OpenAIModel`.")
    lines.append("")
    if sdk_diff.has_drift:
        _render_diff(
            lines,
            sdk_diff,
            add_label=(
                "**Candidate additions** - present in `openai.types.ChatModel`, "
                "missing from our `Literal`:"
            ),
            remove_label=(
                "**Review removals** - in our `Literal`, no longer in "
                "`openai.types.ChatModel`:"
            ),
        )
    else:
        lines.append(
            "No SDK drift. `openai.types.ChatModel` matches the project's "
            "`Literal` after filtering."
        )
        lines.append("")

    lines.append("Live `/v1/models` availability check: advisory only.")
    if live.skipped is not None:
        lines.append(f"- Skipped: {live.skipped}.")
    elif live.error is not None:
        lines.append(f"- Failed: {live.error}.")
    else:
        lines.append(f"- Returned {len(live.models)} filtered OpenAI model IDs.")
    lines.append("")


def _render_live(lines: list[str], check: LiveCheck) -> None:
    lines.append(f"## {check.provider}")
    lines.append("")
    lines.append(f"Source: `{check.source}`.")
    lines.append(f"Target: `{check.target}`.")
    lines.append(f"Docs: {check.docs_url}")
    for note in check.notes:
        lines.append(f"Note: {note}")
    lines.append("")

    if check.skipped is not None:
        lines.append(f"Skipped: {check.skipped}.")
        lines.append("")
        return
    if check.error is not None:
        lines.append(f"Failed: {check.error}.")
        lines.append("")
        return
    if check.diff is None:
        lines.append("No live check result.")
        lines.append("")
        return
    if not check.diff.has_drift and not check.deprecations:
        lines.append("No live drift against the current curated literals.")
        lines.append("")
        return

    _render_diff(
        lines,
        check.diff,
        add_label=(
            "**Available but not tracked** - present in the provider response, "
            "missing from our curated literals:"
        ),
        remove_label=(
            "**Tracked but absent from this API response** - may be deprecated, "
            "region/account-gated, or filtered by this script:"
        ),
    )

    if check.deprecations:
        lines.append(
            "**Tracked but deprecated** - still served, but the provider has "
            "scheduled removal; migrate before the retirement date:"
        )
        lines.extend(
            f"- `{model}` - retires {date}"
            for model, date in sorted(
                check.deprecations.items(), key=lambda kv: (kv[1], kv[0])
            )
        )
        lines.append("")


def _render(
    openai_sdk_diff: Diff,
    openai_live: RemoteModels,
    live_checks: list[LiveCheck],
) -> str:
    lines: list[str] = [
        "# Provider Model Drift Report",
        "",
        "Live provider checks are advisory because model-list responses can "
        "depend on account, region, and API permissions.",
        "",
    ]
    _render_openai(lines, openai_sdk_diff, openai_live)
    for check in live_checks:
        _render_live(lines, check)

    lines.extend(
        [
            "## Review Checklist",
            "",
            "- Confirm new model IDs are public/stable aliases, not dated snapshots.",
            "- Confirm each model belongs in this package's curated typing surface.",
            "- Update runtime prefixes if a new provider naming family appears.",
            "- Run `uv run pyright`, `uv run ruff check .`, and relevant tests.",
            "",
        ]
    )
    return "\n".join(lines)


cli = Cli(
    description="Check provider model drift against the project's model Literals."
)


@cli.main(help="Generate a Markdown drift report.")
def check(
    out: MaybePath | None = Option(
        None, "--out", help="Write the Markdown report to this path."
    ),
) -> None:
    openai_sdk = _openai_sdk_aliases()
    openai_sdk_diff = _diff(openai_sdk, OPENAI_MODELS)
    openai_live = _openai_live_aliases()

    mistral_known = (
        MISTRAL_MODELS
        | set(typing.get_args(MistralEmbeddingModel))
        | set(typing.get_args(MistralAudioModel))
    )
    live_checks = [
        _live_check(
            "Gemini",
            "GET https://generativelanguage.googleapis.com/v1beta/models",
            "padwan_llm/gemini/client.py::GeminiModel",
            "https://ai.google.dev/gemini-api/docs/models",
            _gemini_live_models(),
            GEMINI_MODELS,
            notes=(
                "Only models supporting `generateContent` are considered. "
                "Embedding, image-generation, video, TTS, robotics, old, and "
                "versioned names are filtered.",
            ),
        ),
        _live_check(
            "Mistral",
            "GET https://api.mistral.ai/v1/models",
            (
                "padwan_llm/mistral/client.py::MistralModel, "
                "MistralEmbeddingModel, MistralAudioModel"
            ),
            "https://docs.mistral.ai/getting-started/models/models_overview/",
            _mistral_live_models(),
            mistral_known,
            notes=(
                "Fine-tuned, archived, and dated snapshot models are filtered.",
                "Deprecation dates come from the API's `deprecation` field; "
                "models stay listed until their retirement date.",
            ),
        ),
        _live_check(
            "Grok",
            "GET https://api.x.ai/v1/language-models",
            "padwan_llm/grok/client.py::GrokModel",
            "https://docs.x.ai/docs/models",
            _grok_live_models(),
            {model for model in GROK_MODELS if _is_grok_public_model(model)},
            notes=(
                "Beta, experimental, multi-agent, and dated/versioned language "
                "model names are filtered.",
            ),
        ),
    ]

    report = _render(openai_sdk_diff, openai_live, live_checks)
    print(report)
    if out:
        out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    cli.run()
