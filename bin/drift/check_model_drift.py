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
source of truth, because it ships a static Literal. Gemini, Mistral, xAI, and
Anthropic do not expose equivalent SDK Literals, so this script optionally calls
their live model-list endpoints when API keys are present in the environment.

Always exits 0. The workflow decides whether to open a PR based on ``git status``
and the generated report.
"""

import os
import re
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import niquests
from openai.types import ChatModel
from piou import Cli, MaybePath, Option

from padwan_llm.anthropic.client import ANTHROPIC_MODELS, ANTHROPIC_VERSION
from padwan_llm.gemini.client import GEMINI_MODELS
from padwan_llm.gemini.realtime import DEFAULT_LIVE_MODEL, GeminiLiveModel
from padwan_llm.grok.client import GROK_MODELS
from padwan_llm.mistral.client import (
    MISTRAL_MODELS,
    MistralAudioModel,
    MistralEmbeddingModel,
)
from padwan_llm.openai.client import _OPENAI_PREFIXES, OPENAI_MODELS
from padwan_llm.openai.realtime import _VOICES, DEFAULT_REALTIME_MODEL

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
# Streaming specialists that are not general speech-to-speech models.
_GEMINI_REALTIME_BLACKLIST = ("translate",)

_MISTRAL_BLACKLIST = ("mistral-vibe-cli",)

_GROK_BLACKLIST = (
    "-beta",
    "experimental",
    "-gv2",
    "multi-agent",
)

# Anthropic snapshot ids carry an 8-digit date (claude-haiku-4-5-20251001);
# the project tracks the undated aliases.
_ANTHROPIC_DATE_SUFFIX = re.compile(r"-\d{8}$")
_ANTHROPIC_OLD_PREFIXES = ("claude-2", "claude-3-", "claude-instant")


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


def _is_gemini_realtime_model(model_id: str) -> bool:
    if not _is_gemini_text_model(model_id):
        return False
    return not any(b in model_id for b in _GEMINI_REALTIME_BLACKLIST)


def _is_grok_public_model(model_id: str) -> bool:
    if not model_id.startswith("grok-"):
        return False
    if _DATE_SUFFIX.search(model_id):
        return False
    if _VERSION_SEGMENT.search(model_id):
        return False
    return not any(b in model_id for b in _GROK_BLACKLIST)


def _is_anthropic_public_model(model_id: str) -> bool:
    if not model_id.startswith("claude-"):
        return False
    return not model_id.startswith(_ANTHROPIC_OLD_PREFIXES)


def _literal_strings(annotation: Any) -> set[str]:
    """Collect string members of every Literal nested inside *annotation*.

    E.g. ``Optional[Union[str, Literal["marin", "cedar"]]]`` -> {"marin", "cedar"}.
    """
    if typing.get_origin(annotation) is typing.Literal:
        return {a for a in typing.get_args(annotation) if isinstance(a, str)}
    out: set[str] = set()
    for arg in typing.get_args(annotation):
        out |= _literal_strings(arg)
    return out


def _openai_realtime_sdk() -> tuple[set[str], set[str]]:
    """Return (model ids, voices) from the SDK's realtime session types.

    Field annotations are walked instead of importing named aliases because the
    SDK moved the Literals between modules across major versions.
    """
    from openai.types.realtime.realtime_audio_config_output import (
        RealtimeAudioConfigOutput,
    )
    from openai.types.realtime.realtime_session_create_request import (
        RealtimeSessionCreateRequest,
    )

    models = _literal_strings(
        RealtimeSessionCreateRequest.model_fields["model"].annotation
    )
    voices = _literal_strings(
        RealtimeAudioConfigOutput.model_fields["voice"].annotation
    )
    return models, voices


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


def _gemini_models(
    method: str, keep_model: typing.Callable[[str], bool]
) -> RemoteModels:
    """List Gemini models supporting *method*, filtered by *keep_model*."""
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
                if method not in supported:
                    continue

                names = [
                    _string(item.get("name")),
                    _string(item.get("baseModelId")),
                ]
                for name in names:
                    if name is None:
                        continue
                    model_id = name.removeprefix("models/")
                    if keep_model(model_id):
                        keep.add(model_id)

            raw_token = data.get("nextPageToken") if isinstance(data, dict) else None
            page_token = _string(raw_token)
            if page_token is None:
                break
    except FetchError as e:
        return RemoteModels(set(), error=str(e))

    return RemoteModels(keep)


def _gemini_live_models() -> RemoteModels:
    return _gemini_models("generateContent", _is_gemini_text_model)


def _gemini_realtime_models() -> RemoteModels:
    return _gemini_models("bidiGenerateContent", _is_gemini_realtime_model)


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


def _anthropic_live_models() -> RemoteModels:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return RemoteModels(set(), skipped="ANTHROPIC_API_KEY is not configured")

    keep: set[str] = set()
    after_id: str | None = None
    try:
        while True:
            params = {"limit": "100"}
            if after_id:
                params["after_id"] = after_id
            data = _json_get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
                params=params,
            )
            for item in _items(data, "data"):
                if not isinstance(item, dict):
                    continue
                model_id = _string(item.get("id"))
                if model_id is None:
                    continue
                alias = _ANTHROPIC_DATE_SUFFIX.sub("", model_id)
                if _is_anthropic_public_model(alias):
                    keep.add(alias)
            has_more = bool(data.get("has_more")) if isinstance(data, dict) else False
            after_id = _string(data.get("last_id")) if isinstance(data, dict) else None
            if not has_more or after_id is None:
                break
    except FetchError as e:
        return RemoteModels(set(), error=str(e))
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


def _render_openai_realtime(
    lines: list[str], sdk_models: set[str], voice_diff: Diff
) -> None:
    lines.append("## OpenAI Realtime")
    lines.append("")
    lines.append(
        "Source: `model`/`voice` Literals in the installed SDK's "
        "`openai.types.realtime` session types."
    )
    lines.append(
        "Target: `padwan_llm/openai/realtime.py::DEFAULT_REALTIME_MODEL, _VOICES`."
    )
    lines.append("")
    if DEFAULT_REALTIME_MODEL in sdk_models:
        lines.append(
            f"`DEFAULT_REALTIME_MODEL` (`{DEFAULT_REALTIME_MODEL}`) is a valid SDK model id."
        )
    else:
        lines.append(
            f"**`DEFAULT_REALTIME_MODEL` (`{DEFAULT_REALTIME_MODEL}`) is missing "
            "from the SDK's realtime model Literal** - pick a current alias:"
        )
        lines.extend(f"- `{model}`" for model in sorted(sdk_models))
    lines.append("")
    if voice_diff.has_drift:
        _render_diff(
            lines,
            voice_diff,
            add_label=(
                "**Candidate voice additions** - in the SDK's voice Literal, "
                "missing from `_VOICES`:"
            ),
            remove_label=(
                "**Review voice removals** - in `_VOICES`, no longer in the "
                "SDK's voice Literal:"
            ),
        )
    else:
        lines.append("No voice drift. `_VOICES` matches the SDK's voice Literal.")
        lines.append("")


def _render(
    openai_sdk_diff: Diff,
    openai_live: RemoteModels,
    realtime_models: set[str],
    realtime_voice_diff: Diff,
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
    _render_openai_realtime(lines, realtime_models, realtime_voice_diff)
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


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MISTRAL_DEPRECATIONS = _REPO_ROOT / "padwan_llm" / "mistral" / "_deprecations.py"


def _write_deprecations(
    path: Path, provider: str, deprecations: Mapping[str, str]
) -> None:
    """Emit the generated runtime deprecation map consumed by the provider client.

    Writes ``DEPRECATED: dict[str, str]`` mapping each still-served, scheduled-for-
    retirement model id to its date. ``refresh-llms.sh`` runs ``ruff format`` over
    the result, so the layout here only needs to be valid Python.
    """
    if deprecations:
        body = "".join(
            f'    "{model}": "{date}",\n'
            for model, date in sorted(deprecations.items())
        )
        table = f"{{\n{body}}}"
    else:
        table = "{}"
    path.write_text(
        "# Generated by bin/drift/check_model_drift.py — do not edit by hand.\n"
        f"# Maps a still-served {provider} model id to its ISO-8601 retirement date.\n"
        "\n"
        f"DEPRECATED: dict[str, str] = {table}\n",
        encoding="utf-8",
    )


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
    realtime_models, realtime_voices = _openai_realtime_sdk()
    realtime_voice_diff = _diff(realtime_voices, set(_VOICES))

    mistral_known = (
        MISTRAL_MODELS
        | set(typing.get_args(MistralEmbeddingModel))
        | set(typing.get_args(MistralAudioModel))
    )
    mistral_check = _live_check(
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
            "Gemini Live (realtime)",
            "GET https://generativelanguage.googleapis.com/v1beta/models",
            "padwan_llm/gemini/realtime.py::GeminiLiveModel, DEFAULT_LIVE_MODEL",
            "https://ai.google.dev/gemini-api/docs/live",
            _gemini_realtime_models(),
            set(typing.get_args(GeminiLiveModel)) | {DEFAULT_LIVE_MODEL},
            notes=(
                "Only models supporting `bidiGenerateContent` are considered, "
                "with the text-model filters plus streaming specialists "
                "(translate) removed.",
                "Grok voice models cannot be checked: the xAI API does not "
                "list them; `grok-voice-latest` is a rolling alias maintained "
                "upstream.",
            ),
        ),
        mistral_check,
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
        _live_check(
            "Anthropic",
            "GET https://api.anthropic.com/v1/models",
            "padwan_llm/anthropic/client.py::AnthropicModel",
            "https://platform.claude.com/docs/en/about-claude/models/overview",
            _anthropic_live_models(),
            ANTHROPIC_MODELS,
            notes=(
                "Dated snapshot ids are normalized to their alias "
                "(-YYYYMMDD stripped); claude-2/claude-3-era families are "
                "filtered. Gated models (e.g. claude-mythos-5) only appear "
                "for enrolled accounts.",
            ),
        ),
    ]

    # Refresh the runtime deprecation map only when the live check actually ran,
    # so a missing API key never wipes the committed deprecations.
    if mistral_check.skipped is None and mistral_check.error is None:
        _write_deprecations(
            _MISTRAL_DEPRECATIONS, "Mistral", mistral_check.deprecations
        )

    report = _render(
        openai_sdk_diff, openai_live, realtime_models, realtime_voice_diff, live_checks
    )
    print(report)
    if out:
        out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    cli.run()
