# Multimodal Content

User messages can carry a list of content parts (text and images) instead of a plain string. Parts use the OpenAI content-part shape: OpenAI-compatible providers (OpenAI, Mistral, Grok) receive them verbatim, and the Gemini client converts them to `inlineData` parts.

## Building parts

`content_parts` infers the part type per item:

```python
from pathlib import Path

from padwan_llm import content_parts

parts = content_parts(
    "What is in this screenshot?",  # str -> text part
    Path("shot.png"),  # image extension -> base64 data: URL image part
    Path("notes.md"),  # any other file -> inlined text file part
)
```

Strings are always text — never treated as paths — so message text that mentions a filename is safe; wrap files in `Path` to have them read. Files are classified by extension: an image MIME type yields an image part, anything else is inlined as text.

The explicit builders remain for full control:

```python
from padwan_llm import image_part, text_file_part, text_part

parts = [
    text_part("What is in this screenshot?"),
    image_part("shot.png"),  # reads the file into a base64 data: URL
    text_file_part("notes.md"),  # inlines a text file, labelled with its name
]
```

`image_part` accepts a path (`str | Path`) and guesses the MIME type from the file name; pass `mime=` to override. Unknown extensions fall back to `image/png`. `text_file_part` reads the file as UTF-8 (`encoding=` overrides) and prefixes the text with `--- <name> ---` so the model can tell files apart.

## Sending images

```python
from padwan_llm import ConversationState, LLMClient, image_part, text_part

state = ConversationState()
state.add_user_message(
    [
        text_part("Describe this image."),
        image_part("shot.png"),
    ]
)

async with LLMClient("gpt-4o") as client:
    response, usage = await client.complete_chat(state.messages)
```

## Checking vision support

`supports_vision` is a best-effort, curated check of whether a model accepts image input:

```python
from padwan_llm import supports_vision

supports_vision("gpt-4o")  # True
supports_vision("mistral-large-latest")  # False - text-only
supports_vision("some-local-model")  # True - unknown models are attempted
```

Each provider package exposes its own `supports_vision` (e.g. `padwan_llm.mistral.supports_vision`); the top-level function dispatches to them in the same order as `LLMClient` routing. Unknown models default to `True` so the request is attempted and the provider surfaces the real error instead of the check guessing wrong.

| Provider | Image input |
|----------|-------------|
| OpenAI | All chat models except a curated text-only set: `gpt-4`, `o1-mini`, `o1-preview`, `o3-mini`, `codex-mini-latest`, and the `gpt-oss-*` / `gpt-3.5-*` families |
| Gemini | All current chat models; parts are converted to `inlineData` |
| Mistral | `pixtral-*` models only |
| Grok | All current chat models |
| Anthropic | Model support assumed, but the client does not convert parts yet - see Limitations |

## Limitations

- Image input only: no audio or video parts, and no image generation.
- Content parts are for user messages; assistant messages stay text-only.
- The Anthropic client passes message content through unconverted, so image parts are not usable with Claude models yet.
