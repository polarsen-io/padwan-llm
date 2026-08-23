window.BENCHMARK_DATA = {
  "lastUpdate": 1787465138214,
  "repoUrl": "https://github.com/polarsen-io/padwan-llm",
  "entries": {
    "Import Performance": [
      {
        "commit": {
          "author": {
            "email": "julien.brayere@obitrain.com",
            "name": "Julien Brayere",
            "username": "Andarius"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dad7148968a9864fbf1297c09c606bbebd14156f",
          "message": "feat: audio input content parts across providers (#43)\n\n* feat: audio input content parts across providers\n\nAdd ContentAudioPart (OpenAI input_audio shape, wav/mp3) with an audio_part\nbuilder and content_parts inference. OpenAI/Grok receive parts verbatim,\nGemini converts to inlineData, Mistral rewrites to its base64 input_audio\nchunk via a new _prepare_messages hook. supports_audio mirrors\nsupports_vision with per-provider curated checks.\n\n* feat: accept str paths in audio_part like image_part\n\n* docs: use parentheses instead of em-dashes in audio docs\n\n* feat: per-provider audio format support\n\nWiden AudioFormat to wav/mp3/flac/ogg/aac/aiff/m4a and make supports_audio\nformat-aware (fmt param). Curated per provider: OpenAI wav/mp3 (API schema),\nGemini all formats, Mistral voxtral wav/mp3/flac/ogg (verified against the\nchat API; m4a rejected). Providers expose AUDIO_FORMATS.\n\n* test(otel): multimodal parts captured without binary payloads\n\n* fix: lazy-module registration and otel fixture rename after rebase\n\nRegister padwan_llm.audio in the top-level __lazy_modules__ set (Python\n3.15) and adapt the binary-parts capture test to the otel_logging fixture.",
          "timestamp": "2026-08-23T08:04:56+02:00",
          "tree_id": "be8dc374e3b4a3c08717137427581df6fc264dfa",
          "url": "https://github.com/polarsen-io/padwan-llm/commit/dad7148968a9864fbf1297c09c606bbebd14156f"
        },
        "date": 1787465136855,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "padwan_llm (facade)",
            "value": 217.05,
            "unit": "ms",
            "range": 5.21
          },
          {
            "name": "padwan_llm.openai",
            "value": 217.6,
            "unit": "ms",
            "range": 4.34
          },
          {
            "name": "padwan_llm.otel",
            "value": 234.23,
            "unit": "ms",
            "range": 6.07
          }
        ]
      }
    ]
  }
}