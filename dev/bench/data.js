window.BENCHMARK_DATA = {
  "lastUpdate": 1787583674033,
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
      },
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
          "id": "99394ca05996f5e3699aae9fbcf94f8f2f748593",
          "message": "chore(release): release 0.9.0 (#45)",
          "timestamp": "2026-08-23T08:15:33+02:00",
          "tree_id": "631e8b661f2d02d36a318d6da02e3f768264e2ee",
          "url": "https://github.com/polarsen-io/padwan-llm/commit/99394ca05996f5e3699aae9fbcf94f8f2f748593"
        },
        "date": 1787465776650,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "padwan_llm (facade)",
            "value": 217.23,
            "unit": "ms",
            "range": 5.21
          },
          {
            "name": "padwan_llm.openai",
            "value": 213.91,
            "unit": "ms",
            "range": 2.62
          },
          {
            "name": "padwan_llm.otel",
            "value": 232.3,
            "unit": "ms",
            "range": 4.86
          }
        ]
      },
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
          "id": "cbeec528ea996530aca98fdcda6ee1943cc3d81a",
          "message": "fix: close half-read SSE streams before returning connections to the pool (#46)\n\n* fix: close half-read SSE streams before returning connections to the pool\n\nA stream that breaks on [DONE] (or is abandoned by its consumer) left the\nSSE response half-read; the pooled connection could then hang the next\nstream request. Abort the extension in a finally across all providers.\n\n* refactor: dedupe the SSE stream loop into LLMClientBase._iter_sse\n\next.close() alone releases the pooled connection (verified against a live\nSSE backend on HTTP/1.1 and HTTP/2), so drop the raw-response teardown and\nthe mock-only regression test. Providers wrap the shared iterator in\naclosing so an abandoned stream closes deterministically.",
          "timestamp": "2026-08-23T15:30:06+02:00",
          "tree_id": "0433f9b8ccda881b64b9afdfce3de0d876aa4458",
          "url": "https://github.com/polarsen-io/padwan-llm/commit/cbeec528ea996530aca98fdcda6ee1943cc3d81a"
        },
        "date": 1787491846516,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "padwan_llm (facade)",
            "value": 228.07,
            "unit": "ms",
            "range": 1.62
          },
          {
            "name": "padwan_llm.openai",
            "value": 229.37,
            "unit": "ms",
            "range": 2.23
          },
          {
            "name": "padwan_llm.otel",
            "value": 242.42,
            "unit": "ms",
            "range": 2.51
          }
        ]
      },
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
          "id": "101126627a97f748a515d5fafa6b56bd4c34450d",
          "message": "chore(release): release 0.9.1 (#47)",
          "timestamp": "2026-08-24T07:01:01+02:00",
          "tree_id": "923f2aa8c4df21aad9522d6cef8b2eb5f52516e8",
          "url": "https://github.com/polarsen-io/padwan-llm/commit/101126627a97f748a515d5fafa6b56bd4c34450d"
        },
        "date": 1787547693881,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "padwan_llm (facade)",
            "value": 210.55,
            "unit": "ms",
            "range": 1.54
          },
          {
            "name": "padwan_llm.openai",
            "value": 211.36,
            "unit": "ms",
            "range": 3.3
          },
          {
            "name": "padwan_llm.otel",
            "value": 223.83,
            "unit": "ms",
            "range": 0.94
          }
        ]
      },
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
          "id": "943aabe4da8e06752e31853089e969c24ce38d7e",
          "message": "fix: best-effort SSE cleanup; tolerate HTTP/2 abort quirk (#48)\n\nClosing an already-finished HTTP/2 stream raises KeyError inside\nurllib3-future; the unguarded finally corrupted every successful stream\nover h2 backends (missing message end, spurious error event). Cleanup is\nnow best-effort per step and restores the raw-response teardown and pool\nrelease that the 0.9.1 refactor dropped (the ext-only close leaves the\nlease held until GC on HTTP/1.1).",
          "timestamp": "2026-08-24T16:49:18+02:00",
          "tree_id": "2d5f3f865af799f7e26f91ef5ae080070a843a02",
          "url": "https://github.com/polarsen-io/padwan-llm/commit/943aabe4da8e06752e31853089e969c24ce38d7e"
        },
        "date": 1787583003526,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "padwan_llm (facade)",
            "value": 209.33,
            "unit": "ms",
            "range": 2.16
          },
          {
            "name": "padwan_llm.openai",
            "value": 210,
            "unit": "ms",
            "range": 6.78
          },
          {
            "name": "padwan_llm.otel",
            "value": 224.53,
            "unit": "ms",
            "range": 2.15
          }
        ]
      },
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
          "id": "a6aff107fdfbdb9c4037918daacfd43e5c3c25d7",
          "message": "chore(release): release 0.9.2 (#50)",
          "timestamp": "2026-08-24T17:00:32+02:00",
          "tree_id": "0aff4bc9436e73d803c41d4b4e18ca3ed82b4bd3",
          "url": "https://github.com/polarsen-io/padwan-llm/commit/a6aff107fdfbdb9c4037918daacfd43e5c3c25d7"
        },
        "date": 1787583672929,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "padwan_llm (facade)",
            "value": 225.45,
            "unit": "ms",
            "range": 3.85
          },
          {
            "name": "padwan_llm.openai",
            "value": 224.44,
            "unit": "ms",
            "range": 3.56
          },
          {
            "name": "padwan_llm.otel",
            "value": 238.08,
            "unit": "ms",
            "range": 3.21
          }
        ]
      }
    ],
    "Import Performance (3.15)": [
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
        "date": 1787465444619,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "padwan_llm (facade)",
            "value": 52.15,
            "unit": "ms",
            "range": 0.46
          },
          {
            "name": "padwan_llm.openai",
            "value": 197.98,
            "unit": "ms",
            "range": 1.07
          },
          {
            "name": "padwan_llm.otel",
            "value": 229.12,
            "unit": "ms",
            "range": 1.61
          }
        ]
      },
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
          "id": "99394ca05996f5e3699aae9fbcf94f8f2f748593",
          "message": "chore(release): release 0.9.0 (#45)",
          "timestamp": "2026-08-23T08:15:33+02:00",
          "tree_id": "631e8b661f2d02d36a318d6da02e3f768264e2ee",
          "url": "https://github.com/polarsen-io/padwan-llm/commit/99394ca05996f5e3699aae9fbcf94f8f2f748593"
        },
        "date": 1787466096437,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "padwan_llm (facade)",
            "value": 53.48,
            "unit": "ms",
            "range": 0.59
          },
          {
            "name": "padwan_llm.openai",
            "value": 202.72,
            "unit": "ms",
            "range": 2.16
          },
          {
            "name": "padwan_llm.otel",
            "value": 235.91,
            "unit": "ms",
            "range": 2.64
          }
        ]
      },
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
          "id": "cbeec528ea996530aca98fdcda6ee1943cc3d81a",
          "message": "fix: close half-read SSE streams before returning connections to the pool (#46)\n\n* fix: close half-read SSE streams before returning connections to the pool\n\nA stream that breaks on [DONE] (or is abandoned by its consumer) left the\nSSE response half-read; the pooled connection could then hang the next\nstream request. Abort the extension in a finally across all providers.\n\n* refactor: dedupe the SSE stream loop into LLMClientBase._iter_sse\n\next.close() alone releases the pooled connection (verified against a live\nSSE backend on HTTP/1.1 and HTTP/2), so drop the raw-response teardown and\nthe mock-only regression test. Providers wrap the shared iterator in\naclosing so an abandoned stream closes deterministically.",
          "timestamp": "2026-08-23T15:30:06+02:00",
          "tree_id": "0433f9b8ccda881b64b9afdfce3de0d876aa4458",
          "url": "https://github.com/polarsen-io/padwan-llm/commit/cbeec528ea996530aca98fdcda6ee1943cc3d81a"
        },
        "date": 1787491885242,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "padwan_llm (facade)",
            "value": 53.76,
            "unit": "ms",
            "range": 1.31
          },
          {
            "name": "padwan_llm.openai",
            "value": 207.85,
            "unit": "ms",
            "range": 6.13
          },
          {
            "name": "padwan_llm.otel",
            "value": 234.55,
            "unit": "ms",
            "range": 2.64
          }
        ]
      },
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
          "id": "101126627a97f748a515d5fafa6b56bd4c34450d",
          "message": "chore(release): release 0.9.1 (#47)",
          "timestamp": "2026-08-24T07:01:01+02:00",
          "tree_id": "923f2aa8c4df21aad9522d6cef8b2eb5f52516e8",
          "url": "https://github.com/polarsen-io/padwan-llm/commit/101126627a97f748a515d5fafa6b56bd4c34450d"
        },
        "date": 1787548018489,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "padwan_llm (facade)",
            "value": 54.72,
            "unit": "ms",
            "range": 1
          },
          {
            "name": "padwan_llm.openai",
            "value": 210.45,
            "unit": "ms",
            "range": 3.08
          },
          {
            "name": "padwan_llm.otel",
            "value": 242.97,
            "unit": "ms",
            "range": 2.35
          }
        ]
      },
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
          "id": "943aabe4da8e06752e31853089e969c24ce38d7e",
          "message": "fix: best-effort SSE cleanup; tolerate HTTP/2 abort quirk (#48)\n\nClosing an already-finished HTTP/2 stream raises KeyError inside\nurllib3-future; the unguarded finally corrupted every successful stream\nover h2 backends (missing message end, spurious error event). Cleanup is\nnow best-effort per step and restores the raw-response teardown and pool\nrelease that the 0.9.1 refactor dropped (the ext-only close leaves the\nlease held until GC on HTTP/1.1).",
          "timestamp": "2026-08-24T16:49:18+02:00",
          "tree_id": "2d5f3f865af799f7e26f91ef5ae080070a843a02",
          "url": "https://github.com/polarsen-io/padwan-llm/commit/943aabe4da8e06752e31853089e969c24ce38d7e"
        },
        "date": 1787583054559,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "padwan_llm (facade)",
            "value": 58.71,
            "unit": "ms",
            "range": 2.28
          },
          {
            "name": "padwan_llm.openai",
            "value": 218.23,
            "unit": "ms",
            "range": 6.12
          },
          {
            "name": "padwan_llm.otel",
            "value": 271.52,
            "unit": "ms",
            "range": 23.46
          }
        ]
      }
    ]
  }
}