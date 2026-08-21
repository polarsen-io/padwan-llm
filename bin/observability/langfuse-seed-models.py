#!/usr/bin/env python3
# DESCRIPTION: Seed the local Langfuse with prices for the models padwan-llm uses,
#              so generation costs stop reading 0 (self-hosted ships a near-empty registry).
# USAGE: LANGFUSE_BASE_URL=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... ./langfuse-seed-models.py
import base64
import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3001")

# USD per token; sources: provider pricing pages, 2026-08-21.
# The public API only takes flat input/output prices, so cache-read/-write
# tokens go unpriced — costs are close enough for a local dev stack.
PER_M = 1e-6
MODELS: list[dict] = [
    {
        "modelName": "claude-haiku-4-5",
        "matchPattern": r"(?i)^(claude-haiku-4-5)$",
        "inputPrice": 1.00 * PER_M,
        "outputPrice": 5.00 * PER_M,
    },
    {
        "modelName": "gpt-4o",
        "matchPattern": r"(?i)^(gpt-4o)$",
        "inputPrice": 2.50 * PER_M,
        "outputPrice": 10.00 * PER_M,
    },
    {
        "modelName": "gpt-4o-mini",
        "matchPattern": r"(?i)^(gpt-4o-mini)$",
        "inputPrice": 0.15 * PER_M,
        "outputPrice": 0.60 * PER_M,
    },
    {
        "modelName": "gemini-2.5-flash",
        "matchPattern": r"(?i)^(gemini-2\.5-flash)$",
        "inputPrice": 0.30 * PER_M,
        "outputPrice": 2.50 * PER_M,
    },
    {
        "modelName": "gemini-3-flash-preview",
        "matchPattern": r"(?i)^(gemini-3-flash-preview)$",
        "inputPrice": 0.50 * PER_M,
        "outputPrice": 3.00 * PER_M,
    },
    {
        "modelName": "mistral-small-latest",
        "matchPattern": r"(?i)^(mistral-small-latest|mistral-small-[0-9]+)$",
        "inputPrice": 0.15 * PER_M,
        "outputPrice": 0.60 * PER_M,
    },
    {
        # unverified against the pricing page; magistral-small is only used in one e2e test
        "modelName": "magistral-small-latest",
        "matchPattern": r"(?i)^(magistral-small-latest|magistral-small-[0-9]+)$",
        "inputPrice": 0.50 * PER_M,
        "outputPrice": 1.50 * PER_M,
    },
    {
        "modelName": "grok-4-1-fast-non-reasoning",
        "matchPattern": r"(?i)^(grok-4-1-fast(-non-reasoning|-reasoning)?)$",
        "inputPrice": 0.20 * PER_M,
        "outputPrice": 0.50 * PER_M,
    },
    {
        "modelName": "grok-3-mini",
        "matchPattern": r"(?i)^(grok-3-mini)$",
        "inputPrice": 0.30 * PER_M,
        "outputPrice": 0.50 * PER_M,
    },
]


def request(method: str, path: str, body: dict | None = None) -> dict:
    auth = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read() or "{}")


def existing_custom_models() -> dict[str, dict]:
    models: dict[str, dict] = {}
    page = 1
    while True:
        data = request("GET", f"/api/public/models?page={page}&limit=100")
        for model in data["data"]:
            if not model["isLangfuseManaged"]:
                models[model["modelName"]] = model
        if page >= data["meta"]["totalPages"]:
            return models
        page += 1


def main() -> int:
    current = existing_custom_models()
    for spec in MODELS:
        found = current.get(spec["modelName"])
        if found is not None:
            if (found["inputPrice"], found["outputPrice"]) == (
                spec["inputPrice"],
                spec["outputPrice"],
            ):
                print(f"  {spec['modelName']}: up to date")
                continue
            request("DELETE", f"/api/public/models/{found['id']}")
        request("POST", "/api/public/models", {"unit": "TOKENS", **spec})
        print(f"  {spec['modelName']}: seeded")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as e:
        print(f"error: {e} — {e.read().decode()[:300]}", file=sys.stderr)
        sys.exit(1)
