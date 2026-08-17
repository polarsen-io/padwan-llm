#!/usr/bin/env bash
# Generate TypedDict types from OpenAI's OpenAPI spec
#
# Usage: ./bin/gen-openai-types.sh
#
# Requires: uv

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_FILE="padwan_llm/openai/types.py"
SPEC_URL="https://raw.githubusercontent.com/openai/openai-openapi/main/openapi.yaml"

echo "Spec URL: $SPEC_URL"

# Download spec to temp file
TEMP_SPEC=$(mktemp --suffix=.yml)
trap "rm -f $TEMP_SPEC" EXIT

echo "Downloading spec..."
curl -sL "$SPEC_URL" -o "$TEMP_SPEC"
echo "Downloaded to: $TEMP_SPEC"

# Generate TypedDict types
echo "Generating TypedDict types to: $OUTPUT_FILE"
uvx --from 'datamodel-code-generator[ruff]' datamodel-codegen \
    --input "$TEMP_SPEC" \
    --input-file-type openapi \
    --output "$OUTPUT_FILE" \
    --output-model-type typing.TypedDict \
    --target-python-version 3.14 \
    --use-standard-collections \
    --use-union-operator \
    --use-type-alias \
    --use-schema-description \
    --collapse-root-models \
    --strip-default-none \
    --no-use-closed-typed-dict \
    --formatters ruff-format ruff-check \
    --openapi-scopes paths \
    --openapi-include-paths '/chat/completions' '/files' '/batches'

# Fix TypedDict inheritance issue: child cannot narrow type (int | None -> int)
sed -i 's/top_logprobs: NotRequired\[int\]$/top_logprobs: NotRequired[int | None]/' "$OUTPUT_FILE"

# Absolutize the spec's site-relative doc links
sed -i 's|](/docs/|](https://platform.openai.com/docs/|g' "$OUTPUT_FILE"

echo "Done! Generated: $OUTPUT_FILE"
