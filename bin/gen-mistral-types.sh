#!/usr/bin/env bash
# Generate TypedDict types from Mistral's OpenAPI spec
#
# Usage: ./bin/gen-mistral-types.sh
#
# Requires: uv

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_FILE="padwan_llm/mistral/types.py"
SPEC_URL="https://raw.githubusercontent.com/mistralai/platform-docs-public/main/openapi.yaml"

# Download spec to temp file
TEMP_SPEC=$(mktemp --suffix=.yml)
trap "rm -f $TEMP_SPEC" EXIT

echo "Downloading spec from: $SPEC_URL"
curl -sL "$SPEC_URL" -o "$TEMP_SPEC"

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
    --openapi-include-paths '/v1/chat/completions' '/v1/embeddings' '/v1/audio/transcriptions'

# Fix TypedDict inheritance issue: child cannot redefine NotRequired as Required
sed -i 's/class ChatCompletionResponse(ChatCompletionResponseBase):/class ChatCompletionResponse(TypedDict):/' "$OUTPUT_FILE"
sed -i 's/class EmbeddingResponse(ResponseBase):/class EmbeddingResponse(TypedDict):/' "$OUTPUT_FILE"

echo "Done! Generated: $OUTPUT_FILE"
