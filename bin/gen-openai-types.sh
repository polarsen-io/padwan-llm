#!/usr/bin/env bash
# Generate TypedDict types from OpenAI's OpenAPI spec
#
# Usage: ./bin/gen-openai-types.sh
#
# Requires: uv, gh

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_FILE="padwan_llm/openai/types.py"
STATS_URL="https://github.com/openai/openai-python/blob/main/.stats.yml"

# Fetch .stats.yml from GitHub
echo "Fetching OpenAPI spec URL from: $STATS_URL"
STATS_CONTENT=$(gh api repos/openai/openai-python/contents/.stats.yml --jq '.content' | base64 -d)
SPEC_URL=$(echo "$STATS_CONTENT" | grep 'openapi_spec_url:' | cut -d' ' -f2-)

# URL decode (replace %2F with /)
SPEC_URL=$(echo "$SPEC_URL" | sed 's/%2F/\//g')
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
    --formatters ruff-format ruff-check \
    --openapi-scopes paths \
    --openapi-include-paths '/chat/completions'

# Fix TypedDict inheritance issue: child cannot narrow type (int | None -> int)
sed -i 's/top_logprobs: NotRequired\[int\]$/top_logprobs: NotRequired[int | None]/' "$OUTPUT_FILE"

# Remove typing_extensions import (not needed on Python 3.14)
sed -i '/from typing_extensions import/d' "$OUTPUT_FILE"

# Append Error/ErrorResponse types (not included by --openapi-scopes paths)
ERRORS_SPEC=$(uvx yq -y '{
  openapi: .openapi, info: .info, paths: {},
  components: {schemas: {
    Error: .components.schemas.Error,
    ErrorResponse: .components.schemas.ErrorResponse
  }}
}' "$TEMP_SPEC")

echo "$ERRORS_SPEC" | uvx --from 'datamodel-code-generator[ruff]' datamodel-codegen \
    --input-file-type openapi \
    --output-model-type typing.TypedDict \
    --target-python-version 3.14 \
    --use-standard-collections \
    --use-union-operator \
    --collapse-root-models \
    --strip-default-none \
    --formatters ruff-format ruff-check \
  | grep -v '^#\|^from typing' >> "$OUTPUT_FILE"

echo "Done! Generated: $OUTPUT_FILE"
