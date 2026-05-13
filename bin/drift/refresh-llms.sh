#!/usr/bin/env bash
# DESCRIPTION: Bump every package in the [dependency-groups.llms] table to its
# latest compatible version, regenerate the OpenAI and Mistral OpenAPI TypedDicts,
# run pyright + ruff, and write a provider model drift report.
#
# Designed to be runnable both locally (eyeball the diff before committing) and
# from the weekly model-drift GitHub Action.
#
# USAGE: ./bin/drift/refresh-llms.sh [--out PATH]
#
# EXAMPLES:
#   ./bin/drift/refresh-llms.sh                       # report to stdout
#   ./bin/drift/refresh-llms.sh --out drift-report.md # also write Markdown to PATH
#
# Requires: uv, gh (the latter only for ./bin/gen-openai-types.sh)

set -euo pipefail

cd "$(dirname "$0")/../.."

out=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    -h|--help) sed -n '/^# DESCRIPTION/,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Discover llms group members
mapfile -t pkgs < <(uv tree --only-group llms --depth 1 2>/dev/null \
  | awk '/├──|└──/ {print $2}')
if [[ ${#pkgs[@]} -eq 0 ]]; then
  echo "error: no packages found in [dependency-groups.llms]" >&2
  exit 1
fi
echo "==> Refreshing llms group: ${pkgs[*]}"

upgrade_args=()
for p in "${pkgs[@]}"; do upgrade_args+=(--upgrade-package "$p"); done

# Bump lockfile to latest within pyproject floors
uv lock "${upgrade_args[@]}"

# Install everything pyright/ruff/the gen scripts need
echo "==> Installing dependencies"
uv sync --group llms --group dev --all-extras

# Catch breaking SDK changes before generating anything
echo "==> Running pyright"
uv run pyright

# Refresh OpenAPI-derived TypedDicts
echo "==> Regenerating OpenAPI TypedDicts"
./bin/gen-openai-types.sh
./bin/gen-mistral-types.sh

# Guard against the generators leaving the tree unformatted
echo "==> Running ruff"
uv run ruff check .
uv run ruff format --check .

# Diff the SDK's ChatModel Literal against our OpenAIModel
echo "==> Running drift report"
if [[ -n "$out" ]]; then
  ./bin/drift/check_model_drift.py --out "$out"
else
  ./bin/drift/check_model_drift.py
fi
