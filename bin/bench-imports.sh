#!/usr/bin/env bash
# DESCRIPTION
#   Benchmark import times for padwan_llm (facade, single provider, otel).
#
# USAGE
#   ./bin/bench-imports.sh              # Pretty-print results
#   ./bin/bench-imports.sh --json       # JSON for github-action-benchmark
#   ./bin/bench-imports.sh --profile    # Top offenders via python -X importtime
#
# EXAMPLES
#   RUNS=20 ./bin/bench-imports.sh
#
# Requires: hyperfine, jq, uv

set -euo pipefail

export LC_NUMERIC=C

WARMUP="${WARMUP:-3}"
RUNS="${RUNS:-10}"
MODE=pretty

while [[ $# -gt 0 ]]; do
    case $1 in
        --json) MODE=json; shift ;;
        --profile) MODE=profile; shift ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

for cmd in uv jq; do
    command -v "$cmd" >/dev/null || { echo "Missing required tool: $cmd" >&2; exit 1; }
done
if [[ $MODE != profile ]] && ! command -v hyperfine >/dev/null; then
    echo "Missing required tool: hyperfine (https://github.com/sharkdp/hyperfine)" >&2
    exit 1
fi

# Resolve the venv interpreter once so uv startup is not part of the measurement
PY=$(uv run --frozen python -c 'import sys; print(sys.executable)')

if [[ $MODE == profile ]]; then
    echo "Top cumulative import times for 'import padwan_llm' (µs):"
    "$PY" -X importtime -c "import padwan_llm" 2>&1 \
        | grep "^import time:" \
        | sort -t'|' -k2 -rn \
        | head -20
    exit 0
fi

RESULTS=$(mktemp)
trap 'rm -f "$RESULTS"' EXIT

hyperfine --warmup "$WARMUP" --min-runs "$RUNS" \
    --export-json "$RESULTS" \
    -n "padwan_llm (facade)" "$PY -c 'import padwan_llm'" \
    -n "padwan_llm.openai" "$PY -c 'from padwan_llm.openai import OpenAIClient'" \
    -n "padwan_llm.otel" "$PY -c 'import padwan_llm.otel'" \
    >/dev/null 2>&1

if [[ $MODE == json ]]; then
    jq '[.results[] | {
        name: .command,
        unit: "ms",
        value: (.mean * 1000 * 100 | round / 100),
        range: (.stddev * 1000 * 100 | round / 100)
    }]' "$RESULTS"
else
    printf "\n📊 Import Benchmark Results:\n"
    printf "============================================================\n"
    jq -r '.results[] | [.command, (.mean * 1000), (.stddev * 1000)] | @tsv' "$RESULTS" \
        | while IFS=$'\t' read -r name mean std; do
            printf "  %-30s %9.2fms (±%.2fms)\n" "$name" "$mean" "$std"
        done
    printf "============================================================\n"
fi
