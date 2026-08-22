set quiet

# List available recipes
default:
    @just --list


# Run unit tests
[group('dev')]
test *args:
    uv run pytest {{ args }}

# Run unit tests on the lowest supported Python (matches CI matrix floor)
[group('dev')]
test-min *args:
    uv run --python 3.13 pytest {{ args }}

# Run e2e tests (copy env.template to .env first, then fill in keys)
[group('dev')]
e2e env=".env" *args:
    uv run pytest tests/e2e/ -m e2e --env-file {{ env }} {{ args }}

compose := "docker compose -p padwan-obs --env-file bin/observability/langfuse.env -f bin/observability/docker-compose.yml"

# Start the local OTel stack (Grafana on :3000, OTLP on :4317/:4318, dashboards provisioned)
[group('otel')]
otel-up:
    #!/usr/bin/env bash
    set -euo pipefail
    {{ compose }} --profile otel up -d
    until curl -sf http://localhost:3000/api/health >/dev/null; do sleep 1; done
    echo "Grafana: http://localhost:3000 (admin/admin)"

# Stop the local OTel stack (discards collected telemetry)
[group('otel')]
otel-down:
    {{ compose }} --profile otel down

# Run e2e tests instrumented, exporting traces and metrics to the local OTel stack
[group('otel')]
e2e-otel env=".env" *args: otel-up
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 just e2e {{ env }} --otel {{ args }}

# Start the local Langfuse stack (UI on :3001, dev project + model prices seeded)
[group('langfuse')]
langfuse-up:
    #!/usr/bin/env bash
    set -euo pipefail
    {{ compose }} --profile langfuse up -d
    until curl -sf http://localhost:3001/api/public/health >/dev/null; do sleep 2; done
    echo "Langfuse: http://localhost:3001 (dev@example.com / padwan-dev)"

# Stop the local Langfuse stack (add -v to also drop its volumes)
[group('langfuse')]
langfuse-down *args:
    {{ compose }} --profile langfuse down {{ args }}

# Run e2e tests instrumented through the Langfuse adapter
[group('langfuse')]
e2e-langfuse env=".env" *args: langfuse-up
    #!/usr/bin/env bash
    set -euo pipefail
    set -a && source bin/observability/langfuse.env && set +a
    export LANGFUSE_BASE_URL=http://localhost:3001
    export LANGFUSE_PUBLIC_KEY=$LANGFUSE_INIT_PROJECT_PUBLIC_KEY
    export LANGFUSE_SECRET_KEY=$LANGFUSE_INIT_PROJECT_SECRET_KEY
    just e2e {{ env }} --langfuse {{ args }}

# Type check
[group('dev')]
check:
    uv run pyright padwan_llm/

# Lint
[group('dev')]
lint:
    uv run ruff check padwan_llm/ tests/

# Format
[group('dev')]
fmt:
    uv run ruff format padwan_llm/ tests/

# Fix lint issues where possible
[group('dev')]
fix:
    uv run ruff check --fix padwan_llm/ tests/
    uv run ruff format padwan_llm/ tests/

# Lint + type check + test
[group('dev')]
ci: lint check test

# Benchmark import times (add --profile for a per-module breakdown)
[group('dev')]
bench-imports *args:
    ./bin/bench-imports.sh {{ args }}

# Serve docs locally with hot reload
[group('docs')]
docs:
    uv run --group docs zensical serve -f zensical.toml

# Build docs
[group('docs')]
docs-build:
    uv run --group docs zensical build -f zensical.toml

# Regenerate OpenAI TypedDict types from upstream spec
[group('gen')]
gen-openai:
    ./bin/gen-openai-types.sh

# Regenerate Mistral TypedDict types from upstream spec
[group('gen')]
gen-mistral:
    ./bin/gen-mistral-types.sh

# Regenerate all provider types
[group('gen')]
gen-all: gen-openai gen-mistral

# Regenerate docs/static/favicon.png (64x64) from the full-resolution logo
[group('gen')]
gen-favicon:
    uv run --with pillow python -c "from PIL import Image; Image.open('docs/static/logo-hood.png').resize((64, 64), Image.Resampling.LANCZOS).save('docs/static/favicon.png', optimize=True)"
    echo "regenerated docs/static/favicon.png"

# Bump version (commitizen — updates pyproject.toml and CHANGELOG)
[group('release')]
bump *args:
    uv run --group bump cz bump {{ args }}
