set quiet

e2e_env := ".env"

# List available recipes
default:
    @just --list


# Run unit tests
[group('dev')]
test *args:
    uv run pytest {{ args }}

# Run e2e tests (copy env.template to .env first, then fill in keys)
[group('dev')]
e2e *args:
    uv run pytest tests/e2e/ -m e2e --env-file {{ e2e_env }} {{ args }}

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
