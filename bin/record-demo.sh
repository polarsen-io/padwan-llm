#!/usr/bin/env bash
# DESCRIPTION: Record a TUI demo GIF using VHS
# USAGE: ./bin/record-demo.sh
set -euo pipefail

cd "$(dirname "$0")/.."

vhs docs/static/chat-demo.tape