#!/usr/bin/env bash
# DESCRIPTION: Detect whether the model-drift refresh changed the worktree.
#
# Designed for GitHub Actions and local use. In Actions it writes a `changed`
# output when GITHUB_OUTPUT is set; locally it prints the same value.
#
# USAGE:
#   ./bin/drift/has-changes.sh [--include-untracked]
#
# EXAMPLES:
#   ./bin/drift/has-changes.sh
#   ./bin/drift/has-changes.sh --include-untracked

set -euo pipefail

cd "$(dirname "$0")/../.."

include_untracked=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-untracked) include_untracked=true; shift ;;
    -h|--help) sed -n '/^# DESCRIPTION/,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

changed=false
if ! git diff --quiet --exit-code; then
  changed=true
fi

if $include_untracked && [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  changed=true
fi

echo "changed=$changed"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "changed=$changed" >> "$GITHUB_OUTPUT"
fi

if [[ "$changed" == "true" ]]; then
  git diff --stat
  if $include_untracked; then
    git ls-files --others --exclude-standard | sed 's/^/?? /'
  fi
fi
