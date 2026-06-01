#!/usr/bin/env bash
# DESCRIPTION: Create, refresh, or close the weekly model-drift automation PR.
#
# This wraps the GitHub Actions PR-management logic so it can be tested locally.
# Use --dry-run to print commands without mutating branches, pushing, or calling gh.
#
# USAGE:
#   ./bin/drift/pr.sh open-or-refresh [options]
#   ./bin/drift/pr.sh close-stale [options]
#
# OPTIONS:
#   --branch NAME       PR branch (default: automation/model-update)
#   --base REF          PR base branch (default: ${GITHUB_REF_NAME:-master})
#   --body PATH         PR body markdown file (default: drift-report.md)
#   --title TITLE       PR title
#   --label LABEL       PR label (default: automation)
#   --reviewer NAME     Optional reviewer, e.g. @copilot
#   --assignee NAME     Optional assignee login
#   --dry-run           Print commands instead of running them
#
# EXAMPLES:
#   ./bin/drift/pr.sh open-or-refresh --dry-run
#   ./bin/drift/pr.sh close-stale --dry-run

set -euo pipefail

cd "$(dirname "$0")/../.."

usage() {
  sed -n '/^# DESCRIPTION/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

quote_cmd() {
  {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  } >&2
}

run() {
  if [[ "$dry_run" == "true" ]]; then
    quote_cmd "$@"
  else
    "$@"
  fi
}

request_reviewer() {
  local pr_ref="$1"
  if [[ -z "$reviewer" ]]; then
    return 0
  fi
  run gh pr edit "$pr_ref" --add-reviewer "${reviewer#@}" || true
}

find_existing_pr() {
  if [[ "$dry_run" == "true" ]]; then
    quote_cmd gh pr list --head "$branch" --state open --json number --jq '.[0].number // empty'
  else
    gh pr list --head "$branch" --state open --json number --jq '.[0].number // empty'
  fi
}

cmd="${1:-}"
if [[ "$cmd" == "-h" || "$cmd" == "--help" ]]; then
  usage
  exit 0
fi
if [[ -z "$cmd" ]]; then
  usage
  exit 2
fi
shift

branch="${MODEL_DRIFT_BRANCH:-automation/model-update}"
base="${GITHUB_REF_NAME:-master}"
body="drift-report.md"
title="chore: weekly LLM SDK refresh"
label="automation"
reviewer="${MODEL_DRIFT_REVIEWER:-}"
assignee="${MODEL_DRIFT_ASSIGNEE:-}"
git_user_name="${MODEL_DRIFT_GIT_NAME:-github-actions[bot]}"
git_user_email="${MODEL_DRIFT_GIT_EMAIL:-41898282+github-actions[bot]@users.noreply.github.com}"
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch) branch="$2"; shift 2 ;;
    --base) base="$2"; shift 2 ;;
    --body) body="$2"; shift 2 ;;
    --title) title="$2"; shift 2 ;;
    --label) label="$2"; shift 2 ;;
    --reviewer) reviewer="$2"; shift 2 ;;
    --assignee) assignee="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

commit_message="chore: weekly LLM SDK refresh

- Bump openai, google-genai, xai-sdk, mcp to latest
- Regenerate OpenAI/Mistral OpenAPI TypedDicts
- Include provider model drift report
"

case "$cmd" in
  open-or-refresh)
    if [[ ! -f "$body" ]]; then
      echo "error: PR body file does not exist: $body" >&2
      exit 1
    fi

    run git config user.name "$git_user_name"
    run git config user.email "$git_user_email"
    if [[ -n "${MODEL_DRIFT_SSH_SIGNING_KEY:-}" ]]; then
      key_file="$(mktemp)"
      printf '%s\n' "$MODEL_DRIFT_SSH_SIGNING_KEY" > "$key_file"
      chmod 600 "$key_file"
      run git config gpg.format ssh
      run git config user.signingkey "$key_file"
      run git config commit.gpgsign true
    fi
    run git checkout -B "$branch"
    run git fetch origin "$branch" || true
    run git add -A
    if [[ "$dry_run" != "true" ]] && git diff --cached --quiet --exit-code; then
      echo "No staged changes; skipping PR refresh."
      exit 0
    fi
    run git commit -m "$commit_message"
    run git push --force-with-lease -u origin "$branch"

    if ! run gh label create "$label" \
      --color 1D76DB \
      --description "Maintenance PRs opened by scheduled workflows"; then
      true
    fi

    existing="$(find_existing_pr)"
    if [[ -z "$existing" ]]; then
      create_args=(
        gh pr create
        --title "$title"
        --body-file "$body"
        --label "$label"
        --base "$base"
        --head "$branch"
      )
      [[ -n "$assignee" ]] && create_args+=(--assignee "${assignee#@}")
      if [[ "$dry_run" == "true" ]]; then
        run "${create_args[@]}"
        request_reviewer "<created-pr>"
      else
        created="$("${create_args[@]}")"
        printf '%s\n' "$created"
        request_reviewer "$created"
      fi
    else
      edit_args=(gh pr edit "$existing" --body-file "$body")
      [[ -n "$assignee" ]] && edit_args+=(--add-assignee "${assignee#@}")
      run "${edit_args[@]}"
      request_reviewer "$existing"
      ts="$(date -u +%Y-%m-%dT%H:%MZ)"
      run gh pr comment "$existing" \
        --body "Refreshed at ${ts} (run #${GITHUB_RUN_NUMBER:-local}). See PR body for the latest drift report."
    fi
    ;;
  close-stale)
    existing="$(find_existing_pr)"
    if [[ -n "$existing" ]]; then
      ts="$(date -u +%Y-%m-%dT%H:%MZ)"
      run gh pr close "$existing" \
        --comment "No drift, no SDK/type changes as of ${ts} (run #${GITHUB_RUN_NUMBER:-local})."
    else
      echo "No open stale model-drift PR found for $branch."
    fi
    ;;
  *)
    echo "unknown command: $cmd" >&2
    usage
    exit 2
    ;;
esac
