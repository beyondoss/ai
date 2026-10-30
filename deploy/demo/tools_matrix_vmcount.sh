#!/usr/bin/env bash
# How many E2B sandboxes (running or paused) carry this front session in their metadata.
#     tools_matrix_vmcount.sh <session>      needs E2B_API_KEY
set -euo pipefail
printf 'header = "X-API-Key: %s"\n' "$E2B_API_KEY" \
  | curl -sS -m 60 -K - -G "https://api.e2b.dev/v2/sandboxes" \
    --data-urlencode "state=running,paused" --data-urlencode "limit=100" \
  | jq --arg s "$1" --arg n "${DEMO_NAME:-}" \
    'if type=="array" then [.[] | select(.metadata.session==$s and (.metadata["beyond-ai-demo"]==$n))] | length else error("unexpected list answer") end'
