#!/usr/bin/env bash
# Run a small FrontierHarness v1.0 Terminal-Bench subset through Harbor.
# Verifier pass/fail, turns, tokens, $. Not the Code Mode MCP smokes.
#
# Prerequisites: docker, harbor (`uv tool install harbor`), OPENROUTER_API_KEY,
# and a musl release binary:
#   cargo build -p beyond-ai-agent --release --target x86_64-unknown-linux-musl
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}/crates/agent/eval${PYTHONPATH:+:${PYTHONPATH}}"

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
: "${BEYOND_AI_AGENT_BIN:=${ROOT}/target/x86_64-unknown-linux-musl/release/beyond-ai-agent}"
if [[ ! -f "${BEYOND_AI_AGENT_BIN}" ]]; then
  echo "missing binary: ${BEYOND_AI_AGENT_BIN}" >&2
  echo "build: cargo build -p beyond-ai-agent --release --target x86_64-unknown-linux-musl" >&2
  exit 1
fi
export BEYOND_AI_AGENT_BIN

JOBS_DIR="${JOBS_DIR:-/tmp/beyond-frontier-jobs}"
MODEL="${MODEL:-moonshotai/kimi-k3}"
# FrontierHarness v1.0 TB slice: four medium tasks with prebuilt images.
# Full set is 21 TB + 9 DeepSWE; DeepSWE needs Pier, not this script.
TASKS=(
  openssl-selfsigned-cert
  sanitize-git-repo
  polyglot-c-py
  sqlite-db-truncate
)

mkdir -p "${JOBS_DIR}"
task_args=()
for t in "${TASKS[@]}"; do
  task_args+=(--include-task-name "${t}")
done

exec harbor run \
  --dataset terminal-bench/terminal-bench-2-1 \
  "${task_args[@]}" \
  --agent harbor_beyond_agent:BeyondAiAgent \
  --model "${MODEL}" \
  --allow-agent-host openrouter.ai \
  --ae AI_PROVIDER=openrouter \
  --ae AI_DIRECT=1 \
  --ae "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}" \
  --n-concurrent 1 \
  --jobs-dir "${JOBS_DIR}" \
  "$@"
