#!/usr/bin/env bash
# Run a small FrontierHarness v1.0 Terminal-Bench subset through Harbor.
# Verifier pass/fail, turns, tokens, $. Not the Code Mode MCP smokes.
#
# Three arms on Terminal-Bench (no MCP catalog):
#   HARNESS=beyond              density-default musl binary (no QuickJS)
#   HARNESS=beyond-code-mode    --features code-mode binary + --code-mode
#   HARNESS=pi                  Harbor Pi 0.84.2 (FrontierHarness pin)
#
# Code Mode *with* MCP (local Harbor task, stdlib AST server — not the echo fixture):
#   HARNESS=beyond-mcp              flattened mcp__ast__* tools
#   HARNESS=beyond-mcp-code-mode    same catalog behind execute
#
# beyond* advertised default is read/write/edit/bash/ls/grep/find/todo. `web` stays
# in the binary and is opt-in via `--tools`.
#
# Self-improvement loop: find the expensive task, change one thing, re-run
# that task against Pi. Override TASKS (space or comma separated):
#   TASKS=terminal-bench/polyglot-c-py HARNESS=beyond JOBS_DIR=/tmp/poly-beyond \
#     crates/agent/eval/run_frontier_subset.sh
#   TASKS=terminal-bench/polyglot-c-py HARNESS=pi JOBS_DIR=/tmp/poly-pi \
#     crates/agent/eval/run_frontier_subset.sh
#   HARNESS=beyond-mcp JOBS_DIR=/tmp/ast-flat crates/agent/eval/run_frontier_subset.sh
#   HARNESS=beyond-mcp-code-mode JOBS_DIR=/tmp/ast-code crates/agent/eval/run_frontier_subset.sh
#
# Terminal-Bench has no MCP catalog. beyond vs beyond-code-mode is a
# regression/overhead check (empty `execute` still registers). Pi is the
# harness baseline. This is not the 30-task FrontierHarness board.
#
# Prerequisites: docker, harbor (`uv tool install harbor`), OPENROUTER_API_KEY.
# beyond:        cargo build -p beyond-ai-agent --release --target x86_64-unknown-linux-musl
# beyond-code-mode: same plus --features code-mode, then copy the artifact to
#   target/x86_64-unknown-linux-musl/release/beyond-ai-agent-code-mode
#   (cargo overwrites the same filename).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}/crates/agent/eval${PYTHONPATH:+:${PYTHONPATH}}"

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"

JOBS_DIR="${JOBS_DIR:-/tmp/beyond-frontier-jobs}"
MODEL="${MODEL:-moonshotai/kimi-k3}"
HARNESS="${HARNESS:-beyond}"
DEFAULT_BIN="${ROOT}/target/x86_64-unknown-linux-musl/release/beyond-ai-agent"
CODE_MODE_BIN="${BEYOND_AI_AGENT_CODE_MODE_BIN:-${ROOT}/target/x86_64-unknown-linux-musl/release/beyond-ai-agent-code-mode}"

# FrontierHarness v1.0 TB slice: four medium tasks with prebuilt images.
# Full set is 21 TB + 9 DeepSWE; DeepSWE needs Pier, not this script.
# Override with TASKS=terminal-bench/polyglot-c-py (space/comma separated).
if [[ -n "${TASKS:-}" ]]; then
  IFS=$' ,\n' read -r -a TASK_LIST <<< "${TASKS}"
else
  TASK_LIST=(
    terminal-bench/openssl-selfsigned-cert
    terminal-bench/sanitize-git-repo
    terminal-bench/polyglot-c-py
    terminal-bench/sqlite-db-truncate
  )
fi

require_bin() {
  local path="$1"
  local hint="$2"
  if [[ ! -f "${path}" ]]; then
    echo "missing binary: ${path}" >&2
    echo "${hint}" >&2
    exit 1
  fi
}

mkdir -p "${JOBS_DIR}"
task_args=()
for t in "${TASK_LIST[@]}"; do
  task_args+=(--include-task-name "${t}")
done

common=(
  --dataset terminal-bench/terminal-bench-2-1
  "${task_args[@]}"
  --n-concurrent 1
  --jobs-dir "${JOBS_DIR}"
  --allow-agent-host openrouter.ai
)

case "${HARNESS}" in
  beyond)
    export BEYOND_AI_AGENT_BIN="${BEYOND_AI_AGENT_BIN:-${DEFAULT_BIN}}"
    require_bin "${BEYOND_AI_AGENT_BIN}" \
      "build: cargo build -p beyond-ai-agent --release --target x86_64-unknown-linux-musl"
    exec harbor run \
      "${common[@]}" \
      --agent harbor_beyond_agent:BeyondAiAgent \
      --model "${MODEL}" \
      --ae AI_PROVIDER=openrouter \
      --ae AI_DIRECT=1 \
      --ae "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}" \
      "$@"
    ;;
  beyond-code-mode)
    # Do not reuse BEYOND_AI_AGENT_BIN: that env often points at the density
    # binary, and `--code-mode` on a default build is a hard CLI error.
    export BEYOND_AI_AGENT_BIN="${CODE_MODE_BIN}"
    require_bin "${BEYOND_AI_AGENT_BIN}" \
      "build: cargo build -p beyond-ai-agent --release --target x86_64-unknown-linux-musl --features code-mode && cp target/x86_64-unknown-linux-musl/release/beyond-ai-agent ${CODE_MODE_BIN}"
    exec harbor run \
      "${common[@]}" \
      --agent harbor_beyond_agent:BeyondAiAgentCodeMode \
      --model "${MODEL}" \
      --ak code_mode=true \
      --ae AI_PROVIDER=openrouter \
      --ae AI_DIRECT=1 \
      --ae "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}" \
      "$@"
    ;;
  pi)
    # FrontierHarness v1.0 pin. Model id is provider/model for Harbor's Pi adapter.
    # nvm + Node 22 + npm install runs in each task container; 360s default setup
    # timeout is tight on nested Docker / vfs.
    exec harbor run \
      "${common[@]}" \
      --agent pi \
      --model "openrouter/${MODEL}" \
      --ak version=0.84.2 \
      --agent-setup-timeout-multiplier 4 \
      --ae "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}" \
      "$@"
    ;;
  beyond-mcp|beyond-mcp-code-mode)
    ast_task="${ROOT}/crates/agent/eval/tasks/ast-hotspots"
    mcp_common=(
      --path "${ast_task}"
      --n-concurrent 1
      --jobs-dir "${JOBS_DIR}"
      --allow-agent-host openrouter.ai
    )
    if [[ "${HARNESS}" == "beyond-mcp-code-mode" ]]; then
      export BEYOND_AI_AGENT_BIN="${CODE_MODE_BIN}"
      require_bin "${BEYOND_AI_AGENT_BIN}" \
        "build: cargo build -p beyond-ai-agent --release --target x86_64-unknown-linux-musl --features code-mode && cp target/x86_64-unknown-linux-musl/release/beyond-ai-agent ${CODE_MODE_BIN}"
      exec harbor run \
        "${mcp_common[@]}" \
        --agent harbor_beyond_agent:BeyondAiAgentCodeMode \
        --model "${MODEL}" \
        --ak code_mode=true \
        --ak mcp_ast=true \
        --ae AI_PROVIDER=openrouter \
        --ae AI_DIRECT=1 \
        --ae "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}" \
        "$@"
    fi
    export BEYOND_AI_AGENT_BIN="${BEYOND_AI_AGENT_BIN:-${DEFAULT_BIN}}"
    require_bin "${BEYOND_AI_AGENT_BIN}" \
      "build: cargo build -p beyond-ai-agent --release --target x86_64-unknown-linux-musl"
    exec harbor run \
      "${mcp_common[@]}" \
      --agent harbor_beyond_agent:BeyondAiAgent \
      --model "${MODEL}" \
      --ak mcp_ast=true \
      --ae AI_PROVIDER=openrouter \
      --ae AI_DIRECT=1 \
      --ae "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}" \
      "$@"
    ;;
  *)
    echo "HARNESS must be beyond, beyond-code-mode, pi, beyond-mcp, or beyond-mcp-code-mode (got ${HARNESS})" >&2
    exit 1
    ;;
esac
