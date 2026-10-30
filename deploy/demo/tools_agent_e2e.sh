#!/usr/bin/env bash
# REAL end-to-end: beyond-agent → tools-mcp in an E2B sandbox → get_weather.
#
# Same shape as production sandbox MCP: agent dials an HTTP MCP URL with private
# loopback allowed (in-VM), model is real (OpenAI / OpenRouter / Anthropic from .env).
#
#     ./deploy/demo/tools_agent_e2e.sh            # build musl agent if needed, then prove
#     ./deploy/demo/tools_agent_e2e.sh --no-build
#     ./deploy/demo/tools_agent_e2e.sh --keep
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PKG=$ROOT/sdk/ts/agent-tools-mcp
AGENT_DIR=$HERE/agents/tools
BRIDGE_PORT=${BRIDGE_PORT:-8931}
E2B_TIMEOUT=${E2B_TIMEOUT:-1200}
TARGET=x86_64-unknown-linux-musl
PROFILE=${PROFILE:-release}
KEEP=0
NO_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --keep) KEEP=1 ;;
    --no-build) NO_BUILD=1 ;;
    -h|--help) sed -n '2,14p' "$0" | tr -d '#'; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
fi

say() { echo "── $* ──"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

ID=""
STAGE=""
cleanup() {
  if [ "$KEEP" = 1 ] && [ -n "$ID" ]; then
    echo "KEEP: sandbox $ID still live" >&2
    return
  fi
  [ -n "$ID" ] && e2b -o /dev/null -X DELETE "https://api.e2b.dev/sandboxes/$ID" 2>/dev/null || true
  [ -n "$STAGE" ] && rm -rf "$STAGE"
}
trap cleanup EXIT

e2b() { printf 'header = "X-API-Key: %s"\n' "$E2B_API_KEY" | curl -sS -m "${E2B_CURL_MAX:-120}" -K - "$@"; }
vm_put() {
  curl -sS -m 300 -X POST "https://49983-$ID.e2b.app/files?path=$1&username=user" \
    -H "X-Access-Token: $TOK" -F "file=@$2;filename=$(basename "$1")" \
    | jq -e '.[0].path' >/dev/null
}
vm_get() {
  curl -sS -m 120 "https://49983-$ID.e2b.app/files?path=$1&username=user" \
    -H "X-Access-Token: $TOK"
}
vm_bash() {
  local req len rf script=$1 marker out rc
  marker=$(printf '/tmp/vm_bash.%s.rc' "$RANDOM$RANDOM")
  rf=$(mktemp)
  script="set +e
trap 'printf %s \$? >'$marker'' EXIT
$script"
  req=$(jq -nc --arg s "$script" '{process:{cmd:"/bin/bash",args:["-c",$s]}}')
  len=${#req}
  {
    printf '\x00'
    # shellcheck disable=SC2059
    printf "$(printf '\\x%02x\\x%02x\\x%02x\\x%02x' $((len >> 24 & 255)) $((len >> 16 & 255)) $((len >> 8 & 255)) $((len & 255)))"
    printf '%s' "$req"
  } >"$rf"
  out=$(curl -sS -m "${E2B_CURL_MAX:-600}" -X POST "https://49983-$ID.e2b.app/process.Process/Start" \
    -H "X-Access-Token: $TOK" -H 'Content-Type: application/connect+json' \
    -H 'Connect-Protocol-Version: 1' --data-binary "@$rf" \
    | tr -d '\000-\010\013-\037' | grep -o '"stdout":"[^"]*"' | sed 's/"stdout":"//;s/"$//' \
    | while read -r b; do printf '%s' "$b" | base64 -d 2>/dev/null; done)
  rm -f "$rf"
  printf '%s' "$out"
  rc=$(vm_get "$marker" 2>/dev/null | tr -d '[:space:]')
  [[ "$rc" =~ ^[0-9]+$ ]] || rc=1
  return "$rc"
}

: "${E2B_API_KEY:?set E2B_API_KEY in the environment or repo .env}"

# Prefer Anthropic for reliable tool-calling in this tiny proof; override with MODEL=/PROVIDER_ENV=.
PROVIDER_ENV=${PROVIDER_ENV:-}
MODEL=${MODEL:-}
if [ -z "$PROVIDER_ENV" ]; then
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    PROVIDER_ENV=ANTHROPIC_API_KEY
    MODEL=${MODEL:-claude-haiku-4-5}
  elif [ -n "${OPENAI_API_KEY:-}" ]; then
    PROVIDER_ENV=OPENAI_API_KEY
    MODEL=${MODEL:-gpt-4o-mini}
  elif [ -n "${OPENROUTER_API_KEY:-}" ]; then
    PROVIDER_ENV=OPENROUTER_API_KEY
    MODEL=${MODEL:-openai/gpt-4o-mini}
  else
    fail "need ANTHROPIC_API_KEY or OPENAI_API_KEY or OPENROUTER_API_KEY"
  fi
fi
pass "provider=$PROVIDER_ENV model=$MODEL"

# --- 0) musl agent --------------------------------------------------------------------------------
say "linux musl agent"
export CC_x86_64_unknown_linux_musl=${CC_x86_64_unknown_linux_musl:-x86_64-linux-musl-gcc}
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=${CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER:-x86_64-linux-musl-gcc}
export RUSTFLAGS="${RUSTFLAGS:--C target-feature=+crt-static -C link-arg=-static}"
command -v "$CC_x86_64_unknown_linux_musl" >/dev/null || fail "need $CC_x86_64_unknown_linux_musl"

if [ "$NO_BUILD" = 0 ]; then
  (cd "$ROOT" && cargo build -p beyond-agent --target "$TARGET" --"$PROFILE") || fail "cargo build agent"
fi
AGENT_BIN=$ROOT/target/$TARGET/$PROFILE/agent
[ -x "$AGENT_BIN" ] || fail "missing $AGENT_BIN"
pass "agent=$(ls -lh "$AGENT_BIN" | awk '{print $5}')"

# --- 1) E2B + tools-mcp ---------------------------------------------------------------------------
say "E2B sandbox"
STAGE=$(mktemp -d)
cp -R "$AGENT_DIR/." "$STAGE/"
# HTTP MCP connector the agent will dial (in-VM loopback; --mcp-allow-private).
cat >"$STAGE/.mcp.json" <<EOF
{
  "mcpServers": {
    "tools": {
      "type": "http",
      "url": "http://127.0.0.1:${BRIDGE_PORT}/mcp"
    }
  }
}
EOF

resp=$(e2b -X POST https://api.e2b.dev/sandboxes -H 'Content-Type: application/json' \
  -d "$(jq -nc --argjson t "$E2B_TIMEOUT" '{templateID:"base",timeout:$t,secure:true,allow_internet_access:true,metadata:{"beyond-ai-demo":"tools-agent-e2e"}}')")
ID=$(jq -r '.sandboxID // empty' <<<"$resp")
TOK=$(jq -r '.envdAccessToken // empty' <<<"$resp")
[ -n "$ID" ] && [ -n "$TOK" ] || fail "create sandbox: $(jq -c 'del(.envdAccessToken)' <<<"$resp" | head -c 300)"
pass "sandbox $ID"

echo ok >"$STAGE/keep"
for i in $(seq 1 30); do
  vm_put /home/user/.keep "$STAGE/keep" 2>/dev/null && break
  sleep 1
done

say "install bun + tools-mcp + workspace"
vm_bash '
set -e
curl -fsSL https://bun.sh/install | bash
export PATH="$HOME/.bun/bin:$PATH"
bun --version
mkdir -p /home/user/tools-mcp/src /home/user/workspace/tools /home/user/bin
' >/dev/null || fail "install bun"

for f in cli.ts define-tool.ts load.ts manifest.ts rpc.ts run-tool.ts schema.ts serve-http.ts serve-stdio.ts eve-tools.ts index.ts bun-shim.d.ts; do
  vm_put "/home/user/tools-mcp/src/$f" "$PKG/src/$f"
done
vm_put /home/user/tools-mcp/package.json "$PKG/package.json"
vm_put /home/user/workspace/tools/get_weather.ts "$STAGE/tools/get_weather.ts"
vm_put /home/user/workspace/package.json "$STAGE/package.json"
vm_put /home/user/workspace/AGENTS.md "$STAGE/AGENTS.md"
vm_put /home/user/workspace/.mcp.json "$STAGE/.mcp.json"
vm_bash '
set -e
export PATH="$HOME/.bun/bin:$PATH"
cd /home/user/tools-mcp && bun install --silent
mkdir -p /home/user/workspace/node_modules/eve
printf "%s\n" "{\"name\":\"eve\",\"type\":\"module\",\"exports\":{\"./tools\":\"/home/user/tools-mcp/src/eve-tools.ts\"}}" > /home/user/workspace/node_modules/eve/package.json
cd /home/user/workspace && bun add zod@^4.1.5 --silent
' >/dev/null || fail "bun install workspace"

# Provider key as a file — never on argv.
# shellcheck disable=SC2086
eval "KEY_VAL=\$$PROVIDER_ENV"
printf '%s' "$KEY_VAL" >"$STAGE/provider.key"
vm_put /home/user/bin/provider.key "$STAGE/provider.key"

say "upload agent (gzip)"
gzip -c -9 "$AGENT_BIN" >"$STAGE/agent.gz"
vm_put /home/user/bin/agent.gz "$STAGE/agent.gz"
vm_bash "
set +e
exec >/tmp/place-agent.log 2>&1
gzip -dc /home/user/bin/agent.gz > /home/user/bin/agent || exit 12
chmod 755 /home/user/bin/agent
/home/user/bin/agent --help >/dev/null || exit 16
echo PLACE_OK
" || {
  vm_get /tmp/place-agent.log 2>/dev/null | tail -n 40 >&2 || true
  fail "place agent"
}
pass "agent on guest"

say "start tools-mcp"
vm_bash "
set +e
export PATH=\"\$HOME/.bun/bin:\$PATH\"
cd /home/user/tools-mcp
bun run src/cli.ts serve --dir /home/user/workspace/tools --http 127.0.0.1:$BRIDGE_PORT \
  >/tmp/tools-mcp.log 2>&1 </dev/null &
BPID=\$!
for i in \$(seq 1 100); do
  code=\$(curl -s -o /dev/null -w '%{http_code}' -m 2 http://127.0.0.1:$BRIDGE_PORT/healthz || echo err)
  if [ \"\$code\" = 200 ]; then
    echo READY >/tmp/tools-mcp.ready
    exit 0
  fi
  kill -0 \$BPID 2>/dev/null || { echo bun_died; break; }
  sleep 0.1
done
exit 1
" || {
  vm_get /tmp/tools-mcp.log 2>/dev/null | tail -n 80 >&2 || true
  fail "start tools-mcp"
}
vm_get /tmp/tools-mcp.ready 2>/dev/null | grep -q READY || fail "no READY"
pass "tools-mcp on 127.0.0.1:$BRIDGE_PORT"

# --- 2) Real agent → sandbox MCP ------------------------------------------------------------------
say "agent run → sandbox get_weather (live $MODEL)"
# Export only the chosen provider var inside the guest.
case "$PROVIDER_ENV" in
  OPENAI_API_KEY) EXPORT_LINE='export OPENAI_API_KEY=$(cat /home/user/bin/provider.key)' ;;
  OPENROUTER_API_KEY) EXPORT_LINE='export OPENROUTER_API_KEY=$(cat /home/user/bin/provider.key); export AI_PROVIDER=openrouter' ;;
  ANTHROPIC_API_KEY) EXPORT_LINE='export ANTHROPIC_API_KEY=$(cat /home/user/bin/provider.key)' ;;
esac

vm_bash "
set +e
exec >/tmp/agent-run.out 2>&1
export HOME=/home/user
export TMPDIR=/tmp
export PATH=\"\$HOME/.bun/bin:\$PATH\"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY='*'
$EXPORT_LINE
cd /home/user/workspace
: >/tmp/agent-run.jsonl
: >/tmp/agent-run.err
curl -sf -m 2 http://127.0.0.1:$BRIDGE_PORT/healthz >/tmp/hz.out || { echo 'tools-mcp down' >>/tmp/agent-run.err; echo 3 >/tmp/agent-run.rc; exit 3; }
# Substrate MCP round-trip (same URL as .mcp.json) before the agent spends a model call.
MCP=http://127.0.0.1:$BRIDGE_PORT/mcp
CH='content-type: application/json'
CA='accept: application/json, text/event-stream'
curl -sS -H \"\$CH\" -H \"\$CA\" -D /tmp/mcp.hdr -o /tmp/mcp.init \
  -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"probe\",\"version\":\"0\"}}}' \
  \"\$MCP\" >/dev/null
SID=\$(grep -i '^mcp-session-id:' /tmp/mcp.hdr | tr -d '\\r' | awk '{print \$2}')
curl -sS -H \"\$CH\" -H \"\$CA\" -H \"mcp-session-id: \$SID\" \
  -d '{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}' \"\$MCP\" >/dev/null
curl -sS -H \"\$CH\" -H \"\$CA\" -H \"mcp-session-id: \$SID\" -o /tmp/mcp.call \
  -d '{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}}' \
  \"\$MCP\"
if ! jq -e '.result.structuredContent.temperatureF == 72 and .result.structuredContent.city == \"SF\"' /tmp/mcp.call >/dev/null; then
  echo PROBE_FAIL >>/tmp/agent-run.err
  cat /tmp/mcp.call >>/tmp/agent-run.err
  echo 4 >/tmp/agent-run.rc
  exit 4
fi
echo PROBE_OK >>/tmp/agent-run.err

export AI_LOG=info
# .mcp.json is merged only for a *persisted* trust entry (effective_settings_for_cwd),
# not for one-off --trust-project. Trust the workspace, then run.
/home/user/bin/agent trust /home/user/workspace >>/tmp/agent-run.err 2>&1
/home/user/bin/agent run \
  --no-memory \
  --mcp-tool-loading eager \
  --model '$MODEL' \
  --max-steps 6 \
  --json \
  --prompt-guideline 'The only tool you have is mcp__tools__get_weather. You must call it. Never invent weather numbers.' \
  'Call mcp__tools__get_weather with {\"city\":\"SF\"}. Then reply with exactly: TEMP=<temperatureF> COND=<condition> CITY=<city> using values from the tool result only.' \
  >/tmp/agent-run.jsonl 2>>/tmp/agent-run.err
echo \$? >/tmp/agent-run.rc
" || true

vm_get /tmp/agent-run.jsonl >"$STAGE/agent-run.jsonl" 2>/dev/null || echo '{}' >"$STAGE/agent-run.jsonl"
vm_get /tmp/agent-run.err >"$STAGE/agent-run.err" 2>/dev/null || true
vm_get /tmp/agent-run.rc >"$STAGE/agent-run.rc" 2>/dev/null || echo 1 >"$STAGE/agent-run.rc"
RC=$(tr -d '[:space:]' <"$STAGE/agent-run.rc")
echo "── agent stderr ──"
cat "$STAGE/agent-run.err" 2>/dev/null | tail -n 80 || true
echo "── tool-ish jsonl lines ──"
rg -n 'tool_use|tool_result|get_weather|temperatureF|\"72\"|sunny' "$STAGE/agent-run.jsonl" | head -n 40 || true
[[ "$RC" =~ ^[0-9]+$ ]] || fail "bad rc: $RC"
[ "$RC" = 0 ] || fail "agent run exited $RC"
grep -q PROBE_OK "$STAGE/agent-run.err" || fail "substrate MCP probe failed"
# Tool invocation + stub payload (get_weather.ts returns temperatureF: 72).
rg -q 'mcp__tools__get_weather' "$STAGE/agent-run.jsonl" \
  || fail "no mcp__tools__get_weather in the transcript"
rg -q 'temperatureF.:.?72|\"72\"|TEMP=72' "$STAGE/agent-run.jsonl" \
  || fail "stub temperatureF=72 never appeared (tool may have failed)"
pass "agent → E2B tools-mcp get_weather (live model)"
echo "OK"
