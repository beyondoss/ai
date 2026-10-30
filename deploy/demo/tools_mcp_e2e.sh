#!/usr/bin/env bash
# End-to-end proof: Eve-shaped tools/*.ts → tools-mcp → E2B → MCP tools/call.
#
# Composes:
#   - sdk/ts/agent-tools-mcp (defineTool, serve, compile AOT→JIT fallback)
#   - E2B REST + envd (same style as deploy/demo/mcp-outside-agent/run.sh)
#   - deploy/demo/agents/tools
#
# Host: prove AOT compile (or --jit fallback) + manifest + local call.
# E2B: install Bun in guest, run tools-mcp from uploaded sources, MCP round-trip.
#
#     ./deploy/demo/tools_mcp_e2e.sh
#     ./deploy/demo/tools_mcp_e2e.sh --jit
#     ./deploy/demo/tools_mcp_e2e.sh --local
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PKG=$ROOT/sdk/ts/agent-tools-mcp
AGENT_DIR=$HERE/agents/tools
BRIDGE_PORT=${BRIDGE_PORT:-8931}
E2B_TIMEOUT=${E2B_TIMEOUT:-900}

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
fi

MODE=e2b
FORCE_JIT=0
for arg in "$@"; do
  case "$arg" in
    --local) MODE=local ;;
    --jit) FORCE_JIT=1 ;;
    -h|--help) sed -n '2,16p' "$0" | tr -d '#'; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

say() { echo "── $* ──"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

ID=""
STAGE=""
cleanup() {
  [ -n "$ID" ] && e2b -o /dev/null -X DELETE "https://api.e2b.dev/sandboxes/$ID" 2>/dev/null || true
  [ -n "$STAGE" ] && rm -rf "$STAGE"
}
trap cleanup EXIT

e2b() { printf 'header = "X-API-Key: %s"\n' "$E2B_API_KEY" | curl -sS -m "${E2B_CURL_MAX:-120}" -K - "$@"; }

vm_put() { # path-on-vm local-file
  curl -sS -m 120 -X POST "https://49983-$ID.e2b.app/files?path=$1&username=user" \
    -H "X-Access-Token: $TOK" -F "file=@$2;filename=$(basename "$1")" \
    | jq -e '.[0].path' >/dev/null
}

vm_bash() { # script
  local req len rf
  rf=$(mktemp)
  req=$(jq -nc --arg s "$1" '{process:{cmd:"/bin/bash",args:["-c",$s]}}')
  len=${#req}
  {
    printf '\x00'
    # shellcheck disable=SC2059
    printf "$(printf '\\x%02x\\x%02x\\x%02x\\x%02x' $((len >> 24 & 255)) $((len >> 16 & 255)) $((len >> 8 & 255)) $((len & 255)))"
    printf '%s' "$req"
  } >"$rf"
  curl -sS -m "${E2B_CURL_MAX:-180}" -X POST "https://49983-$ID.e2b.app/process.Process/Start" \
    -H "X-Access-Token: $TOK" -H 'Content-Type: application/connect+json' \
    -H 'Connect-Protocol-Version: 1' --data-binary "@$rf" \
    | tr -d '\000-\010\013-\037' | grep -o '"stdout":"[^"]*"' | sed 's/"stdout":"//;s/"$//' \
    | while read -r b; do printf '%s' "$b" | base64 -d 2>/dev/null; done
  rm -f "$rf"
}

# --- 1) Host compile ------------------------------------------------------------------------------
say "compile host tools-mcp"
rm -f "$PKG/bin/tools-mcp" "$PKG/bin/tools-mcp.mode"
if [ "$FORCE_JIT" = 1 ]; then
  FORCE_NO_AOT=1 OUT="$PKG/bin/tools-mcp" bash "$PKG/scripts/compile.sh"
  [ "$(cat "$PKG/bin/tools-mcp.mode")" = jit ] || fail "expected jit"
else
  OUT="$PKG/bin/tools-mcp" bash "$PKG/scripts/compile.sh"
fi
HOST_BIN=$PKG/bin/tools-mcp
COMPILE_MODE=$(cat "$PKG/bin/tools-mcp.mode")
pass "host binary mode=$COMPILE_MODE size=$(ls -lh "$HOST_BIN" | awk '{print $5}')"

# --- 2) Manifest + local call (real eve/tools + Zod, not a plain-object fake) --------------------
say "manifest + local call"
STAGE=$(mktemp -d)
cp -R "$AGENT_DIR/." "$STAGE/"
# Resolve `import … from "eve/tools"` + zod. STAGE is under /tmp so file: relatives
# in package.json break — link abs paths instead (same modules an install would get).
(cd "$PKG" && npm install --silent)
ZOD_DIR=$(cd "$PKG" && node -e "console.log(require('path').dirname(require.resolve('zod')))")
# Walk up from the package entry to its package root (exports often hide package.json).
SPEC_DIR=$(cd "$PKG" && node -e "let p=require.resolve('@standard-schema/spec');const fs=require('fs');const path=require('path');for(;;){if(fs.existsSync(path.join(p,'package.json'))){console.log(p);break}const n=path.dirname(p);if(n===p)process.exit(1);p=n}")
mkdir -p "$STAGE/node_modules/@beyond.dev" "$STAGE/node_modules/@standard-schema"
ln -sfn "$PKG" "$STAGE/node_modules/@beyond.dev/agent-tools-mcp"
ln -sfn "$PKG/eve-shim" "$STAGE/node_modules/eve"
ln -sfn "$ZOD_DIR" "$STAGE/node_modules/zod"
ln -sfn "$SPEC_DIR" "$STAGE/node_modules/@standard-schema/spec"
"$HOST_BIN" manifest --dir "$STAGE/tools" --name tools -o "$STAGE/.mcp-tools.json"
jq -e '.tools | length == 1 and .[0].remote_name == "get_weather"' "$STAGE/.mcp-tools.json" >/dev/null \
  || fail "manifest"
# Zod → JSON Schema must have landed in the cached tool list.
jq -e '.tools[0].input_schema.type == "object"' "$STAGE/.mcp-tools.json" >/dev/null || fail "zod json schema"
CALL=$("$HOST_BIN" call --dir "$STAGE/tools" get_weather '{"city":"NYC"}')
echo "$CALL" | jq -e '.result.structuredContent.temperatureF == 72' >/dev/null || fail "call: $CALL"
BAD=$("$HOST_BIN" call --dir "$STAGE/tools" get_weather '{"city":""}' || true)
echo "$BAD" | jq -e '.result.isError == true' >/dev/null || fail "expected Zod reject empty city: $BAD"
pass "local manifest + call (Zod validate)"

if [ "$MODE" = local ]; then
  say "local-only done"
  exit 0
fi

# --- 3) E2B (mcp-outside-agent style: no flock / execd required for MCP-only) --------------------
say "E2B sandbox"
: "${E2B_API_KEY:?set E2B_API_KEY in the environment or repo .env}"

resp=$(e2b -X POST https://api.e2b.dev/sandboxes -H 'Content-Type: application/json' \
  -d "$(jq -nc --argjson t "$E2B_TIMEOUT" '{templateID:"base",timeout:$t,secure:true,allow_internet_access:true,metadata:{"beyond-ai-demo":"tools-mcp-e2e"}}')")
ID=$(jq -r '.sandboxID // empty' <<<"$resp")
TOK=$(jq -r '.envdAccessToken // empty' <<<"$resp")
[ -n "$ID" ] && [ -n "$TOK" ] || fail "create sandbox: $(jq -c 'del(.envdAccessToken)' <<<"$resp" | head -c 300)"
pass "sandbox $ID"

say "install bun + upload sources"
echo ok >"$STAGE/keep"
for i in $(seq 1 30); do
  vm_put /home/user/.keep "$STAGE/keep" 2>/dev/null && break
  sleep 1
done
vm_bash '
set -e
curl -fsSL https://bun.sh/install | bash
export PATH="$HOME/.bun/bin:$PATH"
bun --version
mkdir -p /home/user/tools-mcp/src /home/user/workspace/tools
' >/dev/null || fail "install bun"

for f in cli.ts context.ts define-tool.ts load.ts manifest.ts rpc.ts run-tool.ts sandbox-handle.ts schema.ts serve-http.ts serve-stdio.ts eve-tools.ts index.ts bun-shim.d.ts; do
  vm_put "/home/user/tools-mcp/src/$f" "$PKG/src/$f"
done
jq -n '{
  name: "@beyond.dev/agent-tools-mcp",
  version: "0.1.0-dev",
  type: "module",
  dependencies: {
    "@standard-schema/spec": "^1.0.0",
    "zod": "4.1.5"
  }
}' >"$STAGE/tools-mcp.package.json"
vm_put /home/user/tools-mcp/package.json "$STAGE/tools-mcp.package.json"
vm_put /home/user/workspace/tools/get_weather.ts "$STAGE/tools/get_weather.ts"
printf '%s\n' '{"type":"module","dependencies":{"zod":"4.1.5"}}' >"$STAGE/guest-package.json"
vm_put /home/user/workspace/package.json "$STAGE/guest-package.json"
printf '%s\n' 'export { createToolContext, defineTool } from "/home/user/tools-mcp/src/eve-tools.ts";' \
  >"$STAGE/eve-tools.ts"
printf '%s\n' '{"name":"eve","type":"module","exports":{"./tools":"./tools.ts"}}' \
  >"$STAGE/eve-package.json"
pass "sources uploaded"

say "start tools-mcp"
START_OUT=$(vm_bash "
set -e
export PATH=\"\$HOME/.bun/bin:\$PATH\"
cd /home/user/tools-mcp && bun install --silent
cd /home/user/workspace && bun install --silent
mkdir -p /home/user/workspace/node_modules/eve
echo DEPS_OK
") || fail "bun install guest"
vm_put /home/user/workspace/node_modules/eve/tools.ts "$STAGE/eve-tools.ts"
vm_put /home/user/workspace/node_modules/eve/package.json "$STAGE/eve-package.json"
START_OUT=$(vm_bash "
set -e
export PATH=\"\$HOME/.bun/bin:\$PATH\"
cd /home/user/workspace
setsid nohup bun /home/user/tools-mcp/src/cli.ts serve --dir /home/user/workspace/tools --http 0.0.0.0:$BRIDGE_PORT \
  >/tmp/tools-mcp.log 2>&1 </dev/null &
for i in \$(seq 1 200); do
  curl -sf -m 2 localhost:$BRIDGE_PORT/healthz && echo READY && exit 0
  sleep 0.1
done
tail -n 80 /tmp/tools-mcp.log >&2 || true
exit 1
") || fail "start tools-mcp"
echo "$START_OUT" | grep -q READY || fail "no READY: $START_OUT"
pass "listening on :$BRIDGE_PORT"

MCP_URL="https://${BRIDGE_PORT}-${ID}.e2b.app/mcp"
HDR=(-H 'content-type: application/json' -H 'accept: application/json, text/event-stream')

say "MCP round-trip"
INIT_CODE=$(curl -sS "${HDR[@]}" -D "$STAGE/hdrs" -o "$STAGE/init.json" -w '%{http_code}' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"e2e","version":"0"}}}' \
  "$MCP_URL")
[ "$INIT_CODE" = 200 ] || fail "initialize HTTP $INIT_CODE $(head -c 300 "$STAGE/init.json")"
SID=$(grep -i '^mcp-session-id:' "$STAGE/hdrs" | tr -d '\r' | awk '{print $2}')
[ -n "$SID" ] || fail "missing Mcp-Session-Id"
pass "initialize"

curl -sS "${HDR[@]}" -H "mcp-session-id: $SID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' "$MCP_URL" >/dev/null || true

LIST=$(curl -sS "${HDR[@]}" -H "mcp-session-id: $SID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' "$MCP_URL")
echo "$LIST" | jq -e '.result.tools[]|select(.name=="get_weather")' >/dev/null || fail "list: $LIST"
pass "tools/list"

CALL=$(curl -sS "${HDR[@]}" -H "mcp-session-id: $SID" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_weather","arguments":{"city":"SF"}}}' \
  "$MCP_URL")
echo "$CALL" | jq -e '.result.structuredContent.city=="SF" and .result.structuredContent.source=="beyond-agent-tools-mcp"' >/dev/null \
  || fail "call: $CALL"
pass "tools/call on E2B"

say "all good (host_compile=$COMPILE_MODE)"
echo "OK"
