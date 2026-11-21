#!/usr/bin/env bash
# Live check of sandbox-mcp's Streamable HTTP mode with curl and a real Claude Code client.
# usage: live-http-check.sh <sandbox-mcp-bin> <execd-bin> [--no-claude]
# Starts a private execd and a sandbox-mcp on loopback, asserts:
#   Origin: none / http://localhost:PORT -> accepted; https://evil.example -> 403 (EXPECT_ORIGIN_BLOCK=0 inverts, for a build without the check)
#   tools/call of an unknown tool -> JSON-RPC error -32602 (EXPECT_UNKNOWN_TOOL=rpc, or "iserror" for the older shape)
#   claude -p calls the exec tool through `claude mcp add --transport http` config and prints the output.
# Claude's MCP config is written by `claude mcp add --scope project` into a throwaway cwd and passed with
# --strict-mcp-config, so the user's ~/.claude.json mcpServers is never written; its hash is checked before/after.
set -u
MCP=$1; EXECD=$2; WITH_CLAUDE=1; [ "${3:-}" = --no-claude ] && WITH_CLAUDE=0
BLOCK=${EXPECT_ORIGIN_BLOCK:-1}; UNK=${EXPECT_UNKNOWN_TOOL:-rpc}
W=$(mktemp -d); PIDS=(); fail=0
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; rm -rf "$W"; }
trap cleanup EXIT
ok() { echo "PASS $1"; }; bad() { echo "FAIL $1 ($2)"; fail=1; }
mcpsum() { python3 -I -c 'import json,sys,hashlib;print(hashlib.sha256(json.dumps(json.load(open(sys.argv[1])).get("mcpServers"),sort_keys=True).encode()).hexdigest()[:12])' "$HOME/.claude.json" 2>/dev/null; }

TOKEN=$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n'); umask 077; printf %s "$TOKEN" > "$W/bearer"
EP=$((20000 + RANDOM % 20000))
EXECD_LISTEN=127.0.0.1:$EP EXECD_ROOT=$W EXECD_BEARER_FILE=$W/bearer "$EXECD" >"$W/execd.log" 2>&1 & PIDS+=($!)
sleep 1
SANDBOX_URL=http://127.0.0.1:$EP SANDBOX_TOKEN=$TOKEN SANDBOX_MCP_LISTEN=127.0.0.1:0 "$MCP" 2>"$W/mcp.err" >/dev/null & PIDS+=($!)
for _ in $(seq 50); do URL=$(sed -n 's/.*listening on //p' "$W/mcp.err" | head -1); [ -n "$URL" ] && break; sleep 0.2; done
[ -n "${URL:-}" ] || { echo "FAIL start (no listen line)"; cat "$W/mcp.err"; exit 1; }
PORT=${URL##*:}; PORT=${PORT%%/*}; echo "server $URL"

H=(-H 'content-type: application/json' -H 'accept: application/json, text/event-stream')
INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"live-check","version":"0"}}}'
code() { curl -s -o "$W/body" -w '%{http_code}' "${H[@]}" "$@" -d "$INIT" "$URL"; }
c=$(code); [ "$c" = 200 ] && ok "no Origin -> 200" || bad "no Origin -> 200" "$c"
c=$(code -H "Origin: http://localhost:$PORT"); [ "$c" = 200 ] && ok "Origin localhost -> 200" || bad "Origin localhost -> 200" "$c"
c=$(code -H "Origin: https://evil.example")
if [ "$BLOCK" = 1 ]; then [ "$c" = 403 ] && ok "Origin evil -> 403" || bad "Origin evil -> 403" "$c"
else [ "$c" = 200 ] && ok "Origin evil -> 200 (no check)" || bad "Origin evil -> 200" "$c"; fi

# unknown tool, on a real session
curl -s -D "$W/hdr" -o "$W/body" "${H[@]}" -d "$INIT" "$URL"; SID=$(awk -F': ' 'tolower($1)=="mcp-session-id"{print $2}' "$W/hdr" | tr -d '\r')
S=(-H "mcp-session-id: $SID" -H 'mcp-protocol-version: 2025-06-18')
curl -s -o /dev/null "${H[@]}" "${S[@]}" -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' "$URL"
curl -s "${H[@]}" "${S[@]}" -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"no_such_tool","arguments":{}}}' "$URL" > "$W/unk"
echo "unknown-tool reply: $(tr -d '\n' < "$W/unk" | cut -c1-300)"
if [ "$UNK" = rpc ]; then grep -q '"code":-32602' "$W/unk" && ok "unknown tool -> -32602" || bad "unknown tool -> -32602" "see reply"
else grep -q '"isError":true' "$W/unk" && ok "unknown tool -> isError result" || bad "unknown tool -> isError" "see reply"; fi

if [ $WITH_CLAUDE = 1 ]; then
  before=$(mcpsum)
  mkdir "$W/cwd"; (cd "$W/cwd" && claude mcp add --transport http --scope project sbx "$URL" >"$W/add.log" 2>&1)
  out=$(cd "$W/cwd" && claude -p --strict-mcp-config --mcp-config "$W/cwd/.mcp.json" --allowedTools 'mcp__sbx__exec' --model haiku \
    'Call the sbx exec tool with command "echo live-check-$((6*7))" and reply with only the stdout it returned.' 2>&1)
  echo "claude said: $out"
  echo "$out" | grep -q 'live-check-42' && ok "claude -p exec tool call" || bad "claude -p exec tool call" "no live-check-42"
  [ "$before" = "$(mcpsum)" ] && ok "~/.claude.json mcpServers unchanged" || bad "~/.claude.json mcpServers unchanged" "changed"
fi
exit $fail
