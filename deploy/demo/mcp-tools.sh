#!/usr/bin/env bash
# The tool list of a stdio MCP server, as a bundle saves it: run the server once, ask it for `tools/list`
# and print `{"<name>": [{remote_name, description, input_schema}, ...]}`. A bundled agent advertises
# that list at open without dialing the server (the file is `.mcp-tools.json` at the bundle root), so it
# is generated from the server itself, never written by hand.
#
#     deploy/demo/mcp-tools.sh <connector-name> <server-binary> [args...]
set -euo pipefail
name=$1
shift
out=$(mktemp)
trap 'rm -f "$out"' EXIT
{
  printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"bundle-build","version":"0"}}}'
  printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/initialized"}'
  printf '%s\n' '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
  sleep 2
} | timeout 20 "$@" >"$out" || true
jq -ce --arg n "$name" 'select(.id == 2 and .result.tools != null)
  | {($n): [.result.tools[] | {remote_name: .name, description: (.description // ""), input_schema: .inputSchema}]}' "$out" \
  | head -n1 | grep . || { echo "mcp-tools: $* did not answer tools/list" >&2; exit 1; }
