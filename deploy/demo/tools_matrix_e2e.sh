#!/usr/bin/env bash
# REAL end-to-end matrix for `@beyond.dev/tools` endsTurn / availableInSubagents / workspace root /
# cancellation / session gate, through the production path:
#
#   beyond-agent (serve --service, minted grant) -> sandbox MCP connector -> exec front (front.py,
#   `/mcp/<session>?bundle=&tools=`) -> E2B microVM -> the COMPILED `tools-mcp` binary -> tools.
#
# The model is scripted (so model requests are counted) except the last row, which uses OPENAI_API_KEY
# with a cheap model. The driver is crates/agent/tests/agent_tools_matrix.rs.
#
#     ./deploy/demo/tools_matrix_e2e.sh                # everything
#     ./deploy/demo/tools_matrix_e2e.sh r03 r10        # only tests whose name starts with these
#
# Keys come only from the repo's .env. Every sandbox this run makes is tagged with the run name and
# deleted on exit (and the listing is checked). Output: $MATRIX_OUT (default: a temp dir).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PKG=$ROOT/sdk/ts/agent-tools-mcp
FRONT_PORT=${FRONT_PORT:-18084}
E2B_TIMEOUT=${E2B_TIMEOUT:-1200}
LOCK=${BUILD_LOCK:-$HOME/.cache/beyond-locks/rust-build.lock}
RUN=$(date +%H%M%S)
export DEMO_NAME=tools-matrix-$RUN
OUT=${MATRIX_OUT:-$(mktemp -d)}
mkdir -p "$OUT"
WORK=$(mktemp -d)

# The repo's .env (in a linked worktree: the main checkout's).
ENV_FILE=$ROOT/.env
[ -f "$ENV_FILE" ] || ENV_FILE=$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)/../.env
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }
: "${E2B_API_KEY:?E2B_API_KEY missing (repo .env)}"
export E2B_API_KEY E2B_TIMEOUT

say() { echo "-- $* --"; }
e2b() { printf 'header = "X-API-Key: %s"\n' "$E2B_API_KEY" | curl -sS -m 120 -K - "$@"; }
# bun is pinned in mise.toml; use it from there when it is not already on PATH.
command -v bun >/dev/null || PATH="$(dirname "$(cd "$ROOT" && mise which bun)"):$PATH"

FRONT_PID=""
HOST_ID=""
cleanup() {
  set +e
  [ -n "$FRONT_PID" ] && kill "$FRONT_PID" 2>/dev/null
  # Every sandbox of this run (the front's VMs and the file host), by tag.
  for id in $(e2b -G https://api.e2b.dev/v2/sandboxes --data-urlencode "state=running,paused" --data-urlencode limit=100 \
      | jq -r --arg n "$DEMO_NAME" '.[]|select(.metadata["beyond-ai-demo"]==$n)|.sandboxID'); do
    e2b -o /dev/null -w "deleted $id: %{http_code}\n" -X DELETE "https://api.e2b.dev/sandboxes/$id"
  done
  left=$(e2b -G https://api.e2b.dev/v2/sandboxes --data-urlencode "state=running,paused" --data-urlencode limit=100 \
      | jq -r --arg n "$DEMO_NAME" '[.[]|select(.metadata["beyond-ai-demo"]==$n)]|length')
  echo "sandboxes of $DEMO_NAME left: $left" | tee -a "$OUT/cleanup.log"
  rm -rf "$WORK"
}
trap 'echo "failed at line $LINENO: $BASH_COMMAND" >&2' ERR
trap cleanup EXIT

# --- 1) the compiled binary, execd, bundles ---------------------------------------------------------
say "compile tools-mcp (bun build --compile)"
(cd "$PKG" && flock -o "$LOCK" env OUT="$WORK/tools-mcp" bash scripts/compile.sh) >"$OUT/compile.log" 2>&1
BIN=$WORK/tools-mcp
[ -x "$BIN" ] || { echo "no binary"; exit 1; }
sha256sum "$BIN" | tee "$OUT/binary.sha256"

say "execd (musl) for the VMs"
EXECD_DIR=${EXECD_TARGET_DIR:-$HOME/.cache/e2e645-execd}
(cd "$ROOT" && flock -o "$LOCK" env RUSTFLAGS="-C relocation-model=static" CC_x86_64_unknown_linux_musl=musl-gcc \
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=musl-gcc \
  cargo build -q --release --locked --target x86_64-unknown-linux-musl -p beyond-execd --target-dir "$EXECD_DIR") >"$OUT/execd-build.log" 2>&1
EXECD=$EXECD_DIR/x86_64-unknown-linux-musl/release/execd

say "stage the agent folder + node_modules"
STAGE=$WORK/stage
mkdir -p "$STAGE"
cp -R "$HERE/agents/tools/." "$STAGE/"
rm -f "$STAGE/package.json"
[ -d "$PKG/../node_modules/zod" ] || (cd "$PKG/.." && npm ci --silent >/dev/null)
ZOD=$(cd "$PKG" && node -e "console.log(require('path').dirname(require.resolve('zod')))")
SPEC=$(cd "$PKG" && node -e "let p=require.resolve('@standard-schema/spec');const fs=require('fs');const path=require('path');for(;;){if(fs.existsSync(path.join(p,'package.json'))){console.log(p);break}const n=path.dirname(p);if(n===p)process.exit(1);p=n}")
mkdir -p "$STAGE/node_modules/@beyond.dev/tools" "$STAGE/node_modules/@standard-schema" "$STAGE/.claude/agents"
cp -a "$PKG/package.json" "$PKG/src" "$STAGE/node_modules/@beyond.dev/tools/"
cp -a "$PKG/eve-shim" "$STAGE/node_modules/eve"
cp -a "$ZOD" "$STAGE/node_modules/zod"
cp -a "$SPEC" "$STAGE/node_modules/@standard-schema/spec"
echo '{"type":"module"}' >"$STAGE/package.json"
echo "RIGHT-workspace" >"$STAGE/seed.txt"
cat >"$STAGE/AGENTS.md" <<'MD'
---
tools: mcp__tools__get_weather, mcp__tools__finish, mcp__tools__decide, mcp__tools__boom, mcp__tools__quiet, mcp__tools__read_seed, mcp__tools__slow, subagent
max-steps: 8
---

You are a terse helper. Use the tools you are asked to use.
MD
printf -- '---\nname: scout\ndescription: recon helper\n---\nYou are SCOUT. Reply in one line.\n' >"$STAGE/.claude/agents/scout.md"

tarit() { tar --format=ustar -C "$1" -cf "$2" . ; }
# Without a saved menu:
tarit "$STAGE" "$WORK/bundle-without.tar"
# With the menu the compiled binary itself stamps from the tool modules:
"$BIN" manifest --dir "$STAGE/tools" --name tools -o "$STAGE/.mcp-tools.json" >"$OUT/manifest.log" 2>&1
jq -c '.tools|map(.remote_name)' "$STAGE/.mcp-tools.json" | tee -a "$OUT/manifest.log"
tarit "$STAGE" "$WORK/bundle-with.tar"
ls -l "$WORK"/bundle-*.tar

# --- 2) a file host (VMs fetch the binary and the bundles over https, as they do from S3) ----------
say "file host sandbox"
resp=$(e2b -X POST https://api.e2b.dev/sandboxes -H 'Content-Type: application/json' \
  -d "$(jq -nc --arg n "$DEMO_NAME" '{templateID:"base",timeout:1200,secure:true,allow_internet_access:true,metadata:{"beyond-ai-demo":$n,session:"filehost"}}')")
HOST_ID=$(jq -r '.sandboxID // empty' <<<"$resp"); HTOK=$(jq -r '.envdAccessToken // empty' <<<"$resp")
[ -n "$HOST_ID" ] && [ -n "$HTOK" ] || { echo "file host: $(head -c 200 <<<"$resp")"; exit 1; }
put() { curl -sS -m 600 -X POST "https://49983-$HOST_ID.e2b.app/files?path=$1&username=user" -H "X-Access-Token: $HTOK" -F "file=@$2;filename=$(basename "$1")" | jq -e '.[0].path' >/dev/null; }
for i in $(seq 1 30); do put /home/user/srv/.keep "$WORK/bundle-with.tar" 2>/dev/null && break; sleep 1; done
put /home/user/srv/tools-mcp "$BIN"
put /home/user/srv/bundle-with.tar "$WORK/bundle-with.tar"
put /home/user/srv/bundle-without.tar "$WORK/bundle-without.tar"
python3 - "$HOST_ID" "$HTOK" <<'PY'
import base64, json, struct, sys, urllib.request
host, tok = sys.argv[1:3]
req = json.dumps({"process": {"cmd": "/bin/bash", "args": ["-c", "cd /home/user/srv && (setsid nohup python3 -m http.server 8000 >/tmp/http.log 2>&1 </dev/null &) ; sleep 1; curl -sf -m 3 -o /dev/null localhost:8000/ && echo up"]}}).encode()
body = b"\x00" + struct.pack(">I", len(req)) + req
r = urllib.request.Request(f"https://49983-{host}.e2b.app/process.Process/Start", data=body, headers={"X-Access-Token": tok, "Content-Type": "application/connect+json", "Connect-Protocol-Version": "1"})
urllib.request.urlopen(r, timeout=120).read()
PY
BASE=https://8000-$HOST_ID.e2b.app
for i in $(seq 1 30); do [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$BASE/bundle-with.tar")" = 200 ] && break; sleep 1; done
[ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' -I "$BASE/tools-mcp")" = 200 ] || { echo "file host not serving"; exit 1; }

# --- 3) the exec front ------------------------------------------------------------------------------
say "front on :$FRONT_PORT"
mkdir -p "$WORK/vm"
FRONT_LOG=$OUT/front.log
FRONT_PORT=$FRONT_PORT FRONT_VMDIR=$WORK/vm FRONT_E2B_SH=$HERE/e2b.sh FRONT_EXECD=$EXECD FRONT_CA_FILE=$WORK/ca.pem \
  CA_PEM="# none" python3 "$HERE/front.py" >"$FRONT_LOG" 2>"$OUT/front.err" &
FRONT_PID=$!
for i in $(seq 1 50); do curl -sf -m 1 "http://127.0.0.1:$FRONT_PORT/healthz" >/dev/null && break; sleep 0.2; done

# --- 4) the matrix ----------------------------------------------------------------------------------
say "matrix (run $RUN)"
export MATRIX_FRONT=http://127.0.0.1:$FRONT_PORT MATRIX_TOOLS_URL=$BASE/tools-mcp \
  MATRIX_BUNDLE_WITH_PATH=$WORK/bundle-with.tar MATRIX_BUNDLE_WITH_URL=$BASE/bundle-with.tar \
  MATRIX_BUNDLE_WITHOUT_PATH=$WORK/bundle-without.tar MATRIX_BUNDLE_WITHOUT_URL=$BASE/bundle-without.tar \
  MATRIX_FRONT_LOG=$FRONT_LOG MATRIX_VMCOUNT=$HERE/tools_matrix_vmcount.sh MATRIX_RUN=$RUN
FILTERS=()
for f in "$@"; do FILTERS+=("agent_tools_matrix::$f"); done
[ ${#FILTERS[@]} -gt 0 ] || FILTERS=("agent_tools_matrix::")
rc=0
(cd "$ROOT" && flock -o "$LOCK" cargo test -p beyond-agent --test it -- "${FILTERS[@]}" --test-threads=1 --nocapture) 2>&1 | tee "$OUT/matrix.log" || rc=$?
grep -E '^\[matrix |^test ' "$OUT/matrix.log" >"$OUT/matrix.summary" || true
echo "front events:"; jq -r 'select(.event!="mcp_request")|"\(.ts_ms) \(.event) \(.detail.session // "")"' "$FRONT_LOG" 2>/dev/null | tail -40
exit "$rc"
