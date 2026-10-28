#!/usr/bin/env bash
# An outside agent (OpenAI Agents SDK, which we did not write) does a coding task in a real E2B VM through
# sandbox-mcp alone. Needs E2B_API_KEY and OPENAI_API_KEY in the environment (nothing is read from a file).
#   MODEL=gpt-5-mini  COST_CAP_USD=0.25  VERSION=v0.1.16  E2B_TIMEOUT=1200  ./run.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${E2B_API_KEY:?}" "${OPENAI_API_KEY:?}"
VERSION=${VERSION:-v0.1.16}; export MODEL=${MODEL:-gpt-5-mini}; E2B_TIMEOUT=${E2B_TIMEOUT:-1200}
WORK=${WORK:-$(mktemp -d)}; export OUT_DIR=$WORK/out; mkdir -p "$OUT_DIR" "$WORK/dl"
REL=https://github.com/beyondoss/ai/releases/download/$VERSION
e2b() { printf 'header = "X-API-Key: %s"\n' "$E2B_API_KEY" | curl -sS -m 60 -K - "$@"; }

for n in execd sandbox-mcp; do
  f=$n-$VERSION-linux-amd64.tar.gz
  curl -fsSL -o "$WORK/dl/$f" "$REL/$f"; curl -fsSL -o "$WORK/dl/$f.sha256" "$REL/$f.sha256"
  (cd "$WORK/dl" && sha256sum -c "$f.sha256") >&2
  mkdir -p "$WORK/$n" && tar -xzf "$WORK/dl/$f" -C "$WORK/$n"
done
EXECD=$(find "$WORK/execd" -type f -name execd | head -1); export SANDBOX_MCP_BIN=$(find "$WORK/sandbox-mcp" -type f -name sandbox-mcp | head -1)

[ -d "$WORK/venv" ] || { python3 -m venv "$WORK/venv"; "$WORK/venv/bin/pip" install -q openai-agents; }

BEARER=$(openssl rand -hex 24); ID=""
cleanup() { [ -z "$ID" ] || e2b -o /dev/null -X DELETE "https://api.e2b.dev/sandboxes/$ID" || true
  echo "E2B still running/paused: $(e2b -G https://api.e2b.dev/v2/sandboxes --data-urlencode state=running,paused | jq -r --arg i "$ID" '[.[]|select(.sandboxID==$i)]|length')" >&2; }
trap cleanup EXIT
resp=$(e2b -X POST https://api.e2b.dev/sandboxes -H 'Content-Type: application/json' \
  -d "{\"templateID\":\"base\",\"timeout\":$E2B_TIMEOUT,\"secure\":true,\"allow_internet_access\":true,\"metadata\":{\"beyond-ai-demo\":\"mcp-outside-agent\"}}")
ID=$(jq -r '.sandboxID // empty' <<<"$resp"); TOK=$(jq -r '.envdAccessToken // empty' <<<"$resp")
[ -n "$ID" ] && [ -n "$TOK" ] || { echo "no sandbox" >&2; exit 1; }
for i in $(seq 30); do curl -sS -m 120 -X POST "https://49983-$ID.e2b.app/files?path=/home/user/execd&username=user" -H "X-Access-Token: $TOK" -F "file=@$EXECD;filename=execd" >/dev/null 2>&1 && break; sleep 1; done
printf '%s' "$BEARER" > "$WORK/bearer"
curl -sS -m 60 -X POST "https://49983-$ID.e2b.app/files?path=/home/user/bearer&username=user" -H "X-Access-Token: $TOK" -F "file=@$WORK/bearer;filename=bearer" >/dev/null; rm -f "$WORK/bearer"
START='chmod +x /home/user/execd; mkdir -p /home/user/workspace; sudo chown root:root /home/user/execd /home/user/bearer && sudo chmod 600 /home/user/bearer; sudo env -u SUDO_COMMAND -u SUDO_USER -u SUDO_UID -u SUDO_GID EXECD_LISTEN=0.0.0.0:8081 EXECD_ROOT=/home/user/workspace EXECD_HOME=/home/user EXECD_RUN_AS=user EXECD_BEARER_FILE=/home/user/bearer setsid nohup /home/user/execd >/tmp/execd.log 2>&1 < /dev/null & sleep 1'
req=$(jq -nc --arg s "$START" '{process:{cmd:"/bin/bash",args:["-c",$s]}}'); len=${#req}
{ printf '\x00'; printf "$(printf '\\x%02x\\x%02x\\x%02x\\x%02x' $((len>>24&255)) $((len>>16&255)) $((len>>8&255)) $((len&255)))"; printf '%s' "$req"; } > "$WORK/req.bin"
curl -sS -m 60 -X POST "https://49983-$ID.e2b.app/process.Process/Start" -H "X-Access-Token: $TOK" -H 'Content-Type: application/connect+json' -H 'Connect-Protocol-Version: 1' --data-binary "@$WORK/req.bin" >/dev/null
export SANDBOX_URL=https://8081-$ID.e2b.app/exec
for i in $(seq 40); do [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' -X POST "$SANDBOX_URL" -d '{}')" = 401 ] && break; sleep 1; done
export SANDBOX_TOKEN=$BEARER

"$WORK/venv/bin/python" "$HERE/agent.py" | tee "$OUT_DIR/agent.log"

echo "=== independent check: raw exec, not through the agent ===" | tee "$OUT_DIR/verify.log"
rx() { curl -sS -m 120 -X POST "$SANDBOX_URL" -H "Authorization: Bearer $BEARER" -d "$(jq -nc --arg c "$1" '{command:"/bin/bash",args:["-c",$c],cwd:"/home/user/workspace"}')" \
  | jq -r '"exit=\(.exit_code // .code // "?")\n" + ((.stdout // "")|@base64d) + ((.stderr // "")|@base64d)'; }
rx 'find . -path ./.pytest_cache -prune -o -path "*/__pycache__" -prune -o -type f -print | sort; python3 -m pytest -q 2>&1 | tail -5' | tee -a "$OUT_DIR/verify.log"
echo "results in $OUT_DIR"
