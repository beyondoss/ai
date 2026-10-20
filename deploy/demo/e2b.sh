#!/usr/bin/env bash
# shellcheck disable=SC2034  # a sourced library
# The E2B sandbox option (`SANDBOX=e2b`): one E2B microVM per demo session, made through E2B's REST
# API and its in-VM daemon (envd). Sourced by lib.sh; it runs nothing.
#
# A VM is created for each session the first time that session needs one, carries the metadata
# `beyond-ai-demo=<name>` (that is how `down.sh` finds it) and a timeout, runs the same static
# `exec-stub` as the Fargate sandbox, and requires the bearer its session's grant carries. Its public
# port URL (https://<port>-<id>.e2b.app) is the session's exec URL.

E2B_API=https://api.e2b.dev
E2B_TIMEOUT=${E2B_TIMEOUT:-1800}   # seconds a VM lives if nothing kills it: the demo plus a margin
E2B_TEMPLATE=${E2B_TEMPLATE:-base}
VM_HOME=/home/user
VM_PORT=$SANDBOX_PORT

# The key goes to curl as a config on stdin, never on a command line `ps` can see.
e2b_curl() { printf 'header = "X-API-Key: %s"\n' "$E2B_API_KEY" | curl -sS -m "${E2B_CURL_MAX:-60}" -K - "$@"; }

e2b_need_key() {
  [ -n "${E2B_API_KEY:-}" ] || { echo "SANDBOX=e2b needs E2B_API_KEY in the environment" >&2; return 2; }
}

# Every sandbox of this deployment that exists now (running or paused), one id per line. Filters the
# full listing on the metadata here instead of trusting a server-side filter.
e2b_ids() {
  e2b_curl -G "$E2B_API/v2/sandboxes" --data-urlencode "state=running,paused" --data-urlencode "limit=100" \
    | jq -r --arg k "$TAG_KEY" --arg n "$NAME" 'if type=="array" then .[] | select(.metadata[$k]==$n) | .sandboxID else error("unexpected list answer") end'
}
e2b_count() { # prints a number, or fails if the listing could not be read
  local out
  out=$(e2b_ids) || return 1
  printf '%s' "$out" | grep -c . || true
}
e2b_kill() { # id
  e2b_curl -o /dev/null -w '%{http_code}' -X DELETE "$E2B_API/sandboxes/$1"
}

# --- the static exec-stub that runs in each VM ------------------------------------------------------
stub_bin() { # builds it once per run; prints the path
  local dir bin
  dir=$(cd "$ROOT" && cargo metadata --format-version 1 --no-deps 2>/dev/null | jq -r .target_directory)
  bin=$dir/e2b-stub/x86_64-unknown-linux-musl/release/beyond-ai-demo
  # A static musl binary: the VM's libc is not ours. The same flag the sandbox image's build uses,
  # because a static-pie musl binary segfaults at startup here.
  (cd "$ROOT" && RUSTFLAGS="-C relocation-model=static" CC_x86_64_unknown_linux_musl=musl-gcc \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=musl-gcc \
    cargo build -q --release --target x86_64-unknown-linux-musl -p beyond-ai-demo --no-default-features \
    --target-dir "$dir/e2b-stub") >&2 || return 1
  echo "$bin"
}

# Run a command in a VM through envd's process API (Connect, server-streaming). Prints stdout.
vm_run() { # id envd-token script [env-json]
  local id=$1 tok=$2 script=$3 envs=${4:-'{}'} req len
  req=$(jq -nc --arg s "$script" --argjson e "$envs" '{process:{cmd:"/bin/bash",args:["-c",$s],envs:$e}}')
  len=${#req}
  {
    printf '\x00'
    # shellcheck disable=SC2059
    printf "$(printf '\\x%02x\\x%02x\\x%02x\\x%02x' $((len >> 24 & 255)) $((len >> 16 & 255)) $((len >> 8 & 255)) $((len & 255)))"
    printf '%s' "$req"
  } > "$VMDIR/req.$$.bin"
  curl -sS -m "${E2B_CURL_MAX:-60}" -X POST "https://49983-$id.e2b.app/process.Process/Start" -H "X-Access-Token: $tok" \
    -H 'Content-Type: application/connect+json' -H 'Connect-Protocol-Version: 1' --data-binary "@$VMDIR/req.$$.bin" \
    | tr -d '\000-\010\013-\037' | grep -o '"stdout":"[^"]*"' | sed 's/"stdout":"//;s/"$//' | while read -r b; do printf '%s' "$b" | base64 -d 2>/dev/null; done
  rm -f "$VMDIR/req.$$.bin"
}
vm_put() { # id token local-file remote-path
  curl -sS -m 120 -X POST "https://49983-$1.e2b.app/files?path=$4&username=user" -H "X-Access-Token: $2" \
    -F "file=@$3;filename=$(basename "$4")" | jq -e '.[0].path' >/dev/null
}
vm_get() { # id token remote-path
  curl -sS -m 30 "https://49983-$1.e2b.app/files?path=$3&username=user" -H "X-Access-Token: $2"
}

# One VM for a session: created on first need, remembered in $VMDIR/<session>.json (a file, because
# grants are minted inside command substitutions). Prints nothing on stdout.
#
# With VM_PRIVATE=1 the VM's ports are not public: E2B answers 403 to any request without the VM's
# traffic token (header e2b-traffic-access-token), which the session's grant carries for the exec
# endpoint and the browser connector. The use-case agents run this way; the original checks do not.
vm_for() { # session
  local s=$1 f="$VMDIR/$1.json" resp id tok bearer traffic net='{}' hdr=()
  [ -f "$f" ] && return 0
  (
    flock 9
    [ -f "$f" ] && exit 0
    echo "  + E2B sandbox for session $s" >&2
    bearer=$(openssl rand -hex 24)
    [ -z "${VM_PRIVATE:-}" ] || net='{"allowPublicTraffic":false}'
    resp=$(e2b_curl -X POST "$E2B_API/sandboxes" -H 'Content-Type: application/json' -d "$(jq -nc --arg t "$E2B_TEMPLATE" \
      --argjson timeout "$E2B_TIMEOUT" --arg k "$TAG_KEY" --arg n "$NAME" --arg s "$s" --argjson net "$net" \
      '{templateID:$t, timeout:$timeout, secure:true, allow_internet_access:true, metadata:{($k):$n, session:$s}} + (if $net == {} then {} else {network:$net} end)')")
    id=$(jq -r '.sandboxID // empty' <<<"$resp")
    tok=$(jq -r '.envdAccessToken // empty' <<<"$resp")
    traffic=$(jq -r '.trafficAccessToken // empty' <<<"$resp")
    [ -z "${VM_PRIVATE:-}" ] || [ -n "$traffic" ] || { echo "E2B: a private sandbox came back without a traffic token" >&2; exit 1; }
    [ -z "$traffic" ] || hdr=(-H "e2b-traffic-access-token: $traffic")
    [ -n "$id" ] && [ -n "$tok" ] || { echo "E2B: the sandbox was not created: $(jq -c 'del(.envdAccessToken)' <<<"$resp" | head -c 300)" >&2; exit 1; }
    # Recorded at once, so a failure below still leaves a VM that down.sh (and this run) can find.
    jq -nc --arg id "$id" --arg tok "$tok" --arg b "$bearer" --arg s "$s" --arg tr "$traffic" --argjson created "$(date +%s%3N)" \
      '{session:$s, id:$id, envd_token:$tok, bearer:$b, traffic_token:$tr, url:("https://'"$VM_PORT"'-"+$id+".e2b.app/exec"), created_ms:$created}' > "$f.part"
    local i
    for i in $(seq 1 30); do
      vm_put "$id" "$tok" "$STUB" "$VM_HOME/exec-stub" 2>/dev/null && break
      sleep 1
    done
    vm_put "$id" "$tok" "$CA_FILE" "$VM_HOME/ca.pem" || { echo "E2B: could not upload to $id" >&2; exit 1; }
    vm_run "$id" "$tok" "chmod +x $VM_HOME/exec-stub; mkdir -p $VM_HOME/workspace
setsid nohup $VM_HOME/exec-stub exec-stub --listen 0.0.0.0:$VM_PORT --root $VM_HOME/workspace --home $VM_HOME >/tmp/stub.log 2>&1 < /dev/null &
sleep 1" "$(jq -nc --arg b "$bearer" '{DEMO_EXEC_BEARER:$b, SSL_CERT_FILE:"'"$VM_HOME"'/ca.pem"}')" >/dev/null
    # Up when the public URL answers, and answers 401 to a caller without the bearer.
    for i in $(seq 1 40); do
      [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' -X POST "${hdr[@]}" "https://$VM_PORT-$id.e2b.app/exec" -d '{}')" = 401 ] && break
      sleep 1
    done
    [ "$i" -lt 40 ] || { echo "E2B: the stub in $id never answered" >&2; exit 1; }
    mv "$f.part" "$f"
  ) 9> "$VMDIR/lock" || return 1
}
vm_field() { jq -r ".$2" "$VMDIR/$1.json"; }

# The stubs' own log lines, in CloudWatch's shape ([{t, m}]) so the callers need not care where the
# sandbox ran. Optional $1 filters on a substring, $2 on a minimum time.
vm_logs() { # [substring] [since-ms]
  local f id tok
  for f in "$VMDIR"/*.json; do
    [ -e "$f" ] || continue
    id=$(jq -r .id "$f")
    tok=$(jq -r .envd_token "$f")
    vm_get "$id" "$tok" /tmp/stub.log | jq -Rc --arg vm "$id" --arg sub "${1:-}" --argjson since "${2:-0}" \
      'select(startswith("{") and contains($sub)) | fromjson? | select(.ts_ms >= $since) | .vm = $vm | {t:.ts_ms, m:tojson}'
  done | jq -sc 'sort_by(.t)'
}

# --- the use cases' VMs -----------------------------------------------------------------------------

# Put a local directory's contents (dotfiles included) into a VM at a remote directory.
vm_upload_dir() { # id token local-dir remote-dir
  local tar="$VMDIR/up.$$.tgz"
  tar -C "$3" -czf "$tar" --exclude=__pycache__ .
  vm_put "$1" "$2" "$tar" "$VM_HOME/upload.tgz" || { rm -f "$tar"; return 1; }
  rm -f "$tar"
  vm_run "$1" "$2" "mkdir -p '$4' && tar -xzf $VM_HOME/upload.tgz -C '$4' && rm -f $VM_HOME/upload.tgz" >/dev/null
}

# rustwright-mcp, built as a static musl binary from the repo's `mcp/` workspace root. The same flag as
# the stub's: a static-pie musl binary segfaults at startup here.
RUSTWRIGHT_REPO=${RUSTWRIGHT_REPO:-https://github.com/beyondoss/rustwright}
RUSTWRIGHT_REV=${RUSTWRIGHT_REV:-main}
rustwright_bin() { # builds it once; prints the path
  local cache=${DEMO_CACHE:-$HOME/.cache/beyond-ai-demo} src
  src=$cache/rustwright
  mkdir -p "$cache"
  if [ ! -d "$src/.git" ]; then
    git clone -q --depth 1 "$RUSTWRIGHT_REPO" "$src" >&2 || return 1
  fi
  if [ "$RUSTWRIGHT_REV" != main ]; then
    git -C "$src" fetch -q --depth 1 origin "$RUSTWRIGHT_REV" >&2 && git -C "$src" checkout -q FETCH_HEAD >&2 || return 1
  fi
  (cd "$src/mcp" && RUSTFLAGS="-C relocation-model=static" CC_x86_64_unknown_linux_musl=musl-gcc \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=musl-gcc \
    cargo build -q --release --target x86_64-unknown-linux-musl --target-dir "$cache/rustwright-target") >&2 || return 1
  echo "$cache/rustwright-target/x86_64-unknown-linux-musl/release/rustwright-mcp"
}

CHROME_VERSION=${CHROME_VERSION:-155.0.8059.12}
SUPERGATEWAY_VERSION=${SUPERGATEWAY_VERSION:-4.2.0}
BRIDGE_PORT=8931

# The browser stack inside a VM: a lean Chromium (Chrome for Testing's headless shell), rustwright-mcp
# (a stdio MCP server) and supergateway, which serves it as streamable HTTP, one stdio child per MCP
# session (`--stateful`). Hosted sessions skip `command` entries in .mcp.json, so this bridge is how a
# stdio server becomes a URL a grant can name. `vm.overcommit_memory=1`: a small VM refuses the
# renderer's large mappings otherwise and the renderer dies.
vm_browser_up() { # id token rustwright-binary
  vm_put "$1" "$2" "$3" "$VM_HOME/rustwright-mcp" || return 1
  vm_run "$1" "$2" "chmod +x $VM_HOME/rustwright-mcp
sudo sysctl -w vm.overcommit_memory=1 >/dev/null
export DEBIAN_FRONTEND=noninteractive
(sudo apt-get update -qq && sudo apt-get install -y -qq --no-install-recommends unzip libnss3 libatk1.0-0 libatk-bridge2.0-0 \\
  libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 \\
  libasound2 fonts-liberation libx11-6 libxcb1 libxext6 libglib2.0-0 libdbus-1-3 libexpat1 libatspi2.0-0 libudev1 >/tmp/apt.log 2>&1) &
(cd $VM_HOME && curl -sSL https://storage.googleapis.com/chrome-for-testing-public/$CHROME_VERSION/linux64/chrome-headless-shell-linux64.zip -o c.zip) &
(mkdir -p $VM_HOME/sg && cd $VM_HOME/sg && npm init -y >/dev/null && npm i supergateway@$SUPERGATEWAY_VERSION >/tmp/npm.log 2>&1) &
wait
cd $VM_HOME && unzip -q -o c.zip && rm -f c.zip
test -x $VM_HOME/chrome-headless-shell-linux64/chrome-headless-shell && test -x $VM_HOME/sg/node_modules/.bin/supergateway && echo browser-stack-ready" | grep -q browser-stack-ready
}
# Start the bridge (and, if given, a static file server for a directory on localhost:8000).
vm_bridge_start() { # id token [site-dir]
  vm_run "$1" "$2" "${3:+cd $3 && (setsid nohup python3 -m http.server 8000 --bind 127.0.0.1 >/tmp/site.log 2>&1 </dev/null &)}
cd $VM_HOME
export RUSTWRIGHT_CHROMIUM=$VM_HOME/chrome-headless-shell-linux64/chrome-headless-shell
(setsid nohup $VM_HOME/sg/node_modules/.bin/supergateway --stdio $VM_HOME/rustwright-mcp --outputTransport streamableHttp \\
  --stateful --port $BRIDGE_PORT --streamableHttpPath /mcp --healthEndpoint /healthz >/tmp/bridge.log 2>&1 </dev/null &)
for i in \$(seq 1 30); do curl -sf -m 2 localhost:$BRIDGE_PORT/healthz >/dev/null && break; sleep 1; done
curl -sf -m 2 localhost:$BRIDGE_PORT/healthz" | grep -q ok
}
vm_bridge_url() { echo "https://$BRIDGE_PORT-$(vm_field "$1" id).e2b.app/mcp"; }

# Pause a VM (E2B keeps its memory and processes) and resume it; each prints how long the call took.
vm_pause() { # id
  local t0 code
  t0=$(date +%s%3N)
  code=$(e2b_curl -o /dev/null -w '%{http_code}' -X POST "$E2B_API/sandboxes/$1/pause")
  [ "$code" = 204 ] || { echo "pause: HTTP $code" >&2; return 1; }
  echo $(($(date +%s%3N) - t0))
}
vm_resume() { # id -> prints resume latency in ms
  local t0 resp
  t0=$(date +%s%3N)
  resp=$(e2b_curl -X POST "$E2B_API/sandboxes/$1/connect" -H 'Content-Type: application/json' -d "{\"timeout\":$E2B_TIMEOUT}")
  jq -e '.sandboxID' <<<"$resp" >/dev/null || { echo "resume: $resp" | head -c 300 >&2; return 1; }
  echo $(($(date +%s%3N) - t0))
}
