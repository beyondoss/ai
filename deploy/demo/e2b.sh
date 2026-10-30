#!/usr/bin/env bash
# shellcheck disable=SC2034  # a sourced library
# The E2B sandbox option (`SANDBOX=e2b`): one E2B microVM per demo session, made through E2B's REST
# API and its in-VM daemon (envd). Sourced by lib.sh; it runs nothing.
#
# A VM is created for each session the first time that session needs one, carries the metadata
# `beyond-ai-demo=<name>` (that is how `down.sh` finds it) and a timeout, runs the same static
# `execd` as the Fargate sandbox, and requires the bearer its session's grant carries. Its public
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

# --- the static execd that runs in each VM ---------------------------------------------------------
execd_bin() { # builds it once per run; prints the path
  local dir bin
  dir=$(cd "$ROOT" && cargo metadata --format-version 1 --no-deps 2>/dev/null | jq -r .target_directory)
  bin=$dir/e2b-execd/x86_64-unknown-linux-musl/release/execd
  # A static musl binary: the VM's libc is not ours. The same flag the sandbox image's build uses,
  # because a static-pie musl binary segfaults at startup here.
  (cd "$ROOT" && RUSTFLAGS="-C relocation-model=static" CC_x86_64_unknown_linux_musl=musl-gcc \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=musl-gcc \
    cargo build -q --release --locked --target x86_64-unknown-linux-musl -p beyond-execd \
    --target-dir "$dir/e2b-execd") >&2 || return 1
  echo "$bin"
}

# Run a command in a VM through envd's process API (Connect, server-streaming). Prints stdout.
vm_run() { # id envd-token script [env-json]
  local id=$1 tok=$2 script=$3 envs=${4:-'{}'} req len rf="$VMDIR/req.$BASHPID.bin"
  req=$(jq -nc --arg s "$script" --argjson e "$envs" '{process:{cmd:"/bin/bash",args:["-c",$s],envs:$e}}')
  len=${#req}
  {
    printf '\x00'
    # shellcheck disable=SC2059
    printf "$(printf '\\x%02x\\x%02x\\x%02x\\x%02x' $((len >> 24 & 255)) $((len >> 16 & 255)) $((len >> 8 & 255)) $((len & 255)))"
    printf '%s' "$req"
  } > "$rf"
  curl -sS -m "${E2B_CURL_MAX:-60}" -X POST "https://49983-$id.e2b.app/process.Process/Start" -H "X-Access-Token: $tok" \
    -H 'Content-Type: application/connect+json' -H 'Connect-Protocol-Version: 1' --data-binary "@$rf" \
    | tr -d '\000-\010\013-\037' | grep -o '"stdout":"[^"]*"' | sed 's/"stdout":"//;s/"$//' | while read -r b; do printf '%s' "$b" | base64 -d 2>/dev/null; done
  rm -f "$rf"
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
    [ -n "$id" ] && [ -n "$tok" ] || { echo "E2B: the sandbox was not created: $(jq -c 'del(.envdAccessToken)' <<<"$resp" | head -c 300)" >&2; exit 1; }
    # From here the VM exists and costs until it is killed: any failure below kills it, so a retry does
    # not leave a live VM behind for the rest of E2B_TIMEOUT.
    trap 'e2b_kill "$id" >/dev/null 2>&1; rm -f "$f.part"' EXIT
    [ -z "${VM_PRIVATE:-}" ] || [ -n "$traffic" ] || { echo "E2B: a private sandbox came back without a traffic token" >&2; exit 1; }
    [ -z "$traffic" ] || hdr=(-H "e2b-traffic-access-token: $traffic")
    # Recorded at once, so the rest of this function (and its failure trap) can find the VM.
    jq -nc --arg id "$id" --arg tok "$tok" --arg b "$bearer" --arg s "$s" --arg tr "$traffic" --argjson created "$(date +%s%3N)" \
      '{session:$s, id:$id, envd_token:$tok, bearer:$b, traffic_token:$tr, url:("https://'"$VM_PORT"'-"+$id+".e2b.app/exec"), created_ms:$created}' > "$f.part"
    local i
    for i in $(seq 1 30); do
      vm_put "$id" "$tok" "$EXECD" "$VM_HOME/execd" 2>/dev/null && break
      sleep 1
    done
    vm_put "$id" "$tok" "$CA_FILE" "$VM_HOME/ca.pem" || { echo "E2B: could not upload to $id" >&2; exit 1; }
    # The bearer goes up as a file, never on a command line; below it becomes root's, mode 0600, so the
    # commands (which run as the VM's user) cannot read it, and it is in no command's environment.
    (umask 077; printf '%s' "$bearer" > "$f.bearer")
    vm_put "$id" "$tok" "$f.bearer" "$VM_HOME/bearer" || { rm -f "$f.bearer"; echo "E2B: could not upload to $id" >&2; exit 1; }
    rm -f "$f.bearer"
    # execd runs as root (through the VM's passwordless sudo) with EXECD_RUN_AS=user, so the commands it
    # runs are the VM's ordinary unprivileged user and cannot read execd's memory, its environment or its
    # bearer file: the bearer file and the binary are root's, mode 0600 and 0755 (the file stays: the user
    # cannot read it).
    # EXECD_KEPT_JOBS_SURVIVE_IDLE=1: this VM is not stopped while a kept job runs, so it advertises job notify.
    # SSL_CERT_FILE is the demo CA that signed the agent's https callback listener; execd's callback client
    # takes its roots from the OS trust store, which honors it.
    vm_run "$id" "$tok" "chmod +x $VM_HOME/execd; mkdir -p $VM_HOME/workspace
sudo chown root:root $VM_HOME/execd $VM_HOME/bearer && sudo chmod 600 $VM_HOME/bearer || exit 1
sudo env -u SUDO_COMMAND -u SUDO_USER -u SUDO_UID -u SUDO_GID EXECD_LISTEN=0.0.0.0:$VM_PORT EXECD_ROOT=$VM_HOME/workspace EXECD_HOME=$VM_HOME EXECD_RUN_AS=user EXECD_BEARER_FILE=$VM_HOME/bearer EXECD_KEPT_JOBS_SURVIVE_IDLE=1 SSL_CERT_FILE=$VM_HOME/ca.pem setsid nohup $VM_HOME/execd >/tmp/execd.log 2>&1 < /dev/null &
sleep 1" >/dev/null
    # Up when the public URL answers, and answers 401 to a caller without the bearer.
    for i in $(seq 1 40); do
      [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' -X POST "${hdr[@]}" "https://$VM_PORT-$id.e2b.app/exec" -d '{}')" = 401 ] && break
      sleep 1
    done
    [ "$i" -lt 40 ] || { echo "E2B: execd in $id never answered" >&2; exit 1; }
    mv "$f.part" "$f"
    trap - EXIT
  ) 9> "$VMDIR/lock" || return 1
}
vm_field() { jq -r ".$2" "$VMDIR/$1.json"; }

# The sandboxes' own log lines (execd's, turned by execd.jq into the events the checks and the report
# read), in CloudWatch's shape ([{t, m}]) so the callers need not care where the sandbox ran. Optional $1
# filters on a substring, $2 on a minimum time.
vm_logs() { # [substring] [since-ms]
  local f id tok
  for f in "$VMDIR"/*.json; do
    [ -e "$f" ] || continue
    id=$(jq -r .id "$f")
    tok=$(jq -r .envd_token "$f")
    vm_get "$id" "$tok" /tmp/execd.log | jq -L "$HERE" -Rc --arg vm "$id" --arg sub "${1:-}" --argjson since "${2:-0}" \
      'include "execd"; execd_events | select(.ts_ms >= $since) | .vm = $vm | select(tojson | contains($sub)) | {t:.ts_ms, m:tojson}'
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
# execd's: a static-pie musl binary segfaults at startup here.
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
CDP_PORT=9222

# The browser stack inside a VM: a lean Chromium (Chrome for Testing's headless shell), rustwright-mcp
# (a stdio MCP server) and supergateway, which serves it as streamable HTTP, one stdio child per MCP
# session (`--stateful`). Hosted sessions skip `command` entries in .mcp.json, so this bridge is how a
# stdio server becomes a URL a grant can name. `vm.overcommit_memory=1`: a small VM refuses the
# renderer's large mappings otherwise and the renderer dies.
browser_up_script() { # rustwright-binary-url-or-empty
  cat <<EOS
$([ -z "$1" ] || echo "test -x $VM_HOME/rustwright-mcp || curl -fsS -m 120 -o $VM_HOME/rustwright-mcp '$1' || exit 1")
chmod +x $VM_HOME/rustwright-mcp
sudo sysctl -w vm.overcommit_memory=1 >/dev/null
if ! { test -x $VM_HOME/chrome-headless-shell-linux64/chrome-headless-shell && test -x $VM_HOME/sg/node_modules/.bin/supergateway; }; then
$(browser_install_script)
fi
EOS
}
# The binary: put from a local file, or fetched by the VM itself from a URL.
browser_binary_put() { # id token rustwright-binary-or-url
  [[ $3 == http* ]] || vm_put "$1" "$2" "$3" "$VM_HOME/rustwright-mcp"
}
vm_browser_up() { # id token rustwright-binary-or-url
  browser_binary_put "$@" || return 1
  vm_run "$1" "$2" "$(browser_up_script "$([[ $3 == http* ]] && echo "$3")")
echo browser-stack-ready" | grep -q browser-stack-ready
}
# The install, as one script: run in a plain VM at its first browser request, or once at template build.
browser_install_script() {
  cat <<EOS
export DEBIAN_FRONTEND=noninteractive
(sudo apt-get update -qq && sudo apt-get install -y -qq --no-install-recommends unzip libnss3 libatk1.0-0 libatk-bridge2.0-0 \\
  libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 \\
  libasound2 fonts-liberation libx11-6 libxcb1 libxext6 libglib2.0-0 libdbus-1-3 libexpat1 libatspi2.0-0 libudev1 >/tmp/apt.log 2>&1) &
(cd $VM_HOME && curl -sSL https://storage.googleapis.com/chrome-for-testing-public/$CHROME_VERSION/linux64/chrome-headless-shell-linux64.zip -o c.zip) &
(mkdir -p $VM_HOME/sg && cd $VM_HOME/sg && npm init -y >/dev/null && npm i supergateway@$SUPERGATEWAY_VERSION >/tmp/npm.log 2>&1) &
wait
cd $VM_HOME && unzip -q -o c.zip && rm -f c.zip
chown -R user:user $VM_HOME 2>/dev/null || true
test -x $VM_HOME/chrome-headless-shell-linux64/chrome-headless-shell && test -x $VM_HOME/sg/node_modules/.bin/supergateway
EOS
}

# The browser stack baked into an E2B template, so a session's VM boots with it instead of installing it
# in the first request: Chromium, supergateway with its node modules, and the rustwright-mcp binary, so
# a first browser call only launches processes. Built once and reused: the name carries a hash of what the
# build installs (the install script and the binary's sha256), so a changed version builds a new one and an
# unchanged one is found by name. The binary reaches the build by a presigned link from the bundles bucket
# (which must exist). A template belongs to the account, not to a run: `down.sh` keeps it. Prints the
# template name. Needs E2B_API_KEY.
e2b_browser_template() {
  local name row tid bid resp body status i rw rwsha rwurl start ready
  rw=$(rustwright_bin) || return 1
  rwsha=$(sha256sum "$rw" | cut -d' ' -f1)
  start=$(browser_bridge_script; browser_warm_script)
  ready="test -e /tmp/browser-warm && curl -sf -m 2 localhost:$BRIDGE_PORT/healthz"
  name="beyond-ai-demo-browser-$({ browser_install_script; echo "$rwsha"; echo "$start"; echo "$ready"; } | sha256sum | cut -c1-10)"
  row=$(e2b_curl "$E2B_API/templates" | jq -r --arg n "$name" '[.[]|select((.aliases // [])|index($n))][0] // empty | "\(.templateID) \(.buildStatus)"') || return 1
  case "$row" in
    *" ready") echo "$name"; return 0 ;;
  esac
  # A template whose build never finished is removed and made again.
  [ -z "$row" ] || e2b_curl -o /dev/null -X DELETE "$E2B_API/templates/${row%% *}" >&2
  resp=$(e2b_curl -X POST "$E2B_API/v3/templates" -H 'Content-Type: application/json' -d "$(jq -nc --arg n "$name" '{name:$n, cpuCount:2, memoryMB:2048}')")
  tid=$(jq -r '.templateID // empty' <<<"$resp")
  bid=$(jq -r '.buildID // empty' <<<"$resp")
  [ -n "$tid" ] && [ -n "$bid" ] || { echo "E2B: the template was not created: $(head -c 300 <<<"$resp")" >&2; return 1; }
  echo "  + building E2B template $name ($tid)" >&2
  command aws --region "$REGION" s3 cp "$rw" "s3://$(bundle_bucket)/rustwright-mcp-$rwsha" --only-show-errors || return 1
  rwurl=$(command aws --region "$REGION" s3 presign "s3://$(bundle_bucket)/rustwright-mcp-$rwsha" --expires-in 3600) || return 1
  body=$(jq -nc --arg s "$(browser_install_script)" --arg b "curl -fsS -m 120 -o $VM_HOME/rustwright-mcp '$rwurl' && echo '$rwsha  $VM_HOME/rustwright-mcp' | sha256sum -c - >/dev/null && chmod 755 $VM_HOME/rustwright-mcp && chown user:user $VM_HOME/rustwright-mcp" \
    --arg start "$start" --arg ready "$ready" \
    '{fromImage:"e2bdev/base", steps:[{type:"RUN", args:[$s, "root"]}, {type:"RUN", args:[$b, "root"]}], startCmd:$start, readyCmd:$ready}')
  e2b_curl -X POST "$E2B_API/v2/templates/$tid/builds/$bid" -H 'Content-Type: application/json' -d "$body" >&2 || return 1
  for i in $(seq 1 180); do
    status=$(e2b_curl "$E2B_API/templates/$tid/builds/$bid/status")
    case $(jq -r .status <<<"$status") in
      ready) echo "$name"; return 0 ;;
      error) echo "E2B: the template build failed: $(jq -c .reason <<<"$status" | head -c 400)" >&2; return 1 ;;
    esac
    sleep 5
  done
  echo "E2B: the template build timed out" >&2
  return 1
}
# A static file server for a directory, on localhost:8000.
vm_site_start() { # id token dir
  vm_run "$1" "$2" "cd $3 && (setsid nohup python3 -m http.server 8000 --bind 127.0.0.1 >/tmp/site.log 2>&1 </dev/null &)
for i in \$(seq 1 50); do curl -sf -m 1 -o /dev/null localhost:8000/ && break; sleep 0.1; done
curl -sf -m 1 -o /dev/null localhost:8000/ && echo up" | grep -q up
}
# The browser and the bridge's launch, idempotent: each part does nothing if it already answers (a VM made
# from the browser template resumes with both running). Chromium starts first and keeps running on a
# debugging port; rustwright-mcp attaches to it (RUSTWRIGHT_MCP_CDP_ENDPOINT) instead of launching one, so
# the model's first navigate does not pay a browser start. Used by vm_bridge_start and, in the template, as
# the start command. One Chromium serves the VM's sessions, which is how the demo uses it.
browser_bridge_script() {
  cat <<EOS
cd $VM_HOME
sudo sysctl -w vm.overcommit_memory=1 >/dev/null
curl -sf -m 2 localhost:$CDP_PORT/json/version >/dev/null || (setsid nohup $VM_HOME/chrome-headless-shell-linux64/chrome-headless-shell --no-sandbox \\
  --disable-gpu --remote-debugging-port=$CDP_PORT --user-data-dir=/tmp/chrome-profile about:blank >/tmp/chrome.log 2>&1 </dev/null &)
export RUSTWRIGHT_MCP_CDP_ENDPOINT=http://127.0.0.1:$CDP_PORT
curl -sf -m 2 localhost:$BRIDGE_PORT/healthz >/dev/null || (setsid nohup $VM_HOME/sg/node_modules/.bin/supergateway --stdio $VM_HOME/rustwright-mcp --outputTransport streamableHttp \\
  --stateful --port $BRIDGE_PORT --streamableHttpPath /mcp --healthEndpoint /healthz >/tmp/bridge.log 2>&1 </dev/null &)
for i in \$(seq 1 300); do curl -sf -m 2 localhost:$CDP_PORT/json/version >/dev/null && curl -sf -m 2 localhost:$BRIDGE_PORT/healthz >/dev/null && break; sleep 0.1; done
EOS
}
# One throwaway browser session through the bridge (initialize, navigate, close), then a marker: the
# rustwright-mcp, Chromium and node pages it touches are in memory when the template is snapshotted.
browser_warm_script() {
  cat <<EOS
H='-H Content-Type:application/json -H Accept:application/json,text/event-stream'
init='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"warm","version":"0"}}}'
sid=\$(curl -sS -D - -o /dev/null \$H localhost:$BRIDGE_PORT/mcp -d "\$init" | tr -d '\\r' | grep -i '^mcp-session-id' | cut -d' ' -f2)
curl -sS -o /dev/null \$H -H "Mcp-Session-Id: \$sid" localhost:$BRIDGE_PORT/mcp -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'
curl -sS -o /dev/null -m 60 \$H -H "Mcp-Session-Id: \$sid" localhost:$BRIDGE_PORT/mcp -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"browser_navigate","arguments":{"url":"data:text/html,<h1>warm</h1>"}}}'
curl -sS -o /dev/null -X DELETE -H "Mcp-Session-Id: \$sid" localhost:$BRIDGE_PORT/mcp
touch /tmp/browser-warm
EOS
}
# Start the bridge.
vm_bridge_start() { # id token
  vm_run "$1" "$2" "$(browser_bridge_script)
curl -sf -m 2 localhost:$BRIDGE_PORT/healthz" | grep -q ok
}
# Both in one round trip: the front's path. In a VM made from the browser template the binary, the stack,
# Chromium and the bridge are all there already and this only checks them.
vm_browser_start() { # id token rustwright-binary-or-url
  browser_binary_put "$@" || return 1
  vm_run "$1" "$2" "$(browser_up_script "$([[ $3 == http* ]] && echo "$3")")
$(browser_bridge_script)
curl -sf -m 2 localhost:$BRIDGE_PORT/healthz" | grep -q ok
}

# Start @beyond.dev/agent-tools-mcp (AOT linux binary) after the bundle is seeded.
# $3 is an https URL of the binary (presigned), or a local path already on the VM.
vm_tools_start() { # id token tools-mcp-binary-or-url
  local id=$1 tok=$2 src=$3
  vm_run "$id" "$tok" "
set -e
BRIDGE_PORT=${BRIDGE_PORT:-8931}
VM_HOME=\${VM_HOME:-/home/user}
BIN=\$VM_HOME/tools-mcp
if [ ! -x \"\$BIN\" ]; then
  if [[ \"$src\" == http* ]]; then
    curl -fsS -m 120 -o \"\$BIN\" '$src'
  else
    cp '$src' \"\$BIN\"
  fi
  chmod 755 \"\$BIN\"
fi
# tools/ lives in the seeded workspace (bundle root).
TOOLS=\$VM_HOME/workspace/tools
test -d \"\$TOOLS\" || { echo \"tools dir missing: \$TOOLS\" >&2; exit 1; }
curl -sf -m 2 localhost:\$BRIDGE_PORT/healthz >/dev/null || (
  setsid nohup \"\$BIN\" serve --dir \"\$TOOLS\" --http 0.0.0.0:\$BRIDGE_PORT \
    >/tmp/tools-mcp.log 2>&1 </dev/null &
)
for i in \$(seq 1 150); do
  curl -sf -m 2 localhost:\$BRIDGE_PORT/healthz >/dev/null && exit 0
  sleep 0.1
done
echo \"tools-mcp failed to listen; log:\" >&2
tail -n 40 /tmp/tools-mcp.log >&2 || true
exit 1
" | grep -q . || return 1
  vm_run "$id" "$tok" "curl -sf -m 2 localhost:${BRIDGE_PORT:-8931}/healthz" | grep -q ok
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

# A VM that something else made (the exec front makes a coding session's on its first request), found by
# the session in its metadata and recorded the way vm_for records its own, so vm_field, vm_run and
# vm_logs work on it. `connect` returns the VM's tokens. Returns 1 if the session has no VM (yet).
vm_adopt() { # session
  local s=$1 f="$VMDIR/$1.json" row id resp
  row=$(e2b_curl -G "$E2B_API/v2/sandboxes" --data-urlencode "state=running,paused" --data-urlencode "limit=100" \
    | jq -c --arg k "$TAG_KEY" --arg n "$NAME" --arg s "$s" 'if type=="array" then [.[] | select(.metadata[$k]==$n and .metadata.session==$s)][0] // empty else error("unexpected list answer") end') || return 1
  [ -n "$row" ] || return 1
  id=$(jq -r .sandboxID <<<"$row")
  resp=$(e2b_curl -X POST "$E2B_API/sandboxes/$id/connect" -H 'Content-Type: application/json' -d "{\"timeout\":$E2B_TIMEOUT}")
  jq -e '.envdAccessToken' <<<"$resp" >/dev/null || return 1
  jq -nc --arg s "$s" --arg id "$id" --argjson r "$resp" --argjson row "$row" \
    '{session:$s, id:$id, envd_token:$r.envdAccessToken, bearer:"", traffic_token:($r.trafficAccessToken // ""),
      url:("https://'"$VM_PORT"'-"+$id+".e2b.app/exec"), created_ms:0, started_at:$row.startedAt, adopted:true}' > "$f.part" && mv "$f.part" "$f"
}
# How many VMs this deployment has for a session right now (running or paused).
vm_count_for() { # session
  e2b_curl -G "$E2B_API/v2/sandboxes" --data-urlencode "state=running,paused" --data-urlencode "limit=100" \
    | jq --arg k "$TAG_KEY" --arg n "$NAME" --arg s "$1" 'if type=="array" then [.[] | select(.metadata[$k]==$n and .metadata.session==$s)] | length else error("unexpected list answer") end'
}
