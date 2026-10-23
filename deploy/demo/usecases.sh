#!/usr/bin/env bash
# shellcheck shell=bash disable=SC2034,SC2016,SC2154  # sourced by demo.sh: its variables are ours, queries are single-quoted
# The three use cases: an agent is a folder, and each folder here is run for real, against the
# deployment `up.sh` made, with named checks like the rest of the demo.
#
#     ./deploy/demo/demo.sh assistant    no sandbox at all: memory, a public MCP server (DeepWiki), a
#                                        managed connector that holds no key, and a grant's ceilings
#     ./deploy/demo/demo.sh coding       a sandbox per session: a skill, a subagent, a background app,
#                                        a browser, all from the folder
#     ./deploy/demo/demo.sh browser      a browser in a sandbox that does not exist until the model uses
#                                        it, then sleeps: no held connection, state that survives a pause
#
# Each needs `SANDBOX=both ./deploy/demo/up.sh`. This file only defines: the check lists and titles, each
# use case's preparation, its grants and its checks. demo.sh sources it and supplies the runner
# (`chat`, `result`, `tl`, `run_check`, ...). Adding a check to a use case is the same two steps as
# anywhere else: write `check_<name>()` and add the name to its `CHECKS_<usecase>` list.

UC_AGENTS=$HERE/agents

CHECKS_assistant=(persona-applied remote-mcp-real managed-mcp-no-key memory-across-sessions limits-enforced skill-from-bundle no-sandbox-at-all)
CHECKS_coding=(vm-created-on-first-tool skill-used fix-verified subagent-from-folder app-as-background-job browser-sees-localhost)
CHECKS_browser=(vm-created-on-first-tool browser-reads-page no-held-connection idle-cost state-survives-pause state-survives-idle-reap)
CHECK_TITLE+=(
  [persona-applied]="AGENTS.md sets the model and the tools"
  [remote-mcp-real]="A public remote MCP server (DeepWiki), from .mcp.json"
  [managed-mcp-no-key]="A managed connector works, and no key is in the agent"
  [memory-across-sessions]="A brand-new session remembers"
  [limits-enforced]="A grant's ceiling refuses a model outside it"
  [skill-from-bundle]="The model reads a skill from the bundle, with no sandbox"
  [no-sandbox-at-all]="The agent has no sandbox: its grant names none, and none saw a request"
  [vm-created-on-first-tool]="The sandbox VM is created by the first sandbox tool call"
  [skill-used]="The model reads the folder's skill and runs its script"
  [fix-verified]="The failing test passes, checked in the sandbox"
  [subagent-from-folder]="The reviewer subagent from .agents/agents is called"
  [app-as-background-job]="The app runs as a background job; a short job's end is pushed"
  [browser-sees-localhost]="The browser reads the page the sandbox serves"
  [browser-reads-page]="The browser reads a page served inside the sandbox"
  [no-held-connection]="No connection is held to a sleeping sandbox"
  [state-survives-pause]="Page state survives a paused sandbox"
  [state-survives-idle-reap]="Page state survives the agent's MCP idle window"
  [idle-cost]="What an idle browser costs"
)

declare -A UC_TITLE=(
  [assistant]="assistant: an agent with no sandbox and no keys"
  [coding]="coding: an agent that fixes, runs and reviews code in its own sandbox"
  [browser]="browser: a real browser in a sandbox that sleeps"
)
declare -A UC_SUMMARY=(
  [assistant]="A folder with an AGENTS.md, a .mcp.json and a skill, deployed as a bundle. The agent has no sandbox at all: its grant names no exec endpoint. It has memory, a read-only view of its own folder (the skill), a public MCP server and one managed connector (a stand-in endpoint on AWS that holds the real key); the grant caps its models."
  [coding]="A folder with a failing Python app, a skill with a script, a reviewer subagent and an AGENTS.md, deployed as a bundle. The agent's own E2B microVM does not exist until the first sandbox tool call makes it (and seeds it from the bundle); then it fixes the bug, reviews it, runs the app as a background job and looks at the page with a browser in the same VM."
  [browser]="A folder with an AGENTS.md, a .mcp.json and a page, deployed as a bundle together with the browser server's tool list (saved by running the server once at bundle build). The browser (rustwright-mcp behind a supergateway bridge) runs inside an E2B microVM that does not exist until the model first calls a browser tool, and is then paused and resumed; the grant names it as a connector that holds no standing stream."
)

# ---- helpers shared by the use cases -----------------------------------------------------------------

# The folder as the report shows it: every file with its size and, for the small text ones, the text.
uc_tree() { # name
  local f
  (cd "$UC_AGENTS/$1" && find . -type f ! -name '*.pyc' | sort | while read -r f; do
    jq -nc --arg p "${f#./}" --rawfile t "$f" --argjson b "$(stat -c %s "$f")" '{path:$p, bytes:$b, text:($t|.[0:2600])}'
  done) | jq -sc .
}

# tool calls of one chat summary, by name (a regex), as compact JSON
uc_tools() { # summary-json name-regex
  jq -c --arg re "$2" '[.tools[]|select(.name|test($re))]' <<<"$1"
}
# the text of a chat summary's tool output for the first call matching
uc_tool_out() { # summary-json name-regex
  jq -r --arg re "$2" '[.tools[]|select(.name|test($re))][0].output // ""' <<<"$1"
}
uc_tool_ms() { # summary-json name-regex
  jq -r --arg re "$2" '[.tools[]|select(.name|test($re))][0] | if . == null then 0 else ((.end_ms // 0) - .t_ms) end' <<<"$1"
}

# ai.usage rows the gateway logged since a time, once they have arrived (waits for at least $2 of them)
uc_rows_since() { # since-ms min-rows
  local rows n
  for _ in $(seq 1 25); do
    rows=$(cw gateway/ "$1" '"ai.usage"' | jq -c '[.[]|.m|fromjson?|select(.target=="ai.usage")|.fields]')
    n=$(jq length <<<"$rows")
    [ "$n" -ge "${2:-1}" ] && break
    sleep 3
  done
  echo "$rows"
}

# Optional string arguments the model sent as "" to browser tools, over every frame recorded so far:
# [{tool, arg}]. A schema that makes optional arguments required pushes the model to fill them with "".
uc_empty_args() {
  jq -cR 'fromjson?' "$EVENTS" | jq -sc '[.[] | select(.frame.event.kind == "tool_start" and (.frame.event.name | test("^mcp__browser__")))
    | .frame.event as $e | ($e.input | to_entries[] | select(.value == "")) | {tool: $e.name, arg: .key}]'
}

uc_grant() { "$GRANT_FN" "$@"; }

# ---- the grants ------------------------------------------------------------------------------------------

_grant_assistant() { # session [tamper]: no sandbox at all: the folder is the bundle, and the grant names no exec endpoint
  # The managed connector: a URL on the stand-in endpoint and a bmc_v1 token for this session. The
  # third party's key is not an input to the minter.
  "$DEMO" mint-grant --tenant "$UC_TENANT" --session "$1" --shard demo --workspace /workspace/assistant \
    --exec-url "" --bundle "$BUNDLE_URL=$BUNDLE_SHA" \
    --mcp "vault=http://$STANDIN_IP:$STANDIN_PORT/managed/vault/mcp" --mcp-managed vault \
    --limits-models "gpt-4.1-mini,gpt-5-mini" ${2:+--tamper}
}

# The coding agent's exec endpoint is the front, keyed on the first session's name: a VM that does not
# exist yet is made when the first request arrives, and seeded from the bundle (front.py).
_front_url() { echo "http://sandbox.$DNS:$FRONT_PORT/exec/${1:-$SESS_C}?bundle=$(jq -rn --arg u "$BUNDLE_URL" '$u|@uri')"; }
_grant_coding() { # session [tamper]
  "$DEMO" mint-grant --tenant "$UC_TENANT" --session "$1" --shard demo --workspace "$VM_HOME/workspace" \
    --exec-url "$(_front_url)" --bundle "$BUNDLE_URL=$BUNDLE_SHA" --limits-models gpt-5-mini ${2:+--tamper}
}
# The same VM, and now the browser in it: a second session whose grant also names the bridge. The
# bridge's address exists only once the VM does.
_grant_coding_browser() { # session [tamper]
  local traffic
  traffic=$(vm_field "$SESS_C" traffic_token)
  DEMO_MCP_HEADER_BROWSER=$traffic "$DEMO" mint-grant --tenant "$UC_TENANT" --session "$1" --shard demo --workspace "$VM_HOME/workspace" \
    --exec-url "$(_front_url)" --bundle "$BUNDLE_URL=$BUNDLE_SHA" \
    --mcp "browser=$(vm_bridge_url "$SESS_C")" --mcp-sandbox browser --mcp-header browser=e2b-traffic-access-token \
    --limits-models gpt-5-mini ${2:+--tamper}
}

# The browser agent: the exec endpoint and the browser's MCP address are both the front, keyed on its
# session. Neither names a VM, so neither costs one until the model calls a browser tool: the front makes
# the VM on that first request, seeds it from the bundle, installs the browser stack (the VM fetches the
# rustwright-mcp binary itself) and forwards.
_front_mcp_url() { echo "http://sandbox.$DNS:$FRONT_PORT/mcp/$SESS_B?bundle=$(jq -rn --arg u "$BUNDLE_URL" '$u|@uri')&browser=$(jq -rn --arg u "$RW_URL" '$u|@uri')&site=site"; }
_grant_browser() { # session [tamper]
  "$DEMO" mint-grant --tenant "$UC_TENANT" --session "$1" --shard demo --workspace "$VM_HOME/workspace" \
    --exec-url "$(_front_url "$SESS_B")" --bundle "$BUNDLE_URL=$BUNDLE_SHA" \
    --mcp "browser=$(_front_mcp_url)" --mcp-sandbox browser --limits-models gpt-5-mini ${2:+--tamper}
}

# ---- preparation -------------------------------------------------------------------------------------------

uc_prepare() { # called by demo.sh once, before the checks
  UC_NAME=$MODE
  UC_OPENS=$WORK/opens.ndjson
  : > "$UC_OPENS"
  UC_TENANT=uc-$MODE-$SID
  UC_JSON=$(jq -nc --arg n "$MODE" --arg t "${UC_TITLE[$MODE]}" --arg s "${UC_SUMMARY[$MODE]}" --argjson tree "$(uc_tree "$MODE")" \
    '{name:$n, title:$t, summary:$s, folder:("deploy/demo/agents/"+$n), tree:$tree}')
  case "$MODE" in
    assistant) uc_prepare_assistant ;;
    coding) uc_prepare_coding ;;
    browser) uc_prepare_browser ;;
  esac
}

uc_need() { # what is missing -> exits
  echo "demo: $MODE needs $1. Bring the deployment up with: SANDBOX=both ./deploy/demo/up.sh" >&2
  exit 1
}

uc_prepare_assistant() {
  [ "$USECASES" = 1 ] || uc_need "the stand-in endpoint and the bundles bucket"
  STANDIN_IP=$(svc_ip standin)
  [ -n "$STANDIN_IP" ] || uc_need "the stand-in managed-MCP endpoint (not running)"
  for _ in $(seq 1 30); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' -X POST -d '{}' "http://$STANDIN_IP:$STANDIN_PORT/managed/vault/mcp")" = 401 ] && break
    sleep 2
  done
  echo "  stand-in managed endpoint  http://$STANDIN_IP:$STANDIN_PORT (answers 401 without a token)"
  DEMO_MANAGED_SECRET=$(ssm_get managed-secret)
  export DEMO_MANAGED_SECRET
  # The "third party's" real key: read here only so the checks can look for it. It is never an input
  # to the minter, never printed and never written to the record.
  PROVIDER_SECRET=$(ssm_get provider-secret)
  GRANT_FN=_grant_assistant
  # The folder is the deployment: this run's marker goes into the skill, and the bundle carries it.
  uc_stage
  SKILL_MARKER="heron-$SID-$RANDOM"
  sed -i "s/MARKER_PLACEHOLDER/$SKILL_MARKER/" "$STAGE/.agents/skills/house-style/SKILL.md"
  uc_bundle || { echo "demo: could not publish the bundle" >&2; exit 1; }
  SESS_A1=demo.u${SID}a
  SESS_A2=demo.u${SID}b
  SESS_A3=demo.u${SID}c
  # Open each session once without a turn: this is where a session without a bundle used to spend ~30
  # sandbox round trips reading its own folder, and where this one has no sandbox to ask.
  for s in "$SESS_A1" "$SESS_A2" "$SESS_A3"; do
    uc_open "$s"
  done
  echo "  three sessions opened (t=$UC_READY_MS)"
  tl demo sessions_opened '["you","agent"]' null "$UC_READY_MS"
}

# The folder as a bundle: a copy with this run's stamps in it, as a plain ustar tar named by its hash,
# uploaded to the deployment's private bucket and handed to the replica as a presigned link. The
# tar is also what a sandbox VM is seeded from.
uc_stage() {
  STAGE=$WORK/stage/$MODE
  mkdir -p "$WORK/stage"
  rm -rf "$STAGE"
  cp -a "$UC_AGENTS/$MODE" "$STAGE"
  find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
}
uc_bundle() {
  BUNDLE_TAR=$WORK/bundle-$MODE.tar
  (cd "$STAGE" && find . -mindepth 1 | sed 's|^\./||' | LC_ALL=C sort \
    | tar --format=ustar --owner=0 --group=0 --numeric-owner --mtime=@0 --no-recursion -cf "$BUNDLE_TAR" -T -) || return 1
  BUNDLE_SHA=$(sha256sum "$BUNDLE_TAR" | cut -d' ' -f1)
  BUNDLE_BYTES=$(stat -c %s "$BUNDLE_TAR")
  local bucket
  bucket=$(bundle_bucket)
  command aws --region "$REGION" s3 cp "$BUNDLE_TAR" "s3://$bucket/$BUNDLE_SHA.tar" --only-show-errors || return 1
  BUNDLE_URL=$(command aws --region "$REGION" s3 presign "s3://$bucket/$BUNDLE_SHA.tar" --expires-in 21600) || return 1
  echo "  bundle  $(tar -tf "$BUNDLE_TAR" | grep -vc '/$') files, $BUNDLE_BYTES bytes, sha256 ${BUNDLE_SHA:0:12}…, in s3://$bucket (presigned link, 6 h)"
  UC_JSON=$(jq -c --arg sha "$BUNDLE_SHA" --argjson b "$BUNDLE_BYTES" --arg bucket "$bucket" \
    '.bundle = {sha256:$sha, bytes:$b, store:("a private S3 bucket (" + $bucket + "), fetched with a presigned link")}' <<<"$UC_JSON")
}

# Open a session with no turn and record how long that took (connect + the first answer) and when, so the
# record can say how many requests the sandbox side saw while it happened.
uc_open() { # session
  local out t0 ready
  t0=$(now_ms)
  out=$(chat "$1" --command '{"type":"get_state","id":"open"}' --timeout-secs 90) || out='{}'
  ready=$(now_ms)
  jq -nc --arg s "$1" --argjson t0 "$t0" --argjson ready "$ready" --argjson c "$(jq '.connected_ms // 0' <<<"$out")" --argjson t "$(jq '.total_ms // 0' <<<"$out")" \
    '{session:$s, started_ms:$t0, ready_ms:$ready, connected_ms:$c, state_ms:$t, open_ms:($c+$t)}' >> "$UC_OPENS"
  UC_READY_MS=$ready
  echo "  opened $1 in $(( $(jq '.connected_ms // 0' <<<"$out") + $(jq '.total_ms // 0' <<<"$out") )) ms"
}

# The record's "session open" section: per session, the open latency and the sandbox requests seen while
# it was opening (the sandbox's own log, every VM's stub and the Fargate one; for the coding agent also
# what its exec front forwarded), and whether a VM existed then.
uc_session_open_json() {
  local reqs fronts
  sleep 6   # the stubs' lines reach CloudWatch a few seconds late
  reqs=$(cw sandbox/ "$((START_MS - 120000))" '"exec_request"' | jq -c '[.[]|(.m|fromjson? // empty)|select(.event=="exec_request")|.ts_ms]')
  fronts='[]'
  [ "$MODE" != coding ] && [ "$MODE" != browser ] || fronts=$(cw front/ "$((START_MS - 120000))" '"exec_request"' | jq -c '[.[]|(.m|fromjson? // empty)|select(.event=="exec_request" or .event=="mcp_request")|.ts_ms]')
  jq -sc --argjson r "$reqs" --argjson f "$fronts" --arg mode "$MODE" --argjson vm "${VM_AT_OPEN:-null}" '
    map(. as $s | $s + {exec_requests: ([$r[] | select(. >= ($s.started_ms - 500) and . <= ($s.ready_ms + 1500))] | length),
        front_forwarded: ([$f[] | select(. >= ($s.started_ms - 500) and . <= ($s.ready_ms + 1500))] | length)})
    | {sessions: ., exec_requests_at_open: (map(.exec_requests) | add // 0), sessions_opened: length,
       per_session: ((map(.exec_requests) | add // 0) / (length | if . == 0 then 1 else . end)),
       open_ms_mean: ((map(.open_ms) | add // 0) / (length | if . == 0 then 1 else . end) | floor),
       vms_at_open: $vm}' "$UC_OPENS"
}

uc_prepare_coding() {
  [ "$USE_E2B" = 1 ] && [ "$USECASES" = 1 ] || uc_need "E2B and the exec front"
  SESS_C=demo.u${SID}c
  SESS_C2=demo.u${SID}e
  FRONT_PORT=8084
  # The folder, with this run's build line in it; the VM is seeded from the bundle when it is made.
  uc_stage
  BUILD_STAMP=build-$SID
  echo "$BUILD_STAMP" > "$STAGE/app/BUILD"
  # The suite as the agent will find it, run here on the same files (no VM exists yet to run it in).
  TESTS_BEFORE=$(cd "$STAGE/app" && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest 2>&1 | tail -n 3; echo "exit=${PIPESTATUS[0]}")
  TESTS_ORIG_SUM=$(md5sum "$STAGE/app/test_server.py" | cut -d' ' -f1)
  uc_bundle || { echo "demo: could not publish the bundle" >&2; exit 1; }
  rustwright_bin >/dev/null || { echo "demo: could not build rustwright-mcp" >&2; exit 1; }
  GRANT_FN=_grant_coding
  echo "  before: $(tr '\n' ' ' <<<"$TESTS_BEFORE")"
  # Open the session. There is no sandbox VM for it yet, and opening must not make one.
  VM_AT_OPEN=$(vm_count_for "$SESS_C") || VM_AT_OPEN=null
  uc_open "$SESS_C"
  VM_AT_OPEN_AFTER=$(vm_count_for "$SESS_C") || VM_AT_OPEN_AFTER=null
  echo "  E2B sandboxes for this session before the open: $VM_AT_OPEN, after it: $VM_AT_OPEN_AFTER"
  VM_AT_OPEN=${VM_AT_OPEN_AFTER}
}

uc_prepare_browser() {
  [ "$USE_E2B" = 1 ] && [ "$USECASES" = 1 ] || uc_need "E2B and the exec front"
  SESS_B=demo.u${SID}d
  FRONT_PORT=8084
  uc_stage
  PAGE_STAMP=stamp-$SID-$RANDOM
  sed -i "s/STAMP_PLACEHOLDER/$PAGE_STAMP/" "$STAGE/site/index.html"
  local rw bucket sha
  rw=$(rustwright_bin) || { echo "demo: could not build rustwright-mcp" >&2; exit 1; }
  # The browser server's tool list, from the server itself, goes into the bundle: the agent offers these
  # tools at open without starting the server (or the VM it will run in).
  "$HERE/mcp-tools.sh" browser "$rw" > "$STAGE/.mcp-tools.json" 2>/dev/null || { echo "demo: could not list the browser server's tools" >&2; exit 1; }
  echo "  saved tool list  $(jq '.browser|length' "$STAGE/.mcp-tools.json") browser tools, from running the server once"
  uc_bundle || { echo "demo: could not publish the bundle" >&2; exit 1; }
  # The binary the VM will fetch for itself, by a presigned link (nothing passes through the front).
  sha=$(sha256sum "$rw" | cut -d' ' -f1)
  bucket=$(bundle_bucket)
  command aws --region "$REGION" s3 cp "$rw" "s3://$bucket/rustwright-mcp-$sha" --only-show-errors || { echo "demo: could not publish the browser server" >&2; exit 1; }
  RW_URL=$(command aws --region "$REGION" s3 presign "s3://$bucket/rustwright-mcp-$sha" --expires-in 21600) || exit 1
  GRANT_FN=_grant_browser
  # Open the session. There is no sandbox VM for it, and opening must not make one.
  VM_AT_OPEN=$(vm_count_for "$SESS_B") || VM_AT_OPEN=null
  uc_open "$SESS_B"
  VM_AT_OPEN_AFTER=$(vm_count_for "$SESS_B") || VM_AT_OPEN_AFTER=null
  echo "  E2B sandboxes for this session before the open: $VM_AT_OPEN, after it: $VM_AT_OPEN_AFTER"
  VM_AT_OPEN=${VM_AT_OPEN_AFTER}
}

# ============================================================================================================
# assistant
# ============================================================================================================

check_persona_applied() {
  local s=$SESS_A1 t0 out mcp offered want text rows models allowed_ok
  t0=$(now_ms)
  out=$(chat "$s" --prompt "Introduce yourself in one sentence. Then list, comma separated, the exact names of every tool you can call (function names only).")
  note_calls "$s" "$out"
  mcp=$(chat "$s" --command '{"type":"get_mcp","id":"m"}' --timeout-secs 30)
  offered=$(jq -c '[.responses[]|select(.command=="get_mcp")|.data.tools[]]|sort' <<<"$mcp")
  want=$(sed -n 's/^tools: *//p' "$UC_AGENTS/assistant/AGENTS.md" | tr ',' '\n' | sed 's/ //g' | grep '^mcp__' | jq -R . | jq -sc 'sort')
  text=$(jq -r .text <<<"$out")
  rows=$(uc_rows_since "$t0" 2)
  models=$(jq -c '[.[].requested_model]|unique' <<<"$rows")
  local frontmatter_model
  frontmatter_model=$(sed -n 's/^model: *//p' "$UC_AGENTS/assistant/AGENTS.md" | head -1)
  echo "  frontmatter: model $frontmatter_model, tools $want"
  echo "  offered by get_mcp: $offered"
  echo "  models in the gateway's billing rows: $models ($(jq length <<<"$rows") rows)"
  allowed_ok=$(jq -r --argjson m "$models" --arg fm "$frontmatter_model" '$m == [$fm]' <<<null)
  if [ "$offered" = "$want" ] && [ "$allowed_ok" = true ] && [[ $offered != *read_wiki_contents* ]]; then
    result 1 "the model was $frontmatter_model in every billing row (the replica's own default is gpt-5-mini), and the tools offered are exactly the allow-list: DeepWiki's read_wiki_contents is not among them" \
      "$(jq -nc --argjson o "$offered" --argjson w "$want" --argjson m "$models" --argjson n "$(jq length <<<"$rows")" --arg fm "$frontmatter_model" --arg says "$(head -c 400 <<<"$text")" \
        '{frontmatter_model:$fm, billing_rows:$n, requested_models:$m, tools_allow_list:$w, tools_offered:$o, replica_default_model:"gpt-5-mini", reply_signed_by_persona:($says|contains("Juniper")), model_says:$says}')" \
      "$(jq -nc --arg m "$frontmatter_model" '{label:"model from AGENTS.md",value:$m,unit:"in every billing row"}')"
  else
    result 0 "offered=$offered want=$want models=$models persona_text=${text:0:120}"
  fi
}

check_remote_mcp_real() {
  local out tool text ms
  out=$(chat "$SESS_A1" --prompt "Use DeepWiki's ask_wiki_question tool to ask about the repository modelcontextprotocol/rust-sdk: what does it provide? Answer in two sentences.")
  note_calls "$SESS_A1" "$out"
  tool=$(uc_tool_out "$out" '^mcp__deepwiki__ask_wiki_question$')
  ms=$(uc_tool_ms "$out" '^mcp__deepwiki__ask_wiki_question$')
  text=$(jq -r .text <<<"$out")
  echo "  DeepWiki answered with ${#tool} bytes in $ms ms"
  if [ "${#tool}" -gt 300 ] && [[ $tool == *rmcp* || $tool == *"Model Context Protocol"* ]] && [ -n "$text" ]; then
    result 1 "mcp.deepwiki.com answered ${#tool} bytes of real documentation about the repository, in $ms ms; the server came from .mcp.json" \
      "$(jq -nc --argjson b "${#tool}" --argjson ms "$ms" --arg ex "$(head -c 300 <<<"$tool")" --arg url "$(jq -r '.mcpServers.deepwiki.url' "$UC_AGENTS/assistant/.mcp.json")" \
        '{server_url:$url, answer_bytes:$b, tool_round_trip_ms:$ms, excerpt:$ex}')" \
      "$(jq -nc --argjson b "${#tool}" '{label:"DeepWiki answer",value:$b,unit:"bytes"}')"
  else
    result 0 "DeepWiki's answer was not real content: ${tool:0:200}"
  fi
}

# Count the lines of $1 (a file) holding the secret; the secret reaches grep through a 0600 file.
count_secret_in() { # file
  grep -cF -f "$WORK/needle" "$1" 2>/dev/null || true
}

check_managed_mcp_no_key() {
  local s=$SESS_A1 t0 out tool ref calls scan_logs scan_msgs g gj msgs_file text scan_mem ref_in_text
  t0=$(now_ms)
  printf '%s' "$PROVIDER_SECRET" > "$WORK/needle"
  chmod 600 "$WORK/needle"
  out=$(chat "$s" --prompt "Look up the account status for the customer acme and quote its vault_ref.")
  note_calls "$s" "$out"
  tool=$(uc_tool_out "$out" '^mcp__vault__')
  text=$(jq -r .text <<<"$out")
  ref=$(grep -o 'VAULT-[0-9a-f]*' <<<"$tool" | head -1)
  echo "  the managed tool answered: ${tool:0:120}"
  # what the stand-in saw: the bmc_v1 token verified, the call forwarded upstream with the real key
  for _ in $(seq 1 12); do
    calls=$(cw standin/ "$t0" '"managed_call"' | jq -c '[.[]|.m|fromjson|.detail]')
    [ "$(jq length <<<"$calls")" -ge 1 ] && break
    sleep 5
  done
  # 1. the grant: unsealed with the agent's own verifier
  g=$(DEMO_GRANT=$(uc_grant "$s") DEMO_NEEDLE=$PROVIDER_SECRET "$DEMO" inspect-grant)
  # 2. the agent task's logs, since this run began
  # the whole life of the agent task (not only this run): every line it has written
  cw agent/ "$((START_MS - 4 * 3600 * 1000))" | jq -r '.[].m' > "$WORK/agent.log"
  scan_logs=$(count_secret_in "$WORK/agent.log")
  # 3. the transcript the agent persisted for this session, as the agent returns it
  chat "$s" --command '{"type":"get_messages","id":"gm"}' --timeout-secs 60 | jq -c '.responses[]|select(.command=="get_messages")|.data' > "$WORK/messages.json"
  scan_msgs=$(count_secret_in "$WORK/messages.json")
  local msgs_have_ref
  msgs_have_ref=$(grep -c 'vault_ref=VAULT-' "$WORK/messages.json" || true)
  # 4. the agent process's own memory, through ECS Exec
  scan_mem=$(scan_agent_memory)
  echo "  grant: $(jq -c '{managed:.connectors[0].managed, holds_a_bmc_token:.connectors[0].holds_a_bmc_token, key_inside:(.connectors[0].header_values_contain_needle or .exec_header_values_contain_needle or .token_text_contains_needle)}' <<<"$g")"
  local log_ctl
  log_ctl=$(grep -c 'websocket listening' "$WORK/agent.log" || true)
  echo "  agent logs: $(wc -l < "$WORK/agent.log") lines ($log_ctl is the agent's startup line, the control), $scan_logs hold the key; transcript: $(wc -c < "$WORK/messages.json") bytes, $scan_msgs hold it"
  echo "  agent memory: $scan_mem"
  rm -f "$WORK/needle"
  gj=$(jq -c '{managed:.connectors[0].managed, holds_a_bmc_token:.connectors[0].holds_a_bmc_token, key_inside:(([.connectors[].header_values_contain_needle]|any) or .exec_header_values_contain_needle or .gateway_key_contains_needle or .token_text_contains_needle), header_names:.connectors[0].header_names}' <<<"$g")
  local mem_hits mem_control mem_ok
  mem_hits=$(jq -r '.hits // empty' <<<"$scan_mem")
  mem_control=$(jq -r '.control_hits // empty' <<<"$scan_mem")
  mem_ok=$(jq -r '.scanned // false' <<<"$scan_mem")
  ref_in_text=$([[ -n $ref && $text == *"$ref"* ]] && echo yes || echo no)
  local mem_phrase
  mem_phrase="the agent process's memory was not scanned ($(jq -r '.why // "ECS Exec unavailable"' <<<"$scan_mem"))"
  [ "$mem_ok" != true ] || mem_phrase="the agent process's memory ($(jq -r '.mapped_mb' <<<"$scan_mem") MB read through ECS Exec; the control string, this run's tenant id, was found $mem_control times) holds it $mem_hits times"
  if [ -n "$ref" ] && [ "$(jq 'length' <<<"$calls")" -ge 1 ] && [ "$(jq -r '.managed and .holds_a_bmc_token and (.key_inside|not)' <<<"$gj")" = true ] \
    && [ "${scan_logs:-1}" = 0 ] && [ "${scan_msgs:-1}" = 0 ] && [ "${msgs_have_ref:-0}" -ge 1 ] && { [ "$mem_ok" != true ] || [ "${mem_hits:-1}" = 0 ]; }; then
    result 1 "the managed tool answered ($ref); the real key is not in the grant, the agent's logs ($(wc -l < "$WORK/agent.log") lines), the session transcript, and $mem_phrase" \
      "$(jq -nc --argjson g "$gj" --argjson calls "$calls" --argjson sl "$scan_logs" --argjson sm "$scan_msgs" --argjson mem "$scan_mem" --arg ref "$ref" --arg said "$ref_in_text" --argjson lines "$(wc -l < "$WORK/agent.log")" --argjson bytes "$(wc -c < "$WORK/messages.json")" \
        '{vault_ref:$ref, reply_quotes_it:($said=="yes"), grant:$g, standin_calls:$calls, scanned:{grant:"unsealed with the agent verifier", agent_log_lines:$lines, agent_log_hits:$sl, transcript_bytes:$bytes, transcript_hits:$sm, agent_memory:$mem}}')" \
      "$(jq -nc --argjson n "$(( ${scan_logs:-0} + ${scan_msgs:-0} + ${mem_hits:-0} ))" '{label:"copies of the key in the agent",value:$n,unit:"found"}')"
  else
    result 0 "ref=$ref standin_calls=$(jq length <<<"$calls") grant=$gj logs=$scan_logs transcript=$scan_msgs memory=$scan_mem"
  fi
}

# Read the agent process's memory through ECS Exec and count a needle and a control. The needle is the
# fixed prefix of the real key (never the key itself, so the key never goes near the agent task); the
# control is the managed token's prefix, which the agent does hold: it proves the scan reads that memory.
scan_agent_memory() {
  local task script out
  task=$(q ecs list-tasks --cluster "$CLUSTER" --service-name "$P-agent" --desired-status RUNNING --query 'taskArns[0]')
  [ -n "$task" ] || { echo '{"scanned":false,"why":"no running agent task"}'; return; }
  script='pid=""
for p in /proc/[0-9]*; do [ "$(readlink $p/exe 2>/dev/null)" = /usr/local/bin/beyond-ai-agent ] && pid=${p#/proc/}; done
[ -n "$pid" ] || { echo "SCAN-ERROR no agent pid"; exit 0; }
: > /tmp/scan.hits; kb=0; bad=0
while read -r range perms rest; do
  case "$perms" in rw*) ;; *) continue ;; esac
  s=$((0x${range%-*})); e=$((0x${range#*-}))
  # the heap is a 1 GiB reservation, mostly never touched, and reading untouched pages is slow: its first
  # 64 MiB are read, every other writable mapping in full
  [ $((e - s)) -le 67108864 ] || e=$((s + 67108864))
  dd if=/proc/$pid/mem bs=4096 skip=$((s / 4096)) count=$(((e - s) / 4096)) 2>/tmp/scan.err | tr -c "[:print:]" "\n" | grep -o -e bsk_live_demo_ -e "bmc_v1\\." -e "uc-assistant-" >> /tmp/scan.hits
  if grep -q "can.t open\\|ermission" /tmp/scan.err; then bad=$((bad + 1)); else kb=$((kb + (e - s) / 1024)); fi
done < /proc/$pid/maps
hits=$(grep -c bsk_live_demo_ /tmp/scan.hits); ctl=$(grep -c "uc-assistant-" /tmp/scan.hits); bmc=$(grep -c "bmc_v1" /tmp/scan.hits)
rm -f /tmp/scan.hits /tmp/scan.err /tmp/scan.runs
echo "SCAN-RESULT pid=$pid kb=$kb hits=$hits control=$ctl bmc=$bmc unreadable=$bad"'
  local b64
  b64=$(printf '%s\n' "$script" | base64 -w0)
  out=$(command aws --region "$REGION" ecs execute-command --cluster "$CLUSTER" --task "$task" --container agent --interactive \
    --command "/bin/sh -c 'echo $b64 | base64 -d | sh'" 2>&1 | tr -d '\r') || true
  local line
  line=$(grep -o 'SCAN-RESULT.*' <<<"$out" | head -1)
  if [ -z "$line" ]; then
    jq -nc --arg why "$(grep -o 'SCAN-ERROR.*\|[A-Za-z ]*[Ee]rror[^"]*' <<<"$out" | head -1 | head -c 200)" '{scanned:false, why:($why // "no result")}'
    return
  fi
  local kb hits ctl bad
  kb=$(sed -n 's/.* kb=\([0-9]*\).*/\1/p' <<<"$line")
  hits=$(sed -n 's/.* hits=\([0-9]*\).*/\1/p' <<<"$line")
  ctl=$(sed -n 's/.* control=\([0-9]*\).*/\1/p' <<<"$line")
  bad=$(sed -n 's/.* unreadable=\([0-9]*\).*/\1/p' <<<"$line")
  local bmc
  bmc=$(sed -n 's/.* bmc=\([0-9]*\).*/\1/p' <<<"$line")
  # A scan that read nothing, or could not find the control, proves nothing.
  if [ "${kb:-0}" -gt 0 ] && [ "${ctl:-0}" -gt 0 ]; then
    jq -nc --argjson kb "$kb" --argjson h "$hits" --argjson c "$ctl" --argjson bad "${bad:-0}" --argjson bmc "${bmc:-0}" \
      '{scanned:true, mapped_mb:($kb/1024|floor), hits:$h, control_hits:$c, bmc_token_prefix_hits:$bmc, unreadable_ranges:$bad, needle:"bsk_live_demo_ (the real key'"'"'s prefix)", control:"uc-assistant- (this run'"'"'s tenant id, which the agent holds)", note:"every writable mapping was read in full except the heap reservation (1 GiB, mostly untouched), whose first 64 MiB were read"}'
  else
    jq -nc --argjson kb "${kb:-0}" --argjson c "${ctl:-0}" '{scanned:false, why:("the scan read "+($kb|tostring)+" KB and found the control "+($c|tostring)+" times, so it proves nothing")}'
  fi
}

check_memory_across_sessions() {
  local code="orchid-$RANDOM$RANDOM" o1 o2 mem_calls recalled t2 msgs
  o1=$(chat "$SESS_A1" --prompt "Remember this for later: my project codename is $code. Save it with your memory tool: command create, path /memories/MEMORY.md (the path argument, not new_path), file_text a line stating the codename. Then confirm in one sentence.")
  note_calls "$SESS_A1" "$o1"
  mem_calls=$(uc_tools "$o1" '^memory$' | jq length)
  # A brand-new session: another id, another grant, another process-side session, the same tenant.
  o2=$(chat "$SESS_A2" --prompt "What is my project codename? Check your memory first with your memory tool: command view, path /memories/MEMORY.md. Then answer in one sentence.")
  note_calls "$SESS_A2" "$o2"
  recalled=$(jq -r '.text // ""' <<<"$o2")
  msgs=$(chat "$SESS_A2" --command '{"type":"get_messages","id":"gm"}' --timeout-secs 60 | jq -c '.responses[]|select(.command=="get_messages")|.data.messages')
  # The code must not have been in the new session's own conversation before the question was asked.
  local prior
  # what the session had in its conversation before the assistant first spoke (the question, nothing else)
  prior=$(jq -r --arg c "$code" '((map(.role)|index("assistant")) // length) as $i | [.[0:$i][] | tostring | select(contains($c))] | length' <<<"$msgs")
  echo "  session 1 stored it with $mem_calls memory call(s); session 2 said: $(head -c 160 <<<"$recalled")"
  if [ "$mem_calls" -ge 1 ] && [[ $recalled == *"$code"* ]] && [ "$prior" = 0 ] && [ "$SESS_A1" != "$SESS_A2" ]; then
    result 1 "session 1 saved the codename with the memory tool; a brand-new session (a different session id) answered it, and no user message in that session had said it" \
      "$(jq -nc --arg a "$SESS_A1" --arg b "$SESS_A2" --argjson m "$mem_calls" --argjson t2 "$(uc_tools "$o2" '^memory$' | jq length)" \
        '{session_one:$a, session_two:$b, memory_calls_in_session_one:$m, memory_calls_in_session_two:$t2, code_in_session_two_user_messages:0}')" \
      '{"label":"recalled in a new session","value":"yes","unit":""}'
  else
    result 0 "stored=$mem_calls recalled=${recalled:0:200} prior=$prior"
  fi
}

check_limits_enforced() {
  local s=$SESS_A3 refused ok state r_ok r_err
  refused=$(chat "$s" --command '{"type":"set_model","id":"sm1","model":"gpt-5-nano"}' --timeout-secs 30)
  r_ok=$(jq -r '[.responses[]|select(.command=="set_model")][0].success' <<<"$refused")
  r_err=$(jq -r '[.responses[]|select(.command=="set_model")][0].error // ""' <<<"$refused")
  state=$(chat "$s" --command '{"type":"get_state","id":"gs"}' --timeout-secs 30 | jq -r '.responses[]|select(.command=="get_state")|.data.model')
  ok=$(chat "$s" --command '{"type":"set_model","id":"sm2","model":"gpt-5-mini"}' --timeout-secs 30 | jq -r '[.responses[]|select(.command=="set_model")][0].success')
  echo "  set_model gpt-5-nano -> success=$r_ok: ${r_err:0:200}"
  echo "  the model is still $state; set_model gpt-5-mini (inside the ceiling) -> success=$ok"
  if [ "$r_ok" = false ] && [[ $r_err == *gpt-5-nano* ]] && [ "$state" = "$(sed -n 's/^model: *//p' "$UC_AGENTS/assistant/AGENTS.md" | head -1)" ] && [ "$ok" = true ]; then
    result 1 "the grant allows gpt-4.1-mini and gpt-5-mini: set_model to gpt-5-nano was refused by name and the model stayed $state; gpt-5-mini was accepted" \
      "$(jq -nc --arg e "$r_err" --arg st "$state" '{refused_model:"gpt-5-nano", refusal:$e, model_after_refusal:$st, allowed_model_accepted:true, limits_models:["gpt-4.1-mini","gpt-5-mini"]}')" \
      '{"label":"set_model gpt-5-nano","value":"refused","unit":""}'
  else
    result 0 "refused=$r_ok err=$r_err state=$state allowed=$ok"
  fi
}

check_skill_from_bundle() {
  local s=$SESS_A3 out reads read_out text ms
  out=$(chat "$s" --prompt "Your folder has a skill about the team's house style. Read its SKILL.md with the read tool, then quote its HOUSE-STYLE-MARKER line exactly, character for character.")
  note_calls "$s" "$out"
  reads=$(jq -c '[.tools[]|select(.name=="read" and ((.input|tostring)|contains("SKILL.md")))]' <<<"$out")
  read_out=$(jq -r '[.[]|.output // ""]|join("\n")' <<<"$reads")
  ms=$(jq -r '[.[]|((.end_ms // 0) - .t_ms)]|first // 0' <<<"$reads")
  text=$(jq -r .text <<<"$out")
  echo "  read calls on the skill: $(jq length <<<"$reads") ($ms ms); the marker is in the file's text: $([[ $read_out == *"$SKILL_MARKER"* ]] && echo yes || echo no); the reply quotes it: $([[ $text == *"$SKILL_MARKER"* ]] && echo yes || echo no)"
  if [ "$(jq length <<<"$reads")" -ge 1 ] && [[ $read_out == *"HOUSE-STYLE-MARKER: $SKILL_MARKER"* ]] && [[ $text == *"$SKILL_MARKER"* ]]; then
    result 1 "the model read .agents/skills/house-style/SKILL.md through the read tool in $ms ms, from the bundle, and quoted this run's marker ($SKILL_MARKER), which exists only in the bundle; the session has no sandbox" \
      "$(jq -nc --arg m "$SKILL_MARKER" --argjson n "$(jq length <<<"$reads")" --argjson ms "$ms" --arg p "$(jq -r '.[0].input|tostring' <<<"$reads" | head -c 200)" '{marker:$m, read_calls:$n, read_ms:$ms, read_input:$p, marker_in_tool_output:true, marker_in_reply:true}')" \
      "$(jq -nc --argjson ms "$ms" '{label:"skill read from the bundle",value:$ms,unit:"ms"}')"
  else
    result 0 "reads=$(jq length <<<"$reads") tool_output=${read_out:0:200} reply=${text:0:200}"
  fi
}

check_no_sandbox_at_all() {
  local g s grants tools_used reqs listening total used_reaching
  # 1. The grants: each session's, unsealed with the agent's own verifier.
  grants='[]'
  for s in "$SESS_A1" "$SESS_A2" "$SESS_A3"; do
    g=$(DEMO_GRANT=$(uc_grant "$s") "$DEMO" inspect-grant)
    grants=$(jq -c --arg s "$s" --argjson g "$g" '. + [{session:$s, exec_url_empty:$g.exec_url_empty, exec_header_names:$g.exec_header_names, bundle_sha256:$g.bundle_sha256}]' <<<"$grants")
  done
  # 2. The sandbox side, over the whole run, session open included: every exec request any sandbox
  # logged. The Fargate workspace host is up (its startup line is the control), so a request would show.
  tools_used=$(jq -cR 'fromjson?' "$EVENTS" | jq -r 'select(.frame.event.kind=="tool_start")|.frame.event.name' | sort | uniq -c | awk '{print $2":"$1}' | jq -R . | jq -sc .)
  sleep 12   # the stub's log lines reach CloudWatch a few seconds late
  reqs=$(cw sandbox/ "$((START_MS - 120000))" '"exec_request"' | jq -c '[.[]|(.m|fromjson? // empty)|select(.event=="exec_request")|.ts_ms]')
  listening=$(cw sandbox/ "$((START_MS - 20 * 3600 * 1000))" '"listening"' | jq -c '[.[]|(.m|fromjson? // empty)|select(.event=="listening")]|length')
  total=$(jq length <<<"$reqs")
  used_reaching=$(jq -r '[.[]|select(test("^(bash|write|edit|grep|find|ls|bash_job):"))]|length' <<<"$tools_used")
  echo "  grants: $(jq -c '[.[]|{session, exec_url_empty}]' <<<"$grants")"
  echo "  tools the model called in all sessions: $tools_used"
  echo "  exec requests any sandbox logged since the run began: $total (the Fargate workspace host was up: $listening startup line(s) in its log)"
  if [ "$(jq 'all(.[]; .exec_url_empty == true and (.exec_header_names|length) == 0 and .bundle_sha256 != null)' <<<"$grants")" = true ] && [ "$total" = 0 ] && [ "${listening:-0}" -ge 1 ] && [ "$used_reaching" = 0 ]; then
    result 1 "the three grants name no exec endpoint (and carry the bundle), and the sandbox side logged 0 exec requests over the whole run, session open included; the model's tool calls: $tools_used" \
      "$(jq -nc --argjson g "$grants" --argjson t "$tools_used" --argjson r "$total" --argjson l "${listening:-0}" '{grants:$g, model_tool_calls:$t, sandbox_exec_requests_whole_run:$r, workspace_host_startup_lines_seen:$l}')" \
      '{"label":"sandbox requests, whole run","value":"0","unit":""}'
  else
    result 0 "grants=$grants requests=$total listening=$listening tools=$tools_used"
  fi
}

# ============================================================================================================
# coding
# ============================================================================================================

vm_in() { # session script -> runs it in the session's VM
  vm_run "$(vm_field "$1" id)" "$(vm_field "$1" envd_token)" "$2"
}

check_vm_created_on_first_tool() {
  [ "$MODE" != browser ] || { _vm_created_on_first_browser_tool; return; }
  local ready=$UC_READY_MS out tool t_tool t_end t_sent fl first_req create_start vm_up seeded rows model_done vms_after tool_out ok
  out=$(chat "$SESS_C" --timeout-secs 600 --prompt "Use the bash tool to run: ls app; uname -a   Then tell me in one sentence what you see.")
  note_calls "$SESS_C" "$out"
  tool=$(jq -c '[.tools[]|select(.name=="bash")][0] // null' <<<"$out")
  t_tool=$(jq -r '.t_ms // 0' <<<"$tool")
  t_end=$(jq -r '.end_ms // 0' <<<"$tool")
  t_sent=$(jq -r .sent_at_ms <<<"$out")
  tool_out=$(jq -r '.output // ""' <<<"$tool")
  # The front's own log is the record of when it was first asked and when the VM existed.
  for _ in $(seq 1 20); do
    fl=$(cw front/ "$((ready - 60000))" | jq -c --arg s "$SESS_C" '[.[]|(.m|fromjson? // empty)|select(.src=="front" and .detail.session==$s)]')
    [ "$(jq '[.[]|select(.event=="vm_seeded")]|length' <<<"$fl")" -ge 1 ] && break
    sleep 3
  done
  _fl() { jq -r --arg e "$1" '[.[]|select(.event==$e)][0].ts_ms // 0' <<<"$fl"; }
  first_req=$(_fl first_request)
  create_start=$(_fl vm_create_start)
  vm_up=$(_fl vm_ready)
  seeded=$(_fl vm_seeded)
  vms_after=$(vm_count_for "$SESS_C" || echo -1)
  # Now that it exists, the demo can reach it too (to run the suite itself, and to start the browser).
  if vm_adopt "$SESS_C"; then
    jq --argjson c "$create_start" '.created_ms = $c' "$VMDIR/$SESS_C.json" > "$VMDIR/$SESS_C.json.tmp" && mv "$VMDIR/$SESS_C.json.tmp" "$VMDIR/$SESS_C.json"
  fi
  rows=$(uc_rows_since "$t_sent" 1)
  model_done=$(jq '[.[]|((.start_unix_secs*1000) + .latency_ms)]|min // 0' <<<"$rows")
  echo "  session ready at $ready; VMs for it at open: ${VM_AT_OPEN:-?}"
  echo "  first model call answered ~$model_done; the model's bash call started $t_tool"
  echo "  the front was first asked at $first_req (+$((first_req - ready)) ms after open), VM create began $create_start, VM serving $vm_up (+$((vm_up - create_start)) ms), seeded $seeded (+$((seeded - vm_up)) ms)"
  echo "  the first tool result arrived at $t_end: $((t_end - create_start)) ms after the VM create began; E2B VMs for it now: $vms_after"
  ok=0
  if [ "${VM_AT_OPEN:-1}" = 0 ] && [ "$create_start" -gt "$ready" ] && [ "$first_req" -ge "$((t_tool - 1500))" ] && [ "$create_start" -ge "$((model_done - 1500))" ] \
    && [ "$vms_after" = 1 ] && [ "$t_end" -gt "$create_start" ] && [[ $tool_out == *Linux* && $tool_out == *server.py* ]]; then ok=1; fi
  if [ "$ok" = 1 ]; then
    result 1 "no VM existed when the session opened; the first model call came back, the model called bash, and only then did the front create the VM ($((create_start - ready)) ms after the session was ready), seeded from the bundle; the first tool result came $((t_end - create_start)) ms after the create began" \
      "$(jq -nc --argjson ready "$ready" --argjson fr "$first_req" --argjson cs "$create_start" --argjson up "$vm_up" --argjson sd "$seeded" --argjson te "$t_end" --argjson tt "$t_tool" --argjson md "$model_done" \
        '{vms_at_open:0, session_ready_ms:$ready, first_model_call_answered_ms:$md, tool_call_started_ms:$tt, front_first_request_ms:$fr, vm_create_started_ms:$cs, vm_serving_ms:$up, vm_seeded_ms:$sd, first_tool_result_ms:$te,
          open_to_vm_create_ms:($cs-$ready), vm_provision_ms:($up-$cs), seed_from_bundle_ms:($sd-$up), vm_create_to_first_tool_result_ms:($te-$cs)}')" \
      "$(jq -nc --argjson a "$((create_start - ready))" '{label:"session open to VM create",value:$a,unit:"ms"}')"
  else
    result 0 "vms_at_open=${VM_AT_OPEN:-?} ready=$ready first_req=$first_req create=$create_start tool=$t_tool model_done=$model_done vms_after=$vms_after tool_end=$t_end out=${tool_out:0:120}"
  fi
}

# The browser's version: the model is offered the browser tools at open from the list saved in the bundle,
# and the VM (with the browser stack in it) is made by the front when the first of them is called.
_vm_created_on_first_browser_tool() {
  local ready=$UC_READY_MS out tool t_tool t_end t_sent fl first_req create_start vm_up seeded browser_up model_done vms_after nav ok
  out=$(chat "$SESS_B" --timeout-secs 600 --prompt "Open http://localhost:8000/ in the browser (find the browser tools with tool_search). Tell me the page title and the text of the line that starts with 'stamp', quoting them from the browser's snapshot.")
  note_calls "$SESS_B" "$out"
  tool=$(jq -c '[.tools[]|select(.name|test("browser_navigate"))][0] // null' <<<"$out")
  t_tool=$(jq -r '.t_ms // 0' <<<"$tool")
  t_end=$(jq -r '.end_ms // 0' <<<"$tool")
  t_sent=$(jq -r .sent_at_ms <<<"$out")
  nav=$(jq -r '.output // ""' <<<"$tool")
  for _ in $(seq 1 20); do
    fl=$(cw front/ "$((ready - 60000))" | jq -c --arg s "$SESS_B" '[.[]|(.m|fromjson? // empty)|select(.src=="front" and .detail.session==$s)]')
    [ "$(jq '[.[]|select(.event=="browser_ready")]|length' <<<"$fl")" -ge 1 ] && break
    sleep 3
  done
  _fl() { jq -r --arg e "$1" '[.[]|select(.event==$e)][0].ts_ms // 0' <<<"$fl"; }
  first_req=$(_fl first_request)
  create_start=$(_fl vm_create_start)
  vm_up=$(_fl vm_ready)
  seeded=$(_fl vm_seeded)
  browser_up=$(_fl browser_ready)
  vms_after=$(vm_count_for "$SESS_B" || echo -1)
  # Now that it exists, the demo can reach it too (the later checks pause it and read its processes).
  if vm_adopt "$SESS_B"; then
    jq --argjson c "$create_start" '.created_ms = $c' "$VMDIR/$SESS_B.json" > "$VMDIR/$SESS_B.json.tmp" && mv "$VMDIR/$SESS_B.json.tmp" "$VMDIR/$SESS_B.json"
  fi
  model_done=$(jq '[.[]|((.start_unix_secs*1000) + .latency_ms)]|min // 0' <<<"$(uc_rows_since "$t_sent" 1)")
  echo "  session ready at $ready; VMs for it at open: ${VM_AT_OPEN:-?}"
  echo "  the model's browser_navigate call started $t_tool; the front was first asked at $first_req (+$((first_req - ready)) ms after open)"
  echo "  VM create began $create_start, VM serving $vm_up (+$((vm_up - create_start)) ms), seeded $seeded, browser stack and bridge up $browser_up (+$((browser_up - seeded)) ms)"
  echo "  the first browser result arrived at $t_end: $((t_end - t_tool)) ms after the call started; E2B VMs for it now: $vms_after"
  ok=0
  if [ "${VM_AT_OPEN:-1}" = 0 ] && [ "$create_start" -gt "$ready" ] && [ "$first_req" -ge "$((t_tool - 1500))" ] && [ "$create_start" -ge "$((model_done - 1500))" ] \
    && [ "$vms_after" = 1 ] && [ "$t_end" -gt "$create_start" ] && [[ $nav == *"$PAGE_STAMP"* ]]; then ok=1; fi
  if [ "$ok" = 1 ]; then
    result 1 "no VM existed when the session opened, yet the model had the browser tools (from the list saved in the bundle); only when it called browser_navigate did the front make the VM ($((create_start - ready)) ms after the session was ready), seed it, install the browser stack and forward the call; the first browser result came $((t_end - t_tool)) ms after the call started" \
      "$(jq -nc --argjson ready "$ready" --argjson fr "$first_req" --argjson cs "$create_start" --argjson up "$vm_up" --argjson sd "$seeded" --argjson bu "$browser_up" --argjson te "$t_end" --argjson tt "$t_tool" --argjson md "$model_done" \
        '{vms_at_open:0, session_ready_ms:$ready, first_model_call_answered_ms:$md, tool_call_started_ms:$tt, front_first_request_ms:$fr, vm_create_started_ms:$cs, vm_serving_ms:$up, vm_seeded_ms:$sd, browser_ready_ms:$bu, first_tool_result_ms:$te,
          open_to_vm_create_ms:($cs-$ready), vm_provision_ms:($up-$cs), seed_from_bundle_ms:($sd-$up), browser_stack_ms:($bu-$sd), first_tool_call_to_result_ms:($te-$tt)}')" \
      "$(jq -nc --argjson a "$((t_end - t_tool))" '{label:"first browser call to its result (VM made inside it)",value:$a,unit:"ms"}')"
  else
    result 0 "vms_at_open=${VM_AT_OPEN:-?} ready=$ready first_req=$first_req create=$create_start tool=$t_tool model_done=$model_done vms_after=$vms_after tool_end=$t_end nav=${nav:0:160}"
  fi
}

check_skill_used() {
  local out reads runs script_out
  out=$(chat "$SESS_C" --timeout-secs 400 --prompt "The app's tests are failing. Use the run-tests skill to run them, then fix the bug in app/server.py (never edit the tests), then run them again with the skill. Tell me the final RUN-TESTS-SKILL line.")
  note_calls "$SESS_C" "$out"
  reads=$(jq -c '[.tools[]|select((.name=="read" or .name=="bash") and ((.input|tostring)|contains("run-tests/SKILL.md")))]' <<<"$out")
  runs=$(jq -c '[.tools[]|select(.name=="bash" and ((.input|tostring)|contains("run-tests.sh")))]' <<<"$out")
  script_out=$(jq -r '[.[]|.output // ""]|join("\n")' <<<"$runs")
  echo "  read the skill: $(jq length <<<"$reads") time(s); ran its script: $(jq length <<<"$runs") time(s)"
  echo "  the script's last line: $(grep -o 'RUN-TESTS-SKILL:.*' <<<"$script_out" | tail -1)"
  if [ "$(jq length <<<"$reads")" -ge 1 ] && [ "$(jq length <<<"$runs")" -ge 1 ] && [[ $script_out == *"RUN-TESTS-SKILL:"* ]]; then
    result 1 "the model read .agents/skills/run-tests/SKILL.md, then ran its script in the sandbox $(jq length <<<"$runs") time(s); the script's own marker line came back: $(grep -o 'RUN-TESTS-SKILL:.*' <<<"$script_out" | tail -1)" \
      "$(jq -nc --argjson r "$(jq length <<<"$reads")" --argjson n "$(jq length <<<"$runs")" --arg first "$(grep -o 'RUN-TESTS-SKILL:.*' <<<"$script_out" | head -1)" --arg last "$(grep -o 'RUN-TESTS-SKILL:.*' <<<"$script_out" | tail -1)" --argjson ms "$(jq '.total_ms' <<<"$out")" \
        '{skill_reads:$r, script_runs:$n, first_verdict:$first, last_verdict:$last, turn_ms:$ms}')" \
      "$(jq -nc --argjson n "$(jq length <<<"$runs")" '{label:"script runs in the sandbox",value:$n,unit:""}')"
  else
    result 0 "reads=$(jq length <<<"$reads") runs=$(jq length <<<"$runs") out=${script_out:0:200}"
  fi
}

check_fix_verified() {
  local after sum edited
  # Not the model's word: the demo runs the suite itself, in the VM, and checks the tests were not edited.
  after=$(vm_in "$SESS_C" "cd $VM_HOME/workspace/app && python3 -m unittest 2>&1 | tail -n 3; echo exit=\${PIPESTATUS[0]}")
  sum=$(vm_in "$SESS_C" "md5sum $VM_HOME/workspace/app/test_server.py" | cut -d' ' -f1)
  edited=$(vm_in "$SESS_C" "grep -n 'title' $VM_HOME/workspace/app/server.py | head -3")
  echo "  before: $(tr '\n' ' ' <<<"$TESTS_BEFORE")"
  echo "  after:  $(tr '\n' ' ' <<<"$after")"
  if [[ $TESTS_BEFORE == *"exit=1"* ]] && [[ $after == *"exit=0"* && $after == *OK* ]] && [ "$sum" = "$TESTS_ORIG_SUM" ]; then
    result 1 "the suite failed before the agent started ($(grep -o 'FAILED.*' <<<"$TESTS_BEFORE")) and passes now, run by the demo in the VM; the test file is byte for byte the original" \
      "$(jq -nc --arg b "$TESTS_BEFORE" --arg a "$after" --arg e "$edited" '{before:$b, after:$a, fixed_line:$e, tests_unmodified:true}')" \
      '{"label":"tests","value":"fail to pass","unit":"checked by the demo"}'
  else
    result 0 "before=$TESTS_BEFORE after=$after md5=$sum (orig $TESTS_ORIG_SUM)"
  fi
}

check_subagent_from_folder() {
  local out sub text input
  out=$(chat "$SESS_C" --timeout-secs 300 --prompt "Ask the reviewer agent to review your change to app/server.py, and report its REVIEWER-SIGNOFF line verbatim.")
  note_calls "$SESS_C" "$out"
  sub=$(jq -c '[.tools[]|select(.name=="subagent")][0] // null' <<<"$out")
  input=$(jq -r '.input|tostring' <<<"$sub")
  text=$(jq -r '.output // ""' <<<"$sub")
  echo "  subagent call: ${input:0:140}"
  echo "  reviewer replied: $(grep -o 'REVIEWER-SIGNOFF:.*' <<<"$text" | tail -1)"
  if [ "$sub" != null ] && [[ $input == *reviewer* ]] && [[ $text == *"REVIEWER-SIGNOFF:"* ]]; then
    result 1 "the model called the reviewer subagent that only .agents/agents/reviewer.md defines, and its sign-off line (a marker in that file's body) came back: $(grep -o 'REVIEWER-SIGNOFF:.*' <<<"$text" | tail -1)" \
      "$(jq -nc --arg i "${input:0:300}" --arg s "$(grep -o 'REVIEWER-SIGNOFF:.*' <<<"$text" | tail -1)" --argjson ms "$(uc_tool_ms "$out" '^subagent$')" '{subagent_input:$i, signoff:$s, subagent_ms:$ms}')" \
      "$(jq -nc --argjson ms "$(uc_tool_ms "$out" '^subagent$')" '{label:"reviewer subagent",value:$ms,unit:"ms"}')"
  else
    result 0 "subagent=$sub"
  fi
}

check_app_as_background_job() {
  local out1 bg1 curl1 curl2 out2 m notice woke cb
  out1=$(chat "$SESS_C" --timeout-secs 300 --prompt "Start the app in the background: use bash with run_in_background=true to run exactly: cd app && python3 server.py   Then reply with just the word started.")
  note_calls "$SESS_C" "$out1"
  bg1=$(jq -c '[.tools[]|select(.name=="bash" and ((.input|tostring)|contains("server.py")))][0] // null' <<<"$out1")
  sleep 2
  curl1=$(vm_in "$SESS_C" 'curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:8000/; echo; curl -s -m 5 localhost:8000/ | grep -o "<h1>[^<]*</h1>"')
  echo "  the app answers on localhost:8000 inside the VM: $(tr '\n' ' ' <<<"$curl1")"
  m="job-$RANDOM$RANDOM"
  out2=$(chat "$SESS_C" --timeout-secs 300 --wait-wake-secs 150 --prompt "Use the bash tool with run_in_background=true to run exactly: sleep 15; echo finished-$m . Then reply with just the word started and end your turn. Never call bash_job: do not poll, wait on or kill the job; you will be told when it ends, and then you must reply with the job's exact output line and its exit code.")
  note_calls "$SESS_C" "$out2"
  notice=$(jq -r '[.background[]|select(.reason=="finished")][0].text // ""' <<<"$out2")
  woke=$(jq -r '.woke_at_ms // empty' <<<"$out2")
  curl2=$(vm_in "$SESS_C" 'curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:8000/')
  echo "  the short job's end was pushed to the agent: ${notice:0:140}; the app still answers: $curl2"
  local bg_in
  bg_in=$(jq -r '.input|tostring' <<<"$bg1")
  if [ "$bg1" != null ] && [[ $bg_in == *run_in_background* ]] && [[ $curl1 == 200* && $curl1 == *"Hello, World!"* ]] \
    && [ -n "$woke" ] && [[ $notice == *"finished-$m"* && $notice == *"exited with code 0"* ]] && [ "$curl2" = 200 ]; then
    result 1 "the app was started with run_in_background and answered 200 inside the VM; a 15 s job started next ended and the agent was told (exit 0, its real output) while the app kept running" \
      "$(jq -nc --arg c1 "$curl1" --arg c2 "$curl2" --arg n "$notice" --argjson wake "$woke" --argjson sent "$(jq .sent_at_ms <<<"$out2")" --argjson ms "$(jq .total_ms <<<"$out2")" \
        '{app_answers:$c1, app_after_the_short_job:$c2, job_end_notice:$n, job_end_reported_after_ms:($wake-$sent), first_turn_ms:$ms}')" \
      "$(jq -nc --argjson w "$(( woke - $(jq .sent_at_ms <<<"$out2") ))" '{label:"short job started to its end reported",value:($w/1000|floor),unit:"s"}')"
  else
    result 0 "bg=$bg_in curl1=$curl1 woke=$woke notice=${notice:0:200} curl2=$curl2"
  fi
}

check_browser_sees_localhost() {
  local out nav snap text ms rw id tok t0
  # The browser connector names the VM's ports, which exist only now: a second session, in the same VM
  # (the front keys on the first session's name), carries a grant that also names the bridge.
  rw=$(rustwright_bin)
  id=$(vm_field "$SESS_C" id)
  tok=$(vm_field "$SESS_C" envd_token)
  t0=$(now_ms)
  if vm_browser_up "$id" "$tok" "$rw" && vm_bridge_start "$id" "$tok"; then
    echo "  browser stack installed in $(( ($(now_ms) - t0) / 1000 )) s"
  else
    result 0 "the browser stack did not come up in the VM"
    return
  fi
  GRANT_FN=_grant_coding_browser uc_open "$SESS_C2"
  out=$(GRANT_FN=_grant_coding_browser chat "$SESS_C2" --timeout-secs 400 --prompt "Use the browser tools (find them with tool_search) to open http://localhost:8000/ and tell me the heading and the build line of the page, quoting them from what the browser shows.")
  note_calls "$SESS_C2" "$out"
  nav=$(jq -r '[.tools[]|select(.name|test("browser_navigate"))][0].output // ""' <<<"$out")
  ms=$(uc_tool_ms "$out" 'browser_navigate')
  text=$(jq -r .text <<<"$out")
  echo "  the browser's snapshot: $(tr '\n' ' ' <<<"${nav:0:260}")"
  if [[ $nav == *"Hello, World!"* && $nav == *"$BUILD_STAMP"* ]] && [[ $text == *"$BUILD_STAMP"* ]]; then
    result 1 "the browser inside the VM opened localhost:8000, the page the agent's own app serves, and the snapshot showed the fixed heading and this run's build line ($BUILD_STAMP)" \
      "$(jq -nc --arg s "${nav:0:600}" --argjson ms "$ms" '{navigate_ms:$ms, snapshot:$s}')" \
      "$(jq -nc --argjson ms "$ms" '{label:"navigate + snapshot",value:$ms,unit:"ms"}')"
  else
    result 0 "snapshot=${nav:0:300} text=${text:0:200}"
  fi
}

# ============================================================================================================
# browser
# ============================================================================================================

check_browser_reads_page() {
  local out nav ms text
  out=$(chat "$SESS_B" --timeout-secs 400 --prompt "Open http://localhost:8000/ in the browser. Tell me the page title, the heading, and the text of the line that starts with 'stamp', quoting them from the browser's snapshot.")
  note_calls "$SESS_B" "$out"
  nav=$(jq -r '[.tools[]|select(.name|test("browser_navigate"))][0].output // ""' <<<"$out")
  ms=$(uc_tool_ms "$out" 'browser_navigate')
  text=$(jq -r .text <<<"$out")
  echo "  snapshot: $(tr '\n' ' ' <<<"${nav:0:300}")"
  if [[ $nav == *"Field Notes"* && $nav == *"$PAGE_STAMP"* && $nav == *heading* ]] && [[ $text == *"$PAGE_STAMP"* ]]; then
    local empties
    empties=$(uc_empty_args)
    result 1 "navigate to the in-VM page plus its accessibility snapshot took $ms ms; the snapshot held the heading and this run's stamp ($PAGE_STAMP), and the reply quoted it. Browser tool calls with an empty-string optional argument so far: $(jq length <<<"$empties")" \
      "$(jq -nc --arg s "${nav:0:700}" --argjson ms "$ms" --argjson tb "${#nav}" --argjson turn "$(jq .total_ms <<<"$out")" --argjson e "$empties" '{navigate_ms:$ms, snapshot_bytes:$tb, snapshot:$s, turn_ms:$turn, empty_string_args:{count:($e|length), calls:$e}}')" \
      "$(jq -nc --argjson ms "$ms" '{label:"navigate + snapshot",value:$ms,unit:"ms"}')"
  else
    result 0 "snapshot=${nav:0:300} reply=${text:0:200}"
  fi
}

check_no_held_connection() {
  local id tok holder samples n nonzero
  id=$(vm_field "$SESS_B" id)
  tok=$(vm_field "$SESS_B" envd_token)
  # The session is attached for the whole gap (so it is not simply gone), and idle: no prompt, no call.
  NOLOG=1 chat "$SESS_B" --wait-wake-secs 40 --timeout-secs 60 >/dev/null 2>&1 &
  holder=$!
  sleep 3
  tl demo idle_gap_started null null
  samples=$(vm_run "$id" "$tok" 'for i in 1 2 3 4 5 6; do printf "%s " $(ss -Htn state established "( sport = :'"$BRIDGE_PORT"' )" | wc -l); sleep 5; done')
  tl demo idle_gap_ended null null
  wait "$holder" 2>/dev/null
  n=$(wc -w <<<"$samples")
  nonzero=$(tr ' ' '\n' <<<"$samples" | grep -c '^[1-9]' || true)
  echo "  established connections to the bridge port, 6 samples over 30 s of idle: $samples"
  if [ "$n" = 6 ] && [ "$nonzero" = 0 ]; then
    result 1 "0 established connections to the bridge port in all 6 samples across a 30 s idle gap, with the session attached and the browser holding a page" \
      "$(jq -nc --arg s "$samples" '{samples:($s|split(" ")|map(select(.!="")|tonumber)), gap_s:30, port:'"$BRIDGE_PORT"'}')" \
      '{"label":"connections held while idle","value":"0","unit":""}'
  else
    result 0 "samples: $samples"
  fi
}

check_state_survives_pause() {
  local id m set_out pause_ms resume_ms t_pause t_resume out2 snap first_ms navs set_snap
  id=$(vm_field "$SESS_B" id)
  m="M$RANDOM$RANDOM"
  # State that lives only in the page: what is typed into its field, and the line the page's own
  # script sets from it. (Typed by the model through the browser's own tools, as a person would.)
  set_out=$(chat "$SESS_B" --timeout-secs 300 --prompt "On the page that is open, type exactly $m into the 'Your note' textbox: take a browser_snapshot to get its ref, then use browser_type with that ref and the text. Then take another browser_snapshot, use browser_get_text with the selector #echo, and tell me what it says.")
  note_calls "$SESS_B" "$set_out"
  set_snap=$(jq -r '[.tools[]|select(.name|test("^mcp__browser__"))|.output // ""]|join("\n")' <<<"$set_out")
  echo "  before the pause the page shows: $(grep -o "echo:$m" <<<"$set_snap" | tail -1) and a field holding $(grep -o "value=\"$m\"" <<<"$set_snap" | tail -1)"
  [[ $set_snap == *"echo:$m"* ]] || { result 0 "the note was not typed: $(head -c 300 <<<"$set_snap")"; return; }
  # How many MCP sessions the bridge has been asked to open (each is a new stdio child, a new browser).
  local init_before init_after
  init_before=$(vm_in "$SESS_B" 'grep -c "\"method\":\"initialize\"" /tmp/bridge.log' | tr -dc 0-9)
  tl demo vm_pause_requested null null
  pause_ms=$(vm_pause "$id") || { result 0 "E2B would not pause the sandbox"; return; }
  t_pause=$(now_ms)
  echo "  paused ($pause_ms ms); waiting 30 s"
  sleep 30
  resume_ms=$(vm_resume "$id") || { result 0 "E2B would not resume the sandbox"; return; }
  t_resume=$(now_ms)
  tl demo vm_resumed null "$(jq -nc --argjson p "$(( (t_resume - t_pause) / 1000 ))" '{paused_s:$p}')"
  echo "  resumed after $(( (t_resume - t_pause) / 1000 )) s ($resume_ms ms)"
  out2=$(chat "$SESS_B" --timeout-secs 300 --prompt "Do not call browser_navigate or browser_reload. Take a browser_snapshot of the page that is open (the whole page, no target) and use browser_get_text with the selector #echo. Tell me what #echo says and what the 'Your note' textbox contains.")
  note_calls "$SESS_B" "$out2"
  snap=$(jq -r '[.tools[]|select(.name|test("^mcp__browser__"))|.output // ""]|join("\n")' <<<"$out2")
  first_ms=$(uc_tool_ms "$out2" 'mcp__browser__browser_(snapshot|get_text)')
  navs=$(uc_tools "$out2" 'browser_navigate' | jq length)
  init_after=$(vm_in "$SESS_B" 'grep -c "\"method\":\"initialize\"" /tmp/bridge.log' | tr -dc 0-9)
  echo "  MCP sessions opened on the bridge: $init_before before the pause, $init_after after the resume"
  echo "  after the resume, without navigating ($navs navigate calls): $(grep -o "echo:$m" <<<"$snap" | head -1); first browser call took $first_ms ms"
  if [[ $snap == *"echo:$m"* ]] && [ "$navs" = 0 ] && [ "$init_before" = "$init_after" ]; then
    result 1 "after a $(( (t_resume - t_pause) / 1000 )) s pause the page still showed the line its own script had written when the text was typed (echo:$m), read without navigating; resume took $resume_ms ms and the first browser call after it $first_ms ms" \
      "$(jq -nc --arg m "$m" --argjson pause "$pause_ms" --argjson resume "$resume_ms" --argjson first "$first_ms" --argjson paused "$(( (t_resume - t_pause) / 1000 ))" --argjson turn "$(jq .total_ms <<<"$out2")" --argjson ib "${init_before:-0}" --argjson ia "${init_after:-0}" \
        '{typed:$m, echo_after_resume:("echo:"+$m), pause_api_ms:$pause, paused_s:$paused, resume_api_ms:$resume, first_call_after_resume_ms:$first, first_turn_after_resume_ms:$turn, navigations_after_resume:0, mcp_sessions_opened_before:$ib, mcp_sessions_opened_after:$ia, same_mcp_session:true}')" \
      "$(jq -nc --argjson r "$resume_ms" '{label:"resume",value:$r,unit:"ms"}')"
  else
    result 0 "after the resume: navigations $navs, MCP sessions opened $init_before -> $init_after (a higher number means the agent re-dialed the bridge, which starts a new browser); $(head -c 200 <<<"$snap")"
  fi
}

# A sandbox connector is never idle-reaped. The agent's window for an unused MCP server is 120 s
# (BEYOND_AI_AGENT_MCP_IDLE_SECS, unset on this replica, so the default); the session is held attached
# and idle for longer than that, and the page must still hold what was typed, through the same MCP
# session. The window is not lowered for the run.
IDLE_REAP_HOLD_S=${IDLE_REAP_HOLD_S:-150}
check_state_survives_idle_reap() {
  local m set_out set_snap init_before init_after out2 snap navs holder empties
  m="R$RANDOM$RANDOM"
  set_out=$(chat "$SESS_B" --timeout-secs 300 --prompt "On the page that is open, replace the contents of the 'Your note' textbox with exactly $m: take a browser_snapshot to get its ref, then use browser_type with that ref and the text. Then use browser_get_text with the selector #echo and tell me what it says.")
  note_calls "$SESS_B" "$set_out"
  set_snap=$(jq -r '[.tools[]|select(.name|test("^mcp__browser__"))|.output // ""]|join("\n")' <<<"$set_out")
  [[ $set_snap == *"echo:$m"* ]] || { result 0 "the note was not typed: $(head -c 300 <<<"$set_snap")"; return; }
  init_before=$(vm_in "$SESS_B" 'grep -c "\"method\":\"initialize\"" /tmp/bridge.log' | tr -dc 0-9)
  tl demo idle_reap_hold_started null null
  # attached the whole time, so the session is not simply gone; no prompt, no call
  NOLOG=1 chat "$SESS_B" --wait-wake-secs "$IDLE_REAP_HOLD_S" --timeout-secs "$((IDLE_REAP_HOLD_S + 20))" >/dev/null 2>&1 &
  holder=$!
  echo "  holding the session idle for $IDLE_REAP_HOLD_S s (the agent's MCP idle window is 120 s)"
  wait "$holder" 2>/dev/null
  tl demo idle_reap_hold_ended null null
  out2=$(chat "$SESS_B" --timeout-secs 300 --prompt "Do not call browser_navigate or browser_reload. Use browser_get_text with the selector #echo on the page that is open, and tell me what it says.")
  note_calls "$SESS_B" "$out2"
  snap=$(jq -r '[.tools[]|select(.name|test("^mcp__browser__"))|.output // ""]|join("\n")' <<<"$out2")
  navs=$(uc_tools "$out2" 'browser_navigate' | jq length)
  init_after=$(vm_in "$SESS_B" 'grep -c "\"method\":\"initialize\"" /tmp/bridge.log' | tr -dc 0-9)
  empties=$(uc_empty_args)
  echo "  MCP sessions opened on the bridge: $init_before before the hold, $init_after after"
  echo "  after the hold, without navigating ($navs navigate calls): $(grep -o "echo:$m" <<<"$snap" | head -1)"
  echo "  browser tool calls with an empty-string argument in this run: $(jq length <<<"$empties")"
  if [[ $snap == *"echo:$m"* ]] && [ "$navs" = 0 ] && [ -n "$init_before" ] && [ "$init_before" = "$init_after" ]; then
    result 1 "after $IDLE_REAP_HOLD_S s idle (the agent's MCP idle window is 120 s) the page still showed what had been typed (echo:$m), read without navigating, and the bridge saw no new initialize ($init_before before, $init_after after): the connector was not reaped. Browser tool calls carrying an empty-string optional argument in this run: $(jq length <<<"$empties")" \
      "$(jq -nc --arg m "$m" --argjson hold "$IDLE_REAP_HOLD_S" --argjson ib "$init_before" --argjson ia "$init_after" --argjson e "$empties" \
        '{typed:$m, idle_hold_s:$hold, agent_mcp_idle_window_s:120, window_lowered:false, echo_after_hold:("echo:"+$m), navigations_after_hold:0, mcp_sessions_opened_before:$ib, mcp_sessions_opened_after:$ia, same_mcp_session:true, empty_string_args:{count:($e|length), calls:$e}}')" \
      "$(jq -nc --argjson h "$IDLE_REAP_HOLD_S" '{label:"idle, page state kept",value:$h,unit:"s"}')"
  else
    result 0 "after the hold: navigations $navs, MCP sessions opened $init_before -> $init_after; $(head -c 200 <<<"$snap")"
  fi
}

check_idle_cost() {
  local id tok rep
  id=$(vm_field "$SESS_B" id)
  tok=$(vm_field "$SESS_B" envd_token)
  # the session stays attached while it is measured, so the agent does not reap it (a reap ends the
  # browser, which is the point of reaping, not an idle cost)
  NOLOG=1 chat "$SESS_B" --wait-wake-secs 25 --timeout-secs 40 >/dev/null 2>&1 &
  local holder=$!
  sleep 8
  rep=$(vm_run "$id" "$tok" 'for name in rustwright-mcp node chrome-headless; do
  n=0; rss=0; pss=0; anon=0
  for d in /proc/[0-9]*; do
    [ "$(cat $d/comm 2>/dev/null)" = "$name" ] || continue
    n=$((n + 1))
    set -- $(awk "/^Rss:/{r=\$2} /^Pss:/{p=\$2} /^Anonymous:/{a=\$2} END{print r, p, a}" $d/smaps_rollup 2>/dev/null)
    rss=$((rss + ${1:-0})); pss=$((pss + ${2:-0})); anon=$((anon + ${3:-0}))
  done
  echo "$name $n $rss $pss $anon"
done
echo "vm $(free -m | awk "/^Mem:/{print \$2, \$3}")"')
  wait "$holder" 2>/dev/null
  printf '%s\n' "$rep" | sed 's/^/  /'
  local procs
  procs=$(jq -Rn '[inputs | split(" ") | select(.[0] != "vm") | {process:.[0], processes:(.[1]|tonumber), rss_mb:((.[2]|tonumber)/1024*10|round/10), pss_mb:((.[3]|tonumber)/1024*10|round/10), anon_mb:((.[4]|tonumber)/1024*10|round/10)}]' <<<"$rep")
  local vm_total vm_used
  vm_total=$(awk '/^vm /{print $2}' <<<"$rep")
  vm_used=$(awk '/^vm /{print $3}' <<<"$rep")
  if [ "$(jq 'length' <<<"$procs")" = 3 ] && [ "$(jq '[.[]|select(.processes>0)]|length' <<<"$procs")" = 3 ]; then
    local anon_total
    anon_total=$(jq '[.[].anon_mb]|add*10|round/10' <<<"$procs")
    result 1 "idle, with a page open: rustwright-mcp $(jq -r '.[]|select(.process=="rustwright-mcp")|.anon_mb' <<<"$procs") MB anon, the bridge (node) $(jq -r '.[]|select(.process=="node")|.anon_mb' <<<"$procs") MB, Chromium ($(jq -r '.[]|select(.process=="chrome-headless")|.processes' <<<"$procs") processes) $(jq -r '.[]|select(.process=="chrome-headless")|.anon_mb' <<<"$procs") MB; $anon_total MB anonymous in all, of a $vm_total MB VM" \
      "$(jq -nc --argjson p "$procs" --argjson t "$anon_total" --argjson vt "$vm_total" --argjson vu "$vm_used" '{processes:$p, anon_total_mb:$t, vm_memory_mb:$vt, vm_used_mb:$vu, note:"measured, not compared: no threshold is claimed"}')" \
      "$(jq -nc --argjson t "$anon_total" '{label:"idle anon, browser stack",value:$t,unit:"MB"}')"
  else
    result 0 "could not find all three processes: $rep"
  fi
}
