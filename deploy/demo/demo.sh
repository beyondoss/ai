#!/usr/bin/env bash
# A scripted conversation with the agent running on AWS, as a list of named checks.
#
#     ./deploy/demo/demo.sh         the core checks: a real model turn, a command in the sandbox, a
#                                   background job whose end is pushed to the agent (exec v1.4), a
#                                   forged callback, sessions sealed on EFS, and one billing row per call
#     ./deploy/demo/demo.sh kill    start a background job, SIGKILL the agent, watch ECS replace it, and
#                                   see the same session, and the job's end, come back
#     ./deploy/demo/demo.sh all     both, in one run record
#
# Every run writes results/<UTC timestamp>/run.json (the record) and report.html (one self-contained
# page rendered from it, replaying the run). Neither ever holds a key or a grant.
#
# ## Adding a check
#
#  1. Write `check_<name_with_underscores>() { …; result PASS|FAIL "one line" '{"measurement": 1}'; }`.
#     Talk to the agent with `chat <session> --prompt …` (see the existing checks); it prints the
#     step's JSON summary on stdout. `result` takes 1 for pass, 0 for fail.
#  2. Give it a title in CHECK_TITLE, and add its name to the list(s) below.
#
# That is all: the runner times it, records it, and the report draws it.
set -uo pipefail
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# ---- the checks, in one place --------------------------------------------------------------------
CHECKS_core=(grant-refused real-model-turn sandbox-exec background-job-notify notify-forged-refused sealed-at-rest billing-rows)
CHECKS_kill=(kill-and-recover billing-rows)
CHECKS_all=(grant-refused real-model-turn sandbox-exec background-job-notify notify-forged-refused kill-and-recover sealed-at-rest billing-rows)
declare -A CHECK_TITLE=(
  [grant-refused]="A tampered session grant is refused at connect"
  [real-model-turn]="A real model answers, through the real gateway"
  [sandbox-exec]="The model runs a command in the sandbox"
  [background-job-notify]="A background job ends and the agent is told (exec v1.4)"
  [notify-forged-refused]="A forged job_ended is refused and changes nothing"
  [kill-and-recover]="The agent is SIGKILLed mid-job; the session survives"
  [sealed-at-rest]="Session bytes on EFS are ciphertext"
  [billing-rows]="Every model call has exactly one ai.usage row"
  [teardown-zero]="Teardown leaves nothing behind"
)

MODE=${1:-core}
case "$MODE" in
  core | kill | all) ;;
  *) echo "usage: demo.sh [core|kill|all]" >&2; exit 2 ;;
esac
declare -n CHECKS="CHECKS_$MODE"

for tool in aws jq curl cargo; do
  command -v "$tool" >/dev/null || { echo "demo: $tool is required" >&2; exit 2; }
done
now_ms() { date +%s%3N; }
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=${DEMO_RESULTS:-$HERE/results}/$RUN_ID
mkdir -p "$RUN_DIR"
WORK=$(mktemp -d)
chmod 700 "$WORK"
trap 'rm -rf "$WORK"' EXIT
START_MS=$(now_ms)
TENANT=demo-tenant
SID=$(date -u +%m%d%H%M%S)          # sessions: demo.r<SID><letter>, one per scenario
EVENTS=$WORK/events.jsonl           # every frame the client saw
TIMELINE=$WORK/timeline.ndjson      # one line per key event, any source
CHECKS_OUT=$WORK/checks.ndjson
CALLS=$WORK/calls.ndjson            # model calls the agent says it made, per session
touch "$EVENTS" "$TIMELINE" "$CHECKS_OUT" "$CALLS"

tl() { # src event [path-json] [detail-json] [t_ms]
  jq -nc --arg s "$1" --arg e "$2" --argjson p "${3:-null}" --argjson d "${4:-null}" --argjson t "${5:-$(now_ms)}" \
    '{t_ms:$t,src:$s,event:$e,path:$p,detail:$d}' >> "$TIMELINE"
}
rule() { printf '\n%s\n' "────────────────────────────────────────────────────────────"; }

# ---- find the deployment ---------------------------------------------------------------------------
say "the deployment ($NAME, $REGION)"
wait_agent() {
  local ip
  for _ in $(seq 1 90); do
    ip=$(agent_ip)
    if [ -n "$ip" ] && curl -fsS -m 3 "http://$ip:$AGENT_PORT/readyz" >/dev/null 2>&1; then
      echo "$ip"
      return 0
    fi
    sleep 2
  done
  return 1
}
IP=$(wait_agent) || {
  echo "demo: the agent is not answering. Run ./deploy/demo/up.sh first; if it is up, your address may have" >&2
  echo "      changed (the agent port is open to one address): rerun up.sh, which adds yours." >&2
  exit 1
}
MODEL=$(q ecs describe-task-definition --task-definition "$P-agent" \
  --query "taskDefinition.containerDefinitions[0].environment[?name=='MODEL'].value|[0]")
PROVIDER=openai
command aws --region "$REGION" ssm get-parameter --name "$SSM/openrouter-key" >/dev/null 2>&1 && PROVIDER=openrouter
echo "  agent    ws://$IP:$AGENT_PORT"
echo "  model    $MODEL (provider $PROVIDER, through the gateway at gateway.$DNS)"
DEMO=$(demo_bin) || exit 1
DEMO_GRANT_SEED=$(ssm_get grant-seed)
DEMO_FLEET_SECRET=$(ssm_get fleet-secret)
DEMO_DEK=$(ssm_get dek)
DEMO_GATEWAY_KEY=$(ssm_get gateway-key)
export DEMO_GRANT_SEED DEMO_FLEET_SECRET DEMO_DEK DEMO_GATEWAY_KEY

grant() { # session [tamper]
  "$DEMO" mint-grant --tenant "$TENANT" --session "$1" --shard demo --workspace /workspace \
    --exec-url "http://sandbox.$DNS:$SANDBOX_PORT/exec" ${2:+--tamper}
}
# One conversation step; the JSON summary on stdout, the live view on stderr.
chat() { # session chat-args...
  local s=$1
  shift
  DEMO_GRANT=$(grant "$s") "$DEMO" chat --host "$IP" --port "$AGENT_PORT" --session "$s" --events "$EVENTS" "$@"
}
note_calls() { # session summary-json: remember how many model calls the agent says the session made
  jq -c --arg s "$1" '{session:$s, assistant_messages: ([.prompt_response.data.assistant_messages, .wake_response.data.assistant_messages]|map(select(.!=null))|max)}' <<<"$2" >> "$CALLS"
}

# ---- the runner --------------------------------------------------------------------------------------
PASS=0 DETAIL="" MEAS='{}' HEADLINE='{}'
result() { # 1|0 "detail" '{measurements}' '{"label":…,"value":…,"unit":…}'
  PASS=$1 DETAIL=$2 MEAS=${3:-'{}'} HEADLINE=${4:-'{}'}
}
run_check() {
  local name=$1 fn t0 t1
  fn=check_${name//-/_}
  rule
  echo "▶ ${CHECK_TITLE[$name]}   [$name]"
  PASS=0 DETAIL="check did not report" MEAS='{}' HEADLINE='{}'
  t0=$(now_ms)
  "$fn"
  t1=$(now_ms)
  jq -nc --arg n "$name" --arg t "${CHECK_TITLE[$name]}" --argjson ok "$PASS" --arg d "$DETAIL" --argjson m "$MEAS" \
    --argjson h "$HEADLINE" --argjson a "$t0" --argjson b "$t1" \
    '{name:$n,title:$t,pass:($ok==1),started_ms:$a,ended_ms:$b,duration_ms:($b-$a),detail:$d,headline:$h,measurements:$m}' >> "$CHECKS_OUT"
  if [ "$PASS" = 1 ]; then echo "✔ $name — $DETAIL"; else echo "✘ $name — $DETAIL"; fi
}

# CloudWatch, filtered: [{t, m}] for a stream prefix since a time.
cw() { # stream-prefix since-ms [filter-pattern]
  command aws --region "$REGION" --output json logs filter-log-events --log-group-name "$LOG_GROUP" \
    ${1:+--log-stream-name-prefix "$1"} --start-time "$2" ${3:+--filter-pattern "$3"} \
    --query 'events[].{t:timestamp,m:message}' 2>/dev/null || echo '[]'
}

# ---- the checks --------------------------------------------------------------------------------------
M1="pelican-$RANDOM$RANDOM"         # the plaintext the user sees, and later looks for on disk
SESS_MAIN=demo.r${SID}a
SESS_JOB=demo.r${SID}b
SESS_KILL=demo.r${SID}k

check_grant_refused() {
  local out
  out=$(DEMO_GRANT=$(grant "$SESS_MAIN" tamper) "$DEMO" chat --host "$IP" --port "$AGENT_PORT" --session "$SESS_MAIN" \
    --command '{"type":"get_state","id":"g"}' --timeout-secs 20)
  echo "  tampered grant -> HTTP $(jq -r .status <<<"$out")"
  tl you grant_refused '["you","agent"]' "$(jq -c '{status}' <<<"$out")"
  if [ "$(jq -r .refused <<<"$out")" = true ] && [ "$(jq -r .status <<<"$out")" = 401 ]; then
    result 1 "refused with 401 before any session opened" "$(jq -c '{http_status:.status}' <<<"$out")" \
      '{"label":"tampered grant","value":"401","unit":""}'
  else
    result 0 "expected a 401, got: $out"
  fi
}

check_real_model_turn() {
  local out text
  out=$(chat "$SESS_MAIN" --prompt "Reply with exactly this text and nothing else: SEALED-$M1")
  note_calls "$SESS_MAIN" "$out"
  text=$(jq -r .text <<<"$out")
  if [[ $text == *"SEALED-$M1"* ]]; then
    result 1 "answered in $(jq -r .total_ms <<<"$out") ms, first token after $(jq -r .first_token_ms <<<"$out") ms" \
      "$(jq -c '{first_token_ms, total_ms, connected_ms, input_tokens:.prompt_response.data.input_tokens, output_tokens:.prompt_response.data.output_tokens, cache_read_tokens:.prompt_response.data.cache_read_tokens, model_calls:.prompt_response.data.assistant_messages}' <<<"$out")" \
      "$(jq -c '{label:"first token",value:.first_token_ms,unit:"ms"}' <<<"$out")"
  else
    result 0 "the model did not return the phrase: $text"
  fi
}

check_sandbox_exec() {
  local out tool output
  out=$(chat "$SESS_MAIN" --prompt 'Use the bash tool to run: uname -a; ls /   Then tell me in one sentence what you see.')
  note_calls "$SESS_MAIN" "$out"
  tool=$(jq -c '[.tools[]|select(.name=="bash")][0]' <<<"$out")
  output=$(jq -r '.output // ""' <<<"$tool")
  # Only the sandbox container has /workspace and /mnt/efs; the agent's own container has neither.
  if [[ $output == *Linux* && $output == *workspace* && $output == *mnt* ]]; then
    result 1 "ran in the sandbox container (it lists /workspace and /mnt/efs)" \
      "$(jq -c --argjson t "$tool" '{tool_ms: (($t.end_ms // 0) - $t.t_ms), total_ms, first_token_ms}' <<<"$out")" \
      "$(jq -c --argjson t "$tool" '{label:"tool round trip",value:(($t.end_ms // 0) - $t.t_ms),unit:"ms"}' <<<"$out")"
  else
    result 0 "the command did not run in the sandbox: $output"
  fi
}

check_background_job_notify() {
  local m2 out bg cb job_cb woke
  m2="job-$RANDOM$RANDOM"
  out=$(chat "$SESS_JOB" --wait-wake-secs 120 --prompt "Use the bash tool with run_in_background=true to run exactly: sleep 20; echo finished-$m2 . Then reply with just the word started and end your turn. Never poll, wait on or kill the job: you will be told when it ends, and then you must reply with the job's exact output line and its exit code.")
  note_calls "$SESS_JOB" "$out"
  FORGE_SESSION=$SESS_JOB
  woke=$(jq -r '.woke_at_ms // empty' <<<"$out")
  bg=$(jq -c '[.background[]|select(.reason=="finished")][0] // null' <<<"$out")
  # The sandbox's own record of the callback it sent: when, and what the agent answered.
  job_cb=null
  for _ in $(seq 1 30); do
    cb=$(cw sandbox/ "$(jq -r .sent_at_ms <<<"$out")" '"job_ended_callback"' | jq -c '[.[]|{t, d:(.m|fromjson)}|select(.d.detail.status==200)]')
    job_cb=$(jq -c 'sort_by(.t)|last // null' <<<"$cb")
    [ "$(jq 'length' <<<"$cb")" -ge 2 ] && break
    sleep 3
  done
  local cb_t cb_ms
  cb_t=$(jq -r '.t // empty' <<<"$job_cb")
  cb_ms=$(jq -r '.d.detail.took_ms // empty' <<<"$job_cb")
  if [ -n "$cb_t" ]; then
    echo "  job_ended reached the agent at $(date -u -d "@$((cb_t / 1000))" +%T).$(printf %03d $((cb_t % 1000)))Z (callback answered 200 in ${cb_ms} ms)"
  fi
  local notice
  notice=$(jq -r '.text // ""' <<<"$bg")
  if [ -n "$woke" ] && [[ $notice == *"finished-$m2"* && $notice == *"exited with code 0"* ]] && [ -n "$cb_t" ]; then
    result 1 "the agent was told the job ended (exit 0, its real output) $((woke - cb_t)) ms after the sandbox called back" \
      "$(jq -c --argjson cb "$cb_t" --argjson cbms "${cb_ms:-0}" --argjson woke "$woke" \
        '{job_ended_callback_ms:$cb, callback_http_ms:$cbms, agent_event_ms:$woke, notify_latency_ms:($woke-$cb), model_followup_ms:(.wake_done_ms-$woke), first_turn_ms:.total_ms, model_restated_output:(.text|contains("finished-'"$m2"'"))}' <<<"$out")" \
      "$(jq -nc --argjson l "$((woke - cb_t))" '{label:"notify latency",value:$l,unit:"ms"}')"
  else
    result 0 "job end not reported as expected (woke=$woke callback=$cb_t bg=$bg)"
  fi
}

check_notify_forged_refused() {
  local s=${FORGE_SESSION:-$SESS_MAIN} before after status token body nt
  before=$(chat "$s" --command '{"type":"get_state","id":"g"}' --timeout-secs 30 | jq -c '.responses[0].data|{steps,message_count}')
  # A connection is held open meanwhile, so the session is live on the agent: the callback is
  # refused for its token, not merely for finding no session.
  chat "$s" --wait-wake-secs 25 --timeout-secs 40 >/dev/null 2>&1 &
  local holder=$!
  sleep 4
  # The right kid and scope, a wrong tag: what someone who knows the format but not the key can make.
  nt=$(DEMO_FLEET_SECRET=$DEMO_FLEET_SECRET "$DEMO" notify-token --tenant "$TENANT" --session "$s" --job 0123456789abcdef --forge)
  token=$(jq -r .token <<<"$nt")
  body=$(jq -nc --arg s "$(jq -r .scope <<<"$nt")" '{type:"job_ended",scope:$s,id:"0123456789abcdef"}')
  status=$(curl -sS -m 15 -o "$WORK/forged.body" -w '%{http_code}' -X POST -H "Authorization: Bearer $token" \
    -H 'Content-Type: application/json' -d "$body" "http://$IP:$AGENT_PORT/_beyond/agent?session_id=$s&tenant=$TENANT")
  after=$(chat "$s" --command '{"type":"get_state","id":"g"}' --timeout-secs 30 | jq -c '.responses[0].data|{steps,message_count}')
  wait "$holder" 2>/dev/null
  tl sandbox forged_job_ended '["sandbox","agent"]' "$(jq -nc --arg s "$status" '{status:$s}')"
  echo "  forged job_ended -> HTTP $status $(cat "$WORK/forged.body"); session before $before after $after"
  if [ "$status" = 401 ] && [ "$before" = "$after" ]; then
    result 1 "refused with 401 (right kid, wrong tag) while the session was open; its steps and messages did not change" \
      "$(jq -nc --argjson b "$before" --argjson a "$after" --arg s "$status" '{http_status:($s|tonumber), before:$b, after:$a}')" \
      '{"label":"forged callback","value":"401","unit":""}'
  else
    result 0 "status $status, before $before, after $after"
  fi
}

check_kill_and_recover() {
  local m3 out msgs_before old new t_kill t_stopped t_new t_ready resumed msgs_after re st bg
  m3="kill-$RANDOM$RANDOM"
  out=$(chat "$SESS_KILL" --prompt "Use the bash tool with run_in_background=true to run exactly: sleep 70; echo finished-$m3 . Then reply with just the word started and end your turn. Never poll, wait on or kill the job: you will be told when it ends, and then you must reply with the job's exact output line and its exit code.")
  note_calls "$SESS_KILL" "$out"
  FORGE_SESSION=${FORGE_SESSION:-$SESS_KILL}
  msgs_before=$(chat "$SESS_KILL" --command '{"type":"get_state","id":"g"}' --timeout-secs 30 | jq -r '.responses[0].data.message_count')
  old=$(q ecs list-tasks --cluster "$CLUSTER" --service-name "$P-agent" --desired-status RUNNING --query 'taskArns[0]')
  echo
  echo "  >>> SIGKILL the agent task now (message_count before: $msgs_before)"
  t_kill=$(now_ms)
  tl demo agent_killed null "$(jq -nc --arg t "$old" '{task:$t}')" "$t_kill"
  # stop-task sends SIGTERM to pid 1, which is a shell that ignores it; the container is SIGKILLed two seconds later.
  aws ecs stop-task --cluster "$CLUSTER" --task "$old" --reason "demo kill" --query task.lastStatus >/dev/null
  t_stopped=""
  t_new=""
  for _ in $(seq 1 240); do
    if [ -z "$t_stopped" ]; then
      st=$(q ecs describe-tasks --cluster "$CLUSTER" --tasks "$old" --query 'tasks[0].lastStatus')
      [ "$st" = STOPPED ] && { t_stopped=$(now_ms); echo "  agent task stopped after $(((t_stopped - t_kill) / 1000)) s"; tl demo agent_stopped null null "$t_stopped"; }
    fi
    new=$(q ecs list-tasks --cluster "$CLUSTER" --service-name "$P-agent" --desired-status RUNNING --query 'taskArns[0]')
    if [ -n "$new" ] && [ "$new" != "$old" ]; then
      t_new=$(now_ms)
      echo "  ECS started a replacement after $(((t_new - t_kill) / 1000)) s"
      tl demo agent_replacement_running null null "$t_new"
      break
    fi
    sleep 1
  done
  [ -n "$t_new" ] || { result 0 "ECS did not replace the agent task"; return; }
  IP=$(wait_agent) || { result 0 "the replacement agent never became ready"; return; }
  t_ready=$(now_ms)
  tl demo agent_ready null "$(jq -nc --arg ip "$IP" '{ip:$ip}')" "$t_ready"
  echo "  replacement answers /readyz at $IP after $(((t_ready - t_kill) / 1000)) s"
  # Same session id, a fresh grant: the new agent opens the session from EFS and learns the job's fate.
  re=$(chat "$SESS_KILL" --touch-sandbox --wait-wake-secs 150 --timeout-secs 200)
  resumed=$(jq -r .connected_at_ms <<<"$re")
  tl demo session_reattached '["you","agent"]' null "$resumed"
  msgs_after=$(chat "$SESS_KILL" --command '{"type":"get_state","id":"g"}' --timeout-secs 30 | jq -r '.responses[0].data.message_count')
  bg=$(jq -c '[.background[]|select(.reason=="finished")][0] // null' <<<"$re")
  note_calls "$SESS_KILL" "$re"
  [ "$(jq -r '.turns // 0' <<<"$re")" -gt 0 ] && echo '{"title_retry":1}' >> "$CALLS"
  if [ "$(jq -r .refused <<<"$re")" = false ] && [[ $(jq -r '.text // ""' <<<"$re") == *"finished-$m3"* || $(jq -r '.text // ""' <<<"$bg") == *"finished-$m3"* ]] \
    && [ "${msgs_after:-0}" -ge "${msgs_before:-1}" ] 2>/dev/null; then
    result 1 "session reattached $(((resumed - t_kill) / 1000)) s after the kill with $msgs_before -> $msgs_after messages, and the job's real output was reported" \
      "$(jq -nc --argjson k "$t_kill" --argjson s "${t_stopped:-0}" --argjson n "$t_new" --argjson r "$t_ready" --argjson a "$resumed" \
        --argjson mb "$msgs_before" --argjson ma "$msgs_after" --argjson woke "$(jq '.woke_at_ms // 0' <<<"$re")" \
        '{kill_issued_ms:$k, task_stopped_ms:$s, replacement_running_ms:$n, agent_ready_ms:$r, session_reattached_ms:$a,
          kill_to_stopped_s:(if $s>0 then (($s-$k)/1000) else null end), kill_to_replacement_s:(($n-$k)/1000), kill_to_ready_s:(($r-$k)/1000),
          takeover_s:(($a-$k)/1000), messages_before:$mb, messages_after:$ma, job_end_reported_ms:$woke}')" \
      "$(jq -nc --argjson t "$(((resumed - t_kill) / 1000))" '{label:"takeover",value:$t,unit:"s"}')"
  else
    result 0 "did not recover: re=$(jq -c '{refused,text,background}' <<<"$re") before=$msgs_before after=$msgs_after"
  fi
}

check_sealed_at_rest() {
  local s=${1:-$SESS_MAIN} cmd out res hex plain bytes line
  cmd="d=/mnt/efs/$TENANT/sessions/$s; f=\$(ls \$d/*.jsonl | head -1); echo FILE=\$f; echo BYTES=\$(wc -c < \$f); \
echo PLAINTEXT_MATCHES=\$(cat \$d/* | grep -c 'SEALED-$M1'); echo LINE2_PREFIX=\$(sed -n 2p \$f | head -c 24); echo HEX; sed -n 2p \$f | head -c 96 | od -An -tx1"
  out=$(chat "$s" --command "$(jq -nc --arg c "$cmd" '{type:"bash",id:"b",command:$c}')" --timeout-secs 60)
  res=$(jq -r '[.responses[]|select(.command=="bash")][0].data // {} | (.output // .stdout // .result // "")' <<<"$out")
  [ -n "$res" ] || res=$(jq -r '[.tools[]?|.output // ""]|join("\n")' <<<"$out")
  plain=$(sed -n 's/^PLAINTEXT_MATCHES=//p' <<<"$res" | head -1)
  bytes=$(sed -n 's/^BYTES=//p' <<<"$res" | head -1)
  line=$(sed -n 's/^LINE2_PREFIX=//p' <<<"$res" | head -1)
  hex=$(sed -n '/^HEX$/,$p' <<<"$res" | tail -n +2 | tr -s ' \n' ' ' | sed 's/^ //;s/ $//')
  echo "  on EFS: $bytes bytes; plaintext matches: $plain; line 2 begins: $line"
  echo "  hex: ${hex:0:96}…"
  DISK=$(jq -nc --arg s "$s" --arg m "SEALED-$M1" --arg hex "$hex" --arg pre "$line" --argjson bytes "${bytes:-0}" --argjson plain "${plain:-1}" \
    '{session:$s, plaintext_seen_by_user:$m, hex_excerpt:$hex, ascii_prefix:$pre, file_bytes:$bytes, plaintext_matches:$plain}')
  tl agent session_read_from_efs '["efs","sandbox"]' null
  if [ "${plain:-1}" = 0 ] && [ -n "$hex" ] && [ "${bytes:-0}" -gt 0 ]; then
    result 1 "the phrase the user saw appears 0 times in the session's $bytes bytes on EFS" \
      "$(jq -c '{plaintext_matches, file_bytes}' <<<"$DISK")" '{"label":"plaintext on disk","value":"0","unit":"matches"}'
  else
    result 0 "plaintext matches: ${plain:-?}, bytes: ${bytes:-?}; output: ${res:0:300}"
  fi
}

check_billing_rows() {
  local expected rows n tries total
  # What the agent says it asked the model, per session (the latest cumulative count), plus one title
  # call per session: the agent names a session once, with a model call of its own.
  # A session the agent could not title (the title call may spend its budget thinking) is titled
  # again by the next process that runs a turn in it: one more call after a restart.
  expected=$(jq -s '([.[]|select(.session)]|group_by(.session)|map(max_by(.assistant_messages)|.assistant_messages)|(add // 0) + length) + ([.[]|select(.title_retry)]|length)' "$CALLS")
  rows='[]'
  for tries in $(seq 1 40); do
    rows=$(cw gateway/ "$((START_MS - 5000))" '"ai.usage"' | jq -c '[.[]|.m|fromjson?|select(.target=="ai.usage")|.fields]')
    n=$(jq length <<<"$rows")
    [ "$n" -ge "$expected" ] && { sleep 6; rows=$(cw gateway/ "$((START_MS - 5000))" '"ai.usage"' | jq -c '[.[]|.m|fromjson?|select(.target=="ai.usage")|.fields]'); n=$(jq length <<<"$rows"); break; }
    sleep 5
  done
  BILLING=$(jq -c --argjson e "$expected" '{calls_expected:$e, rows_found:length,
    total_cost_micros:(map(.cost_micros)|add // 0), total_price_micros:(map(.price_micros)|add // 0),
    input_tokens:(map(.input_tokens)|add // 0), output_tokens:(map(.output_tokens)|add // 0), cache_read_tokens:(map(.cache_read_tokens)|add // 0),
    rows:map({request_id, t_ms:((.start_unix_secs // 0)*1000), tenant_id, provider, model, requested_model, routed_model, price_model, outcome, price_status,
      input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, cost_micros, price_micros, latency_ms, upstream_request_id})}' <<<"$rows")
  total=$(jq -r '.total_cost_micros' <<<"$BILLING")
  local complete dup
  # Complete: tenant, the requested/routed/price model, tokens and cost present; the cost non-zero
  # wherever a row spent tokens (a title call that produced no output still has priced input).
  complete=$(jq '[.rows[]|select(.tenant_id!=null and .requested_model!=null and .routed_model!=null and .price_model!=null and .outcome=="ok" and .price_status=="priced" and .input_tokens>0 and .cost_micros>0)]|length' <<<"$BILLING")
  dup=$(jq '[.rows[].request_id]|(length - (unique|length))' <<<"$BILLING")
  printf '  %s model calls made, %s ai.usage rows found, %s complete, summed cost $%s\n' "$expected" "$n" "$complete" "$(jq -nr --argjson c "$total" '$c/1000000')"
  local bj title_rows
  title_rows=$(jq '[.rows[]|select(.output_tokens==0)]|length' <<<"$BILLING")
  bj=$(jq -nc --argjson tr "$title_rows"  --argjson e "$expected" --argjson n "$n" --argjson c "$complete" --argjson d "$dup" --argjson t "$total" \
    '{title_calls:$tr, calls_expected:$e, rows_found:$n, rows_complete:$c, duplicate_request_ids:$d, total_cost_micros:$t, total_cost_usd:($t/1000000)}')
  if [ "$n" = "$expected" ] && [ "$complete" = "$n" ] && [ "$dup" = 0 ] && [ "$n" -gt 0 ]; then
    result 1 "$n calls, $n rows, every field present; the demo cost \$$(jq -nr --argjson c "$total" '$c/1000000')" "$bj" \
      "$(jq -nc --argjson t "$total" --argjson n "$n" '{label:"cost",value:($t/1000000),unit:("USD, "+($n|tostring)+" rows")}')"
  else
    result 0 "calls $expected, rows $n, complete $complete, duplicate ids $dup" "$bj"
  fi
}

# ---- run ---------------------------------------------------------------------------------------------
echo
echo "Demo run $RUN_ID: $MODE checks (${CHECKS[*]})"
tl demo run_started null null "$START_MS"
FORGE_SESSION=""
DISK=null
BILLING=null
for c in "${CHECKS[@]}"; do
  run_check "$c"
done
END_MS=$(now_ms)
tl demo run_finished null null "$END_MS"

# ---- the record -----------------------------------------------------------------------------------------
rule
echo "writing the run record…"
sleep 8   # CloudWatch lags a few seconds; the sandbox and gateway lines for the last step
LOGS=$(cw "" "$((START_MS - 5000))")
# Gateway rows and sandbox events join the timeline; the agent's own startup lines mark restarts.
# A job's first callback is the one `keep` triggers (the proof); the later one is its end.
jq -c '[.[]|select(.m|startswith("{"))|{t, d:(.m|fromjson? // empty)}|select(.d.src=="sandbox")|
  {t_ms:.d.ts_ms, src:"sandbox", event:.d.event, path:(if .d.event=="exec_start" then ["agent","sandbox"] elif .d.event=="exec_end" or .d.event=="job_ended_callback" then ["sandbox","agent"] else null end), detail:.d.detail}]
  | sort_by(.t_ms) | reduce .[] as $e ({seen:{},out:[]};
      if $e.event=="job_ended_callback" then ($e.detail.job) as $j | (if .seen[$j] then "job_end" else "keep_time" end) as $k | .seen[$j]=true | .out += [$e|.detail.kind=$k]
      else .out += [$e] end) | .out[]' <<<"$LOGS" >> "$TIMELINE"
jq -c '.[]|{t, d:(.m|fromjson? // empty)}|select(.d.target=="ai.usage")|.d.fields|
  ({t_ms:((.start_unix_secs*1000)), src:"gateway", event:"model_request", path:["agent","gateway","model"], detail:{request_id, model:.requested_model, input_tokens}},
   {t_ms:((.start_unix_secs*1000)+.latency_ms), src:"gateway", event:"model_response", path:["model","gateway","agent"], detail:{request_id, output_tokens, cost_micros}})' <<<"$LOGS" >> "$TIMELINE"
jq -c '.[]|select(.m|startswith("serve: websocket listening"))|{t_ms:.t, src:"agent", event:"agent_listening", path:null, detail:{line:.m}}' <<<"$LOGS" >> "$TIMELINE"
# Client frames: the key ones only (the stream's deltas stay out of the record).
jq -c 'select(.src=="client")|.frame as $f|.session as $s|.ts_ms as $t|
  if $f.type=="demo_prompt_sent" then {t_ms:$t,src:"client",event:"prompt",path:["you","agent"],detail:{session:$s}}
  elif $f.event.kind=="tool_start" then {t_ms:$t,src:"client",event:"tool_start",path:null,detail:{session:$s,name:$f.event.name}}
  elif $f.event.kind=="tool_end" then {t_ms:$t,src:"client",event:"tool_end",path:null,detail:{session:$s,name:$f.event.name}}
  elif $f.event.kind=="background" then {t_ms:$t,src:"client",event:("background_"+($f.event.reason//"")),path:null,detail:{session:$s,wake:$f.event.wake}}
  elif $f.type=="response" and $f.command=="prompt" then {t_ms:$t,src:"client",event:"turn_complete",path:["agent","efs"],detail:{session:$s}}
  else empty end' "$EVENTS" >> "$TIMELINE"

CONV=$(jq -s '
  def close: if .open then .open=false else . end;
  [ .[]|select(.src=="client") ] |
  reduce .[] as $e ({msgs:[], open:false};
    ($e.frame) as $f | ($e.ts_ms) as $t | ($e.session) as $s |
    if $f.type=="demo_prompt_sent" then .msgs += [{t_ms:$t,session:$s,role:"user",text:$f.message}] | .open=false
    elif ($f.event.kind=="stream" and $f.event.type=="text_delta") then
      (if .open then .msgs[-1].text += $f.event.text else .msgs += [{t_ms:$t,session:$s,role:"assistant",text:$f.event.text}] | .open=true end)
    elif $f.event.kind=="tool_start" then .msgs += [{t_ms:$t,session:$s,role:"tool",name:$f.event.name,input:$f.event.input}] | .open=false
    elif $f.event.kind=="tool_end" then (.msgs[-1] |= (if .role=="tool" then .output=($f.event.result|tostring|.[0:600]) | .end_ms=$t else . end)) | .open=false
    elif $f.event.kind=="background" and ($f.event.reason=="finished") then .msgs += [{t_ms:$t,session:$s,role:"notice",text:$f.event.text}] | .open=false
    else . end) | .msgs
  | map(if .input then .input=(.input|tostring|.[0:300]) else . end)' "$EVENTS")

TOTAL_COST=$(jq -r '.total_cost_micros // 0' <<<"${BILLING:-null}" 2>/dev/null || echo 0)
COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
DIRTY=$(git -C "$ROOT" status --porcelain 2>/dev/null | grep -c . || true)
ROWS_JSON=$(cw gateway/ "$((START_MS - 5000))" '"ai.usage"' | jq -c '[.[]|.m|fromjson?|select(.target=="ai.usage")|.fields]')
jq -n --slurpfile checks "$CHECKS_OUT" --slurpfile tl "$TIMELINE" --argjson conv "$CONV" --argjson disk "${DISK:-null}" \
  --argjson rows "$ROWS_JSON" --arg id "$RUN_ID" --arg mode "$MODE" --arg commit "$COMMIT" --argjson dirty "${DIRTY:-0}" \
  --arg region "$REGION" --arg name "$NAME" --arg model "$MODEL" --arg provider "$PROVIDER" \
  --argjson start "$START_MS" --argjson end "$END_MS" '
  ($checks) as $c |
  ($rows | map({request_id, t_ms:((.start_unix_secs // 0)*1000), tenant_id, provider, model, requested_model, routed_model, price_model, outcome, price_status,
      input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, cost_micros, price_micros, latency_ms})) as $r |
  {schema:1,
   run:{id:$id, mode:$mode, started_ms:$start, ended_ms:$end, commit:$commit, uncommitted_files:$dirty, region:$region, name:$name},
   model:{requested:$model, provider:$provider, routed:($r|map(.routed_model)|unique), price_model:($r|map(.price_model)|unique), via:"crates/gateway (managed bai_v1 key)"},
   checks:($c + [{name:"teardown-zero", title:"Teardown leaves nothing behind", pass:null, status:"pending", detail:"filled in by down.sh", measurements:{}}]),
   timeline:($tl|sort_by(.t_ms)),
   conversation:$conv,
   billing:{rows:$r, rows_found:($r|length), total_cost_micros:($r|map(.cost_micros)|add // 0), total_price_micros:($r|map(.price_micros)|add // 0),
            total_cost_usd:(($r|map(.cost_micros)|add // 0)/1000000), input_tokens:($r|map(.input_tokens)|add // 0),
            output_tokens:($r|map(.output_tokens)|add // 0), cache_read_tokens:($r|map(.cache_read_tokens)|add // 0)},
   disk:$disk,
   teardown:null}' > "$RUN_DIR/run.json"
"$HERE/report.sh" "$RUN_DIR" > /dev/null

rule
pass=$(jq '[.checks[]|select(.pass==true)]|length' "$RUN_DIR/run.json")
total=$(jq '[.checks[]|select(.name!="teardown-zero")]|length' "$RUN_DIR/run.json")
failed=$(jq -r '[.checks[]|select(.pass==false)|.name]|join(", ")' "$RUN_DIR/run.json")
echo "$pass of $total checks passed${failed:+ (failed: $failed)}; model spend \$$(jq -r '.billing.total_cost_usd' "$RUN_DIR/run.json")"
echo "record  $RUN_DIR/run.json"
echo "report  $RUN_DIR/report.html"
[ -z "$failed" ]
