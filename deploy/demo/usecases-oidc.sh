#!/usr/bin/env bash
# shellcheck shell=bash disable=SC2034,SC2016,SC2154  # sourced by usecases.sh: its variables are ours, queries are single-quoted
# The oidc use case: an agent acts as a signed-in user against an OAuth / OIDC protected service, through a
# managed connector, and never holds the user's tokens.
#
#     SANDBOX=oidc ./deploy/demo/up.sh && ./deploy/demo/demo.sh oidc
#
# The stack: Keycloak (the identity provider, two users), and one task beside it that holds the broker
# (the managed endpoint that signs users in once and acts as them) and the notes service it fronts
# (a resource server that trusts only a validated JWT). See BROKER.md for the broker's contract.
#
# The scripted user signs in through the provider's real login and consent pages (oidc/login.sh); the
# control plane's calls (start a consent, bind a session, revoke) are curl with the admin credential.

CHECKS_oidc=(user-signs-in-once agent-acts-as-user agent-never-sees-token token-refresh users-isolated scope-enforced revocation)
CHECK_TITLE+=(
  [user-signs-in-once]="A user signs in once, and the broker holds the refresh token sealed"
  [agent-acts-as-user]="The agent acts as the user: the service sees the user's sub in a validated token"
  [agent-never-sees-token]="The agent never holds an access or refresh token"
  [token-refresh]="The access token lapses and the broker refreshes it, invisibly to the agent"
  [users-isolated]="Two users in two sessions each see only their own data"
  [scope-enforced]="A tool needing a scope the connection lacks is refused upstream"
  [revocation]="After the user revokes, the next call says so and the session survives"
)
UC_TITLE[oidc]="oidc: an agent that acts as a signed-in user and never holds their tokens"
UC_SUMMARY[oidc]="A folder with an AGENTS.md, deployed as a bundle; the agent has no sandbox. Its one connector is managed: it holds a bmc_v1 token and nothing else. Behind that connector a broker (the stand-in for Beyond's managed endpoint) signs each user in once with authorization code + PKCE against a real identity provider (Keycloak), keeps their refresh token sealed under a per-tenant key, refreshes the 60-second access tokens itself, and calls a notes service that trusts only a validated JWT. The checks drive two users through it, then look for any token in the agent."

# ---- helpers -----------------------------------------------------------------------------------------------

declare -gA OIDC_CONN OIDC_SUB

# The control plane's calls: the admin credential travels in a 0600 header file, never on a command line.
oidc_ctl() { # method path [json-body]
  curl -sS -m 40 -X "$1" -H @"$WORK/admin.hdr" ${3:+-d "$3"} "http://$OIDC_IP:$OIDC_CONTROL_PORT$2"
}
# The scripted browser reaches the provider (and the broker's callback) by their stable names at today's address.
oidc_to() { echo "$OIDC_HOST:$1:$OIDC_IP:$1"; }

# Broker and resource lines from CloudWatch: [{ts_ms, src, event, detail}], once at least $3 have arrived.
oidc_log() { # since-ms event-regex [min]
  local rows n
  for _ in $(seq 1 20); do
    rows=$(cw broker/ "$1" | jq -c --arg re "$2" '[.[]|.m|fromjson?|select(.event != null and (.event|test($re)))]|sort_by(.ts_ms)')
    n=$(jq length <<<"$rows")
    [ "$n" -ge "${3:-1}" ] && break
    sleep 3
  done
  echo "$rows"
}

oidc_mint() { # session -> a bmc_v1 token for that session's notes connector
  "$DEMO" managed-token --tenant "$UC_TENANT" --session "$1" --connector notes
}
# A session's own token, presented raw to the managed endpoint: what a compromised agent could try.
oidc_probe() { # session path extra-header-or-empty json-body -> "<status> <body>"
  local tok hdr out
  tok=$(oidc_mint "$1")
  hdr=$WORK/probe.hdr
  { printf 'authorization: Bearer %s\ncontent-type: application/json\n' "$tok"; [ -z "$3" ] || printf '%s\n' "$3"; } > "$hdr"
  chmod 600 "$hdr"
  out=$(curl -sS -m 30 -o "$WORK/probe.body" -w '%{http_code}' -H @"$hdr" -d "$4" "http://$OIDC_IP:$STANDIN_PORT$2")
  echo "$out $(head -c 600 "$WORK/probe.body")"
  rm -f "$hdr" "$WORK/probe.body"
}
RPC_MY_NOTES='{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"my_notes","arguments":{}}}'

_grant_oidc() { # session [tamper]: no sandbox; one managed connector; the folder is the bundle
  "$DEMO" mint-grant --tenant "$UC_TENANT" --session "$1" --shard demo --workspace /workspace/oidc \
    --exec-url "" --bundle "$BUNDLE_URL=$BUNDLE_SHA" \
    --mcp "notes=http://$OIDC_IP:$STANDIN_PORT/managed/notes/mcp" --mcp-managed notes \
    --limits-models "gpt-4.1-mini,gpt-5-mini" ${2:+--tamper}
}

uc_prepare_oidc() {
  [ "$OIDC" = 1 ] || { echo "demo: oidc needs its own stack. Bring it up with: SANDBOX=oidc ./deploy/demo/up.sh" >&2; exit 1; }
  OIDC_IP=$(svc_ip oidc)
  [ -n "$OIDC_IP" ] || { echo "demo: the identity provider and broker task is not running" >&2; exit 1; }
  OIDC_HOST=oidc.$DNS
  OIDC_ISSUER=http://$OIDC_HOST:$OIDC_IDP_PORT/realms/demo
  for _ in $(seq 1 90); do
    curl -fsS -m 3 --connect-to "$(oidc_to "$OIDC_IDP_PORT")" "$OIDC_ISSUER/.well-known/openid-configuration" > /dev/null 2>&1 \
      && [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://$OIDC_IP:$OIDC_CONTROL_PORT/healthz")" = 200 ] \
      && [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' -X POST -d '{}' "http://$OIDC_IP:$STANDIN_PORT/managed/notes/mcp")" = 401 ] && break
    sleep 3
  done
  echo "  identity provider  $OIDC_ISSUER   (at $OIDC_IP)"
  echo "  broker             managed http://$OIDC_IP:$STANDIN_PORT (401 without a token), control http://$OIDC_IP:$OIDC_CONTROL_PORT"
  DEMO_MANAGED_SECRET=$(ssm_get managed-secret)
  export DEMO_MANAGED_SECRET
  printf 'authorization: Bearer %s\n' "$(ssm_get broker-admin)" > "$WORK/admin.hdr"
  chmod 600 "$WORK/admin.hdr"
  OIDC_USERS=$(ssm_get oidc-users)
  GRANT_FN=_grant_oidc
  uc_stage
  uc_bundle || { echo "demo: could not publish the bundle" >&2; exit 1; }
  SESS_OA=demo.u${SID}a   # alice's
  SESS_OB=demo.u${SID}b   # bob's
}

# ---- 1. the user signs in once -----------------------------------------------------------------------------

qparam() { sed -n "s/.*[?&]$2=\([^&]*\).*/\1/p" <<<"$1" | head -1; }

check_user_signs_in_once() {
  local t0 u pw start url loc cb scope audit evidence='[]' ok=1 ev started created
  t0=$(now_ms)
  for u in alice bob; do
    pw=$(jq -r --arg u "$u" '.[$u]' <<<"$OIDC_USERS")
    start=$(oidc_ctl POST /connect/start "$(jq -nc --arg t "$UC_TENANT" '{tenant:$t,connector:"notes"}')")
    url=$(jq -r '.authorize_url // empty' <<<"$start")
    [ -n "$url" ] || { result 0 "the broker did not start a consent: $start"; return; }
    # the scripted user: the provider's login form, then its consent screen
    loc=$(LOGIN_PASSWORD=$pw "$HERE/oidc/login.sh" "$url" "$u" - --connect-to "$(oidc_to "$OIDC_IDP_PORT")" 2>&1) || { result 0 "$u could not sign in: $loc"; return; }
    cb=$(curl -sS -m 30 --connect-to "$(oidc_to "$OIDC_CONTROL_PORT")" "$loc")
    OIDC_CONN[$u]=$(jq -r '.connection // empty' <<<"$cb")
    OIDC_SUB[$u]=$(jq -r '.sub // empty' <<<"$cb")
    scope=$(jq -r '.scope // ""' <<<"$cb")
    [ -n "${OIDC_CONN[$u]}" ] || { result 0 "the callback did not create a connection for $u: $cb"; return; }
    audit=$(oidc_ctl GET "/connections/${OIDC_CONN[$u]}?audit=1")
    echo "  $u signed in as ${OIDC_SUB[$u]}: connection ${OIDC_CONN[$u]}, granted: $scope"
    echo "    PKCE ($(qparam "$url" code_challenge_method)), response_type=$(qparam "$url" response_type), redirect_uri=$(qparam "$url" redirect_uri | sed 's/%3A/:/g; s/%2F/\//g')"
    echo "    at rest: $(jq -c '.at_rest|{sealed_bytes, opens_under_the_tenants_key, opens_under_another_tenants_key, plaintext_token_in_file}' <<<"$audit")"
    jq -e '.at_rest.opens_under_the_tenants_key == true and .at_rest.opens_under_another_tenants_key == false and .at_rest.opens_as_another_connection == false
      and .at_rest.plaintext_token_in_file == false and .at_rest.token_tail_in_file == false and .refresh_token_stored == true' <<<"$audit" > /dev/null || ok=0
    [[ $scope == *notes:read* && $scope == *offline_access* && $scope != *notes:write* ]] || ok=0
    [[ $(qparam "$url" code_challenge_method) == S256 && $(qparam "$url" response_type) == code ]] || ok=0
    evidence=$(jq -c --arg u "$u" --arg c "${OIDC_CONN[$u]}" --arg s "${OIDC_SUB[$u]}" --arg sc "$scope" --argjson a "$audit" --arg m "$(qparam "$url" code_challenge_method)" \
      '. + [{user:$u, connection:$c, sub:$s, granted_scope:$sc, pkce_method:$m, at_rest:$a.at_rest, refresh_token_stored:$a.refresh_token_stored}]' <<<"$evidence")
  done
  [ "${OIDC_CONN[alice]}" != "${OIDC_CONN[bob]}" ] || ok=0
  # one session per user, each bound to exactly its user's connection; only then are the sessions opened
  # (opening a session lists its connector's tools, and an unbound session has no one to act as)
  for u in alice bob; do
    local sess=$SESS_OA bound
    [ "$u" = alice ] || sess=$SESS_OB
    bound=$(oidc_ctl POST /sessions/bind "$(jq -nc --arg t "$UC_TENANT" --arg s "$sess" --arg c "${OIDC_CONN[$u]}" '{tenant:$t,session:$s,connector:"notes",connection:$c}')")
    jq -e '.bound == true' <<<"$bound" > /dev/null || { ok=0; echo "  bind failed: $bound"; }
  done
  uc_open "$SESS_OA"
  uc_open "$SESS_OB"
  tl demo sessions_opened '["you","agent"]' null "$UC_READY_MS"
  ev=$(oidc_log "$((t0 - 5000))" '^(consent_started|connection_created|session_bound)$' 6)
  started=$(jq '[.[]|select(.event=="consent_started" and .detail.pkce=="S256")]|length' <<<"$ev")
  created=$(jq -c '[.[]|select(.event=="connection_created")|.detail|{user, sub, connection, scope, refresh_token_stored, refresh_sealed_bytes}]' <<<"$ev")
  echo "  broker log: $started consent(s) started with PKCE S256; created: $created"
  if [ "$ok" = 1 ] && [ "$started" -ge 2 ] && [ "$(jq length <<<"$created")" -ge 2 ]; then
    result 1 "alice and bob each signed in once through the provider's login and consent pages (authorization code + PKCE S256); the broker holds each refresh token sealed (XChaCha20-Poly1305 under a per-tenant key: it opens under the tenant's key, not another's, and the plaintext is not in the file); each granted notes:read and offline_access, not notes:write; each session is bound to exactly its user's connection" \
      "$(jq -nc --argjson e "$evidence" --argjson c "$created" '{users:$e, broker_log:$c}')" \
      '{"label":"refresh tokens stored","value":"2 sealed","unit":""}'
  else
    result 0 "ok=$ok started=$started created=$created evidence=$evidence"
  fi
}

# ---- 2. the agent acts as the user ---------------------------------------------------------------------------

check_agent_acts_as_user() {
  local t0 out text tool ref calls seen sub_ok ref_in_reply
  t0=$(now_ms)
  out=$(chat "$SESS_OA" --prompt "List my notes with your notes tool and quote the passport renewal reference exactly.")
  note_calls "$SESS_OA" "$out"
  text=$(jq -r .text <<<"$out")
  tool=$(uc_tool_out "$out" '^mcp__notes__my_notes$')
  ref=$(grep -o 'PR-[0-9a-f]*' <<<"$tool" | head -1)
  echo "  the tool answered: ${tool:0:160}"
  calls=$(oidc_log "$((t0 - 5000))" '^resource_call$' 1)
  seen=$(jq -c '[.[]|select(.detail.seen.tool=="my_notes")|.detail|{user:.seen.user, sub:.seen.sub, iss:.seen.iss, aud:.seen.aud, azp:.seen.azp, scope:.seen.scope, ttl_s:(.seen.exp-.seen.iat), ref}]' <<<"$calls")
  echo "  what the notes service saw (a validated token's claims): $seen"
  sub_ok=$(jq -r --arg s "${OIDC_SUB[alice]}" --arg iss "$OIDC_ISSUER" --arg ref "$ref" \
    'length >= 1 and all(.[]; .sub == $s and .iss == $iss and .aud == "notes-api" and .azp == "notes-broker" and (.scope|index("notes:read")) != null and .ref == $ref)' <<<"$seen")
  ref_in_reply=$([[ -n $ref && $text == *"$ref"* ]] && echo yes || echo no)
  echo "  reply quotes the reference: $ref_in_reply — ${text:0:160}"
  if [ -n "$ref" ] && [ "$sub_ok" = true ] && [ "$ref_in_reply" = yes ] && [[ $tool == *"notes for ${OIDC_SUB[alice]} (alice)"* ]]; then
    result 1 "the model called the notes tool; the service validated a JWT (signature via JWKS, issuer, audience notes-api, expiry, scope notes:read) whose sub is alice's (${OIDC_SUB[alice]}) and returned alice's notes; the reply quotes her reference $ref" \
      "$(jq -nc --argjson s "$seen" --arg ref "$ref" --arg sub "${OIDC_SUB[alice]}" '{service_saw:$s, reference:$ref, alice_sub:$sub, reply_quotes_reference:true}')" \
      "$(jq -nc --arg sub "${OIDC_SUB[alice]}" '{label:"sub the service saw",value:$sub,unit:""}')"
  else
    result 0 "ref=$ref saw=$seen reply=${text:0:200} tool=${tool:0:200}"
  fi
}

# ---- 3. the agent never sees a token ---------------------------------------------------------------------------

# Read the agent process's memory (and its environment, and what it keeps on EFS) through ECS Exec, for the
# control and the tokens' tails in $1 (a file of `kind tail`). The control is a string this run planted in
# the agent; the scan must find it, or it proves nothing.
scan_agent_oidc() { # needles-file -> the scan's stdout
  local task script out
  task=$(q ecs list-tasks --cluster "$CLUSTER" --service-name "$P-agent" --desired-status RUNNING --query 'taskArns[0]')
  [ -n "$task" ] || { echo "SCAN-ERROR no running agent task"; return; }
  script=$({
    printf 'cat > /tmp/needles <<"EOF_NEEDLES"\n'
    cat "$1"
    printf 'EOF_NEEDLES\ngrep -v "^control" /tmp/needles | cut -d" " -f2 > /tmp/needles.tails\nset -- /usr/local/bin/beyond-ai-agent /tmp/needles\n'
    cat "$HERE/oidc/scan-memory.sh"
    cat <<'EOF'
echo "SCAN-ENV needle_hits=$(tr '\0' '\n' < /proc/$pid/environ | grep -c -F -f /tmp/needles.tails) jwt_like=$(tr '\0' '\n' < /proc/$pid/environ | grep -c 'eyJ[A-Za-z0-9_-]\{8,\}\.')"
echo "SCAN-EFS files=$(find /mnt/efs -type f 2>/dev/null | wc -l) kb=$(du -sk /mnt/efs 2>/dev/null | cut -f1) needle_files=$(find /mnt/efs -type f -exec grep -l -F -f /tmp/needles.tails {} + 2>/dev/null | wc -l)"
rm -f /tmp/needles /tmp/needles.tails
# an ECS Exec session can drop its last output when it closes at once
sleep 5
EOF
  })
  # The script is longer than a command line comfortably is: it goes to the bundles bucket and the agent
  # container fetches it with a presigned link (the way it fetches its own bundle), then runs it.
  local key bucket url
  bucket=$(bundle_bucket)
  key=scan-$(printf '%s' "$script" | sha256sum | cut -c1-16).sh
  printf '%s\n' "$script" > "$WORK/scan.sh"
  command aws --region "$REGION" s3 cp "$WORK/scan.sh" "s3://$bucket/$key" --only-show-errors || { echo "SCAN-ERROR could not stage the scan"; return; }
  url=$(command aws --region "$REGION" s3 presign "s3://$bucket/$key" --expires-in 600)
  # (the session plugin ends the session, and drops its last output, when its stdin closes: hold stdin open)
  out=$(sleep 50 | command aws --region "$REGION" ecs execute-command --cluster "$CLUSTER" --task "$task" --container agent --interactive \
    --command "/bin/sh -c 'wget -q -O /tmp/scan.sh \"$url\" && sh /tmp/scan.sh; rm -f /tmp/scan.sh'" 2>&1 | tr -d '\r') || true
  command aws --region "$REGION" s3 rm "s3://$bucket/$key" --only-show-errors > /dev/null 2>&1 || true
  grep -aoE 'SCAN-[A-Z]+ .*' <<<"$out" || echo "SCAN-ERROR $(head -c 300 <<<"$out" | tr '\n' ' ')"
}

check_agent_never_sees_token() {
  local ctl_sig ctl_jwt out needles="$WORK/needles" tails="$WORK/tails" fp u kinds scan ctl_hits real_hits other_jwts kb bad
  # The control: a fresh token-shaped string, handed to the agent in a prompt, so it is in the agent's
  # memory (and the session's transcript) for certain. The scan has to find it.
  ctl_sig=$(openssl rand -base64 36 | tr '+/' '-_' | tr -d '=\n' | head -c 43)
  CTL_PAYLOAD=eyJjdHJsIjoicGxhbnRlZCJ9   # {"ctrl":"planted"}
  ctl_jwt="eyJhbGciOiJSUzI1NiJ9.$CTL_PAYLOAD.$ctl_sig"
  out=$(chat "$SESS_OA" --prompt "Do not use any tool. This is a format question only: is $ctl_jwt a valid JWT? Answer in one short sentence.")
  note_calls "$SESS_OA" "$out"
  # and a fresh call for each user, so each has an access token in flight right now (the scan below runs
  # within that token's 60 seconds, or the broker refreshes: either way the latest token is on the list)
  for s in "$SESS_OA" "$SESS_OB"; do
    out=$(chat "$s" --prompt "List my notes with your notes tool.")
    note_calls "$s" "$out"
  done
  # every token the broker has handled for the two connections: the last 24 characters of each (enough to
  # recognise a token, never enough to use it)
  : > "$needles"
  chmod 600 "$needles"
  printf 'control %s\n' "${ctl_sig: -24}" >> "$needles"
  for u in alice bob; do
    fp=$(oidc_ctl GET "/connections/${OIDC_CONN[$u]}/fingerprints")
    jq -r '.fingerprints[]|"\(.kind) \(.tail)"' <<<"$fp" >> "$needles"
  done
  grep -v '^control' "$needles" | cut -d' ' -f2 > "$tails"
  chmod 600 "$tails"
  kinds=$(cut -d' ' -f1 "$needles" | sort | uniq -c | awk '{print $2":"$1}' | paste -sd' ' -)
  echo "  looking for: $kinds (tails only)"

  # 1. the grant, unsealed with the agent's own verifier: only a bmc_v1 token, no JWT anywhere in it
  local g_a g_jwt g_bmc grants_ok=1
  for s in "$SESS_OA" "$SESS_OB"; do
    g_a=$(DEMO_GRANT=$(uc_grant "$s") DEMO_NEEDLE=eyJhbGci "$DEMO" inspect-grant)
    g_jwt=$(jq -r '[.connectors[].header_values_contain_needle, .exec_header_values_contain_needle, .gateway_key_contains_needle, .token_text_contains_needle]|any' <<<"$g_a")
    g_bmc=$(jq -r '.connectors[0]|.managed and .holds_a_bmc_token and (.header_names == ["Authorization"])' <<<"$g_a")
    [ "$g_jwt" = false ] && [ "$g_bmc" = true ] || grants_ok=0
  done
  echo "  grants (2 sessions): only a bmc_v1 bearer in the connector's header, no JWT text anywhere: $([ $grants_ok = 1 ] && echo yes || echo NO)"

  # 2. the agent's logs, the whole life of the task
  cw agent/ "$((START_MS - 4 * 3600 * 1000))" | jq -r '.[].m' > "$WORK/agent.log"
  local log_hits log_jwt log_lines startup
  log_hits=$(grep -cF -f "$tails" "$WORK/agent.log" || true)
  log_jwt=$(grep -oE 'eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}' "$WORK/agent.log" | grep -vF "${ctl_sig: -24}" | wc -l)
  log_lines=$(wc -l < "$WORK/agent.log")
  startup=$(grep -c 'websocket listening' "$WORK/agent.log" || true)

  # 3. the session transcripts, as the agent returns them: alice's holds the planted control (the control
  # for this scan); neither holds a real token
  local tr_ok=1 tr_ctl=0 tr_hits=0 tr_jwt=0 tr_bytes=0
  for s in "$SESS_OA" "$SESS_OB"; do
    chat "$s" --command '{"type":"get_messages","id":"gm"}' --timeout-secs 60 | jq -c '.responses[]|select(.command=="get_messages")|.data' > "$WORK/messages.$s.json"
    tr_bytes=$((tr_bytes + $(wc -c < "$WORK/messages.$s.json")))
    tr_hits=$((tr_hits + $(grep -cF -f "$tails" "$WORK/messages.$s.json" | tr -d '\n' || true)))
  done
  tr_ctl=$(grep -cF "$ctl_sig" "$WORK/messages.$SESS_OA.json" || true)
  for s in "$SESS_OA" "$SESS_OB"; do
    tr_jwt=$((tr_jwt + $(grep -oE 'eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}' "$WORK/messages.$s.json" | grep -vF "$ctl_sig" | wc -l)))
  done
  echo "  agent logs: $log_lines lines ($startup startup line = the control that the log was read), $log_hits hold a token tail, $log_jwt JWT-shaped strings"
  echo "  transcripts: $tr_bytes bytes; the planted control is in alice's ($tr_ctl line); $tr_hits real-token tails, $tr_jwt other JWT-shaped strings"

  # 4. the agent process: environment, EFS, and every writable page of memory
  scan=$(scan_agent_oidc "$needles")
  echo "$scan" | sed 's/^/    /'
  local res
  res=$(grep '^SCAN-RESULT' <<<"$scan" | head -1)
  kb=$(sed -n 's/.* kb=\([0-9]*\).*/\1/p' <<<"$res")
  bad=$(sed -n 's/.* unreadable=\([0-9]*\).*/\1/p' <<<"$res")
  ctl_hits=$(grep '^SCAN-NEEDLE control' <<<"$scan" | awk '{print $4}' | head -1)
  real_hits=$(grep -E '^SCAN-NEEDLE (access|refresh) ' <<<"$scan" | awk '{s += $4} END {print s + 0}')
  # the strings this run (and earlier runs against the same agent process) planted all carry the same payload
  other_jwts=$(grep '^SCAN-JWT' <<<"$scan" | grep -vc " $CTL_PAYLOAD\$" || true)
  local env_hits env_jwt efs_files efs_hits
  env_hits=$(sed -n 's/^SCAN-ENV needle_hits=\([0-9]*\).*/\1/p' <<<"$scan")
  env_jwt=$(sed -n 's/.*jwt_like=\([0-9]*\).*/\1/p' <<<"$scan")
  efs_files=$(sed -n 's/^SCAN-EFS files=\([0-9]*\).*/\1/p' <<<"$scan")
  efs_hits=$(sed -n 's/.*needle_files=\([0-9]*\).*/\1/p' <<<"$scan")
  echo "  agent memory: $((${kb:-0} / 1024)) MB read, $bad ranges refused; the planted control was found ${ctl_hits:-0} time(s); real token tails found ${real_hits:-?}; JWT-shaped strings that are not a planted control: ${other_jwts}"
  echo "  agent environment: ${env_hits:-?} token tails, ${env_jwt:-?} JWT-like values; EFS: ${efs_files:-?} files, ${efs_hits:-?} holding a token tail"
  local needle_count
  needle_count=$(grep -vc '^control' "$needles" || true)
  if [ "${kb:-0}" -gt 0 ] && [ "${bad:-1}" = 0 ] && [ "${ctl_hits:-0}" -ge 1 ] && [ "${real_hits:-1}" = 0 ] && [ "${other_jwts:-1}" = 0 ] \
    && [ "${env_hits:-1}" = 0 ] && [ "${env_jwt:-1}" = 0 ] && [ "${efs_hits:-1}" = 0 ] && [ "$grants_ok" = 1 ] \
    && [ "${log_hits:-1}" = 0 ] && [ "${log_jwt:-1}" = 0 ] && [ "${startup:-0}" -ge 1 ] && [ "${tr_ctl:-0}" -ge 1 ] && [ "$tr_hits" = 0 ] && [ "$tr_jwt" = 0 ]; then
    result 1 "$needle_count real token tails (every access and refresh token the broker handled for the two users) were looked for in the grant, the agent's env, its logs ($log_lines lines), both transcripts, the sealed session files and $((kb / 1024)) MB of its memory: none found. The planted control was found in memory ($ctl_hits) and in the transcript ($tr_ctl), so the scans do see what is there" \
      "$(jq -nc --argjson kb "$kb" --argjson ctl "$ctl_hits" --argjson n "$needle_count" --argjson ll "$log_lines" --argjson tb "$tr_bytes" --argjson trc "$tr_ctl" --argjson ef "${efs_files:-0}" --arg kinds "$kinds" \
        '{real_token_tails_searched:$n, kinds:$kinds, agent_memory:{mb_read:($kb/1024|floor), planted_control_found:$ctl, real_token_hits:0, other_jwt_shaped:0}, env:{token_hits:0, jwt_like:0}, grant:"only a bmc_v1 bearer; no JWT text", agent_log:{lines:$ll, token_hits:0}, transcripts:{bytes:$tb, planted_control_found:$trc, token_hits:0}, efs:{files:$ef, token_hits:0, note:"sealed, so ciphertext either way"}}')" \
      "$(jq -nc '{label:"token copies in the agent",value:0,unit:"found"}')"
  else
    result 0 "kb=$kb bad=$bad control=$ctl_hits real=$real_hits other_jwts=$other_jwts env=$env_hits/$env_jwt efs=$efs_hits grants=$grants_ok log=$log_hits/$log_jwt startup=$startup transcript_ctl=$tr_ctl transcript=$tr_hits/$tr_jwt scan=$(tr '\n' ' ' <<<"$scan" | head -c 400)"
  fi
  rm -f "$needles" "$tails"
}

# ---- 4. the access token lapses ---------------------------------------------------------------------------------

check_token_refresh() {
  local t0 o1 o2 first second jti1 jti2 exp1 exp2 wait_s ref1 ref2 refreshed ttl now_s
  t0=$(now_ms)
  o1=$(chat "$SESS_OB" --prompt "List my notes with your notes tool.")
  note_calls "$SESS_OB" "$o1"
  first=$(oidc_log "$((t0 - 5000))" '^resource_call$' 1 | jq -c --arg s "${OIDC_SUB[bob]}" '[.[]|select(.detail.seen.sub==$s)]|last')
  jti1=$(jq -r '.detail.seen.jti' <<<"$first")
  exp1=$(jq -r '.detail.seen.exp' <<<"$first")
  ttl=$(jq -r '.detail.seen.exp - .detail.seen.iat' <<<"$first")
  # the token lives $ttl seconds and the broker renews it in its last 10: wait until it is well past
  now_s=$(( $(now_ms) / 1000 ))
  wait_s=$((exp1 - now_s + 6))
  [ "$wait_s" -gt 0 ] || wait_s=1
  echo "  call 1 used access token $jti1 (lives ${ttl}s, expires in $((exp1 - now_s))s); waiting ${wait_s}s for it to lapse"
  sleep "$wait_s"
  local t1
  t1=$(now_ms)
  o2=$(chat "$SESS_OB" --prompt "List my notes with your notes tool again.")
  note_calls "$SESS_OB" "$o2"
  second=$(oidc_log "$((t1 - 3000))" '^resource_call$' 1 | jq -c --arg s "${OIDC_SUB[bob]}" '[.[]|select(.detail.seen.sub==$s)]|last')
  jti2=$(jq -r '.detail.seen.jti' <<<"$second")
  exp2=$(jq -r '.detail.seen.exp' <<<"$second")
  refreshed=$(oidc_log "$((t1 - 3000))" '^token_refreshed$' 1 | jq -c --arg c "${OIDC_CONN[bob]}" '[.[]|select(.detail.connection==$c)]')
  ref1=$(grep -o 'PR-[0-9a-f]*' <<<"$(uc_tool_out "$o1" '^mcp__notes__my_notes$')" | head -1)
  ref2=$(grep -o 'PR-[0-9a-f]*' <<<"$(uc_tool_out "$o2" '^mcp__notes__my_notes$')" | head -1)
  echo "  call 2 used access token $jti2 (expires $((exp2 - $(now_ms) / 1000))s from now); broker refresh lines: $(jq length <<<"$refreshed")"
  echo "  both calls answered with bob's reference: $ref1 / $ref2; any error in the model's tool results: $(jq -r '[.tools[]|(.output|tostring)|select(test("error|expired|unauthor";"i"))]|length' <<<"$o2")"
  if [ -n "$jti1" ] && [ "$jti1" != "$jti2" ] && [ "${exp2:-0}" -gt "${exp1:-0}" ] && [ "$(jq length <<<"$refreshed")" -ge 1 ] \
    && [ -n "$ref1" ] && [ "$ref1" = "$ref2" ] && [ "$(jq -r '[.tools[]|select(.name=="mcp__notes__my_notes")]|length' <<<"$o2")" -ge 1 ]; then
    result 1 "bob's first call used access token $jti1 (${ttl}s lifetime); after it lapsed the second call still worked: the broker refreshed (${exp2}>${exp1}), the notes service saw a new token $jti2, and the agent saw only the same answer ($ref2) through the same bmc_v1 token" \
      "$(jq -nc --arg a "$jti1" --arg b "$jti2" --argjson e1 "$exp1" --argjson e2 "$exp2" --argjson ttl "$ttl" --argjson r "$refreshed" --arg ref "$ref2" \
        '{first_token_jti:$a, second_token_jti:$b, first_exp:$e1, second_exp:$e2, access_ttl_s:$ttl, broker_refresh:$r, same_answer_ref:$ref, agent_visible_difference:"none"}')" \
      "$(jq -nc --argjson ttl "$ttl" '{label:"access token lifetime",value:$ttl,unit:"s"}')"
  else
    result 0 "jti1=$jti1 jti2=$jti2 exp1=$exp1 exp2=$exp2 refreshed=$refreshed ref1=$ref1 ref2=$ref2"
  fi
}

# ---- 5. two users, two sessions ---------------------------------------------------------------------------------

check_users_isolated() {
  local t0 oa ob ta tb ra rb calls ok=1 detail='[]' p
  t0=$(now_ms)
  oa=$(chat "$SESS_OA" --prompt "List my notes with your notes tool.")
  ob=$(chat "$SESS_OB" --prompt "List my notes with your notes tool.")
  note_calls "$SESS_OA" "$oa"
  note_calls "$SESS_OB" "$ob"
  ta=$(uc_tool_out "$oa" '^mcp__notes__my_notes$')
  tb=$(uc_tool_out "$ob" '^mcp__notes__my_notes$')
  ra=$(grep -o 'PR-[0-9a-f]*' <<<"$ta" | head -1)
  rb=$(grep -o 'PR-[0-9a-f]*' <<<"$tb" | head -1)
  echo "  alice's session saw: ${ta:0:110}"
  echo "  bob's session saw:   ${tb:0:110}"
  [[ $ta == *"(alice)"* && $ta != *"bob"* && $ta != *"$rb"* && $tb == *"(bob)"* && $tb != *"alice"* && $tb != *"$ra"* && -n $ra && -n $rb && $ra != "$rb" ]] || ok=0
  # the broker's own view: every call of a session reached that session's user's connection
  # (the log lines reach CloudWatch a few seconds late: wait until both sessions' tool calls are in)
  for _ in $(seq 1 25); do
    calls=$(oidc_log "$((t0 - 5000))" '^managed_call$' 2 | jq -c '[.[]|.detail|{session, connection, sub, method}]')
    [ "$(jq -r --arg a "$SESS_OA" --arg b "$SESS_OB" '([.[]|select(.method=="tools/call" and .session==$a)]|length) > 0 and ([.[]|select(.method=="tools/call" and .session==$b)]|length) > 0' <<<"$calls")" = true ] && break
    sleep 4
  done
  jq -e --arg a "$SESS_OA" --arg b "$SESS_OB" --arg ca "${OIDC_CONN[alice]}" --arg cb "${OIDC_CONN[bob]}" \
    '([.[]|select(.session==$a)] | length > 0 and all(.[]; .connection == $ca)) and ([.[]|select(.session==$b)] | length > 0 and all(.[]; .connection == $cb))' <<<"$calls" > /dev/null || ok=0
  echo "  broker: every call of alice's session used ${OIDC_CONN[alice]}, bob's ${OIDC_CONN[bob]}: $(jq -r --arg a "$SESS_OA" '[.[]|select(.session==$a)]|length' <<<"$calls") / $(jq -r --arg b "$SESS_OB" '[.[]|select(.session==$b)]|length' <<<"$calls") calls (handshake included)"

  # What alice's session could try with the one credential it has: its own token. Each attempt is made raw,
  # against the broker, as a compromised agent would.
  probe() { # label expected-status expected-text session path header body
    local out status body good=1
    out=$(oidc_probe "$4" "$5" "$6" "$7")
    status=${out%% *}
    body=${out#* }
    [ "$status" = "$2" ] || good=0
    [[ $body == *"$3"* ]] || good=0
    [[ $body != *"${OIDC_SUB[bob]}"* && $body != *"$rb"* ]] || good=0
    echo "    $1 -> HTTP $status $(head -c 110 <<<"$body")  [$([ $good = 1 ] && echo refused || echo WRONG)]"
    [ "$good" = 1 ] || ok=0
    detail=$(jq -c --arg l "$1" --arg s "$status" --arg b "${body:0:160}" --argjson g "$good" '. + [{attempt:$l, http:($s|tonumber), answer:$b, refused:($g==1)}]' <<<"$detail")
  }
  echo "  alice's session token, used to reach bob's connection:"
  probe "names bob's connection in a header" 403 "not bound to that connection" "$SESS_OA" /managed/notes/mcp "x-beyond-connection: ${OIDC_CONN[bob]}" "$RPC_MY_NOTES"
  probe "names bob's connection in the URL" 403 "not bound to that connection" "$SESS_OA" "/managed/notes/mcp?connection=${OIDC_CONN[bob]}" "" "$RPC_MY_NOTES"
  probe "a valid token for a session nobody connected" 200 "nothing to act as" "demo.u${SID}z" /managed/notes/mcp "" "$RPC_MY_NOTES"
  probe "the token, aimed at another connector" 401 "unauthorized" "$SESS_OA" /managed/other/mcp "" "$RPC_MY_NOTES"
  p=$(oidc_ctl POST /sessions/bind "$(jq -nc --arg t "$UC_TENANT" --arg s "$SESS_OA" --arg c "${OIDC_CONN[bob]}" '{tenant:$t,session:$s,connector:"notes",connection:$c}')")
  echo "    re-binding alice's session to bob's connection -> $(jq -c . <<<"$p")  [$(jq -e '.error != null' <<<"$p" > /dev/null && echo refused || echo WRONG)]"
  jq -e '.error != null' <<<"$p" > /dev/null || ok=0
  # and through the model: ask for bob's notes by name
  local oc tc
  oc=$(chat "$SESS_OA" --prompt "Call my_notes with the owner argument set to bob, and tell me what it says.")
  note_calls "$SESS_OA" "$oc"
  tc=$(uc_tool_out "$oc" '^mcp__notes__my_notes$')
  echo "    the model asked the service for bob's notes -> ${tc:0:150}"
  [[ $tc == *"only read your own notes"* && $tc != *"$rb"* && $(jq -r .text <<<"$oc") != *"$rb"* ]] || ok=0
  if [ "$ok" = 1 ]; then
    result 1 "alice's session saw only alice's notes ($ra) and bob's only bob's ($rb); every broker call of a session used that user's connection; alice's own token could not reach bob's connection by any route (named in a header or the URL, a token for an unconnected session, another connector, re-binding), and the service itself refused a request for another user's notes" \
      "$(jq -nc --arg ra "$ra" --arg rb "$rb" --argjson c "$calls" --argjson d "$detail" '{alice_reference:$ra, bob_reference:$rb, broker_calls:$c, attempts_with_alices_token:$d}')" \
      "$(jq -nc --argjson n "$(jq length <<<"$detail")" '{label:"cross-user attempts refused",value:($n+2),unit:""}')"
  else
    result 0 "ra=$ra rb=$rb calls=$calls detail=$detail model=${tc:0:200}"
  fi
}

# ---- 6. scope ----------------------------------------------------------------------------------------------------

check_scope_enforced() {
  local t0 out text tool refused after ev
  t0=$(now_ms)
  out=$(chat "$SESS_OA" --prompt "Use your add_note tool to add the note: buy milk. Then say exactly what happened.")
  note_calls "$SESS_OA" "$out"
  text=$(jq -r .text <<<"$out")
  tool=$(jq -r '[.tools[]|select(.name=="mcp__notes__add_note")][0].output // "" | tostring' <<<"$out")
  echo "  add_note answered: ${tool:0:200}"
  for _ in $(seq 1 15); do
    ev=$(oidc_log "$((t0 - 5000))" '^resource_refused$' 1 | jq -c '[.[]|select(.detail.why=="scope")|.detail]')
    [ "$(jq length <<<"$ev")" -ge 1 ] && break
    sleep 4
  done
  echo "  the notes service refused it itself: $ev"
  after=$(chat "$SESS_OA" --prompt "List my notes with your notes tool.")
  note_calls "$SESS_OA" "$after"
  refused=$([[ $tool == *insufficient_scope* && $tool == *notes:write* ]] && echo yes || echo no)
  echo "  reply to the user: ${text:0:200}"
  if [ "$refused" = yes ] && [ "$(jq length <<<"$ev")" -ge 1 ] && [ "$(jq -r --arg s "${OIDC_SUB[alice]}" 'all(.[]; .sub == $s and .needs == "notes:write" and (.scope|index("notes:write")) == null)' <<<"$ev")" = true ] \
    && [[ $text == *write* || $text == *scope* || $text == *permission* || $text == *"not granted"* || $text == *refus* || $text == *"not allowed"* ]] \
    && [[ $(uc_tool_out "$after" '^mcp__notes__my_notes$') != *"buy milk"* ]]; then
    result 1 "add_note needs notes:write; alice's connection was granted only notes:read. The notes service (not the broker, not the agent) refused it for the missing scope, the tool returned \"insufficient_scope… notes:write… not granted\", the model told the user, and no note was written (reading back shows none)" \
      "$(jq -nc --arg t "${tool:0:300}" --argjson ev "$ev" --arg r "${text:0:300}" '{tool_error:$t, service_refusal:$ev, model_reply:$r, note_written:false}')" \
      '{"label":"add_note","value":"refused upstream","unit":""}'
  else
    result 0 "refused=$refused ev=$ev reply=${text:0:200} tool=${tool:0:200}"
  fi
}

# ---- 7. revocation -----------------------------------------------------------------------------------------------

check_revocation() {
  local t0 rev out text tool state hi bob_out audit
  t0=$(now_ms)
  rev=$(oidc_ctl POST "/connections/${OIDC_CONN[alice]}/revoke")
  echo "  the user revokes alice's connection: $(jq -c . <<<"$rev")"
  out=$(chat "$SESS_OA" --prompt "List my notes with your notes tool.")
  note_calls "$SESS_OA" "$out"
  text=$(jq -r .text <<<"$out")
  tool=$(jq -r '[.tools[]|select(.name=="mcp__notes__my_notes")][0].output // "" | tostring' <<<"$out")
  echo "  next call: ${tool:0:230}"
  echo "  reply: ${text:0:230}"
  # the session survives: it is still there, and it answers
  state=$(chat "$SESS_OA" --command '{"type":"get_state","id":"gs"}' --timeout-secs 40 | jq -r '[.responses[]|select(.command=="get_state")][0].success')
  hi=$(chat "$SESS_OA" --prompt "Do not use any tool. Reply with the single word: alive.")
  note_calls "$SESS_OA" "$hi"
  bob_out=$(chat "$SESS_OB" --prompt "List my notes with your notes tool.")
  note_calls "$SESS_OB" "$bob_out"
  audit=$(oidc_ctl GET "/connections/${OIDC_CONN[alice]}?audit=1")
  echo "  session state ok: $state; it still answers: $(jq -r .text <<<"$hi" | head -c 60); bob's session still works: $(uc_tool_out "$bob_out" '^mcp__notes__my_notes$' | head -c 60)"
  echo "  stored after revocation: $(jq -c '{status, refresh_token_stored, at_rest}' <<<"$audit")"
  if [ "$(jq -r .upstream_revoked <<<"$rev")" = true ] && [ "$(jq -r .refresh_after_revoke <<<"$rev")" = invalid_grant ] \
    && [[ $tool == *"no longer active"* && $tool == *revoked* ]] && [[ ${text,,} == *revok* || ${text,,} == *"no longer"* || ${text,,} == *"sign in"* ]] \
    && [ "$state" = true ] && [[ $(jq -r .text <<<"$hi" | tr 'A-Z' 'a-z') == *alive* ]] \
    && [[ $(uc_tool_out "$bob_out" '^mcp__notes__my_notes$') == *"(bob)"* ]] \
    && [ "$(jq -r '.status' <<<"$audit")" = revoked ] && [ "$(jq -r '.refresh_token_stored' <<<"$audit")" = false ]; then
    result 1 "the user revoked: the broker asked the provider to revoke the refresh token (RFC 7009) and proved it dead (using it gets invalid_grant), then erased the stored copy; alice's next call returned a plain message that the connection was revoked and the model relayed it; her session kept running (it answered afterwards), and bob's connection was untouched" \
      "$(jq -nc --argjson rev "$rev" --arg t "${tool:0:300}" --arg r "${text:0:300}" --argjson a "$audit" '{revocation:$rev, tool_message:$t, model_reply:$r, stored_after:{status:$a.status, refresh_token_stored:$a.refresh_token_stored}, session_survived:true, other_user_unaffected:true}')" \
      '{"label":"next call after revoke","value":"clear message","unit":""}'
  else
    result 0 "rev=$rev tool=${tool:0:200} reply=${text:0:200} state=$state hi=$(jq -r .text <<<"$hi" | head -c 80) audit=$audit"
  fi
}
