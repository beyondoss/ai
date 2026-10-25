#!/usr/bin/env bash
# Prove that one agent replica stays bounded and dense when many tenants share it, on the real AWS stack.
#
#     ./deploy/demo/load.sh up      # one Fargate agent task (the demo's size) + a scripted-model/bundle-host task
#     ./deploy/demo/load.sh run     # the waves, the cold-start burst, the 200-server .mcp.json; a record and a report
#     ./deploy/demo/load.sh down    # delete it all and prove it (down.sh), into the same record
#     ./deploy/demo/load.sh all     # up, run, down
#
# It spends nothing on a model: the replica's model is a scripted stand-in (`beyond-ai-demo load-host`)
# that answers instantly. There is no gateway, no sandbox and no provider key, so the load lands on the
# replica alone. The stack is named `load` (DEMO_NAME to change) and shares nothing with another demo.
#
# What the run does (sizes are flags of `beyond-ai-demo load`, defaults below):
#   - 150 bundles (50 tenants, several versions each, 10 of them with 5 MB of skills) uploaded to the bundle
#     host, and 100 more for the burst;
#   - two waves: ramp to 500 sessions open at once (60% long-lived, 40% churning through the bundles so the
#     cache has something to evict), a few turns each, every turn reading a skill; hold; close all; idle 3 min;
#   - the cold-start burst: 200 sessions on 100 bundles nobody has opened, all at once;
#   - one tenant whose .mcp.json lists 200 servers, with the grant limit unset and at 5.
# The replica is read from inside its task by a sampler container (sampler.sh): memory, threads, fds, the
# bundle cache on disk and the sockets, every 2 seconds. load-analyze.py joins that with what the client saw.
set -euo pipefail
export DEMO_NAME=${DEMO_NAME:-load} SANDBOX=load
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

LOAD_CACHE_MB=${LOAD_CACHE_MB:-16}
LOAD_STORE=${LOAD_STORE:-efs} # efs (the default) or local: the session store on the task's ephemeral disk
export LOAD_CACHE_MB LOAD_STORE
SESSIONS=${LOAD_SESSIONS:-500}
BURST=${LOAD_BURST:-200}
RAMP=${LOAD_RAMP_SECS:-60}
HOLD=${LOAD_HOLD_SECS:-90}
IDLE=${LOAD_IDLE_SECS:-180}
WAVES=${LOAD_WAVES:-2}

# The teardown goes into the load record only: down.sh would otherwise fill every record waiting for one,
# and rebuild it with the demo's page.
merge_teardown() { # teardown json file
  local rec
  for rec in "$HERE"/results/*-load/run.json; do
    [ -f "$rec" ] || continue
    [ "$(jq -r '.teardown == null' "$rec")" = true ] || continue
    jq --slurpfile t "$1" '.teardown = $t[0]
      | .checks |= map(if .name == "teardown-zero" then
          .pass = true | .status = "pass" | .detail = ("every resource type counted 0 at " + $t[0].verified_at)
          | .headline = {label: "left behind", value: 0, unit: "resources"} else . end)' "$rec" > "$rec.tmp" && mv "$rec.tmp" "$rec"
    "$HERE/report.sh" "$(dirname "$rec")" >/dev/null
    echo "teardown recorded in $rec"
  done
}

cmd_up() { "$HERE/up.sh"; }

cmd_down() {
  local tmp rc=0
  tmp=$(mktemp -d)
  DEMO_RESULTS=$tmp DEMO_TEARDOWN_OUT=$tmp/teardown.json "$HERE/down.sh" || rc=$?
  if [ "$rc" = 0 ] && [ -s "$tmp/teardown.json" ]; then merge_teardown "$tmp/teardown.json"; fi
  rm -rf "$tmp"
  return $rc
}

cmd_run() {
  local work run_id dir ip lh bin target start_ms end_ms
  work=$(mktemp -d)
  trap 'rm -rf "$work"' RETURN
  ip=$(agent_ip)
  lh=$(svc_ip loadhost)
  [ -n "$ip" ] && [ -n "$lh" ] || { echo "load: the stack is not up (run ./deploy/demo/load.sh up)" >&2; return 1; }

  (cd "$ROOT" && cargo build -q --release -p beyond-ai-demo)
  target=$(cd "$ROOT" && cargo metadata --format-version 1 --no-deps | jq -r .target_directory)
  bin=$target/release/beyond-ai-demo

  DEMO_GRANT_SEED=$(ssm_get grant-seed)
  DEMO_FLEET_SECRET=$(ssm_get fleet-secret)
  DEMO_DEK=$(ssm_get dek)
  DEMO_GATEWAY_KEY=$(ssm_get gateway-key)
  DEMO_LOAD_TOKEN=$(ssm_get load-token)
  export DEMO_GRANT_SEED DEMO_FLEET_SECRET DEMO_DEK DEMO_GATEWAY_KEY DEMO_LOAD_TOKEN
  ulimit -n 16384 2>/dev/null || true

  run_id=$(date -u +%Y%m%dT%H%M%SZ)
  dir=$HERE/results/$run_id-load
  mkdir -p "$dir"
  say "load: $SESSIONS sessions x $WAVES waves, burst $BURST, idle ${IDLE}s, agent $ip, bundle host $lh:$LOADHOST_PORT"
  start_ms=$(($(date +%s) * 1000))
  "$bin" load --host "$ip" --port "$AGENT_PORT" --loadhost "$lh:$LOADHOST_PORT" --out "$work/driver.json" \
    --sessions "$SESSIONS" --burst "$BURST" --ramp-secs "$RAMP" --hold-secs "$HOLD" --idle-secs "$IDLE" --waves "$WAVES" \
    2> >(tee "$work/driver.log" >&2) >/dev/null
  end_ms=$(($(date +%s) * 1000))

  say "the sampler's log, from the replica's task"
  sleep 5 # CloudWatch lags a few seconds
  command aws --region "$REGION" --output json logs filter-log-events --log-group-name "$LOG_GROUP" \
    --log-stream-name-prefix sampler --filter-pattern LOAD --start-time "$((start_ms - 120000))" --end-time "$((end_ms + 15000))" \
    | jq -r '.events[].message | select(startswith("LOAD ")) | sub("^LOAD "; "")' > "$work/samples.jsonl"
  echo "  $(wc -l < "$work/samples.jsonl") samples"

  say "the replica's own OPEN lines (one per phase half of every open)"
  command aws --region "$REGION" --output json logs filter-log-events --log-group-name "$LOG_GROUP" \
    --log-stream-name-prefix agent --filter-pattern OPEN --start-time "$((start_ms - 120000))" --end-time "$((end_ms + 15000))" \
    | jq -r '.events[].message | select(startswith("OPEN "))' > "$work/opens.log"
  echo "  $(wc -l < "$work/opens.log") lines"

  jq -n --arg commit "$(git -C "$ROOT" rev-parse --short HEAD)" --arg region "$REGION" --arg name "$NAME" \
    --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson budget "$LOAD_CACHE_MB" --arg store "$LOAD_STORE" \
    '{commit:$commit, region:$region, name:$name, started_at:$at, cache_budget_mb:$budget, store:$store,
      task:{vcpu:0.5, memory_mb:1024, flags:"serve --service --bundle-cache-mb \($budget)"}}' > "$work/meta.json"
  python3 -I "$HERE/load-analyze.py" --driver "$work/driver.json" --samples "$work/samples.jsonl" --opens "$work/opens.log" --meta "$work/meta.json" > "$dir/run.json"
  # What the analysis was made from, for re-running it: the client's per-open rows and the replica's lines.
  cp "$work/driver.json" "$work/opens.log" "$dir/"
  "$HERE/report.sh" "$dir"
  jq -r '.checks[] | "  \(.status | ascii_upcase)  \(.name): \(.detail)"' "$dir/run.json"
  echo "load: record $dir/run.json"
}

case "${1:-run}" in
  up) cmd_up ;;
  run) cmd_run ;;
  down) cmd_down ;;
  all) cmd_up && cmd_run; cmd_down ;;
  *) echo "usage: load.sh up|run|down|all" >&2; exit 2 ;;
esac
