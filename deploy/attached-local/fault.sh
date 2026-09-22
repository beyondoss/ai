#!/usr/bin/env bash
# The fault command the simulator invokes: `fault.sh <action> <replica>`.
#
# This is the whole reason the simulator has a `--fault-cmd` seam. Here the actions are local
# mechanisms; the AWS version does the same five things with NACLs and ECS StopTask, and the
# simulator cannot tell the difference. A claim proved through one is proved through the other.
set -euo pipefail
STATE=${STATE:-/var/tmp/fleet-attached}
action=$1; replica=$2
i=${replica#r}; c=$((i - 1)); ns="algate$c"; cif="alc$c"
pidfile="$STATE/pid-$replica"

# Best-effort actions exit 0 explicitly. Left as `[ -f x ] && kill ...`, the list's status leaks out
# as the script's, so asking to kill a replica that is already dead reported a *fault failure* — and
# the scenario then reported "could not stop the replica", which is a harness error wearing a
# claim's clothes.
case "$action" in
  kill)
    p=$(cat "$pidfile" 2>/dev/null || true)
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true
    rm -f "$pidfile"
    exit 0 ;;
  term)
    p=$(cat "$pidfile" 2>/dev/null || true)
    [ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true
    exit 0 ;;
  # Link state, not a firewall rule: it stops traffic *and* lease renewal, which is what a lost
  # mount target does, and it cuts an already-established connection.
  partition) sudo -n ip netns exec "$ns" ip link set "$cif" down ;;
  heal)      sudo -n ip netns exec "$ns" ip link set "$cif" up ;;
  restart)
    # Back at the SAME address: an attached replica's address belongs to whoever runs it, so the
    # ring does not change. That is also true of an ECS task replaced in place.
    eval "$(cat "$STATE/cmd-$replica")" >/dev/null 2>>"$STATE/$replica.log" &
    echo $! > "$pidfile"
    port=$((18100 + i))
    for _ in $(seq 1 80); do
      curl -sf -o /dev/null "http://127.0.0.1:$port/livez" 2>/dev/null && break
      sleep 0.25
    done ;;
  *) echo "unknown action $action" >&2; exit 2 ;;
esac
