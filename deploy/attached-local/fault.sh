#!/usr/bin/env bash
# The fault command the simulator invokes: `fault.sh <action> <replica>`.
#
# This is the whole reason the simulator has a `--fault-cmd` seam. Here the actions are local
# mechanisms; the AWS version does the same five things with NACLs and ECS StopTask, and the
# simulator cannot tell the difference. A claim proved through one is proved through the other.
set -euo pipefail
STATE=${STATE:-/var/tmp/fleet-attached}
# Matches up.sh: a replica must never read a real home directory.
AGENT_HOME=${AGENT_HOME:-/nonexistent-fleet-sim-home}
action=$1; replica=$2
# The held-back capped replica sits after the pool and is named `capped`, not `r4`.
if [ "$replica" = "capped" ]; then i=$((${REPLICAS:-3} + 1)); else i=${replica#r}; fi
c=$((i - 1)); ns="algate$c"; cif="alc$c"
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
  # The same signal, waited on. `term` is defined by the replica still being there; `stop` by it
  # being gone, so they cannot be the same word.
  stop)
    p=$(cat "$pidfile" 2>/dev/null || true)
    [ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true
    for _ in $(seq 1 120); do
      [ -n "$p" ] && kill -0 "$p" 2>/dev/null || break
      sleep 0.5
    done
    exit 0 ;;
  # Link state, not a firewall rule: it stops traffic *and* lease renewal, which is what a lost
  # mount target does, and it cuts an already-established connection.
  partition) sudo -n ip netns exec "$ns" ip link set "$cif" down ;;
  heal)      sudo -n ip netns exec "$ns" ip link set "$cif" up ;;
  restart)
    # Idempotent, and that is not a nicety here: the simulator restores the fleet before every
    # scenario, so `restart` is asked of replicas that are dead, alive, and — after the drain
    # scenario — still finishing what they owned. Starting a second one on top of a draining first
    # would fail to bind, and the readiness probe below would then pass against the *old* process.
    # So: stop whatever is there, wait for the port, then start.
    port=$((18100 + i))
    p=$(cat "$pidfile" 2>/dev/null || true)
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true
    for _ in $(seq 1 80); do
      ss -ltn 2>/dev/null | grep -q ":$port " || break
      sleep 0.25
    done
    # `exec` inside an explicit subshell, so `$!` is the **agent's** pid and not a shell that
    # happens to be its parent. Without the `exec`, `$!` was the subshell: `kill` then killed the
    # shell and left the agent serving, so a scenario that restarts a replica to force a cold open
    # got a replica that had never closed — its sessions still live in memory, still readable with
    # the DEK they were opened under. That surfaced as C10 reporting a *wrong tenant key had read
    # another tenant's content*, which is about as alarming as a harness bug can look, and as C6
    # reporting that a takeover never advanced the epoch, because the owner it killed was not dead.
    #
    # HOME is restored here too: the replica must never fall back to a real home directory, and the
    # saved command line holds arguments only.
    ( eval "HOME=$AGENT_HOME exec $(cat "$STATE/cmd-$replica")" >/dev/null 2>>"$STATE/$replica.log" ) &
    echo $! > "$pidfile"
    # Readiness, not liveness: a replica answers /livez before its shards are usable.
    for _ in $(seq 1 120); do
      curl -sf -o /dev/null "http://127.0.0.1:$port/readyz" 2>/dev/null && break
      sleep 0.25
    done ;;
  # Where it answers now. Locally that is always the same port — the address is in the saved command
  # line — but the simulator asks rather than assumes, because on ECS it is not.
  address)
    echo "127.0.0.1:$((18100 + i))" ;;
  *) echo "unknown action $action" >&2; exit 2 ;;
esac
