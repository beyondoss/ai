#!/usr/bin/env bash
# Stand up a fleet the simulator does NOT own, on this machine.
#
# Same shape as the EFS proof — pre-existing mounts, replicas started by somebody else, faults
# performed by a script — but on the local NFS substrate and for free. It exists to prove the
# `--substrate attached` path with the same nine scenarios *before* any of it depends on AWS, so a
# failure there is about EFS rather than about the harness.
#
# Mirrors crates/fleet-sim/src/substrate.rs's `prepare_nfs`: one netns + UTS namespace per replica,
# because a Linux NFSv4 client identifies itself by its UTS hostname and two mounts from one host
# would otherwise be ONE client sharing ONE lease — which makes partitioning a single replica
# impossible. See that file for the full argument.
set -euo pipefail

REPLICAS=${REPLICAS:-3}
# One more client than there are replicas in the pool: the last is started with
# `--max-live-sessions 1` and kept out of the pool, because the cap is fixed when a replica starts
# and one inside the pool would refuse every other scenario's second session. See `--capped-replica`.
TOTAL=$((REPLICAS + 1))
SHARDS=${SHARDS:-1}
STATE=${STATE:-/var/tmp/fleet-attached}
BIN=${BIN:-/home/jared/ai/target/debug/beyond-ai-agent}
SIM=${SIM:-/home/jared/ai/target/debug/fleet-sim}
MOCK=${MOCK:-127.0.0.1:19000}
OCTET=$((66 + RANDOM % 60))

[ -x "$BIN" ] || { echo "no agent binary at $BIN (cargo build -p beyond-ai-agent)" >&2; exit 1; }

# Refuse to start on top of a previous fleet, loudly.
#
# Silently proceeding is worse than any failure here: the new replicas fail to bind with AddrInUse,
# the health check then passes against the *old* ones — still serving mounts that no longer exist,
# with sessions still live in memory — and the matrix grades claims against a fleet nobody intended.
# That produced a false C10 isolation violation before this check existed.
for i in $(seq 1 "$TOTAL"); do
  for port in $((18100 + i)) $((18200 + i)); do
    if ss -ltn 2>/dev/null | grep -q ":$port "; then
      echo "port $port is already listening — run down.sh first" >&2
      exit 1
    fi
  done
done
mkdir -p "$STATE"/{mnt,keys}
EXPORT="$STATE/export"
mkdir -p "$EXPORT"
echo "$OCTET" > "$STATE/octet"

# Keys first: the replicas need them before the simulator exists.
eval "$("$SIM" keys --dir "$STATE/keys")"
echo "grant key: $AI_AGENT_GRANT_KEY"

for s in $(seq 1 "$SHARDS"); do mkdir -p "$EXPORT/s$s"; done
# One more shard that exists on the storage and is deliberately **not** mounted by any replica.
# That is the whole of what the misdirection scenario needs: a session homed somewhere this replica
# cannot serve. It costs a directory.
UNMOUNTED="s$((SHARDS + 1))"
mkdir -p "$EXPORT/$UNMOUNTED"
sudo -n exportfs -o rw,sync,no_subtree_check,no_root_squash,insecure,fsid=$((RANDOM + 1000)) "10.0.0.0/8:$EXPORT"

for i in $(seq 1 "$TOTAL"); do
  c=$((i - 1)); ns="algate$c"; hif="alh$c"; cif="alc$c"
  hip="10.$OCTET.$c.1"; cip="10.$OCTET.$c.2"
  sudo -n ip netns add "$ns"
  sudo -n ip link add "$hif" type veth peer name "$cif"
  sudo -n ip link set "$cif" netns "$ns"
  sudo -n ip addr add "$hip/30" dev "$hif"; sudo -n ip link set "$hif" up
  sudo -n ip netns exec "$ns" ip addr add "$cip/30" dev "$cif"
  sudo -n ip netns exec "$ns" ip link set "$cif" up
  sudo -n ip netns exec "$ns" ip link set lo up

  shard_args=()
  for s in $(seq 1 "$SHARDS"); do
    mnt="$STATE/mnt/c$c/s$s"; mkdir -p "$mnt"
    # Mount inside net+UTS, OUTSIDE any new mount namespace, so the mount is visible to an ordinary
    # process here while its RPC still travels through this replica's own namespace.
    sudo -n nsenter --net=/var/run/netns/"$ns" unshare --uts sh -c \
      "hostname $ns && mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport $hip:$EXPORT/s$s $mnt"
    shard_args+=(--shard "s$s=$mnt")
  done

  port=$((18100 + i))
  # A pool replica publishes a scrape and has no session cap; the held-back one is the opposite.
  # Locally the scrape needs no sidecar — the simulator is on this host, so loopback is reachable —
  # but the flag it is passed to the matrix through is the same one an ECS sidecar's address goes in.
  # The name the simulator knows this replica by, and the one `fault.sh` is invoked with. The
  # held-back one is `capped`, not `r4`: a fault aimed at it must not be able to land on a pool
  # member by arithmetic.
  if [ "$i" -le "$REPLICAS" ]; then
    name="r$i"
    extra=(--metrics-listen "127.0.0.1:$((18200 + i))")
  else
    name="capped"
    extra=(--max-live-sessions 1)
  fi
  cmd=("$BIN" serve --service --gateway-url "http://$MOCK" --model claude-test
       --listen "127.0.0.1:$port" --grant-key "$AI_AGENT_GRANT_KEY"
       --seal-key "$AI_AGENT_SEAL_KEY" "${shard_args[@]}" --drain-grace 30 "${extra[@]}")
  printf '%q ' "${cmd[@]}" > "$STATE/cmd-$name"
  HOME=/nonexistent-fleet-sim-home "${cmd[@]}" >/dev/null 2>"$STATE/$name.log" &
  echo $! > "$STATE/pid-$name"
  if [ "$i" -le "$REPLICAS" ]; then
    echo "127.0.0.1:$port" >> "$STATE/replicas"
    echo "$((18200 + i))" >> "$STATE/metrics"
  else
    echo "127.0.0.1:$port=1" > "$STATE/capped"
  fi
done

# Wait for every replica to answer before claiming the fleet is up.
for i in $(seq 1 "$TOTAL"); do
  port=$((18100 + i))
  for _ in $(seq 1 80); do
    if curl -sf -o /dev/null "http://127.0.0.1:$port/livez" 2>/dev/null; then break; fi
    sleep 0.25
  done
done

echo
echo "fleet up. run the matrix with:"
echo -n "  $SIM matrix --substrate attached --fault-cmd $(dirname "$0")/fault.sh --mock-listen $MOCK"
for s in $(seq 1 "$SHARDS"); do echo -n " --shard s$s=$EXPORT/s$s"; done
echo -n " --shard $UNMOUNTED=$EXPORT/$UNMOUNTED"
while read -r a; do echo -n " --replica $a"; done < "$STATE/replicas"
while read -r a; do echo -n " --replica-metrics $a"; done < "$STATE/metrics"
echo -n " --capped-replica $(cat "$STATE/capped")"
echo
