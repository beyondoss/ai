#!/usr/bin/env bash
# Give the machine back. Same order the simulator's own teardown uses, for the same reason: bring
# every link UP before unmounting, because an NFS client cannot finish unmounting without reaching
# its server, and unmounting a partitioned one leaves a state-manager kernel thread wedged forever.
set -uo pipefail
STATE=${STATE:-/var/tmp/fleet-attached}
# Pidfiles first, then anything still holding this state dir open.
#
# Pidfiles alone are not enough: a replica the matrix killed and restarted rewrites its pidfile, and
# any sequence that loses track leaves a process behind. One did — still listening on a replica port
# 1034 s later, answering /livez against mounts that had been deleted under it, which made the next
# run's `attach` health check pass against a stale replica whose session was still in memory. C10
# then reported that a wrong-key grant had read the session content: a false isolation violation
# produced entirely by a leaked process.
for f in "$STATE"/pid-*; do [ -f "$f" ] && kill -9 "$(cat "$f")" 2>/dev/null; done
for p in $(pgrep -f "seal-key $STATE" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 1
for ns in $(sudo -n ip netns list 2>/dev/null | awk '{print $1}' | grep '^algate' || true); do
  sudo -n ip netns exec "$ns" ip link set "${ns/algate/alc}" up 2>/dev/null
done
for m in $(mount | grep "$STATE" | awk '{print $3}'); do
  sudo -n umount "$m" 2>/dev/null || sudo -n umount -f -l "$m" 2>/dev/null
done
for ns in $(sudo -n ip netns list 2>/dev/null | awk '{print $1}' | grep '^algate' || true); do
  sudo -n ip link del "${ns/algate/alh}" 2>/dev/null
  sudo -n ip netns del "$ns" 2>/dev/null
done
sudo -n exportfs -u "10.0.0.0/8:$STATE/export" 2>/dev/null
rm -rf "$STATE"
echo "down: mounts=$(mount|grep -c nfs4) netns=$(sudo -n ip netns list 2>/dev/null|wc -l) dstate=$(ps -eo stat --no-headers|awk '$1 ~ /D/'|wc -l)"
