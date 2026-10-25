# The load proof's sampler: one JSON line every $INTERVAL seconds about the agent process in the same
# task (shared process namespace and network). Busybox sh and awk only. Run by the task definition in
# load-tasks.sh; set CACHE_DIR, LOADHOST_PORT, WS_PORT, METRICS_URL and PROC to run it elsewhere.
INTERVAL=${INTERVAL:-2}
CACHE_DIR=${CACHE_DIR:-/scratch/tmp/agent-bundles}
WS_PORT=${WS_PORT:-8080}
METRICS_URL=${METRICS_URL:-http://127.0.0.1:9100/metrics}
PROC=${PROC:-/proc}
AGENT_CMD=${AGENT_CMD:-/usr/local/bin/beyond-ai-agent}  # argv[0] of the process to watch, not the shell that starts it
[ ! -d /scratch ] || chmod 1777 /scratch
LHEX=$(printf '%04X' "$LOADHOST_PORT")
WHEX=$(printf '%04X' "$WS_PORT")
while :; do
  pid=
  for d in "$PROC"/[0-9]*; do
    case "$({ tr '\0' ' ' < "$d/cmdline"; } 2>/dev/null)" in
      "$AGENT_CMD"\ serve*) pid=${d#"$PROC"/}; break ;;
    esac
  done
  if [ -n "$pid" ]; then
    st=$(awk '/^(VmRSS|RssAnon|RssFile|VmHWM|Threads):/ { sub(":", "", $1); printf "\"%s\":%s,", $1, $2 }' "$PROC/$pid/status")
    cpu=$(awk '{ print $14 + $15 }' "$PROC/$pid/stat" 2>/dev/null)
    huge=$(awk '/^AnonHugePages:/ { print $2 }' "$PROC/$pid/smaps_rollup" 2>/dev/null)
    fds=$(ls "$PROC/$pid/fd" 2>/dev/null | wc -l)
    net=$(awk -v l="$LHEX" -v w="$WHEX" 'NR > 1 && $4 == "01" { split($3, r, ":"); split($2, a, ":"); if (r[2] == l) out++; if (a[2] == w) ws++ }
      END { printf "\"to_loadhost\":%d,\"ws_established\":%d", out, ws }' "$PROC/net/tcp")
    kb=$(du -sk "$CACHE_DIR" 2>/dev/null | cut -f1)
    dirs=$(find "$CACHE_DIR" -mindepth 2 -maxdepth 2 -type d ! -name '.*' 2>/dev/null | wc -l)
    trash=$(find "$CACHE_DIR" -mindepth 2 -maxdepth 2 -type d -name '.trash-*' 2>/dev/null | wc -l)
    # The gauges, plus the blocking pool (agent_blocking_*, one key per site and bucket as
    # `<what>|<site>[|<le>]`, zeros left out) and the runtime's worst scheduling lag since the last scrape.
    m=$(wget -q -T 2 -O - "$METRICS_URL" 2>/dev/null | awk '
      /^agent_(sessions_live|threads|runs_in_flight|runtime_lag_max_seconds) / { printf "\"%s\":%s,", $1, $2 }
      /^agent_blocking_/ && $1 !~ /_count[{]/ && $2 != 0 {
        k = $1; sub(/^agent_blocking_/, "", k); sub(/_seconds/, "", k)
        gsub(/[{]site="/, "|", k); gsub(/",le="/, "|", k); gsub(/"[}]/, "", k)
        printf "\"%s\":%s,", k, $2
      }')
    printf 'LOAD {"t":%s000,"pid":%s,%s%s"cpu_ticks":%s,"AnonHugePages":%s,"fds":%s,"cache_kb":%s,"cache_dirs":%s,"cache_trash":%s,%s}\n' \
      "$(date +%s)" "$pid" "$st" "$m" "${cpu:-0}" "${huge:-0}" "$fds" "${kb:-0}" "$dirs" "$trash" "$net"
  fi
  sleep "$INTERVAL"
done
