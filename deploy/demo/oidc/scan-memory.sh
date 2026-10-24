#!/bin/sh
# Read a process's writable memory and look for token-shaped strings and for specific token tails.
# Runs inside the agent's container (through ECS Exec), as the agent's own user, with busybox tools only.
#
#     scan-memory.sh <path of the process's executable> <needles file>
#
# The needles file has one `kind tail` per line: `control` for a string the caller planted in the agent
# (so the scan is shown to find what is there) and `access` / `refresh` for the last 24 characters of
# a real token (enough to recognise it, not to use it). Prints:
#
#     SCAN-RESULT pid=… kb=<memory read> unreadable=<ranges refused> jwt_shaped=<distinct JWT-shaped strings>
#     SCAN-NEEDLE <kind> <tail> <lines holding it>       one per needle
#     SCAN-JWT <last 24 characters> <first 24 of the payload>   one per distinct JWT-shaped string
#
# Every writable mapping is read in full, including the allocator's large reservation: a read of a page
# that was never touched costs a syscall and nothing else. Mappings with no resident page are skipped
# (their `Rss` in smaps is 0), which is what makes that affordable.
exe=$1 needles=$2
pid=""
for p in /proc/[0-9]*; do
  [ "$(readlink "$p/exe" 2>/dev/null)" = "$exe" ] && pid=${p#/proc/}
done
[ -n "$pid" ] || { echo "SCAN-ERROR no process runs $exe"; exit 0; }

hits=/tmp/scan.hits err=/tmp/scan.err
: > "$hits"
alt=$(cut -d' ' -f2 "$needles" | paste -sd'|' -)
kb=0 bad=0
# smaps lists each mapping's header line, then its fields; keep the headers of mappings that are
# writable and have resident pages.
awk '/^[0-9a-f]+-[0-9a-f]+ / { hdr = $1; perms = $2; next } /^Rss:/ { if (perms ~ /^rw/ && $2 > 0) print hdr }' "/proc/$pid/smaps" > /tmp/scan.ranges
while read -r range; do
  s=$((0x${range%-*}))
  e=$((0x${range#*-}))
  dd if="/proc/$pid/mem" bs=4096 skip=$((s / 4096)) count=$(((e - s) / 4096)) 2>"$err" \
    | tr -cs 'A-Za-z0-9_.-' '\n' \
    | grep -E -o "eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}|$alt" >> "$hits"
  if grep -q "can.t open\|ermission\|Input/output" "$err"; then bad=$((bad + 1)); else kb=$((kb + (e - s) / 1024)); fi
done < /tmp/scan.ranges

jwts=$(grep -c '^eyJ' "$hits")
echo "SCAN-RESULT pid=$pid kb=$kb unreadable=$bad jwt_shaped=$(grep '^eyJ' "$hits" | sort -u | wc -l) jwt_hits=$jwts"
while read -r kind tail; do
  echo "SCAN-NEEDLE $kind $tail $(grep -c -F "$tail" "$hits")"
done < "$needles"
grep '^eyJ' "$hits" | sort -u | awk -F. '{ printf "SCAN-JWT %s %s\n", substr($3, length($3) - 23), substr($2, 1, 24) }'
rm -f "$hits" "$err" /tmp/scan.ranges
