#!/bin/sh
# Prove a running execd serves: /livez is open, an exec without the token is refused, and an exec
# with it runs a command. Needs curl, and BASE (http://host:port) and TOKEN in the environment.
set -eu

body='{"command":"sh","args":["-c","echo hi"]}'

code="$(curl -s -o /dev/null -w '%{http_code}' "$BASE/livez")"
[ "$code" = 200 ] || { echo "/livez returned $code" >&2; exit 1; }
echo "ok /livez 200 without auth"

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST -d "$body" "$BASE/")"
[ "$code" = 401 ] || { echo "unauthenticated exec returned $code" >&2; exit 1; }
echo "ok unauthenticated exec 401"

out="$(curl -s -X POST -H "authorization: Bearer $TOKEN" -d "$body" "$BASE/")"
printf "response: %s\n" "$out"
printf "%s" "$out" | grep -Eq '"exit_code": ?0' || { echo "exit_code is not 0" >&2; exit 1; }
printf "%s" "$out" | grep -Fq '"stdout":"hi\n"' || { echo 'stdout is not "hi\n"' >&2; exit 1; }
echo "ok authenticated exec: exit_code 0, stdout hi"
