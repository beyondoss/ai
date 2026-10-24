#!/usr/bin/env bash
# A scripted user at a browser: open an authorization URL, sign in at the identity provider's login
# form, approve the consent screen, and print the URL the provider redirects to (the broker's
# callback, carrying the one-time code). No human, and nothing here the broker or the agent can see.
#
#     login.sh <authorization-url> <username> <password> [--connect-to host:port:ip:port]
#
# The URL's host is the provider's stable in-VPC name; --connect-to is how this script, outside the
# VPC, reaches it at its current public address (the Host header stays the stable name).
# The password is read from the environment variable LOGIN_PASSWORD when the argument is "-".
set -euo pipefail
url=$1 user=$2 pass=$3
shift 3
[ "$pass" != - ] || pass=${LOGIN_PASSWORD:?LOGIN_PASSWORD is not set}
CONNECT_TO=
[ "${1:-}" != --connect-to ] || CONNECT_TO=$2
jar=$(mktemp)
body=$(mktemp)
trap 'rm -f "$jar" "$body"' EXIT
curl_() { curl -sS -m 30 -b "$jar" -c "$jar" ${CONNECT_TO:+--connect-to "$CONNECT_TO"} "$@"; }
unescape() { sed -e 's/&amp;/\&/g' -e 's/&quot;/"/g'; }
# the action of the form with this id
action() { # html-file id
  tr '\n' ' ' <"$1" | grep -o "<form[^>]*id=\"$2\"[^>]*>" | grep -o 'action="[^"]*"' | head -1 | sed 's/^action="//; s/"$//' | unescape || true
}
location() { { grep -i "^location:" || true; } | tr -d "\r" | sed "s/^[Ll]ocation: *//" | head -1; }

curl_ -o "$body" "$url"
act=$(action "$body" kc-form-login)
[ -n "$act" ] || { echo "login: no login form at $url" >&2; exit 1; }
loc=$(curl_ -D - -o "$body" --data-urlencode "username=$user" --data-urlencode "password=$pass" --data-urlencode credentialId= "$act" | location)
# Still on the provider: the consent screen (a required action). Approve everything asked.
origin=${url%%/realms/*}
if [ "${loc#"$origin"}" != "$loc" ]; then
  curl_ -o "$body" "$loc"
  act=$(tr '\n' ' ' <"$body" | grep -o 'action="[^"]*login-actions/consent[^"]*"' | head -1 | sed 's/^action="//; s/"$//' | unescape || true)
  [ -n "$act" ] || { echo "login: no consent form after signing in as $user" >&2; exit 1; }
  code=$(grep -o 'name="code" value="[^"]*"' "$body" | head -1 | sed 's/.*value="//; s/"$//')
  loc=$(curl_ -D - -o /dev/null --data-urlencode "code=$code" --data-urlencode accept=Yes "$origin$act" | location)
fi
[ -n "$loc" ] || { echo "login: the consent was not accepted" >&2; exit 1; }
printf '%s\n' "$loc"
