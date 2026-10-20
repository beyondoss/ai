#!/usr/bin/env bash
# Render a run record into one self-contained page: report.html next to run.json. No network, no
# build step: the record is embedded in the template and the page replays it.
#
#     ./deploy/demo/report.sh deploy/demo/results/<run>
set -euo pipefail
dir=${1:?usage: report.sh <results/run dir>}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$dir/run.json" ] || { echo "report: $dir/run.json not found" >&2; exit 1; }
# `</` becomes `<\/` (still valid JSON) so nothing in the record can close the script element.
data=$(jq -c . "$dir/run.json" | sed 's#</#<\\/#g')
out="$dir/report.html"
while IFS= read -r line; do
  if [ "$line" = '/*RUN_JSON*/' ]; then printf '%s\n' "$data"; else printf '%s\n' "$line"; fi
done < "$here/report.template.html" > "$out.tmp"
mv "$out.tmp" "$out"
echo "$out"
