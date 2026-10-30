#!/usr/bin/env bash
# Build the in-sandbox tools-mcp binary for the current OS/arch with `bun build --compile`.
#
# The executable bundles the CLI and keeps Bun's runtime, so it still imports tools/*.ts from
# the seeded workspace at start. --smol trims the runtime's memory use; --minify shrinks the
# bundled CLI.
#
# --compile-autoload-package-json: the tools import `eve/tools` by bare name from the bundle's
# node_modules. A compiled executable resolves nothing from disk (even a plain package) unless it
# reads package.json at runtime.
#
# Build on the guest OS/arch (linux-x64 Docker for E2B). OUT= overrides the path.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-"$ROOT/bin/tools-mcp"}
ENTRY=${ENTRY:-"$ROOT/src/cli.ts"}
mkdir -p "$(dirname "$OUT")"

if ! command -v bun >/dev/null; then
  echo "compile: bun is required (https://bun.sh)" >&2
  exit 1
fi

bun build --compile \
  --minify \
  --compile-exec-argv=--smol \
  --compile-autoload-package-json \
  "$ENTRY" \
  --outfile "$OUT"

ls -lh "$OUT"
echo "compile: wrote $OUT"
