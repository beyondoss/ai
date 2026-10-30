#!/usr/bin/env bash
# Build the in-sandbox tools-mcp binary for the current OS/arch.
#
# Prefer AOT (`bun compile`): smaller RSS, no JIT. Needs --aot-allow-code-from-disk so the
# guest can import tools/*.ts from the seeded workspace.
# If AOT fails, fall back to JIT-in-binary (`bun compile --no-aot`) without that flag — the JIT
# runtime can load tools from disk. Same --trim and --smol either way.
#
# FORCE_NO_AOT=1 skips straight to JIT (proves the fallback artifact).
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

compile_aot() {
  bun compile \
    --trim \
    --aot-allow-code-from-disk \
    --compile-exec-argv=--smol \
    "$ENTRY" \
    --outfile "$OUT"
}

compile_jit() {
  # No --aot-allow-code-from-disk: that flag requires --aot. JIT executables load disk imports.
  bun compile --no-aot \
    --trim \
    --compile-exec-argv=--smol \
    "$ENTRY" \
    --outfile "$OUT"
}

MODE=aot
if [ "${FORCE_NO_AOT:-}" = 1 ]; then
  echo "compile: FORCE_NO_AOT=1 → JIT-in-binary" >&2
  MODE=jit
  compile_jit
elif ! compile_aot 2>"$ROOT/bin/compile-aot.log"; then
  echo "compile: AOT failed; falling back to --no-aot (JIT-in-binary). Log:" >&2
  tail -n 40 "$ROOT/bin/compile-aot.log" >&2 || true
  MODE=jit
  compile_jit
fi

ls -lh "$OUT"
echo "$MODE" >"${OUT}.mode"
echo "compile: wrote $OUT ($MODE + trim + --smol)"
