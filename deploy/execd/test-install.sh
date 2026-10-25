#!/bin/bash
# Rerunnable checks for execd's distribution: the installer in alpine (busybox wget) and debian
# (curl), and the carrier image's `COPY --from`. Needs docker, python3 and a static musl execd.
#
#   cargo build --release --locked -p beyond-ai-execd --target x86_64-unknown-linux-musl \
#     (with RUSTFLAGS="-C relocation-model=static")
#   deploy/execd/test-install.sh target/x86_64-unknown-linux-musl/release/execd
#
# Set SKIP_IMAGE=1 to skip the Dockerfile.execd build, or BUILD_CONTEXT=<dir> to build that image
# from a different workspace root (default: the repo root), or EXECD_ARGS for extra execd flags.
set -euo pipefail

bin="${1:?usage: test-install.sh <path to static execd binary>}"
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
tag="v0.0.0-test"
uniq="execd-test-$$"
work="$(mktemp -d)"
server_pid=""
cleanup() {
  [ -z "$server_pid" ] || kill "$server_pid" 2>/dev/null || true
  docker rm -f "$uniq-run" >/dev/null 2>&1 || true
  docker rmi -f "$uniq-carrier" "$uniq-consumer" >/dev/null 2>&1 || true
  rm -rf "$work"
}
trap cleanup EXIT

pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

# --- static check ---------------------------------------------------------------------------------
file "$bin"
file "$bin" | grep -q 'statically linked' || fail "$bin is not statically linked"
if ldd "$bin" 2>&1 | grep -vq 'not a dynamic executable\|statically linked'; then fail "ldd says dynamic"; fi
pass "binary is static ($(wc -c <"$bin") bytes)"

# --- fake release dir -----------------------------------------------------------------------------
rel="$work/rel"
mkdir -p "$rel" "$work/pkg"
cp "$bin" "$work/pkg/execd"
for arch in amd64 arm64; do
  name="execd-$tag-linux-$arch.tar.gz"
  tar -C "$work/pkg" -czf "$rel/$name" execd
  (cd "$rel" && sha256sum "$name" >"$name.sha256")
done
cp "$here/install-execd.sh" "$rel/"
# A second release whose checksum is wrong.
bad="v0.0.0-bad"
for arch in amd64 arm64; do
  name="execd-$bad-linux-$arch.tar.gz"
  cp "$rel/execd-$tag-linux-$arch.tar.gz" "$rel/$name"
  printf '%064d  %s\n' 0 "$name" >"$rel/$name.sha256"
done

port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
(cd "$rel" && exec python3 -m http.server "$port" --bind 127.0.0.1 >/dev/null 2>&1) &
server_pid=$!
for _ in $(seq 50); do
  if curl -fs "http://127.0.0.1:$port/install-execd.sh" >/dev/null 2>&1; then break; fi
  sleep 0.1
done

# --- installer, per distro ------------------------------------------------------------------------
# Host networking so the container reaches the server on the host's loopback.
run_in() { # image, then a sh script on stdin
  docker run --rm -i --network host -v "$here/install-execd.sh:/install-execd.sh:ro" \
    -e EXECD_BASE_URL="http://127.0.0.1:$port" "$1" sh -eu
}

check() { # label, image, prelude
  local label="$1" image="$2" prelude="$3"
  echo "== $label ($image)"
  run_in "$image" <<EOF
$prelude
export EXECD_VERSION=$tag EXECD_INSTALL_DIR=/opt/bin

# success: installed, executable, identical to the built binary
sh /install-execd.sh
[ -x /opt/bin/execd ]
first=\$(sha256sum /opt/bin/execd | cut -d' ' -f1)
[ "\$first" = "$(sha256sum "$bin" | cut -d' ' -f1)" ]
echo "ok success"

# rerun: idempotent, same bytes, no leftover temp files
sh /install-execd.sh
[ "\$(sha256sum /opt/bin/execd | cut -d' ' -f1)" = "\$first" ]
[ "\$(ls -A /opt/bin)" = execd ]
echo "ok rerun"

# checksum mismatch into a fresh dir: refused, nothing left behind
if EXECD_VERSION=$bad EXECD_INSTALL_DIR=/opt/fresh sh /install-execd.sh 2>/tmp/err; then echo "bad checksum accepted"; exit 1; fi
grep -q 'checksum mismatch' /tmp/err
[ -z "\$(ls -A /opt/fresh)" ]
echo "ok mismatch refused, no file"

# mismatch over an existing install: the old binary is untouched
if EXECD_VERSION=$bad sh /install-execd.sh 2>/dev/null; then echo "bad checksum accepted"; exit 1; fi
[ "\$(sha256sum /opt/bin/execd | cut -d' ' -f1)" = "\$first" ]
[ "\$(ls -A /opt/bin)" = execd ]
echo "ok mismatch leaves existing binary"

# unsupported architecture is refused clearly
mkdir -p /fake && printf '#!/bin/sh\n[ "\$1" = -m ] && echo riscv64 || echo Linux\n' >/fake/uname && chmod +x /fake/uname
if PATH=/fake:\$PATH sh /install-execd.sh 2>/tmp/err; then echo "riscv accepted"; exit 1; fi
grep -q 'unsupported architecture' /tmp/err
echo "ok unsupported arch refused"

EOF
  pass "$label"
}

check alpine alpine:3.22 "mkdir -p /opt/bin /opt/fresh; command -v wget; ! command -v curl"
check debian debian:bookworm-slim \
  "apt-get update -qq >/dev/null && apt-get install -y -qq --no-install-recommends curl ca-certificates >/dev/null; mkdir -p /opt/bin /opt/fresh; command -v curl"

# --- carrier image --------------------------------------------------------------------------------
if [ -z "${SKIP_IMAGE:-}" ]; then
  echo "== Dockerfile.execd"
  docker build -q -f "$repo/Dockerfile.execd" -t "$uniq-carrier" "${BUILD_CONTEXT:-$repo}" >/dev/null
  docker build -q -t "$uniq-consumer" - >/dev/null <<EOF
FROM alpine:3.22
COPY --from=$uniq-carrier /execd /usr/local/bin/execd
EOF
  cport="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
  # shellcheck disable=SC2206
  execd_args=(${EXECD_ARGS:-})
  docker run -d --name "$uniq-run" -p "127.0.0.1:$cport:8080" "$uniq-consumer" \
    execd --listen 0.0.0.0:8080 "${execd_args[@]}" >/dev/null
  ok=""
  for _ in $(seq 50); do
    if curl -fs "http://127.0.0.1:$cport/livez" >/dev/null; then ok=1; break; fi
    sleep 0.2
  done
  [ -n "$ok" ] || { docker logs "$uniq-run" >&2; fail "execd in the consumer image never answered /livez"; }
  pass "COPY --from carrier works; /livez answered: $(curl -s "http://127.0.0.1:$cport/livez")"
fi
echo "all checks passed"
