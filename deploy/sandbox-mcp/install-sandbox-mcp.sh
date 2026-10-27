#!/bin/sh
# Install sandbox-mcp, the MCP server for a sandbox, from a GitHub release.
#
#   curl -fsSL https://github.com/beyondoss/ai/releases/latest/download/install-sandbox-mcp.sh | sh
#
# Environment:
#   SANDBOX_MCP_VERSION      release tag to install (default: the latest release)
#   SANDBOX_MCP_INSTALL_DIR  where to put the binary (default: /usr/local/bin)
#   SANDBOX_MCP_BASE_URL     directory URL holding sandbox-mcp-<tag>-linux-<arch>.tar.gz and its .sha256
#                      (default: the GitHub release download URL for the tag). Requires
#                      SANDBOX_MCP_VERSION, since the file names carry the tag.
#
# POSIX sh, busybox-compatible. The binary is checksum-verified before it is installed, and lands via
# a temp file in the destination directory plus a rename, so a failure never leaves a partial
# `sandbox-mcp`. Rerunning replaces it with the same bytes.
set -eu

REPO="beyondoss/ai"
INSTALL_DIR="${SANDBOX_MCP_INSTALL_DIR:-/usr/local/bin}"

die() {
  echo "install-sandbox-mcp: $*" >&2
  exit 1
}

[ "$(uname -s)" = Linux ] || die "unsupported OS '$(uname -s)' (need Linux)"
case "$(uname -m)" in
  x86_64 | amd64) arch=amd64 ;;
  aarch64 | arm64) arch=arm64 ;;
  *) die "unsupported architecture '$(uname -m)' (need x86_64 or aarch64)" ;;
esac

if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }
  fetch_stdout() { curl -fsSL --retry 3 "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -q -O "$2" "$1"; }
  fetch_stdout() { wget -q -O - "$1"; }
else
  die "need curl or wget"
fi

version="${SANDBOX_MCP_VERSION:-}"
if [ -z "$version" ]; then
  [ -z "${SANDBOX_MCP_BASE_URL:-}" ] || die "SANDBOX_MCP_BASE_URL requires SANDBOX_MCP_VERSION"
  version="$(fetch_stdout "https://api.github.com/repos/$REPO/releases/latest" |
    sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)" ||
    true
  [ -n "$version" ] || die "could not resolve the latest release; set SANDBOX_MCP_VERSION"
fi
base="${SANDBOX_MCP_BASE_URL:-https://github.com/$REPO/releases/download/$version}"
base="${base%/}"
name="sandbox-mcp-$version-linux-$arch.tar.gz"

mkdir -p "$INSTALL_DIR" || die "cannot create $INSTALL_DIR"
work="$(mktemp -d)" || die "mktemp failed"
tmp_bin=""
trap 'rm -rf "$work"; [ -z "$tmp_bin" ] || rm -f "$tmp_bin"' EXIT
trap 'exit 1' HUP INT TERM

echo "install-sandbox-mcp: downloading $name"
fetch "$base/$name" "$work/$name" || die "download failed: $base/$name"
fetch "$base/$name.sha256" "$work/$name.sha256" || die "download failed: $base/$name.sha256"

# The .sha256 is `<hex>  <file>`; compare just the hash so the check does not depend on the name
# inside it or on `sha256sum -c` flavour.
want="$(awk '{print $1; exit}' "$work/$name.sha256")"
got="$(sha256sum "$work/$name" | awk '{print $1}')"
if [ -z "$want" ] || [ "$want" != "$got" ]; then
  die "checksum mismatch for $name (expected '$want', got '$got')"
fi

tar -xzf "$work/$name" -C "$work" sandbox-mcp || die "archive does not contain sandbox-mcp"

# Same directory as the target so the rename is atomic (same filesystem).
tmp_bin="$(mktemp "$INSTALL_DIR/.sandbox-mcp.XXXXXX")" || die "cannot write to $INSTALL_DIR"
cp "$work/sandbox-mcp" "$tmp_bin"
chmod 755 "$tmp_bin"
mv -f "$tmp_bin" "$INSTALL_DIR/sandbox-mcp"
tmp_bin=""

echo "install-sandbox-mcp: installed $version to $INSTALL_DIR/sandbox-mcp"
