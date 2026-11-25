#!/usr/bin/env bash
# The hash of the sources the job client's kernel artifacts are built from: this crate and the kernel, by
# content (the paths are not hashed), and the compiler's pinned release (`toolchain`), so a rustc bump is a
# visible change and never a surprise red. The WASM module for TypeScript (build.sh) and the native library
# for Python (build-native.sh) carry this one hash, and each SDK's test recomputes it.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$here/.."
# macOS has no sha256sum.
if command -v sha256sum >/dev/null 2>&1; then sum=sha256sum; else sum="shasum -a 256"; fi
cat job-client-ffi/toolchain job-client-ffi/Cargo.toml job-client/Cargo.toml $(LC_ALL=C ls job-client-ffi/src/*.rs job-client/src/*.rs) | $sum | cut -d' ' -f1
