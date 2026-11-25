#!/usr/bin/env bash
# Build the job client's kernel as a native library for Python's ctypes.
#
#   crates/job-client-ffi/build-native.sh            build into the Python package's _native/ directory
#   crates/job-client-ffi/build-native.sh DIR        build into DIR (a wheel build's staging directory)
#
# Writes DIR/libjob_client_ffi.so (.dylib, job_client_ffi.dll) and DIR/kernel.src.sha256, the hash of
# the sources it was built from (source-hash.sh: the same hash the TypeScript module carries). The
# compiler is the one pinned in `toolchain`. `tests/test_kernel.py` fails when a library does not match
# the sources beside it.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-$root/sdk/python/sandbox/src/beyond_sandbox/_native}
mkdir -p "$out"

export RUSTUP_TOOLCHAIN=$(tr -d '[:space:]' <"$here/toolchain")
target=${CARGO_TARGET_DIR:-$root/target}
# Size, and the same bytes from any checkout: the profile is set here, not in the workspace's.
export CARGO_PROFILE_RELEASE_OPT_LEVEL=s CARGO_PROFILE_RELEASE_LTO=true CARGO_PROFILE_RELEASE_PANIC=abort
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_STRIP=true CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=true
flags="--remap-path-prefix=$root=/src --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo --remap-path-prefix=${RUSTUP_HOME:-$HOME/.rustup}=/rustup"
# A musl host links its C runtime statically by default, which cannot make a shared library.
case "$(rustc -vV)" in *-musl*) flags="$flags -C target-feature=-crt-static" ;; esac
export RUSTFLAGS=$flags
# The shared compile cache (mbx) must not serve this build: the library is a pinned artifact.
export MBX_OFF=1
slot=${BUILD_SLOT:-$HOME/.cache/tmp/build-slot.sh}
[[ -x $slot ]] || slot=
(cd "$root" && $slot cargo build --release --locked --lib -p beyond-job-client-ffi)

case "$(uname -s)" in
  Darwin) lib=libjob_client_ffi.dylib ;;
  MINGW* | MSYS* | CYGWIN* | Windows_NT) lib=job_client_ffi.dll ;;
  *) lib=libjob_client_ffi.so ;;
esac
cp "$target/release/$lib" "$out/$lib"
"$here/source-hash.sh" >"$out/kernel.src.sha256"
echo "$lib $(wc -c <"$out/$lib") bytes"
