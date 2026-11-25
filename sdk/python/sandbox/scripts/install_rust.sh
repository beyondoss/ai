#!/bin/sh
# Installs the compiler the job client's kernel is pinned to (crates/job-client-ffi/toolchain), for a wheel
# build in a container (cibuildwheel's before-all). Plain `sh`: the musllinux images have no more.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
toolchain=$(tr -d '[:space:]' <"$here/../../../../crates/job-client-ffi/toolchain")
if ! command -v rustup >/dev/null 2>&1 && [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
fi
PATH="$HOME/.cargo/bin:$PATH"
rustup toolchain install "$toolchain" --profile minimal
