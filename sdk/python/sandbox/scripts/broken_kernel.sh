#!/usr/bin/env bash
# The handle's behaviour comes from the library and from nothing the Python shell holds: build the kernel
# with one constant of its table changed (429 -> 428, so a busy answer is no longer retried) in a scratch
# copy of the repository, and run a test against it. The test must go red there, and be green with the
# package's own library.
#
#   sdk/python/sandbox/scripts/broken_kernel.sh
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-$HOME/.cache}/py-broken-kernel.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

mkdir -p "$scratch/repo"
(cd "$root" && tar --exclude=./target --exclude=./.git --exclude=node_modules --exclude=.venv \
  --exclude=_native --exclude=./sdk/ts -cf - .) | tar -xf - -C "$scratch/repo"
step=$scratch/repo/crates/job-client/src/step.rs
before=$(grep -c 'else if status == 429 ||' "$step")
sed -i 's/else if status == 429 ||/else if status == 428 ||/' "$step"
[[ $before -gt 0 && $(grep -c 'else if status == 428 ||' "$step") == "$before" ]] || { echo "the kernel has no 429 to break: update this script" >&2; exit 1; }

CARGO_TARGET_DIR="$scratch/target" "$scratch/repo/crates/job-client-ffi/build-native.sh" "$scratch/lib" >/dev/null
lib=$(ls "$scratch"/lib/*job_client_ffi.*)

test=tests/test_held.py::test_a_busy_answer_is_waited_out_and_tried_again
cd "$root/sdk/python/sandbox"
echo "== the package's own library"
python -m pytest -q "$test"
echo "== the library with 429 changed to 428: this must fail"
if KERNEL_LIB_UNDER_TEST=$lib python -m pytest -q "$test"; then
  echo "the test stayed green against a broken kernel: the shell decides something" >&2
  exit 1
fi
echo "red against the broken kernel, as it must be"
