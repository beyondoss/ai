# A3 (PTY) state, handoff. Worktree ~/.cache/wt/a3-pty, branch a3-pty, PR #734 (beyondoss/ai).

Build/test recipe: `export PATH=$HOME/.cargo/bin:$PATH CARGO_TARGET_DIR=~/.cache/a3-pty-target CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0`, cargo via `~/.cache/tmp/build-slot.sh`.
Verus: `KERNEL_CRATES=crates/execd-jobs .github/scripts/verify-kernels.sh [--features mut_x]` (97 verified green). Mutant helper: `~/.cache/a3-mut.sh mut_a mut_b` (prints red/green).
Models run locally: `flock ~/.cache/tmp/model-run.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 <cargo test ...>` (coordinator: local is fine). `~/.cache/a3-model-all.sh` = full sweep (slow, lock queue).
dprint: `D=$(ls -d ~/.local/share/mise/installs/dprint/*/|head -1); PATH=$D:$PATH dprint fmt <files>`. node: ~/.local/share/mise/installs/node/*/bin.
Never pkill -f; kill by pid after checking cwd.

## Done since the first report (all committed up to 78596ff3, NOT yet pushed after 5f493b22: push!)
- 503 path: `Options::terminals_max` / env `EXECD_TERMINALS_MAX` (jobs.rs launch_term), test in tests/pty.rs.
- `Options::helper_exe` (keeper binary), used by the in-process mock.
- run_as test (tests/pty.rs, sudo): shell uid 65534, owns tty, keeper is root.
- hung_up reason in TS/Python/Rust SDK (+unit tests).
- sdk_conformance.rs restored to the original assertion; in-process mock (test-support exec_mock.rs `restart_execd`, `kill_keepers`, Ctl) + `MockHost` in the test. `reference_agent_passes_every_scenario` passes (160 s).
- harness hosts: local + e2b (crates/sandbox-conformance/src/e2b.rs `E2bHost`: restart_execd SIGHUP, crash_execd SIGKILL, kill_keepers); new scenario jobs.pty.survives-execd-crash.
- Release keeper (n=30): Rss p50 3932 KiB p95 4132; Pss p50 537 KiB p95 553 KiB (under 1 MiB). Debug build was Pss 1.15 MiB. Log ~/.cache/a3-rel.log. M7 still needs an ASSERT (Pss < 1 MiB) in a release-run test and numbers in the report.
- `Handle::activity().terminals` -> `Terminal{scope,id,pgid,pts_rdev,pts_fs,attached,members}`; test in tests/embed.rs; ARCHITECTURE.md updated (also documents Kill{marked} behaviour change).

## E2B matrix: RUNNING when I stopped
- Started 15:39 PST: `target/debug/sandbox-matrix --registry sdk/conformance/drivers.toml --backend e2b --report ~/.cache/a3-pty-e2b/report.json`, log ~/.cache/a3-pty-e2b/run.log, E2B_EXECD_BIN=~/.cache/a3-pty-musl/x86_64-unknown-linux-musl/debug/execd (built from 5f493b22-era source; REBUILD musl + rerun if kernel/execd changes: `CARGO_TARGET_DIR=~/.cache/a3-pty-musl cargo build --target x86_64-unknown-linux-musl -p beyond-execd --bin execd --locked`).
- VM ib2no4i6zagvu4u3b3ebd (the harness deletes it at the end and its watchdog on death). After the run: confirm no VM left (`beyond-conformance` metadata; list/delete via the harness reaper or E2B API with the key; key only via `set -a; . ~/ai/.env; set +a`, never print).
- Output is only written at the end. Read report.json / run.log tail for the summary; scenarios jobs.pty.survives-execd-restart/crash and keeper-killed-is-gone must PASS on e2b (E2bHost), not unsupported. Fix and rerun if red.
- E2B scenarios are in this PR (a3-pty-sdk merged in; no separate branch needed).

## Open (coordinator's list, from ~/.cache/audit-733-734/audit734.md)
B1 (blocker): `admit_peer(peer_cred, expected) -> Admit` pure fn in crates/execd-jobs (lib.rs, inside verus!): admit iff uid==expected uid && pid==expected pid; soundness ensures + completeness lemma; `mut_admit_any_peer` (add to execd-jobs Cargo.toml, execd Cargo.toml forward, mise task list); call from keeper side (term.rs: socket_peercred check in the accept loop, uid only: expected pid = any? keeper knows only uid; use Option<pid>) and execd side (term.rs connect_sync); test with wrong uid/pid refused (execd-side test with a fake keeper socket via a unix listener in a tempdir whose pid != keeper_pid).
M1: 503 decision (keeper exit code 124 -> 503; terminals_max cap) into a kernel fn (e.g. `start_refusal(keeper_exit: Option<i32>, open, max)`), call from jobs.rs launch_term; merges with the cap test.
M2: mapping keeper answer -> `Keeper::Answered{size}` in kernel (fn from `Said::Ready` fields + peer admit result -> Keeper), so recover's proof covers it; jobs.rs recover currently builds it inline.
M3: resize idempotence lemma (resize to same size twice == once) in lib.rs + mutant (e.g. `mut_resize_not_idempotent`), Verus red.
M5: 3 tests in tests/pty.rs: (a) list shell's fds (/proc/<shell pid>/fd) and find no leaked execd/keeper fd (socket/pipe/spool fds) ; (b) restart execd (SIGHUP) with the keeper already dead -> terminal Gone with status, offsets ok; (c) SIGKILL execd, new execd, terminal survives (like the restart test with kill instead of SIGHUP).
M6: keeper spool write failure currently `Err(_) => break` silently in term.rs master drain loop. Choose fail-closed: on write error to the spool, keeper ends the terminal (kill session group via SIGKILL of the child, write status with a reason / exit) -- or gap. Tell coordinator which. Suggest fail-closed: kill the session child and exit, execd then sees the end (code from status), log a warning.
M7: release-build test asserting keeper Pss < 1 MiB (n>=30), report numbers (537/553 KiB above).
M4: no change; PR body must mention the Kill{marked} behaviour change for every held job (PR body currently does NOT; edit with `gh pr edit 734 --body`).

After fixes: Verus + 20 mutants red (`~/.cache/a3-mut.sh ...`, mise task verify:execd-jobs-term-mutants lists them: add new ones), job_model/keeper_model green (scoped), full `cargo test -p beyond-execd -p beyond-execd-jobs` (skip the_mutation/faithful_), `cargo clippy --workspace --all-targets --locked -- -D warnings`, `cargo fmt --check`, dprint, merge origin/main once, push, `gh pr checks 734`.
Report to coordinator in <=150 words: sha, E2B summary, keeper numbers, test lines.
