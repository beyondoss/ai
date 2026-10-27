# beyond-sandbox-conformance

The proof for the sandbox clients: every client's driver, run against every backend, over the shared scenarios in
`sdk/conformance/scenarios`. `publish = false`; nothing ships from here.

```text
drivers (processes)  x  backends (exec endpoints)  x  scenarios (JSON)  ->  one cell per driver__backend
```

One evaluator serves two callers. The library (`sandbox_conformance`) replays scenarios through a `Link`, anything that answers
one request with one response. The `sandbox-matrix` binary reaches a driver over its stdin and stdout; the agent's reference
test (`crates/agent/tests/sdk_conformance.rs`) reaches the in-process reference driver. There is no second copy of the matchers,
`requires`, `poll`, `$repeat`, derived views or strict `xfail`.

## Modules

| file                              | what it is                                                                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lib.rs`; `bin/sandbox-matrix.rs` | The library root; the CLI (`--driver NAME=CMD`, `--registry`, `--backend`, `--filter`, `--report`, `--verbose`). Exit 1 on any `fail`, 2 on a usage error.          |
| `eval.rs`                         | Pure functions of the scenario language: `${var}` substitution, `$repeat`, `_utf8` / `_len` / `_sha256` views, matchers.                                            |
| `run.rs`                          | `Backend` (a connected driver plus what its probes found), `run_scenario`, the five outcomes, `Tally`. No I/O of its own beyond `Link` / `Connector`.               |
| `relay.rs`                        | Relay scenarios (steps with `via`): which driver a step goes to, and one run of every relay scenario for a pair A, B on the same sandbox.                           |
| `link.rs`                         | `ProcessLink`: `sh -c "exec CMD"` with the `CONFORMANCE_*` environment, one request in flight, a named call timeout, stderr kept for the death note.                |
| `backends.rs`                     | Starts a backend and returns a `Started` (URL, bearer, work directory, extra variables, a guard that stops it).                                                     |
| `reaper.rs`                       | Every child is its own process group, killed on drop; a registry file plus one watchdog kill the groups and remove the containers when the harness dies by SIGKILL. |
| `matrix.rs`                       | The plan: backends in parallel, each started once, its drivers one after another, then every ordered pair of them for the relay scenarios; the registry.            |
| `report.rs`                       | The table and the JSON cell.                                                                                                                                        |
| `bin/sandbox-driver-rust.rs`      | The Rust client's driver. Includes `crates/sandbox/tests/sdk_conformance/driver.rs` by path.                                                                        |
| `bin/sandbox-driver-broken.rs`    | Wraps a driver and corrupts one op's answer. `tests/selftest.rs` proves the harness fails exactly the checks that use that op.                                      |

## Differential testing

`tests/differential.rs` (`mise run test:differential`) is the randomized counterpart of the shared scenarios. Hegel
generates a sequence of operations (`write`, guarded `write_if_unchanged`, `read`, `stat`, `list`, `search`, `mkdir`, raw
`exec` and `exec_stream`, and changes made behind the client's back through `exec`: write, mkdir, symlink, chmod, remove)
over a small tree, using the name, content and query generators of the `ShellFs` property test. Every sequence runs in
lockstep through every driver in `drivers.toml`, each in its own fresh root on one backend, and each answer is checked
two ways:

1. against the other drivers', byte for byte after the normalisations listed in `tests/differential/norm.rs` (request ids, error
   `detail`, the root path, mtimes, directory sizes, listing order, the text of a search `note`), each of which the spec lets differ;
2. on the local backends, against the filesystem: `std::fs` snapshots and the brute-force search of the property test
   (`oracle.rs`, which includes `truth.rs` and `search_truth.rs` from `crates/agent/tests/shellfs_props_env`). Writes must change only
   their target, failed operations nothing, reads must equal the file's bytes, listings the directory's entries.

An op is compared across the drivers whose `hello` lists it. A driver without it (the MCP driver has no `exec_stream` or `write_if_unchanged`) is not asked, is never a divergence, and does not make the others skip the op. When the others applied a `write_if_unchanged`, the driver without it is brought level with a plain `write` of the same bytes, so its tree stays the same.

A sequence ends with a read-back of every path it named, so the final tree is compared too. On the container backends there
is no host filesystem to read, so only (1) applies. Hegel shrinks a difference to a minimal sequence and the test prints it,
and writes it to `target/differential/<backend>.json`, as a scenario file.

Open defects are tolerated by id so one cannot hide the rest: `D2`, `D3`, `D5` (found
here, in `known.rs` and `oracle.rs`), each with a scenario in `differential-found.json` or `files.json`. A fix deletes
its entry and its `xfail` mark.

| variable                                                                                             | what                                                                                            |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `DIFFERENTIAL_CASES`, `DIFFERENTIAL_SEED`, `DIFFERENTIAL_STEPS`                                      | cases per backend, Hegel's seed, operations per sequence                                        |
| `DIFFERENTIAL_BACKENDS`, `DIFFERENTIAL_DRIVERS`                                                      | comma-separated names to run                                                                    |
| `DIFFERENTIAL_SHRINK=0`, `DIFFERENTIAL_NO_KNOWN=1`, `DIFFERENTIAL_BUILD=1`, `DIFFERENTIAL_VERBOSE=1` | no shrinking; report the tolerated defects too; build the registry; print each tolerated defect |

CI runs fixed seeds and case counts. Not covered: jobs (their timing is not deterministic across drivers) and the fault, read-only
and tiny-disk backends.

## Process accounting

Every process the harness starts (drivers, a local `execd`) is its own process-group leader and held by a `reaper::Leader`:
dropping it kills the group. Leaders and containers are also written to a registry file watched by one watchdog process
that, once the harness is gone (SIGKILL included), kills the registered groups and removes the registered containers. The
differential test aborts after `DIFFERENTIAL_DEADLINE_SECS` (1800) and leaves the same cleanup to the watchdog. Killing the test
binary mid-run with SIGKILL left no `execd`, driver or container behind (29 processes and 2 containers before, none after).

## Outcomes

A check is `pass`, `fail`, `known defect` (an `xfail` for this driver, which failed as marked), `unsupported` (the driver's
`hello` lacks an op the scenario uses) or `skipped` (planned, or `requires` unmet). Only `pass` is a pass. An `xfail` that
passes is a `fail`, so a mark cannot outlive the defect. The `xfail` driver name is the `driver` the driver's `hello` reports.
A driver that cannot be started, or answers nothing to `hello`, fails its cell with one `driver/hello` check.

## Backends

A backend is a URL plus how to start and stop it. `Started` holds the guard; dropping it stops the backend.

| name                               | what                                                                                      |
| ---------------------------------- | ----------------------------------------------------------------------------------------- |
| `local` (group)                    | the `execd` binary on loopback, temp root: plain, with a bearer, and the forced profiles  |
| `local-grep`, `local-busybox-grep` | `CONFORMANCE_CAPS` forces the POSIX grep rung, the busybox rung                           |
| `alpine`                           | static `execd` in Alpine: real busybox tools                                              |
| `debian-runas`                     | static `execd` in Debian with `EXECD_RUN_AS=sandbox`                                      |
| `readonly`                         | Debian, run-as: `${RO}` a read-only mount, `${LOCKED}` a directory the user may not write |
| `tinydisk`                         | Debian, run-as: `${TINY}` a 1 MiB tmpfs that fills                                        |

The container backends mount a static (musl) `execd` read-only into the image, so no image is built; the only network use is
pulling `alpine:3.20` and `debian:12-slim`. Each container is named after the process and removed on drop. Readiness is
execd's own `listening` line in `docker logs`, never a port probe (docker-proxy accepts before execd listens).

A backend's extra directories are variables (`${RO}`, `${LOCKED}`, `${TINY}`) that a scenario names in `requires.vars`; the
`environments` suite is skipped, visibly, everywhere else.

## Bounds

| what                                | bound                                             | freed by                                |
| ----------------------------------- | ------------------------------------------------- | --------------------------------------- |
| one driver request                  | `CALL_TIMEOUT` (300 s), then the driver is killed | the call returning, or the driver dying |
| a dead driver's stderr in a failure | the last 2000 bytes                               | the failure line                        |
| backends running at once            | `--parallel` (4)                                  | the backend's task ending               |
| a started endpoint's time to listen | `READY_WITHIN` (60 s)                             | readiness                               |

## Adding a live backend

A live backend (a VM, a vendor sandbox) is a function returning `Started` from a URL and a bearer in the environment, with a
work directory the sandbox user may write; the guard is whatever stops it. Register its name in `backends::start`.
