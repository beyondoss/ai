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

| file                              | what it is                                                                                                                                                 |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lib.rs`; `bin/sandbox-matrix.rs` | The library root; the CLI (`--driver NAME=CMD`, `--registry`, `--backend`, `--filter`, `--report`, `--verbose`). Exit 1 on any `fail`, 2 on a usage error. |
| `eval.rs`                         | Pure functions of the scenario language: `${var}` substitution, `$repeat`, `_utf8` / `_len` / `_sha256` views, matchers.                                   |
| `run.rs`                          | `Backend` (a connected driver plus what its probes found), `run_scenario`, the five outcomes, `Tally`. No I/O of its own beyond `Link` / `Connector`.      |
| `link.rs`                         | `ProcessLink`: `sh -c "exec CMD"` with the `CONFORMANCE_*` environment, one request in flight, a named call timeout, stderr kept for the death note.       |
| `backends.rs`                     | Starts a backend and returns a `Started` (URL, bearer, work directory, extra variables, a guard that stops it).                                            |
| `matrix.rs`                       | The plan: backends in parallel, each started once, its drivers one after another; the registry file.                                                       |
| `report.rs`                       | The table and the JSON cell.                                                                                                                               |
| `bin/sandbox-driver-rust.rs`      | The Rust client's driver. Includes `crates/agent/tests/sdk_conformance_driver.rs` by path until `beyond-sandbox` owns the client.                          |
| `bin/sandbox-driver-broken.rs`    | Wraps a driver and corrupts one op's answer. `tests/selftest.rs` proves the harness fails exactly the checks that use that op.                             |

## Outcomes

A check is `pass`, `fail`, `known defect` (an `xfail` for this driver, which failed as marked), `unsupported` (the driver's
`hello` lacks an op the scenario uses) or `skipped` (planned, or `requires` unmet). Only `pass` is a pass. An `xfail` that
passes is a `fail`, so a mark cannot outlive the defect. The `xfail` driver name is the `driver` the driver's `hello` reports.
A driver that cannot be started, or answers nothing to `hello`, fails its cell with one `driver/hello` check.

## Backends

A backend is a URL plus how to start and stop it. `Started` holds the guard; dropping it stops the backend.

| name                                                                  | what                                                                                                                                                |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `local` (group)                                                       | the `execd` binary on loopback, temp root: plain, with a bearer, and the forced profiles                                                            |
| `local-no-stdin`, `local-grep`, `local-busybox-grep`                  | `CONFORMANCE_CAPS` forces the 48 KiB chunked write, the POSIX grep rung, the busybox rung                                                           |
| `alpine`                                                              | static `execd` in Alpine: real busybox tools                                                                                                        |
| `debian-runas`                                                        | static `execd` in Debian with `EXECD_RUN_AS=sandbox`                                                                                                |
| `readonly`                                                            | Debian, run-as: `${RO}` a read-only mount, `${LOCKED}` a directory the user may not write                                                           |
| `tinydisk`                                                            | Debian, run-as: `${TINY}` a 1 MiB tmpfs that fills                                                                                                  |
| `fault` (group: `fault-no-stdin`, `fault-v11`, `fault-baseline-jobs`) | the in-process fault-injecting endpoint (`beyond_test_support::exec_mock`): a v1 endpoint that ignores stdin, a v1.1-only endpoint, native jobs off |

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
