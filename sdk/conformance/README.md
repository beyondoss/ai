# Sandbox client conformance

One specification, one set of scenarios, and a reference check, for the family of
clients that drive a sandbox over the exec protocol: the Rust crate
`beyond-sandbox`, the MCP server `sandbox-mcp`, `@beyond.dev/sandbox` (TypeScript)
and `beyond-sandbox` (Python). They must behave identically.

| file                                     | what it is                                                                                   |
| ---------------------------------------- | -------------------------------------------------------------------------------------------- |
| [FILES-OVER-EXEC.md](FILES-OVER-EXEC.md) | the normative spec of the three layers and of every helper (stat, read, write, search, jobs) |
| [DRIVER.md](DRIVER.md)                   | the driver protocol, the error vocabulary, and the scenario format                           |
| [scenarios/](scenarios)                  | the shared scenario files: `exec`, `stream`, `jobs`, `files`, `errors`, `helpers`, `bulk`    |

## How the suite composes

```text
clients (drivers)  x  backends (exec endpoints)  x  scenarios
```

A harness starts a driver for each client, points it at each backend with
environment variables, and replays every scenario step as a JSON line. Each
check is `pass`, `fail`, `skip` (planned, or the backend lacks something the
scenario `requires`), `unsupported` (the driver lacks the op) or `known defect`.
Skipped and unsupported checks are printed and never counted as passes.

The three layers are tested separately: `exec`, `stream` and `jobs` are the
transport, `files` the helpers, `helpers` is code a user wrote on the raw exec
calls alone, which every client must be able to run. `bulk` is planned (T7) and
skipped.

## The reference check

`cargo nextest run -p beyond-agent --test it -E 'test(/sdk_conformance/)'`
replays every scenario through the agent's own `ShellFs` and `HttpExecRunner`
against a loopback `execd`, in several backend profiles (plain, bearer, a v1
endpoint without stdin, a v1.1-only endpoint, forced no-stdin, forced `grep` and
busybox-grep rungs, baseline jobs). It proves the scenarios say what the
implementation the spec was read from does today. Known defects of that
implementation are marked `xfail` and listed in FILES-OVER-EXEC.md; they must
fail there and pass everywhere else.

## Adding a client

Write a driver: a process that reads the requests of [DRIVER.md](DRIVER.md) on
stdin and answers on stdout, built on your client's public exec call and helpers
only. Answer `unsupported` for ops you do not have yet. Run the harness against
`execd`; fix what fails, and do not edit a scenario to make a client pass: a
scenario is wrong only if the reference check says so.

## Adding a backend

A backend is a URL (and a bearer, if it has one) of an exec endpoint, and a
directory in the sandbox the harness may use. Start `execd` anywhere (see
`crates/execd/README.md`) or point at any v1.1 to v1.3 endpoint. Scenarios that
need a capability the backend lacks are skipped, visibly.

## Not covered

A client-side request deadline (`timeout`) needs an endpoint that never answers
and a 30 second grace; no scenario waits for it.
