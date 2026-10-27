# beyond-sandbox

**Run commands, edit files and keep jobs in any sandbox, from Rust.** One client for the exec protocol (v1.1 to v1.4),
with the file and job helpers an agent needs built on top of it. It is the code the Beyond agent itself runs in a
sandbox, extracted so anything else can use it.

```rust,no_run
use beyond_sandbox::exec_endpoint::jobs::{JobId, JobScope, JobSpec};
use beyond_sandbox::{FsBackend, Sandbox, TokenSource};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The token source is asked again, once, if the endpoint answers 401 (Beyond's expire within 24 h).
    let sb = Sandbox::connect(
        "https://exec.example/run",
        TokenSource::command(["mint-exec-token"]),
    )
    .await?;
    println!("{}", sb.exec("uname", ["-a"]).run().await?.stdout);
    sb.files()
        .write_bytes("/work/hello.txt".as_ref(), b"hi\n".to_vec())
        .await?;
    let hit = beyond_sandbox::fs::query("hi", "/work".into(), 10);
    println!("{} matches", sb.files().search(&hit).await?.hits.len());
    let (scope, id) = (JobScope::for_session(None, "demo"), JobId::fresh());
    let spec = JobSpec {
        program: "npm".into(),
        args: vec!["run".into(), "dev".into()],
        cwd: Some("/work".into()),
        label: "dev server".into(),
    };
    sb.jobs().start_kept(&scope, &id, &spec).await?; // keeps running after this process exits
    Ok(())
}
```

## What is proven

- **One shared conformance matrix passes: 3,661 checks, 0 failures,** across ten backend profiles (a conforming `execd`
  with and without bearer, no stdin, `rg` / GNU `grep` / busybox `grep`, an HTTP v1 and a v1.1 endpoint, native and
  baseline jobs). The scenarios are the ones in [`sdk/conformance`](../../sdk/conformance), which the TypeScript and
  Python clients are held to as well.
- **The layers compose.** The same files, jobs, helpers and bulk scenarios run over a transport that is nothing but
  `sh` on this host (written inside the test, no HTTP anywhere): 335 checks pass, 31 skipped, identical to the HTTP
  client's counts on the same suites ([`tests/sdk_conformance.rs`](tests/sdk_conformance.rs)).
- **A user-written helper needs only `run`.** A `git status` helper written on raw `Exec` runs unchanged over any
  transport ([`tests/sandbox_surface.rs`](tests/sandbox_surface.rs)).
- **Token renewal is exactly one retry:** a `401` asks the source once for a new token and repeats that request once;
  eight concurrent requests that all hit the `401` share one renewal; a second `401` is final
  ([`tests/token_refresh.rs`](tests/token_refresh.rs)).
- **Parsers are fuzzed:** search records, `stat`, listings, base64, v1.2 frames and v1.3 job answers
  ([`fuzz/`](fuzz), four minutes per target in CI).

## Three layers, each usable alone

| layer      | what                                          | you use                                                                              |
| ---------- | --------------------------------------------- | ------------------------------------------------------------------------------------ |
| 1. exec    | run a command: one-shot, streamed, with stdin | `Exec` (the trait), `HttpExec`, `TemplateRunner` (`ssh`, `docker exec`), or your own |
| 2. helpers | files and search, jobs: built only on layer 1 | `Files::over(exec)`, `Jobs::over(exec)`                                              |
| 3. surface | one object that owns all three                | `Sandbox::connect(url, token)`, `.exec(..)`, `.files()`, `.jobs()`, `.raw()`         |

Write a transport for anything that can run a command (a vendor SDK, a test double) by implementing `Exec::run`; every
other method has a safe default (`Unsupported`, or a deadline), and the helpers probe what the transport can do.

## Protocol and conformance

The wire protocol is specified once, in the module doc of [`src/exec_endpoint.rs`](src/exec_endpoint.rs). The exact
commands and parsing of the file helpers are [`sdk/conformance/FILES-OVER-EXEC.md`](../../sdk/conformance/FILES-OVER-EXEC.md).
To check an endpoint of your own, the verifier is `exec_endpoint::conformance` in `beyond-agent`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how it works.
