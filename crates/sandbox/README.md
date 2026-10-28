# beyond-sandbox

**Run commands, edit files and keep jobs in any sandbox, from Rust.** One client for the exec protocol,
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
    let uname = sb.exec("uname", ["-a"]).run().await?; // stdout and stderr are the bytes the command wrote
    println!("{}", uname.stdout_text());
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
    sb.jobs()?.start_kept(&scope, &id, &spec).await?; // keeps running after this process exits
    Ok(())
}
```

## What is proven

- **One shared conformance matrix passes: 2,164 checks, 0 failures,** across four backend profiles (a conforming `execd` with
  and without bearer, `rg` / GNU `grep` / busybox `grep`) plus a transport that is nothing but `sh`. The scenarios are the
  ones in [`sdk/conformance`](../../sdk/conformance), which the TypeScript, Python and MCP clients are held to as well.
- **The layers compose.** The same files, helpers and bulk scenarios run over a transport that is nothing but
  `sh` on this host (written inside the test, no HTTP anywhere, and no jobs): 263 checks pass, 22 skipped
  ([`tests/sdk_conformance.rs`](tests/sdk_conformance.rs)).
- **A user-written helper needs only `run`.** A `git status` helper written on raw `Exec` runs unchanged over any
  transport ([`tests/sandbox_surface.rs`](tests/sandbox_surface.rs)).
- **Token renewal is exactly one retry:** a `401` asks the source once for a new token and repeats that request once;
  eight concurrent requests that all hit the `401` share one renewal; a second `401` is final
  ([`tests/token_refresh.rs`](tests/token_refresh.rs)).
- **Parsers are fuzzed:** search records, `stat`, listings, base64, streaming frames and jobs job answers
  ([`fuzz/`](fuzz), four minutes per target in CI).

## Three layers, each usable alone

| layer      | what                                          | you use                                                                      |
| ---------- | --------------------------------------------- | ---------------------------------------------------------------------------- |
| 1. exec    | run a command: one-shot, streamed, with stdin | `Exec` (the trait), `HttpExec`, or your own                                  |
| 2. helpers | files and search, jobs: built only on layer 1 | `Files::over(exec)`, `Jobs::over(exec)`                                      |
| 3. surface | one object that owns all three                | `Sandbox::connect(url, token)`, `.exec(..)`, `.files()`, `.jobs()`, `.raw()` |

Write a transport for anything that can speak the exec protocol (a vendor SDK, a test double) by implementing `Exec`:
`run`, `run_streaming`, `run_streaming_tracked` and `run_attached`, plus `sandbox_jobs` for the target's jobs. A transport
with no jobs of its own leaves `sandbox_jobs` at `None`, and `Sandbox::jobs()` answers `Unsupported`.

**Raw output is bytes.** `sb.exec(..).run()` returns the exit code and stdout and stderr as `Vec<u8>`: exactly what the
command wrote. The endpoint carries them as base64, so a `tar`, a `gzip` or a `cat` of a binary comes back intact, and
there is no flag to ask for it. A transport of your own returns its bytes in `ExecResult`; `stdout_text()` is the one
place they become text.

## Protocol and conformance

The wire protocol is specified once, in the module doc of [`src/exec_endpoint.rs`](src/exec_endpoint.rs). The exact
commands and parsing of the file helpers are [`sdk/conformance/FILES-OVER-EXEC.md`](../../sdk/conformance/FILES-OVER-EXEC.md).
To check an endpoint of your own, the verifier is `exec_endpoint::conformance` in `beyond-agent`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how it works.
