# beyond-sandbox

Run commands in a sandbox, and read, write and search its files, over the exec protocol. Pure
Python, no dependencies, sync and asyncio, fully typed.

```python
from beyond_sandbox import Sandbox

sb = Sandbox.connect("https://sandbox.example/exec", token="...")   # nothing is sent yet
sb.exec("git", ["status", "--porcelain"], cwd="/work").stdout_text  # raw exec: a result, any exit code
sb.files.write("/work/hello.py", b"print('hi')\n")                  # atomic, keeps mode, follows symlinks
sb.files.search("TODO", "/work", glob="*.py", before=1).hits        # rg, grep or busybox grep, whichever is there
with sb.stream("make", ["-j8"], cwd="/work") as frames:             # v1.2: output as it happens; leaving kills it
    for frame in frames: print(frame)
sb.jobs.start("my-session", "dev", "npm", ["run", "dev"], cwd="/work")  # the sandbox keeps it, not this process
print(sb.jobs.read("my-session", "dev").data)                       # its output, from any offset, from any process
```

```python
from beyond_sandbox import AsyncSandbox

async with AsyncSandbox.connect("https://sandbox.example/exec", token="...") as sb:
    await sb.files.write("/work/a.txt", b"x")
    results = await asyncio.gather(sb.exec("uname"), sb.files.read("/work/a.txt"))
```

It speaks the exec protocol v1.1 to v1.3 (one `POST` per command; streaming; sandbox-native jobs), as
served by [`execd`](../../../crates/execd) or any endpoint that implements it. Behaviour is specified
by [FILES-OVER-EXEC.md](../../conformance/FILES-OVER-EXEC.md) and proven by the shared conformance
matrix (`mise run test:clients`), which runs this package's driver (sync and async) against every
backend next to the Rust, MCP and TypeScript clients.

## Three layers, each usable alone

| layer       | what                                                                            | import                                     |
| ----------- | ------------------------------------------------------------------------------- | ------------------------------------------ |
| 1 Transport | raw exec: `run`, `stream`, and native jobs when offered                         | `HttpTransport`, `AsyncHttpTransport`      |
| 2 Helpers   | `Files` (stat, read, write, write_if_unchanged, mkdir, list, search) and `Jobs` | `Files`, `Jobs`, `AsyncFiles`, `AsyncJobs` |
| 3 Surface   | `Sandbox` gathers them and always keeps raw exec in reach                       | `Sandbox`, `AsyncSandbox`                  |

A helper is built on a transport's public `run` and nothing else, so everything in layer 2 works over
any transport. `sandbox.transport` is the transport; `sandbox.exec` and `sandbox.stream` are its calls.

Every failure is a `SandboxError` whose `kind` is the spec's error name (`not_found`, `is_directory`,
`permission_denied`, `conflict`, `auth`, `timeout`, `too_large`, `not_offered`, ...); each kind is
also a class (`NotFound`, `Conflict`, `AuthError`, ...). A command that exits non-zero is a result, not an error.

## Authentication

`token` is a string, or a function that returns one. The function is called on first use and once
more if the endpoint answers `401`, after which that one request is sent again. That is the only
retry in the package; every other failure is reported as it happened.

```python
Sandbox.connect(url, token=lambda: fetch_short_lived_token())
AsyncSandbox.connect(url, token=fetch_token_async)          # a coroutine function works too
```

## Bring your own transport

A transport is anything with `run` and `stream`: an SSH session, `docker exec`, a vendor SDK, a test
double. `Files`, `Jobs` and `Sandbox` take it unchanged. This one runs commands locally:

```python
import subprocess
from beyond_sandbox import ExecResult, Files, NotOffered, Sandbox

class LocalTransport:
    def run(self, command, args=(), *, cwd=None, timeout_ms=120_000, stdin=None):
        p = subprocess.run([command, *args], cwd=cwd, input=stdin or b"",
                           capture_output=True, timeout=timeout_ms / 1000)
        return ExecResult(p.returncode, p.stdout, p.stderr)

    def stream(self, command, args=(), *, cwd=None, timeout_ms=None):
        raise NotOffered("this transport does not stream")

sb = Sandbox(LocalTransport())
sb.files.write("/tmp/x", b"hello")     # the same atomic write, search and jobs, with no HTTP at all
```

Without `stream` support `stream` raises `NotOffered`; jobs fall back to a baseline built from plain
commands. A transport that also has `job_offer`, `job_call` and `job_stream` (see `JobTransport`) gets
native v1.3 jobs. `HttpTransport` is the default implementation, not the definition.

## Your own helper on raw exec

The file helpers have no private powers. Anything they do, you can do with `exec`:

```python
def changed_since(sb: Sandbox, root: str, marker: str) -> list[str]:
    r = sb.exec("find", [root, "-type", "f", "-newer", marker])
    return sorted(r.stdout_text.split())
```

Pass anything variable as its own argument. A pipeline is a fixed script run as
`sh -c SCRIPT sh ARG...` with the variable parts arriving as `$1`, `$2`; never build a command line
from a path.

## Behaviour worth knowing

- `files.write` replaces a file atomically (temp beside the file, then `mv`), keeps its mode, writes
  through symlinks, and never leaves a temp behind. Without stdin on the endpoint it sends 48 KiB
  chunks. `files.write_if_unchanged(path, data, expected_mtime)` raises `Conflict` on a mismatch
  (`None` means the file must not exist); it is two round trips, a guard rather than a lock.
- `files.search` picks `rg`, GNU `grep` or busybox `grep`, reports the engine
  (`sandbox.capabilities().search_engine`), sorts by path components, counts only matches toward
  `limit`, and clips long lines around the match. Engines differ in regex dialect and in honouring
  `.gitignore`; the report says which one ran.
- **Raw exec is bytes.** `sandbox.exec()` returns `stdout` and `stderr` as `bytes` and an `exact` flag
  (`stdout_text` and `stderr_text` are the lossy views). A `tar`, a `gzip` or a `cat` of a binary comes
  back intact when the endpoint honours `output_base64` (`execd` does) or streams (v1.2). A plain v1.1
  endpoint can only answer text, with invalid UTF-8 already replaced by U+FFFD; the result then carries
  that text as UTF-8 and `exact=False`, so the client never passes text off as bytes. The file and job
  helpers read text and call `transport.run_text` when the transport has it (`run` otherwise).
- The response of a one-shot call is read incrementally and capped at 16 MiB (of decoded output, with
  `output_base64`); over the cap the call raises `TooLarge` and says the command ran. Streams are never
  buffered whole.
- Closing a stream (`with`, `.close()`) kills the command. Jobs outlive the connection that started
  them: start with `jobs.start`, follow with `jobs.attach` or `jobs.follow`, stop with `jobs.kill`.
- One connection per request, no pooling and no proxy support: the standard library's `socket` and
  `ssl`, or `asyncio` streams, and nothing else. TLS verifies the certificate against the system
  roots; pass `ssl_context=` to change that.
- Probes run once per `Sandbox`, on first use. A probe that fails (unreachable, `401`) raises from the
  call that needed it and is not remembered.

## How it is built

`_wire.py` and `_http.py` are the protocol and an HTTP/1.1 framing with no I/O; `transport.py` moves
bytes with `socket`, `atransport.py` with `asyncio` streams. The helpers (`_fs.py`, `_jobs.py`) are
generators that _ask_ for commands and are driven by `transport.run` or `await transport.run`
(`_drive.py`), so each fallback rung of each helper exists once, for sync and async. The shell scripts
in `_scripts.py` are generated from the reference (`scripts/gen_scripts.py`, checked by a test).

## Develop

```sh
mise run install:py    # .venv with the dev tools (uv if present, else venv + pip)
mise run check:py      # ruff format + lint, mypy --strict
mise run test:py       # pytest; loopback tests run a real execd
mise run test:clients  # the shared conformance matrix, every client
```

Python 3.10 or newer. MIT.
