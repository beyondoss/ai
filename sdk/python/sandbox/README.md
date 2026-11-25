# beyond-sandbox

**Connect to any sandbox that speaks the exec protocol.** This package does **not** create, pause, or
destroy VMs — lifecycle stays with the platform that gave you the URL (Beyond, E2B with
[`execd`](../../../crates/execd) installed, Docker, …). Here you run commands, read, write and search
files, and keep background jobs. Pure Python, no dependencies, sync and asyncio, fully typed.

```python
from beyond_sandbox import Sandbox

sb = Sandbox.connect("https://sandbox.example/exec", token="...")   # probes; bad URL/token fails here
sb.exec("git", ["status", "--porcelain"], cwd="/work").stdout_text  # argv, not a shell string
sb.files.write("/work/hello.py", "print('hi')\n")                   # atomic; str or bytes; keeps mode
sb.files.read_text("/work/hello.py")                                # UTF-8 view of read (lossy, like stdout_text)
sb.files.search("TODO", "/work", glob="*.py", before=1).hits        # rg, grep or busybox grep, whichever is there
with sb.stream("make", ["-j8"], cwd="/work") as frames:             # output as it happens; leaving kills it
    for frame in frames: print(frame)
sb.jobs.start("my-session", "dev", "npm", ["run", "dev"], cwd="/work")  # the sandbox keeps it, not this process
print(sb.jobs.read("my-session", "dev").data)                       # its output, from any offset, from any process
```

```python
from beyond_sandbox import AsyncSandbox

sb = await AsyncSandbox.connect("https://sandbox.example/exec", token="...")
await sb.files.write("/work/a.txt", "x")
results = await asyncio.gather(sb.exec("uname"), sb.files.read_text("/work/a.txt"))
```

It speaks the exec protocol (one `POST` per command; streaming; sandbox-native jobs), as
served by [`execd`](../../../crates/execd) or any endpoint that implements it. Behaviour is specified
by [FILES-OVER-EXEC.md](../../conformance/FILES-OVER-EXEC.md) and proven by the shared conformance
matrix (`mise run test:clients`), which runs this package's driver (sync and async) against every
backend next to the Rust, MCP and TypeScript clients.

## Three layers, each usable alone

| layer       | what                                                                                       | import                                     |
| ----------- | ------------------------------------------------------------------------------------------ | ------------------------------------------ |
| 1 Transport | raw exec: `run`, `stream`, and jobs                                                        | `HttpTransport`, `AsyncHttpTransport`      |
| 2 Helpers   | `Files` (stat, read, read_text, write, write_if_unchanged, mkdir, list, search) and `Jobs` | `Files`, `Jobs`, `AsyncFiles`, `AsyncJobs` |
| 3 Surface   | `Sandbox` gathers them and always keeps raw exec in reach                                  | `Sandbox`, `AsyncSandbox`                  |

A helper is built on a transport's public `run` and nothing else, so everything in layer 2 works over
any transport. `sandbox.transport` is the transport; `sandbox.exec` and `sandbox.stream` are its calls.

Every failure is a `SandboxError` whose `kind` is the spec's error name (`not_found`, `is_directory`,
`permission_denied`, `conflict`, `auth`, `timeout`, `too_large`, `not_offered`, ...); each kind is
also a class (`NotFoundError`, `ConflictError`, `AuthError`, `NotOfferedError`, ...). A command that
exits non-zero is a result, not an error.

## Why argv, not a shell string

`exec("ls", ["-l", path])` takes the program and each argument separately. A path like `` `; rm -rf /` ``
is then one inert argument, never parsed by a shell. That matches the FILES-OVER-EXEC rule every helper
follows. Other sandbox SDKs often take one shell string — convenient until a model-supplied path is
interpolated. When you need a pipeline, run a **fixed** script and pass variables as `$1`, `$2`:

```python
sb.exec("sh", ["-c", 'find "$1" -name "$2"', "sh", dir, "*.py"])
```

## Connect probes

`Sandbox.connect(url, token=...)` builds the client and probes once (positional URL, keyword options —
idiomatic Python). A bad token raises `AuthError` there. `await AsyncSandbox.connect(...)` does the
same under asyncio. Construct `Sandbox(transport)` / `AsyncSandbox(transport)` when you already hold a
transport and do not want a probe (tests, BYO). The TypeScript client uses the same eager promise with
an options-object `await Sandbox.connect({ url, token })`.

`jobs.start(scope, id, command, args, *, cwd=..., label=...)` is positional for the required pieces
and keyword-only for the rest. The TypeScript client takes a `JobSpec` object instead — same wire
protocol, each language shaped for its callers.

The wire's other job options are keyword arguments under the wire's names: `jobs.start_held(...)` (`hold: true`: the job lives
only as long as its event stream unless you `jobs.keep(scope, id, num=, kind=, seen=)` it), `jobs.seen(scope, id, offset)`,
`grace_ms` on `kill` and `wipe` (the SIGTERM grace, default 300 ms) and `heartbeat_seconds` on `follow`, `collect`,
`start` and `start_held` (1 to 240, the `Keepalive-Ping-Interval` header; a stream is presumed dead after two beats and ten
seconds of silence, never under 30 s). `keep`'s `notify` is not offered: it carries a bearer token, which this client does
not hold. `jobs.collect` follows a job to its end and returns everything (it was `attach`).

Job failures are typed, each a subclass of the kind it was before (`except NotFoundError` keeps working): `JobGoneError`
(a `NotFoundError`: `follow` or `collect` of a job that is not there, or a `read` whose job was removed), `JobExistsError`
(a `ConflictError`: `start` of an id that exists), `DiskFullError` (an `IoError`: `start` on a full disk, 507), and for
`write_stdin`: `StdinClosedError` (`.reason` is `"never_opened"` or `"closed"`; retrying cannot succeed), `StdinGapError`
(the bytes before it never came) and `StdinBusyError` (16 writes already wait: the same write, later, is safe).

### Breaking

- `Jobs.attach` / `AsyncJobs.attach` are renamed `collect` (follow to the end, collected); `follow` is unchanged.
- `Jobs.read` of a removed or unknown job raises `JobGoneError` (a `NotFoundError`) instead of returning state `gone`.
- Every `jobs.start` and `jobs.start_held` carries an idempotency key: `idempotency_key` when you pass one (1 to 128 characters of `[0-9A-Za-z._:-]`), else one generated for the call. A start sent again with the same key and command answers the job the first send started and runs nothing, even across an execd restart; the key of another command, or of another id, is a `KeyConflictError` (409). An endpoint that does not echo the key (one that predates start keys) is `NotOfferedError`: the job it started is stopped, because a start sent again for a lost answer would run twice there. Pass your own so a re-run of a whole program returns its job.
- A start of an existing id is `JobExistsError`, a full disk `DiskFullError`, and a stdin write's 409 and 429 are `StdinClosedError`, `StdinGapError` and `StdinBusyError` (all still `ConflictError`s).
- A custom `job_call` / `job_stream` is passed `heartbeat_seconds`, `env`, `command`, `args` and `cwd` only when set.

## Authentication

`token` is a string, or a function that returns one. The function is called on first use and once
more if the endpoint answers `401`, after which that one request is sent again. That is the only
retry in the package; every other failure is reported as it happened.

```python
Sandbox.connect(url, token=lambda: fetch_short_lived_token())
await AsyncSandbox.connect(url, token=fetch_token_async)    # a coroutine function works too
```

## Bring your own transport

A transport is anything with `run` (and `stream`, `job_call`, `job_stream` for the rest): an SSH
session, `docker exec`, a vendor SDK, a test double. `Files` needs only `run`. This one runs commands
locally:

```python
import subprocess
from beyond_sandbox import ExecResult, Files

class LocalTransport:
    def run(self, command, args=(), *, cwd=None, timeout_ms=120_000, stdin=None):
        p = subprocess.run([command, *args], cwd=cwd, input=stdin or b"",
                           capture_output=True, timeout=timeout_ms / 1000)
        return ExecResult(p.returncode, p.stdout, p.stderr)

files = Files(LocalTransport())
files.write("/tmp/x", "hello")      # the same atomic write and search, with no HTTP at all
```

`HttpTransport` is the default implementation, not the definition. `Sandbox` and `Jobs` also need
`stream`, `job_call` and `job_stream` (see `JobTransport`).

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

## Copy whole trees

`upload` and `download` move a directory between this machine and the sandbox as a tar stream over
plain exec, so the sandbox needs only `sh` and a `tar` (GNU or busybox), and this machine none.
Modes, mtimes and symlinks survive; a link is stored, never followed.

```python
sb.upload("./project", "/work/project")            # a directory's contents, or a list of paths
sb.download("/work/project", "./out")              # a directory's contents, or one file
```

- **One request** when the archive fits `chunk_bytes` (16 MiB); larger trees are staged in chunks
  and extracted by one last command. Memory held is about two chunks, however big the tree.
- **Downloads stream**: stdout frames are spooled to an unnamed temporary file as they arrive and read
  back with the standard library's `tarfile` (no dependency), so size is not capped by the one-shot
  response limit and memory stays flat. The extractor applies the rules of the spec itself rather than
  `tarfile`'s `filter="data"`, which refuses symlinks whose target leaves the destination (allowed here:
  a link is never followed); no minimum Python above 3.10 is needed.
- **The sandbox is not trusted.** A member with `..`, an absolute name, or a path below a symlink is
  refused (`protocol`, `unsafe archive member NAME: ...`); nothing is written through a link.
- **Failures are reported**: a destination that is a file is `not_a_directory`, a read-only or full
  filesystem is `io` naming it, a directory this call created is removed on failure.

Both return a `Transfer(files, dirs, links, bytes)`; `AsyncSandbox` has the same two methods.

## Behaviour worth knowing

- `files.write` replaces a file atomically (temp beside the file, then `mv`), keeps its mode, writes
  through symlinks, and never leaves a temp behind; `data` is `str` (UTF-8) or `bytes` and travels as
  the command's stdin. `files.read_text` is `read` decoded the same way as `stdout_text`.
  `files.write_if_unchanged(path, data, expected_mtime)` raises `ConflictError` on a mismatch
  (`None` means the file must not exist); it is two round trips, a guard rather than a lock.
- `files.search` picks `rg`, GNU `grep` or busybox `grep`, reports the engine
  (`sandbox.capabilities().search_engine`), sorts by path components, counts only matches toward
  `limit`, and clips long lines around the match. Engines differ in regex dialect and in honouring
  `.gitignore`; the report says which one ran.
- **Raw exec is bytes.** `sandbox.exec()` returns `stdout` and `stderr` as `bytes` (`stdout_text` and
  `stderr_text` are the lossy views): the endpoint answers both as base64, so a `tar`, a `gzip` or a
  `cat` of a binary comes back intact. The file and job helpers decode that output as text in one
  place, `ExecResult.stdout_text` / `stderr_text`.
- The response of a one-shot call is read incrementally and capped at 16 MiB (of decoded output, stdout and
  stderr together); over the cap the call raises `TooLargeError` and says the command ran. Streams are never
  buffered whole.
- Closing a stream (`with`, `.close()`) kills the command. Jobs outlive the connection that started
  them: start with `jobs.start`, follow with `jobs.collect` (to the end, collected) or `jobs.follow` (events), stop with `jobs.kill`.
- One connection per request, no pooling and no proxy support: the standard library's `socket` and
  `ssl`, or `asyncio` streams, and nothing else. TLS verifies the certificate against the system
  roots (the default context is built once per transport); pass `ssl_context=` to change that.
- The environment (busybox or GNU `find`, `rg` or `grep`, `dd`/`base64` flags) is probed once per
  `Sandbox` in a single `sh -c` round trip: `connect` runs it up front; constructing from a transport
  probes on the first read, list or search. Writes need no probe. A probe that fails (unreachable,
  `401`) raises and is not remembered.

## How it is built

`_wire.py` and `_http.py` are the protocol and an HTTP/1.1 framing with no I/O; `transport.py` moves
bytes with `socket`, `atransport.py` with `asyncio` streams. The helpers (`_fs.py`, `_jobs.py`) are
generators that _ask_ for commands and are driven by `transport.run` or `await transport.run`
(`_drive.py`), so each helper exists once, for sync and async. The shell scripts
in `_scripts.py` are generated from the reference (`scripts/gen_scripts.py`, checked by a test).

## Develop

```sh
mise run install:py    # .venv with the dev tools (uv if present, else venv + pip)
mise run check:py      # ruff format + lint, mypy --strict
mise run test:py       # pytest; loopback tests run a real execd
mise run test:clients  # the shared conformance matrix, every client
```

Python 3.10 or newer. MIT.
