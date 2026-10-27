# Files over exec

The normative specification of the file, search and job helpers that every
Beyond sandbox client implements on top of the exec protocol. It was derived by
reading `ShellFs` (`crates/agent/src/tools/fs/shell.rs`) and `HttpExecRunner`
(`crates/agent/src/exec_endpoint.rs`) as they are, and is proven by the
scenarios in [`scenarios/`](scenarios) (see [README.md](README.md) and
[DRIVER.md](DRIVER.md)). A client in any language can be written from this
document and the exec protocol's own spec (the module doc of
`crates/agent/src/exec_endpoint.rs`) alone.

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

Where the reference implementation has a defect, this document says what a
client MUST do instead and lists it in
[Known deviations](#known-deviations-from-the-reference-implementation). The
scenarios carry an `xfail` mark on exactly those checks, for the reference
driver only.

## 1. The three layers

```text
surfaces   Sandbox object, MCP tools           may use layers 2 and 1
helpers    files, search, jobs, bulk copy      may use layer 1 only
transport  exec: one-shot, stream, jobs        may use nothing of this spec
```

1. **Transport.** A raw `exec` call: one-shot (v1.1), streaming (v1.2) and the
   job calls (v1.3). It is a plain interface (command, args, cwd, timeout,
   stdin in; exit status and bytes out) that a user MUST be able to replace
   with their own implementation (an SSH session, a vendor SDK, a test double).
   The HTTP implementation is the default, not the definition.
2. **Helpers.** Everything in sections 4 to 6. A helper MUST be built only on
   the public layer 1 call. It MUST NOT use a private path, a second
   connection, an endpoint-specific route, or any state of the transport beyond
   what the interface exposes. Consequently every helper works over any
   transport, and a user can write a helper of their own with the same powers.
3. **Surfaces.** The convenient `Sandbox` object and the MCP tools. They
   compose helpers, and they MUST always expose raw exec as well, so that a
   user is never limited to what the surface offers.

The conformance scenarios hold all three to account: the `exec`, `stream` and
`jobs` suites test layer 1, `files` tests layer 2, and `helpers` runs
user-written helpers that use layer 1 only and must work on every client.

## 2. The exec call a helper is built on

A helper calls

```text
run(command, args[], { cwd?, timeout, stdin? }) -> { code, stdout, stderr }
```

or fails with a transport-level error (section 3.2).

- `command` and every entry of `args` are separate strings. A helper MUST NOT
  assemble a command line to be parsed by a shell. Anything variable (a path, a
  pattern, a size) travels as its own argument. A pipeline is only ever a
  **fixed script** run as `sh -c SCRIPT sh ARG...`, where `SCRIPT` is a
  constant in this document and the variable parts arrive as `$1`, `$2`, ...
  (the `sh` after the script is `$0`). A path such as `'; rm -rf /` is then one
  inert argument.
- Helpers send no `cwd`: relative paths are the endpoint's business, and
  helpers SHOULD be given absolute paths.
- `stdin` is bytes. It is sent only by `write` and only after the stdin probe
  proved the endpoint delivers it.
- `stdout` and `stderr` are **text**. Over the one-shot envelope (v1.1) they are
  JSON strings, so bytes that are not valid UTF-8 reach the client as U+FFFD,
  decided by the endpoint. A helper that must move arbitrary bytes MUST encode
  them (base64) inside the command.
- `code` is the exit status. Over v1.1 a command killed by signal `S` has
  `code` `128+S`, and a command that hit `timeout_ms` has `code` 124 with stderr
  ending `command exceeded its timeout`. A command that cannot be started (or
  whose `cwd` is missing) has `code` 127 and the reason on stderr. None of
  these is a transport error.
- Timeouts: helpers use 120 s per command (`timeout_ms` 120000) and 10 s for
  capability probes.

On the wire (the HTTP transport) a call is one `POST` of

```json
{
  "command": "sh",
  "args": ["-c", "...", "sh", "/a"],
  "timeout_ms": 120000,
  "stdin_base64": "..."
}
```

(`cwd` and `stdin_base64` only when present) with every configured header,
among them `Authorization: Bearer <token>`, and the answer
`{"exit_code": N, "stdout": "...", "stderr": "..."}`. The client reads the body
incrementally and stops at a cap (default 16 MiB): over the cap the call fails
with `too_large` and says the command ran. The request deadline is
`timeout_ms` plus 30 s of grace so that the endpoint's own timeout report wins.

## 3. Errors

### 3.1 Vocabulary

Every helper reports failures with one of these kinds, plus a free-text
`detail`. A client MAY add fields; it MUST NOT invent other kinds for these
situations.

| kind                | meaning                                                                     |
| ------------------- | --------------------------------------------------------------------------- |
| `not_found`         | the path (or its parent) does not exist                                     |
| `is_directory`      | a file operation met a directory                                            |
| `not_a_directory`   | a path component that must be a directory is not                            |
| `permission_denied` | the sandbox user may not do it                                              |
| `io`                | a command ran and failed in a way this table does not name                  |
| `invalid_query`     | the search pattern or glob was rejected                                     |
| `conflict`          | `write_if_unchanged`: the file's mtime is not the expected one              |
| `too_large`         | the endpoint's answer exceeded the response cap                             |
| `auth`              | the endpoint refused the credentials (HTTP 401 or 403)                      |
| `timeout`           | the request deadline passed (a transport timeout, not a command's exit 124) |
| `transport`         | the endpoint could not be reached, or a stream broke                        |
| `protocol`          | the endpoint answered with something that is not the protocol               |
| `endpoint`          | the endpoint refused the request with another status (`detail` names it)    |
| `not_offered`       | the endpoint lacks the capability (it does not stream; it has no jobs)      |
| `invalid_request`   | the caller's arguments cannot form a request; nothing was sent              |

### 3.2 From the transport

| what happened                                                       | kind          |
| ------------------------------------------------------------------- | ------------- |
| connection refused, reset, DNS failure                              | `transport`   |
| the request deadline passed                                         | `timeout`     |
| HTTP 401 or 403                                                     | `auth`        |
| HTTP 422 whose body says the output exceeded the cap, or client cap | `too_large`   |
| any other non-2xx                                                   | `endpoint`    |
| 200 whose body is not the expected JSON                             | `protocol`    |
| a v1.2 stream ends without its final frame, or a `{"error"}` frame  | `transport`   |
| a stream asked of an endpoint that did not prove v1.2               | `not_offered` |

A transport-level failure is never a capability: see
[probe errors](#52-probe-errors).

### 3.3 From a command that ran and failed

A helper that gets a non-zero exit takes the **first line of trimmed stderr**
(or `command failed` if there is none) as `detail`, prefixed as each operation
below says, and classifies it case-insensitively by the first rule that
matches:

| stderr contains                                                               | kind                |
| ----------------------------------------------------------------------------- | ------------------- |
| `no such file or directory`, `directory nonexistent`, `nonexistent directory` | `not_found`         |
| `is a directory`                                                              | `is_directory`      |
| `not a directory`                                                             | `not_a_directory`   |
| `permission denied`                                                           | `permission_denied` |
| anything else                                                                 | `io`                |

The patterns cover GNU coreutils, busybox, dash and bash. (The `read` script
prints the three `(os error N)` wordings itself so the text does not depend on
the sandbox's tools.)

## 4. Capability probe

A helper client probes the sandbox once and records the results. Each probe is
one `run` with a 10 s timeout.

| capability    | command                                                                                                                  | present when                                   |
| ------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| `rg`          | `rg --version`                                                                                                           | exit 0                                         |
| `grep_null`   | `grep -rnZ -e __cap_probe__ -- /dev/null`                                                                                | exit 0 or 1                                    |
| `find_printf` | `find /dev/null -maxdepth 0 -printf %y`                                                                                  | exit 0 or 1                                    |
| `stdin`       | `cat` with stdin `stdin-probe`                                                                                           | exit 0 **and** stdout is exactly `stdin-probe` |
| `read_window` | `sh -c 'printf abc \| dd iflag=skip_bytes,count_bytes skip=1 count=1 status=none 2>/dev/null \| base64 -w0 2>/dev/null'` | exit 0 and stdout trimmed is `Yg==`            |

Derived: the **search engine** is `rg` if `rg`; otherwise `grep` (GNU grep:
`grep_null`); otherwise `busybox-grep`.

`stdin` is proven by an echo, never assumed. A v1 endpoint ignores
`stdin_base64`, runs `cat` on empty input and answers with empty stdout, so the
probe fails and writes fall back to chunks. Writing through such an endpoint as
though stdin arrived would replace the file with an empty one.

### 4.1 What a client reports

`capabilities` (DRIVER.md) reports `stdin`, `stream`, `jobs`
(`native`, `baseline` or `none`), `search_engine`, `find_printf` and
`read_window`. `stream` is true only when the v1.2 probe of the exec spec
succeeded.

### 4.2 Probe errors

If a probe request fails at the transport level (unreachable, `auth`,
`endpoint`, `timeout`), the client MUST report that error from the operation
that needed the probe, and MUST NOT cache a capability set learned from failed
probes. (The reference `ShellFs::connect` reads every probe error as "absent",
so with a wrong bearer it concludes the sandbox has no stdin and no `dd`, and
the first read then fails with a message about `dd`. See D3.)

Probe overrides for testing are in DRIVER.md (`CONFORMANCE_CAPS`).

## 5. Operations

All paths are absolute strings. All counts and offsets are non-negative
integers. Byte content is carried as raw bytes inside a client and as base64
only in the driver protocol.

### 5.1 stat

`stat(path) -> None | { kind, size, mtime, writable }`

Runs `sh -c STAT_SCRIPT sh PATH` (script A). Parse stdout:

- if the exit is not 0, or trimmed stdout is empty: **does not exist** (not an
  error; this includes a missing parent, a broken symlink, and a path below a
  file);
- otherwise the first line is `W<TAB>T<TAB>SIZE<TAB>MTIME`: `W` is `w`
  (writable) or `-`; `T` is `f` file, `d` directory, `o` other (FIFO, socket,
  device); `SIZE` an integer (a line that does not parse is also "does not
  exist"); `MTIME` is epoch seconds, optionally with a fractional part, of
  which the **integer part** is kept (truncated, not rounded).

`kind` follows symlinks (`[ -d ]`/`[ -f ]`), `writable` is a real access check
for the calling user (`[ -w ]`), not mode bits. If `stat -c` itself fails the
script prints `0<TAB>0`.

A symlink's `size` and `mtime` MUST be the target's. The reference runs
`stat -c` without `-L` and reports the link's own (D2); a client MUST use
`stat -L -c` (GNU and busybox both accept it).

A path that is not valid UTF-8 cannot be written in a JSON request, so such a
file cannot be addressed. It does appear in a `list` (section 5.6), with U+FFFD
in its name.

### 5.2 read

`read(path, offset = 0, max = unbounded) -> bytes`

Runs one command per window of at most **96 KiB (98304 bytes)**:
`sh -c READ_SCRIPT sh PATH OFFSET WANT` (script B), where
`WANT = min(max - collected, 98304)` and `OFFSET` advances by what was
collected. The script prints, on failure, one of
`No such file or directory (os error 2)`, `Is a directory (os error 21)`,
`Permission denied (os error 13)` on stderr and exits 1; otherwise it prints
the window as base64 on one line (`base64 -w0`).

Loop: decode the trimmed stdout (standard base64, whitespace ignored); if this
is the first chunk and it is short (fewer than `WANT` bytes) or `WANT == max`,
return it; otherwise append it, and stop when the chunk was short or `max`
bytes are collected. A **short chunk is end of file**, which also means a file
whose size is an exact multiple of 98304 costs one extra command that returns
nothing.

Rules:

- A non-zero exit is an error, classified by section 3.3, with detail
  `read PATH: LINE`. A missing file MUST NOT read as an empty file: the pipeline
  ends in `base64`, which succeeds on empty input, so the readability guard at
  the top of script B is load-bearing.
- A client MUST refuse to read when the `read_window` probe failed: error `io`,
  `read PATH: this target's dd or base64 does not support
  iflag=skip_bytes,count_bytes / -w0, so files cannot be read`.
- `offset` past the end, and `max` 0, give empty bytes.
- A window is read by its own command: a file that changes between windows
  shows the new bytes to the later window (no snapshot).
- A symlink is followed; a broken one is `not_found`.
- A caller SHOULD `stat` first and refuse `kind` `other`: `dd` on a FIFO with no
  writer blocks until the command's timeout.

### 5.3 write

`write(path, bytes)`: replace the file's whole content, atomically.

The write goes to a temp file `T.tmp.SUFFIX` **beside the file the path
resolves to**, and is committed with `mv -f`. `SUFFIX` is chosen by the client
per write, unpredictable (at least 64 bits of randomness, lowercase hex), so a
preplanted name cannot redirect the write. The scripts (C and D below) do the
resolution, in order:

1. a path starting with `-` is prefixed with `./`;
2. if it is a symlink, resolve it: `readlink -f`, and if that yields nothing
   (a dangling link) one `readlink` hop, joined to the link's directory when
   relative; the prefix rule of 1 is applied again;
3. the temp is created empty with `set -C` (no-clobber: a preplanted temp is
   refused, not followed);
4. if the target exists, the temp gets its mode **before any content**:
   `chmod --reference`, falling back to `chmod "$(stat -L -c %a TARGET)"`
   (busybox); if neither works the write fails (`cannot give the new file the
   mode of ...`) and the temp is removed;
5. the content is appended; the temp's size is checked against the expected
   byte count; only then `mv -f TEMP TARGET`.

A write is therefore atomic (a reader sees the old file or the new one), keeps
the file's mode (a 0600 file stays 0600, a 0755 file stays executable), writes
through symlinks (the link stays a link; a dangling link's target is created),
creates a new file with the default mode, and never leaves a temp behind on a
failure the script sees.

**With stdin** (the `stdin` probe passed): one command,
`sh -c WRITE_STDIN_SCRIPT sh TARGET SUFFIX SIZE` with the whole content as
stdin (script C). The size check is what turns a lost stdin into a failed
write instead of an empty file: a probe is one sample, and an endpoint behind a
load balancer mid-rollout can pass it on one replica and answer the write from
another.

**Without stdin**: one command per **48 KiB (49152 raw bytes)** chunk, each
`sh -c WRITE_CHUNK_SCRIPT sh TARGET SUFFIX CHUNK_B64 MODE COMMIT SIZE` (script
D) where `CHUNK_B64` is the chunk in standard base64 with padding (at most 64
KiB of text, half of Linux's 128 KiB single-argument limit), `MODE` is `new`
for the first chunk and `append` after, `COMMIT` is `commit` for the last chunk
and the empty string otherwise, and `SIZE` the total byte count. The temp is
created by the `new` command and the commit does the size check and the `mv`.
An empty file is one empty chunk (so the temp is still created and committed).
Chunks are sent sequentially.

**On failure** (non-zero exit of any command, or a transport error part-way):
the client sends one best-effort cleanup command,
`sh -c REMOVE_TEMP_SCRIPT sh TARGET SUFFIX` (script E; its result is ignored),
and then reports the failure: a transport error as it is, otherwise
`write PATH: LINE` classified by section 3.3. Stdin and chunk scripts both
remove the temp themselves on failures they see; the cleanup covers the ones
they cannot (a timeout between chunks).

Further rules:

- The parent directory must exist (`not_found`); `write` does not create
  directories.
- A file the caller may not write is refused with `permission_denied` and
  unchanged: the temp inherits the old file's mode (a read-only file gives a
  read-only temp that cannot be appended to).
- A write onto a **directory** MUST fail and change nothing.
- Two writers to one path leave one complete content, never a mixture.

### 5.4 write_if_unchanged

`write_if_unchanged(path, bytes, expected_mtime: int | null)`

1. `m = stat(path)` and take its `mtime` (absent if the path does not exist).
2. Compare **whole epoch seconds**: they match when `m` and `expected_mtime`
   are both absent, or both present and equal. Otherwise fail with `conflict`
   and write nothing.
3. `write(path, bytes)`.

So `expected_mtime` null means "the file must not exist". The key is required.
The two steps are two round trips and the guard is not a lock: it catches a
person or agent editing between a read and a write, and misses an edit that
leaves the mtime second unchanged (a limit, with a scenario). Through a symlink
the comparison MUST use the target's mtime (D2).

### 5.5 mkdir

`mkdir(path)`: `mkdir -p -- PATH` as raw argv (`command` `mkdir`, `args`
`["-p", "--", PATH]`). A non-zero exit is an error `create PATH: LINE`
classified by section 3.3 (making a directory below a file is
`not_a_directory`). Idempotent.

### 5.6 list

`list(path, include_hidden = false, limit = 10000) -> entries[{ name, kind, size }]`

**With `find_printf`**: one command, `find PATH -mindepth 1 -maxdepth 1 -printf
'%Y\t%s\t%f\0'` (the argument is the literal characters `%Y`, backslash, `t`,
and so on; `find` expands the escapes). Fail with `ls PATH: LINE` when the exit
is non-zero **and** stdout is empty; a non-zero exit with output is a partial
listing and is used. Split stdout on NUL; each non-empty record is
`TYPE<TAB>SIZE<TAB>NAME` (split at the first two tabs only; the name may
contain anything but NUL). Then, in order and stopping at `limit` kept entries:
drop hidden names (starting with `.`) unless `include_hidden` **before**
counting; map `TYPE` `f` to `file`, `d` to `dir`, `N` and `?` to **skipped**
(broken link, unstattable), anything else to `other`; `size` is the integer
(0 if unparseable).

`%Y` is the type after following a symlink, so a link to a directory lists as
`dir` and a broken link is omitted. The size of a symlink in this listing is
the link's own; callers needing it `stat`. Order is unspecified; clients MAY
sort and the scenarios compare unordered.

**Without `find_printf`** (busybox): two commands, `find PATH -mindepth 1
-maxdepth 1 -print0` (same failure rule) and the same with `-type d` added.
Entries are the first listing; `name` is the last path component; an entry is
`dir` if its full path is in the second listing, otherwise `file`; `size` is 0.
(So FIFOs and links to directories are `file` there.)

Listing a **file** gives no entries and no error (`find` finds no children).
A missing directory is `not_found`. Names that are not valid UTF-8 appear with
U+FFFD for the bad bytes.

### 5.7 search

```text
search({ pattern, root, literal, ignore_case, glob?, glob_exclude, limit,
         before, after }) -> { hits[{ path, line, text, is_match }], truncated, note }
```

`before` and `after` are clamped to **100**. `limit` counts matches, not
context lines.

#### Engine arguments

The command is `rg` or `grep`, with **no shell**, each item one argument.

`rg` engine (in this order):

```text
--null --with-filename --line-number --no-heading --color=never --hidden --crlf
[--no-require-git]            unless the root is inside a git repository (below)
[--ignore-case] [--fixed-strings]
[--glob G | --glob !G]        the caller's glob; the second form when excluding
--glob !.git
[-B before] [-A after]        each only when greater than 0
-e PATTERN -- ROOT
```

`grep` engine (GNU grep):

```text
-r -n -H -Z -I --color=never
-F | -E                       -F when literal, else -E
[-i]
[--include=G | --exclude=G]
--exclude-dir=.git            unless the root's last path component is .git
[-B before] [-A after]
-e PATTERN -- ROOT
```

where for a non-literal pattern `PATTERN` is rewritten for CRLF files (below).

`busybox-grep` engine: `-r -n -H`, then `-F` or `-E`, `-i`, context flags,
`-e PATTERN -- ROOT`. It has no `-Z`, `-I`, `--color`, `--include`,
`--exclude-dir`; the glob is applied by the client.

**Git root probe** (`rg` only): `--no-require-git` is added unless the root or
an ancestor holds a `.git` entry, decided **in the sandbox** (never from the
client's filesystem): `sh -c GIT_SCRIPT sh ROOT` (script F), exit 0 meaning
inside a repository, cached per root. A failed probe counts as "not inside".
It makes `.gitignore` honored in a plain directory too.

**CRLF-tolerant ERE** (`grep` and `busybox-grep`, non-literal only): `$` as an
end-of-line anchor becomes `<CR>?$` where `<CR>` is the literal carriage
return byte, so `foo$` matches `foo\r\n` lines. Scan the pattern left to right:
a backslash copies itself and the next character; a bracket expression
(starting at `[`, optionally `^`, a leading `]` literal, `[:class:]`, `[=x=]`
and `[.x.]` nesting) is copied verbatim to its closing `]`; a `$` that is
followed by the end of the pattern, `|` or `)` is replaced; every other
character is copied.

#### Exit codes and errors

Exit 0 (matches) and 1 (none) are success. Exit 124, or the endpoint's own
timeout, is `timeout`. Any other exit is told apart by stderr, since the cases
share exit 2:

- every stderr line is a per-path OS error (`PATH: No such file or directory`,
  `Permission denied`, `Is a directory`, ...; `rg` adds `(os error N)`) and one
  of them names the search root itself as missing: `not_found`;
- every line is a per-path OS error otherwise: success. The hits on stdout are
  returned and `note` says how many paths were skipped and names the first;
- anything else (or empty stderr) is the engine rejecting the pattern:
  `invalid_query` with detail `bad regex: STDERR` (trimmed; `bad regex: PROGRAM
  rejected the pattern` if stderr is empty).

#### Parsing

`rg` and `grep -Z` emit records `PATH<NUL>LINE<SEP>TEXT<LF>`, where `SEP` is `:`
for a match and `-` for a context line. Parse by scanning for the **NUL**, not
for newlines: a path may contain a newline. At each position, first check for
a group separator: the bytes `--` + LF (or `--` at the very end) are skipped.
(ripgrep with `--crlf` writes `--` + CR + LF; a client MUST skip that too:
D5.) Otherwise the path runs to the next NUL, the rest of the record to the
next LF; `LINE` is one or more digits followed by `:` or `-`; a record that
does not fit is dropped.

`busybox-grep` emits `PATH:LINE:TEXT` (match) or `PATH-LINE-TEXT` (context). A
line equal to `--` is skipped. Otherwise split at the **first** position `i`
where the byte is `:` or `-`, followed by one or more digits, followed by the
same byte; the path is everything before `i`. This is wrong for a path
containing such a sequence, and a path containing a newline cannot be
represented, which is why busybox is the last resort. The caller's glob is
applied to the full path with glob semantics of the Rust `globset` crate
(`*` also crosses `/`; `**`, `?`, `[..]`, `{a,b}`), case-insensitive when
`ignore_case`; `glob_exclude` negates.

For every engine, `TEXT` is normalized: remove one trailing LF, then one
trailing CR, then **every remaining CR**. Text that is not valid UTF-8 has
U+FFFD for the bad bytes (stdout is text).

#### Post-processing, in this order

1. Drop every hit whose path has a `.git` component **below** the root (the
   root's own components do not count, so a search rooted in `.git` works).
2. **Clip** long lines (below).
3. Sort by path then line, where paths compare **component by component**
   (split on `/`, each component compared bytewise), not as strings: `a/b.txt`
   sorts before `a.txt`.
4. Apply `limit`: count only `is_match` hits; if the number of matches exceeds
   `limit`, cut the list just before the first hit that is the (`limit`+1)-th
   match, then drop every trailing context hit. The result then ends on a match.
   `truncated` is true when matches exceeded the limit **or** the command's
   output was cut short (never the case over HTTP, where an oversized answer is
   an error).
5. `note` is the first line of the trimmed stderr of a successful run, or null.

Context groups that overlap are merged by the engine; separators are not hits.

#### Clipping

`MAX_LINE` is 500 **characters** (Unicode scalar values, not bytes or UTF-16
units). A line of at most 500 characters is unchanged. For a **context** line
longer than that: the first 500 characters followed by `… [truncated]`
(U+2026, then `[truncated]`). For a **match** line longer than 500: find the
first match of the query in the line (the pattern as a Rust-syntax regular
expression, escaped when `literal`, case-insensitive when `ignore_case`; if
that fails to compile or finds nothing, treat as a context line). With `cs` and
`ce` its start and end character offsets and `total` the line's length: if
`ce <= 500` clip as a context line. Otherwise
`pad = (500 - (ce - cs)) / 2` (integer division, floored at 0),
`start = max(0, cs - pad)`, `end = start + 500`; if `end > total` then
`end = total` and `start = total - 500`. The text is `…` + characters
`[start, end)` + (`…` if `end < total`) + `[truncated; columns
{start+1}-{end} of {total}]`.

#### Engines differ; say which you used

`rg` honors `.gitignore`/`.ignore`, skips binary files (any NUL), uses Rust
regex syntax and searches hidden files. `grep` does none of the ignore
handling, skips binary files with `-I` (in a UTF-8 locale also files with
invalid UTF-8), and uses POSIX ERE; `\d` and non-greedy quantifiers are not
available. `busybox-grep` additionally cannot represent some path names (above)
and may report binary files differently. A client MUST report the engine in use
(`capabilities.search_engine`); scenarios that depend on one declare
`requires.search_engine`.

### 5.8 Not specified here

- `glob` (find paths by pattern): the reference computes it host-side with
  `globset`; surfaces build it on `find`/`rg --files` over raw exec.
- `remove`: no counterpart in the reference. A surface that offers it runs
  `rm -rf -- PATH`-style commands through raw exec.
- `upload_dir` / `download_dir` (tar over exec): planned (T7); the scenarios in
  `bulk.json` are marked planned and are skipped.

## 6. Jobs

Jobs are layer 1, specified by the exec protocol (v1.3 natively; the
**baseline** over plain commands where the endpoint has no native jobs; the
baseline's layout and supervisor are in `crates/agent/src/exec_endpoint/jobs.rs`).
A client MUST support native jobs and SHOULD support the baseline; the driver
ops are in DRIVER.md. Behaviours the scenarios pin, for both:

- a job id is 1 to 32 characters of `[0-9a-z]`, chosen by the caller, unique
  within its scope; a client rejects anything else before sending (`invalid_request`);
- `read` returns `state` (`running`, `exited`, `ended`, `lost`, `gone`), `code`,
  `offset` (where the data starts; later than asked when old output was
  dropped), `end` (the end of the job's output so far, whatever `max` was) and
  the data; an unknown job is `gone` with no data, not an error;
- a job's stdout and stderr are one output, its stdin is empty, its status is
  the exit code or `128+N`;
- `kill` signals the whole group (TERM, a grace, KILL), is a no-op for an ended
  or unknown job, and with `remove` deletes jobs whose group is empty;
  `remove` never deletes a running job; `wipe` kills and deletes the scope;
- scopes are separate namespaces; at least the last 8 MiB of output is kept;
- when the shell exits leaving its group alive the job is `exited` with the
  shell's code until the group is empty (`ended`); `attach` ends only then.

## 7. Appendix: the scripts

The scripts are constants. A client MUST send them byte for byte (comments and
spacing included, so that behaviour on every `sh` is the one tested). Each is
the single `SCRIPT` argument of `sh -c SCRIPT sh ARG...`.

### A. STAT_SCRIPT (`$1` path)

```sh
[ -e "$1" ] || exit 1
if [ -d "$1" ]; then t=d
elif [ -f "$1" ]; then t=f
else t=o
fi
if [ -w "$1" ]; then w=w; else w=-; fi
printf '%s	%s	' "$w" "$t"
stat -c '%s	%Y' -- "$1" 2>/dev/null || printf '0	0'
printf '
'
```

### B. READ_SCRIPT (`$1` path, `$2` offset, `$3` count)

```sh
if [ ! -e "$1" ]; then
  echo "No such file or directory (os error 2)" >&2
  exit 1
fi
if [ -d "$1" ]; then
  echo "Is a directory (os error 21)" >&2
  exit 1
fi
if [ ! -r "$1" ]; then
  echo "Permission denied (os error 13)" >&2
  exit 1
fi
exec dd if="$1" iflag=skip_bytes,count_bytes skip="$2" count="$3" status=none | base64 -w0
```

### C. WRITE_STDIN_SCRIPT (`$1` target, `$2` suffix, `$3` size)

```sh
t="$1"
case "$t" in -*) t="./$t" ;; esac
if [ -L "$t" ]; then
  r=$(readlink -f "$t" 2>/dev/null) || r=
  if [ -z "$r" ]; then
    r=$(readlink "$t") || exit 1
    case "$r" in
      /*) ;;
      *) case "$t" in */*) r="${t%/*}/$r" ;; esac ;;
    esac
  fi
  t="$r"
  case "$t" in -*) t="./$t" ;; esac
fi
tmp="$t.tmp.$2"
set -C
: > "$tmp" || exit 1
set +C
if [ -e "$t" ]; then
  chmod --reference="$t" "$tmp" 2>/dev/null || chmod "$(stat -L -c %a "$t")" "$tmp" || {
    echo "cannot give the new file the mode of $t" >&2
    rm -f "$tmp"
    exit 1
  }
fi
if cat >> "$tmp"; then
  n=$(wc -c < "$tmp")
  if [ $n -eq "$3" ]; then
    mv -f "$tmp" "$t" && exit 0
  else
    echo "short write: $n of $3 bytes arrived on stdin" >&2
  fi
fi
rm -f "$tmp"
exit 1
```

### D. WRITE_CHUNK_SCRIPT (`$1` target, `$2` suffix, `$3` chunk base64, `$4` new|append, `$5` commit or empty, `$6` size)

```sh
t="$1"
case "$t" in -*) t="./$t" ;; esac
if [ -L "$t" ]; then
  r=$(readlink -f "$t" 2>/dev/null) || r=
  if [ -z "$r" ]; then
    r=$(readlink "$t") || exit 1
    case "$r" in
      /*) ;;
      *) case "$t" in */*) r="${t%/*}/$r" ;; esac ;;
    esac
  fi
  t="$r"
  case "$t" in -*) t="./$t" ;; esac
fi
tmp="$t.tmp.$2"
if [ "$4" = new ]; then
set -C
: > "$tmp" || exit 1
set +C
if [ -e "$t" ]; then
  chmod --reference="$t" "$tmp" 2>/dev/null || chmod "$(stat -L -c %a "$t")" "$tmp" || {
    echo "cannot give the new file the mode of $t" >&2
    rm -f "$tmp"
    exit 1
  }
fi
fi
printf %s "$3" | base64 -d >> "$tmp" || { rm -f "$tmp"; exit 1; }
[ -z "$5" ] && exit 0
n=$(wc -c < "$tmp")
if [ $n -eq "$6" ]; then
  mv -f "$tmp" "$t" && exit 0
else
  echo "short write: $n of $6 bytes assembled" >&2
fi
rm -f "$tmp"
exit 1
```

### E. REMOVE_TEMP_SCRIPT (`$1` target, `$2` suffix)

```sh
t="$1"
case "$t" in -*) t="./$t" ;; esac
if [ -L "$t" ]; then
  r=$(readlink -f "$t" 2>/dev/null) || r=
  if [ -z "$r" ]; then
    r=$(readlink "$t") || exit 1
    case "$r" in
      /*) ;;
      *) case "$t" in */*) r="${t%/*}/$r" ;; esac ;;
    esac
  fi
  t="$r"
  case "$t" in -*) t="./$t" ;; esac
fi
tmp="$t.tmp.$2"
rm -f "$tmp"
```

### F. GIT_SCRIPT (`$1` root)

```sh
d="$1"
[ -d "$d" ] || d="$(dirname "$d")"
while [ -n "$d" ]; do
  [ -e "$d/.git" ] && exit 0
  n="$(dirname "$d")"
  [ "$n" = "$d" ] && break
  d="$n"
done
exit 1
```

## Known deviations from the reference implementation

The reference (`ShellFs` as of this writing) behaves as follows where a client
MUST do better. Each has an `xfail` check in the scenarios, marked for the
reference driver, which turns into a plain pass when the reference is fixed.

- **D2. Symlink size and mtime.** `stat` reports the link's own size and mtime
  while `kind` follows the link, so `write_if_unchanged` through a link
  compares the wrong file. Scenarios `files.stat.symlink-size-and-mtime-follow`
  and `files.write-if-unchanged.through-symlink`.
- **D3. Probe errors.** See section 4.2. Scenario `errors.auth.*` (the
  reference driver surfaces them itself, with a first request before probing).
- **D5. Context separator with `rg --crlf`.** The separator is `--` CR LF,
  which the reference parser does not recognise: the next hit's path becomes
  `--\r\n/real/path`, so it sorts and prints wrongly. Affects any search with
  context whose groups are not adjacent. Scenarios `files.search.context-lines`
  and `files.search.limit-counts-only-matches-and-drops-dangling-context`.
