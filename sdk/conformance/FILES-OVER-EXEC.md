# Files over exec

The normative specification of the file, search and job helpers that every
Beyond sandbox client implements on top of the exec protocol. It was derived by
reading `ShellFs` (`crates/sandbox/src/fs/shell.rs`) and `HttpExecRunner`
(`crates/sandbox/src/exec_endpoint.rs`) as they are, and is proven by the
scenarios in [`scenarios/`](scenarios) (see [README.md](README.md) and
[DRIVER.md](DRIVER.md)). A client in any language can be written from this
document and the exec protocol's own spec (the module doc of
`crates/sandbox/src/exec_endpoint.rs`) alone.

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

1. **Transport.** A raw `exec` call: one-shot, streaming and the job calls. It is a plain interface (command, args, cwd, timeout,
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
- `stdin` is bytes. It is sent by `write`, and the endpoint always delivers it.
- `stdout` and `stderr` are **bytes**: exactly what the command wrote, carried
  as standard base64 on the wire. A helper that wants text decodes the bytes in
  one place (UTF-8, invalid sequences replaced by U+FFFD); a helper that must
  move arbitrary bytes keeps them as bytes (the `read` script still encodes
  its file in base64 inside the command, so its stdout is text by design).
- `code` is the exit status. A command killed by signal `S` has
  `code` `128+S`, and a command that hit `timeout_ms` has `code` 124 with stderr
  ending `command exceeded its timeout`. A command that cannot be started (or
  whose `cwd` is missing) has `code` 127 and a reason on stderr that names what
  failed as the caller sent it: `execd: pytest -q: command not found`,
  `execd: cwd /no/dir: No such file or directory` (exec spec: module doc of
  `crates/sandbox/src/exec_endpoint.rs`). None of these is a transport error.
- Timeouts: helpers use 120 s per command (`timeout_ms` 120000) and 10 s for
  the environment probe.

On the wire (the HTTP transport) a call is one `POST` of

```json
{
  "command": "sh",
  "args": ["-c", "...", "sh", "/a"],
  "timeout_ms": 120000,
  "stdin": "..."
}
```

(`cwd` and `stdin` only when present; `stdin` is standard base64) with every
configured header, among them `Authorization: Bearer <token>`, and the answer
`{"exit_code": N, "stdout": "<base64>", "stderr": "<base64>"}` (standard base64
with padding, the empty string for no output). The client reads the body
incrementally and stops at a cap (default 16 MiB of decoded `stdout` plus
`stderr`): over the cap the call fails with `too_large` and says the command ran.
A `stdout` or `stderr` that is missing or not standard base64 is a `protocol`
error, never empty output. The request deadline is
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
| `invalid_request`   | the caller's arguments cannot form a request; nothing was sent              |

### 3.2 From the transport

| what happened                                                       | kind        |
| ------------------------------------------------------------------- | ----------- |
| connection refused, reset, DNS failure                              | `transport` |
| the request deadline passed                                         | `timeout`   |
| HTTP 401 or 403                                                     | `auth`      |
| HTTP 422 whose body says the output exceeded the cap, or client cap | `too_large` |
| any other non-2xx                                                   | `endpoint`  |
| 200 whose body is not the expected JSON                             | `protocol`  |
| a stream request answered 200 without `application/x-ndjson`       | `protocol`  |
| a stream ends without its final frame, or a `{"error"}` frame       | `transport` |

A stream request is answered by a stream or not at all. A 200 to it whose `Content-Type` is not
`application/x-ndjson` is `protocol`, whatever its body: a client never reads that body as a
one-shot answer (an object that looks like a result would be a clean empty exit 0, silent data
loss).

A transport-level failure is never a capability: see
[probe errors](#42-probe-errors).

### 3.3 From a command that ran and failed

A helper that gets a non-zero exit takes the **first line of trimmed stderr**
(lines end at LF and at no other character: CR, VT, FF and U+2028 do not end
one; or `command failed` if there is none) as `detail`, prefixed as each operation
below says, and classifies it case-insensitively by the first rule that
matches. A path may contain a newline, which splits one message over two lines
and leaves the OS error on a later one: when the first line contains none of the
patterns below, `detail` runs through the first line that does (lines joined
with LF), and if no line does it is the first line.

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

## 4. Environment probe

A helper client looks at the sandbox's tools once, in **one** round trip, the
first time an operation needs to know (a write needs nothing and never probes).
The result is cached per sandbox. The probe is a single `run` with a 10 s
timeout of script F of the appendix, `sh -c PROBE_SCRIPT sh`, whose stdout has
one line per capability the sandbox has:

| line          | capability                                         | present when the script finds                              |
| ------------- | -------------------------------------------------- | ---------------------------------------------------------- |
| `rg`          | ripgrep                                            | `rg --version` exits 0                                     |
| `grep_null`   | `grep -Z`                                          | `grep -rnZ -e __cap_probe__ -- /dev/null` exits 0 or 1     |
| `find_printf` | `find -printf`                                     | `find /dev/null -maxdepth 0 -printf %y` prints exactly `c` |
| `read_window` | `dd iflag=skip_bytes,count_bytes` and `base64 -w0` | `printf abc \| dd ... \| base64 -w0` prints `Yg==`         |

A capability whose line is absent is absent. Derived: the **search engine** is
`rg` if `rg`; otherwise `grep` (GNU grep: `grep_null`); otherwise
`busybox-grep`.

### 4.1 What a client reports

`capabilities` (DRIVER.md) reports `search_engine`, `find_printf` and
`read_window`.

### 4.2 Probe errors

If the probe request fails at the transport level (unreachable, `auth`,
`endpoint`, `timeout`), the client MUST report that error from the operation
that needed the probe, and MUST NOT cache a capability set learned from a failed
request. Only a probe that ran and left a line out means absent.

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
  exist"); `MTIME` is `SECS.FRAC`: epoch seconds, a dot, and the fractional
  digits `stat` reports (nine digits of nanoseconds on GNU coreutils and on
  busybox; `0` if it reports none). A client keeps **all of it**, as a number
  (a double is enough: it resolves 0.24 µs at present epochs, finer than any
  filesystem records).

`kind` follows symlinks (`[ -d ]`/`[ -f ]`), `writable` is a real access check
for the calling user (`[ -w ]`), not mode bits. The script takes the whole
seconds from `stat -c %Y` and the fraction from the part after the dot of
`stat -c %y` (`YYYY-MM-DD HH:MM:SS.NNNNNNNNN ZONE` on both GNU and busybox;
`%.9Y` is GNU-only). If `stat -c` itself fails the script prints `0<TAB>0`.

A symlink's `size` and `mtime` are the target's, like `kind`: the script runs
`stat -L -c` (GNU and busybox both accept it). A dangling link fails the
leading `[ -e ]` and is reported as not found.

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

The write goes to a temp file **in the directory of the file the path
resolves to** (so the `mv` is a rename within one filesystem), named
`.tmp.SUFFIX` whatever the target is called, and is committed with `mv -f`.
`SUFFIX` is chosen by the client per write: **exactly 16 lowercase hex digits**
(64 random bits, unpredictable), so a preplanted name cannot redirect the
write. The name is therefore always 21 bytes, and a target whose own name is
anything up to `NAME_MAX` (255 bytes) can be written; a name derived from the
target's (`NAME.tmp.SUFFIX`) overflowed `NAME_MAX` for names over about 230
bytes. The script (C below) does the
resolution, in order:

1. a path starting with `-` is prefixed with `./`;
2. while it is a symlink, follow it one `readlink` hop at a time (live or
   dangling alike), each relative target joined to the directory of the link it
   came from, for at most 40 hops (Linux's MAXSYMLINKS). A client MUST NOT use
   `readlink -f`: it is unbounded in time and memory on an adversarial chain,
   and busybox resolves a dangling link's relative target against the working
   directory instead of the link's. The prefix rule of 1 is applied after every
   hop. If the path is still a link after 40 hops it is a loop: the script
   prints `Too many levels of symbolic links (os error 40)` and exits 1, before
   any temp exists, so nothing changes (kind `io`: section 3.3 has no better
   one). A chain that ends in a missing directory fails at the temp as any
   write into a missing directory does (`not_found`);
3. a target that is a directory is refused (`Is a directory (os error 21)`, exit
   1), before the temp exists, because `mv -f` would move the temp _into_ it
   and report success;
4. a target whose parent directory does not exist is refused before the temp
   exists, in words that name the directory and not a temp file or a script line:
   `parent directory DIR does not exist: No such file or directory (os error 2)`
   (`DIR` is the parent as the path spells it, `/` for a file in the root, `.`
   for a bare name; kind `not_found`). A parent that exists but is not a
   directory fails at the temp as before (`not_a_directory`), and so does a parent
   that cannot be resolved (`Too many levels of symbolic links`): only a parent
   `ls` reports as missing is named here;
5. the temp is created empty with `set -C` (no-clobber: a preplanted temp is
   refused, not followed) under `umask 077` (mode 0600), or
   with the default mode when the target does not exist, so the size check in
   step 6 can read it back whatever the file's mode is: **a write needs no
   read permission on the file** (a mode 0200 file is replaced). Creating the
   temp comes first, so a read-only filesystem reports its own
   `Read-only file system` and a full one its own error. Then, if the target
   exists and `[ -w TARGET ]` is false, the temp is removed and the write is
   refused (`Permission denied (os error 13)`, exit 1);
6. the content is appended; the temp's size is checked against the expected
   byte count; the directory check runs again (a directory created meanwhile
   removes the temp and fails); if the target exists the temp gets its mode
   **now, just before the `mv`**: `chmod --reference`, falling back to
   `chmod "$(stat -L -c %a TARGET)"` (busybox); if neither works the write fails
   (`cannot give the new file the mode of ...`) and the temp is removed; only
   then `mv -f TEMP TARGET`.

A write is therefore atomic (a reader sees the old file or the new one), keeps
the file's mode (a 0600 file stays 0600, a 0755 file stays executable), writes
through symlinks (the link stays a link; a dangling link's target is created),
creates a new file with the default mode, and never leaves a temp behind on a
failure the script sees.

One command, `sh -c WRITE_STDIN_SCRIPT sh TARGET SUFFIX SIZE` with the whole
content as stdin (script C). The size check is what turns a lost stdin (a
transport that closes the command's stdin, as `docker exec` does without `-i`)
into a failed write instead of an empty file.

**On failure** (a non-zero exit, or a transport error):
the client sends one best-effort cleanup command,
`sh -c REMOVE_TEMP_SCRIPT sh TARGET SUFFIX` (script D; its result is ignored),
and then reports the failure: a transport error as it is, otherwise
`write PATH: LINE` classified by section 3.3. The script removes the temp itself on failures it sees; the cleanup covers the
ones it cannot (a timeout).

Further rules:

- The parent directory must exist (`not_found`, detail `write PATH: parent
  directory DIR does not exist: ...`, never a temp's name); `write` does not
  create directories. A surface built for agents may create the missing parents
  itself, with `mkdir -p` (the `create_dir_all` helper), before it writes: the
  MCP server's `write_file` does.
- A file the caller may not write is refused with `permission_denied` and
  unchanged: the script tests `[ -w TARGET ]` once the temp exists, and only then.
- A file the caller may write but not read (mode 0200, 0333) is replaced: the
  temp is 0600 until the `mv`, so the size check can read it.
- A write onto a **directory** MUST fail and change nothing.
- Two writers to one path leave one complete content, never a mixture.

### 5.4 write_if_unchanged

`write_if_unchanged(path, bytes, expected_mtime: int | null)`

1. `m = stat(path)` and take its `mtime` (absent if the path does not exist).
2. Compare the mtimes **at full precision, as numbers**: they match when `m`
   and `expected_mtime` are both absent, or both present and equal (the same
   `SECS.FRAC` parsed to a double gives the same double, so the comparison a
   client makes between two of its own `stat` results is exact). Otherwise fail
   with `conflict` and write nothing.
3. `write(path, bytes)`.

So `expected_mtime` null means "the file must not exist". The key is required,
and a number is the `mtime` of an earlier `stat`, fraction included: a value
truncated to whole seconds does not match a file whose mtime has a fraction.
The two steps are two round trips and the guard is not a lock: it catches a
person or agent editing between a read and a write, including one in the same
second, and misses an edit that restores the exact timestamp (a limit, with a
scenario). Through a symlink the comparison MUST use the target's mtime.

### 5.5 mkdir

`mkdir(path)`: `mkdir -p -- PATH` as raw argv (`command` `mkdir`, `args`
`["-p", "--", PATH]`). A non-zero exit is an error `create PATH: LINE`
classified by section 3.3 (making a directory below a file is
`not_a_directory`). Idempotent.

### 5.6 list

`list(path, include_hidden = false, limit = 10000) -> entries[{ name, kind, size }]`

**With `find_printf`**: one command, `find -H PATH -mindepth 1 -maxdepth 1 -printf
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

**Without `find_printf`** (busybox): two commands, `find -H PATH -mindepth 1
-maxdepth 1 -print0` (same failure rule) and the same with `-type d` added.
Entries are the first listing; `name` is the last path component; an entry is
`dir` if its full path is in the second listing, otherwise `file`; `size` is 0.
(So FIFOs and links to directories are `file` there.)

`-H` makes `find` follow `PATH` itself when it is a symlink (it does not by
default): a link to a directory lists the directory it points to, and each
entry is reported by its name under the link. A link to a file, or a broken
one, lists nothing.

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

#### What the glob decides

The glob restricts or excludes **files**; it never decides whether a directory is
descended. `glob = "*.md"` keeps the files whose path matches and still searches
every directory, matching or not; `glob = "*.txt"` with `glob_exclude` drops
the files that match and still searches a directory named `d.txt`. `rg --glob
'!G'` cannot say this: it also matches directories, so `!*.txt` would skip
everything under `d.txt/`, and no override form re-includes directories
without whitelisting every file and directory past `.gitignore`. So an engine
is asked to exclude only through a flag that is files-only (GNU `--exclude`), and
`rg` is given no excluding glob at all: the client drops the hits whose path
matches the glob (same `globset` matcher as busybox, below, but case-sensitive,
like `rg --glob`), context hits included. Excluded files are still read by `rg` and by busybox `grep` (which filters
client-side too), so their per-path errors about an excluded file are dropped
from stderr before it is read: a record whose path, between an optional `rg:`
or `grep:` and the OS message, matches the glob (the same matcher as the hits).
Never the root's own error (`ROOT:` or `ROOT: IO error for operation on ROOT:`), which
decides `not_found` or `not_a_directory` for the whole search, whatever the root is
called. Only files: the glob never excludes a directory, and an unreadable directory
named like the glob still hides files the caller wanted. So when at least one
record matches the glob, the client runs `sh -c DIRS_SCRIPT sh PATH...` (script
J) once with the paths of those records, and keeps the record of each path it
prints. If that command fails, every record is kept. If no record is left, a run
that exited 2 is a success with no note. A glob that does not parse is
`invalid_query` (`bad glob: ...`) whichever engine runs. A root that is itself a
file is searched whatever the glob says on `rg`; GNU `grep` applies it to a file
named on the command line.

#### Engine arguments

The command is `rg` or `grep`, with **no shell**, each item one argument.

`rg` engine (in this order):

```text
--null --with-filename --line-number --column --no-heading --color=never
--hidden --crlf
[--no-require-git]            unless the root is inside a git repository (below)
[--ignore-case] [--fixed-strings]
[--glob G]                    the caller's glob, only when it restricts (never `!G`, see below)
--glob !.git
[-B before] [-A after]        each only when greater than 0
--no-context-separator        always
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
--no-group-separator          always
-e PATTERN -- ROOT
```

where for a non-literal pattern `PATTERN` is rewritten for CRLF files (below).

`busybox-grep` engine: `-r -n -H`, then `-F` or `-E`, `-i`, context flags (no
`--no-group-separator`: busybox rejects it),
`-e PATTERN -- ROOT`. It has no `-Z`, `-I`, `--color`, `--include`,
`--exclude-dir`; the glob is applied by the client.

**Git root probe** (`rg` only): `--no-require-git` is added unless the root or
an ancestor holds a `.git` entry, decided **in the sandbox** (never from the
client's filesystem): `sh -c GIT_SCRIPT sh ROOT` (script E), exit 0 meaning
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
share exit 2.

Read stderr as **records**, not lines, because a path may contain a newline: a
record runs from the end of the previous one through the first line that ends
in an OS error (`No such file or directory`, `Permission denied`, `Is a
directory`, `Not a directory`, `Too many levels of symbolic links`, `Input/output error`, with a
trailing `(os error N)` that `rg` adds ignored) or in
`stream did not contain valid UTF-8` (`rg` reading a `.git` file, in or above
the root, that is not text: the search itself ran, and no flag turns that read
off without also turning off the `.gitignore` handling the helper promises).

- stderr is non-empty, nothing is left after its last record (every record is a
  per-path OS error), and one record is the search root's own error: it
  contains `: ROOT:` and a space, immediately followed by `No such file or
  directory` or `Not a directory`, optionally with `IO error for operation on
  ROOT:` and a space before it, where `ROOT` is the exact root string given to
  the engine, newlines included (a path under the root continues after `ROOT`,
  so it does not match): `not_found` for `No such file or directory`,
  `not_a_directory` for `Not a directory` (a root below a regular file);
- every record is a per-path OS error otherwise: success. The hits on stdout
  are returned and `note` says how many records were skipped and names the
  first;
- anything else (or empty stderr) is the engine rejecting the pattern, whatever
  stdout holds (`rg: regex parse error:`, `rg: error parsing glob ...`,
  `rg: compiled regex exceeds size limit ...` are such lines):
  `invalid_query` with detail `bad regex: STDERR` (trimmed; `bad regex: PROGRAM
  rejected the pattern` if stderr is empty).

**Unsearchable directories under GNU `grep` with a glob.** A directory that can
be listed but not entered (mode `r--` for the sandbox user) hides what is in it:
`grep` cannot stat the entries and judges one by its name alone, as a file, so
with `--include=G` it skips every entry whose name does not satisfy `G` (and
with `--exclude=G` every one that does) and prints nothing. The run exits 0 or 1
with an empty stderr, and the matches below that directory are missing with
`note` null. So when the engine is GNU `grep`, the search has a glob, the run
succeeded and stderr was empty, the client runs one more command:

```text
find -H ROOT -type d ! -executable -print0
```

(`-executable` is `access(X_OK)` for the sandbox user, so nothing is listed
when that user is root; a `find` without `-executable`, or a transport error,
lists nothing and adds no note; the exit code is not read.) Drop each path
that has a `.git` component below the root. When any is left, `note` is
`N path(s) could not be searched; first: grep: PATH: Permission denied` with `N`
their number and `PATH` the first in `find`'s order. A directory that cannot be
read at all is already reported by `grep` itself, which is why this is asked
only when stderr was empty. The check lists a directory whose entries the glob
would have skipped anyway too, so the note means "may be incomplete". `rg` and
busybox `grep` need none (the first says so itself, the second has no glob).
The check walks the tree once, about 6% of the time the same search takes
(measured on a 105 000 file tree, 0.05 s against 0.8 s).

#### Parsing

`rg` and `grep -Z` emit records `PATH<NUL>LINE<SEP>TEXT<LF>`, where `SEP` is `:`
for a match and `-` for a context line. `rg --column` puts the byte column of the
line's first match between the separator and the text of a **match** record:
`PATH<NUL>LINE:COL:TEXT<LF>` (`COL` is 1-based, in bytes of the raw line, CRs
included; a context record has none). GNU `grep -b` puts the byte offset of the
line start there instead, in match **and** context records
(`PATH<NUL>LINE:OFF:TEXT`, `PATH<NUL>LINE-OFF-TEXT`). The command MUST turn off the group
separator: `rg --no-context-separator` (ripgrep 0.10 and later), GNU
`grep --no-group-separator` (2.5.4 and later). The separator is the only record
without a NUL, and a path may begin with `--` + LF (or `--` + CR LF under
`rg --crlf`), so it cannot be told from the start of such a path; with it off,
the output is path records only. A client MUST NOT skip any `--` bytes. Parse by
scanning for the **NUL**, not for newlines: a path may contain a newline. The
path runs from the start of the record to the next NUL, the rest of the record
to the next LF; `LINE` is one or more digits followed by `:` or `-`, and under `rg` a match
record then has `COL`, one or more digits followed by `:`; a record that does
not fit is dropped. A file named `--` or `--` LF `b` is an ordinary
path.

`rg` also prints, after the records of a file it stopped searching at a NUL
byte (a NUL after a match), one line `PATH: WARNING: stopped searching binary
file after match (found "\0" byte around offset N)` on **stdout**, where `\0` is
a backslash and a zero and `N` digits. It has no NUL, so it would read as the
start of the next record's path. `rg --no-messages` does not suppress it,
`--binary` only rewords it (still on stdout), and `--text` changes the results
(binary files would be searched whole), so the client drops it: a line is that
warning when it is exactly the **previous record's path** followed by the text
above, and a record is parsed from the line after it. Requiring the previous
record's path means a file merely named like the warning is still a path (the
one ambiguity left is a file whose name is the previous record's path plus
that text and a newline, listed right after it). GNU `grep -I` prints nothing
for such a file, and busybox `grep` prints `Binary file PATH matches`, which has
no `LINE:` field and is dropped by its parser.

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

Context groups that overlap are merged by the engine.

#### Clipping

`MAX_LINE` is 500 **characters** (Unicode scalar values, not bytes or UTF-16
units). A line of at most 500 characters is unchanged. A line longer than that
is clipped at its **head** (the first 500 characters followed by `… [truncated]`,
U+2026, space, `[truncated]`) if it is a context line, if the engine reported no
column for it, or if its match starts early (below). A client never searches the
line for the match itself: the position is what the engine reported, so clients
cannot differ on pattern syntax.

For a **match** line with a byte column `COL` (from `rg --column`; for GNU
`grep`, below): `cs` is the number of characters
of the line's UTF-8 encoding before byte `COL - 1`, not counting CRs (the text
has none, see Parsing); a `COL` that is not a character boundary of the line
(possible only when the line had bytes that are not UTF-8) counts as no column.
`start = max(0, cs - 250)`. If `start` is 0, clip at the head. Otherwise
`end = start + 500`; if `end > total` (the line's length in characters) then
`end = total` and `start = total - 500`. The text is `…` + characters
`[start, end)` + (`…` if `end < total`) + `[truncated; columns
{start+1}-{end} of {total}]`.

GNU `grep` has no column field. When the run returned a match line of more than
500 characters, the client runs the same command once more with `-o` added and
no context flags (`-o -b` prints `PATH<NUL>LINE:OFF:MATCH` per match, `OFF` the
byte offset of the match in its file). For each such line the first record with
its path and line number gives `COL = OFF - LINEOFF + 1`, where `LINEOFF` is the
offset the first run printed for the line. A second run that fails, or has no
record for the line, leaves the line clipped at its head.

#### Engines differ; say which you used

`rg` honors `.gitignore`/`.ignore`, skips binary files (any NUL), uses Rust
regex syntax and searches hidden files. `grep` does none of the ignore
handling, skips binary files with `-I` (in a UTF-8 locale also files with
invalid UTF-8), and uses POSIX ERE; `\d` and non-greedy quantifiers are not
available. `busybox-grep` reports no match position and clips every long line
at its head.
`busybox-grep` additionally cannot represent some path names (above)
and may report binary files differently. A client MUST report the engine in use
(`capabilities.search_engine`); scenarios that depend on one declare
`requires.search_engine`.

### 5.8 Bulk copy: upload and download

Two helpers move whole trees between the **client's machine** (the _local_ side)
and the sandbox as a `tar` stream over layer 1. Nothing in the sandbox is
installed: it needs `sh` and a `tar` (GNU or busybox).

```text
upload(sources, remote_dir)        -> { files, dirs, links, bytes }
download(remote_path, local_dir)   -> { files, dirs, links, bytes }
```

`sources` is a local directory (its contents land directly in `remote_dir`) or
a list of local paths (each lands in `remote_dir` under its base name, a
directory with its whole tree). `remote_dir` and `remote_path` are absolute
(`invalid_request` otherwise; so is `/` as `remote_dir`); `local_dir` is where
the client's own process runs, and is created (with its parents) when missing.
The result counts what was written: members that are regular files, directories
(not the root), symbolic or hard links, and the bytes of file content.

A tree is copied as it is: regular files, directories and symbolic links, with
the **permission bits** (`0o777` mask: set-id and sticky bits are not carried)
and the **mtime** of files (directories too on download; on upload the
sandbox's `tar` decides, and busybox does not restore a directory's). A symbolic link is a link, stored
by its target text and never followed on either side. Ownership is not
carried: files belong to whoever extracts them. A hard link is two files on
upload and is recreated as a hard link on download. A FIFO, socket or device is
`invalid_request` on upload (naming the path) and `io` on download
(`unsupported member type`).

#### The archive

Both directions use the same subset of `tar`, written and read by the client
itself (no `tar` on the client's machine is run):

- **Written** (upload): ustar-compatible headers (POSIX `ustar\0` magic, or GNU's `ustar  \0` as the `tar` crate writes it),
  types `0` file, `5` directory (name with a trailing `/`), `2` symlink. A name
  or link target over 100 bytes uses a GNU `L` / `K` extension member (busybox
  and GNU tar both read these; the ustar `prefix` field is never written). `uid`
  and `gid` are 0, `uname` and `gname` empty, `mode` the permission bits,
  `mtime` the local mtime in whole seconds (never negative); a `size` of 8 GiB
  or more is base-256. Names are relative, have no `./`, no `..` and no empty
  component, and are valid UTF-8 (a name that is not is `invalid_request`).
  Members come in pre-order, siblings sorted by name bytewise, so a directory
  precedes its content. There is no member for the root. The archive ends with
  two 512-byte zero blocks and no padding to a record. A file that changes size
  while it is read fails the upload (`io`, `PATH: changed while being
  archived`).
- **Read** (download): ustar, v7, GNU (`L`, `K`, base-256 numbers) and pax
  (`x`, `g`: the `path`, `linkpath` and `size` records), whatever `tar -c` of
  the sandbox wrote. A header whose checksum is wrong, or an archive that ends
  without its end-of-archive block, is `protocol`.

#### Extraction on the local side

The sandbox is not trusted: a tar from it may hold anything. The client
extracts under `local_dir` by these rules and refuses, with `protocol` and a
`detail` `unsafe archive member NAME: WHY`, at the first member that breaks one,
leaving what was already written. `WHY` is one of `it is absolute`,
`it has a ".." component`, `it has a NUL`, `it is not valid UTF-8`,
`a parent is not a directory (a link or a file)`,
`its link target is unsafe:` and one of the first four,
`a hard link must name a file this archive already wrote`,
`its link target is not a file` (the scenarios match these words):

1. A member name is split on `/`; empty and `.` components are dropped; a name
   that is absolute, has a `..` component or a NUL, or is not valid UTF-8, is
   refused. A name that is empty after that (`./`, the root) is skipped: it
   sets no mode or mtime on `local_dir`.
2. Every component above a member's last must be a real directory at that
   moment (an `lstat`, not a `stat`); a missing one is created (mode `0o755`);
   a **symbolic link or file there is refused**: nothing is ever written
   through a link, whether it was in the archive or was already in `local_dir`.
3. A file is created with `O_CREAT|O_EXCL|O_NOFOLLOW` (mode `0o600` until it is
   complete) after an existing non-directory of that name is unlinked; an
   existing directory of that name is an error `PATH: Is a directory (os error
   21)`, kind `is_directory` (section 3.3). A directory member over an existing non-directory unlinks it first. A
   directory is created `0o700`; the **mode and mtime** of every directory
   are applied at the end, deepest first, so a `0o555` directory can still be
   filled. A file gets its mode (set-id and sticky bits masked off) and mtime
   when it is complete.
4. A symlink member is created with the target text unchanged; the target is
   never resolved. A hard link member is created only for a target that, read
   the way rule 1 reads a name, names a regular file this archive already
   wrote; anything else is refused.
5. Any member type other than file (`0`, `\0`, `7`), directory, symlink,
   hard link and the meta types is `io` (`unsupported member type`).

The rules are checks of paths at the time of the call: they do not defend
against another local process changing `local_dir` while a download runs.

Upload needs no such rules for the sandbox: every member name was produced by
the client under rule-1 shape, and the archive never has a member below a
symlink of its own. A symbolic link that **already exists** inside `remote_dir`
is followed by the sandbox's own `tar -x`, as it would be for anyone running
`tar -x` there; the helper does not defend against it. The effect is contained
in the sandbox, which is the isolation boundary: it can write where the
sandbox user could write anyway, and nothing on the client's machine is
reached. (Links in the _local_ source are archived as links and never
followed.)

#### Upload

The archive is produced incrementally and sent in **chunks** of at most
`BULK_CHUNK_BYTES` (16 MiB of raw tar, 21.3 MB of base64: under the 64 MiB an
`execd` accepts and the 16 MiB a one-shot answer may carry). A client holds at
most about two chunks.

- **Fits in one chunk** (the common case): one command,
  `sh -c UPLOAD_SCRIPT sh DEST ""` (script G) with the whole archive as stdin and
  `timeout_ms` `BULK_TIMEOUT_MS`.
- **Larger**: a _stage_ file beside the destination,
  `STAGE = DEST (trailing slashes removed) + ".upload." + SUFFIX`, `SUFFIX` 16
  lowercase hex digits chosen by the client (as in 5.3). Each chunk is one
  command `sh -c STAGE_SCRIPT sh STAGE OFFSET LENGTH` (script I; stdin the chunk;
  120 s) with `OFFSET` the bytes already staged: the first (`OFFSET` 0) creates
  the file (refusing one that exists, creating missing parents), each later one
  first checks that the stage is exactly `OFFSET` bytes, so a repeated or lost
  chunk fails instead of corrupting the archive; each checks the size after its
  append. Then `sh -c UPLOAD_SCRIPT sh DEST STAGE` with no stdin extracts the
  stage and removes it. The number of commands is `ceil(size / BULK_CHUNK_BYTES) + 1`.
  A chunk is never empty except when the archive is an exact multiple of the
  chunk size, in which case the empty tail is not sent.

The script, in this order: refuses a `DEST` that exists and is not a directory
(`Not a directory (os error 20)`, exit 1); creates `DEST` with `mkdir -p` when it
is missing; runs `umask 0` so that `tar` applies the archive's modes exactly
(busybox applies the umask even to preserved modes); `cd`s into `DEST` and runs
`tar -xf -` (or `tar -xf STAGE`). On any failure it removes the stage and,
**when this call created `DEST`, removes `DEST`**; a `DEST` that existed keeps
whatever `tar` had written when it failed (extraction is not atomic; the error
is the report). Success exits 0.

On a failure (a non-zero exit, or a transport error) of any command the client
sends one best-effort `rm -f -- STAGE` (its result is ignored) when it staged,
and reports the failure: a transport error as it is, otherwise
`upload DEST: LINE` classified by section 3.3. A `tar` that is out of space or
on a read-only filesystem says so on stderr and exits non-zero, which is `io`
(`No space left on device`, `Read-only file system`), and an unwritable
destination is `permission_denied`. If the local read fails before the archive
is complete nothing is extracted. A call cancelled in the middle of a staged
upload can leave its stage behind.

#### Download

`sh -c DOWNLOAD_SCRIPT sh PATH` (script H) writes the archive to stdout:
`tar -cf - -C PATH .` for a directory, and for anything else a one-member
archive of `./NAME` (so `download("/w/a.txt", d)` writes `d/a.txt`). A missing
path is `No such file or directory (os error 2)` on stderr and exit 1.

The archive's size is not known in advance, so the client **streams** it: the
stream form of the exec call, stdout frames fed to the extractor as they arrive
(never held whole), stderr kept up to `BULK_STDERR_BYTES`, `timeout_ms`
`BULK_TIMEOUT_MS`.
A client reports a failed command (a non-zero final status, a signal, a
timeout) as `download PATH: LINE` classified by section 3.3 **before** it
reports anything about the archive's content, but an unsafe member is reported
the moment it is seen and the stream is closed. Whatever was extracted before
a failure stays in `local_dir`: a partial download is an error, never a result.
`tar` exits non-zero for a file that vanished or changed while it was read; the
client does not interpret that.

#### Limits

| name                | value  | what it bounds                                                     |
| ------------------- | ------ | ------------------------------------------------------------------ |
| `BULK_CHUNK_BYTES`  | 16 MiB | raw tar bytes per upload request; memory held is about two chunks  |
| `BULK_TIMEOUT_MS`   | 600000 | the extraction command and the download stream (a hung `tar` ends) |
| `BULK_STDERR_BYTES` | 64 KiB | the stderr a download keeps for its error `detail`                 |

A stage append and a cleanup use the helpers' 120 s. There is no limit on the
size of a tree: the chunk count follows from it. A client SHOULD let a caller
lower the chunk size (the matrix does, to walk the staged path with small
trees), and a lower value is never an error.

#### Errors

| situation                                                      | kind                                                                                                                 |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `remote_dir` / `remote_path` not absolute, `/` as `remote_dir` | `invalid_request` (nothing sent)                                                                                     |
| a local source that is a FIFO, socket or device; a bad name    | `invalid_request` (nothing sent)                                                                                     |
| a local source or `local_dir` missing, unreadable, not a dir   | `not_found`, `permission_denied`, `not_a_directory`, ... classified from the OS error, as section 3.3 does from text |
| `DEST` exists and is a file                                    | `not_a_directory`                                                                                                    |
| read-only filesystem, no space, other `tar` failure            | `io` (or `permission_denied` / `not_found` from the text)                                                            |
| an archive member that breaks the extraction rules             | `protocol`, `unsafe archive member NAME: WHY`                                                                        |
| a malformed or truncated archive                               | `protocol`                                                                                                           |
| an unsupported member type                                     | `io`                                                                                                                 |
| transport failures                                             | section 3.2                                                                                                          |

### 5.9 Not specified here

- `glob` (find paths by pattern): the reference computes it host-side with
  `globset`; surfaces build it on `find`/`rg --files` over raw exec.
- `remove`: no counterpart in the reference. A surface that offers it runs
  `rm -rf -- PATH`-style commands through raw exec.

## 6. Jobs

Jobs are layer 1, specified by the exec protocol ("Sandbox jobs" in the module
doc of `crates/sandbox/src/exec_endpoint.rs`): the endpoint runs them, and a
client calls the job operations. (A transport that is not an exec endpoint, a
template such as `ssh` or `docker exec`, has no jobs of its own; the Rust crate
builds them there from plain commands, in
`crates/sandbox/src/exec_endpoint/jobs.rs`. That is not part of this contract.)
The driver ops are in DRIVER.md. Behaviours the scenarios pin:

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
if s=$(stat -L -c '%s %Y %y' -- "$1" 2>/dev/null); then
  set -- $s
  f=${4#*.}
  [ "$f" = "$4" ] && f=0
  printf '%s	%s.%s' "$1" "$2" "$f"
else
  printf '0	0'
fi
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
n=0
while [ -L "$t" ] && [ $n -lt 40 ]; do
  l=$(readlink "$t") || exit 1
  case "$l" in
    /*) t="$l" ;;
    *) case "$t" in */*) t="${t%/*}/$l" ;; *) t="$l" ;; esac ;;
  esac
  case "$t" in -*) t="./$t" ;; esac
  n=$((n + 1))
done
if [ -L "$t" ]; then
  echo "Too many levels of symbolic links (os error 40)" >&2
  exit 1
fi
case "$t" in */*) tmp="${t%/*}/.tmp.$2" ;; *) tmp=".tmp.$2" ;; esac
if [ -d "$t" ]; then
  echo "Is a directory (os error 21)" >&2
  exit 1
fi
case "$t" in */*) d="${t%/*}"; d="${d:-/}" ;; *) d=. ;; esac
if [ ! -e "$d" ]; then
  case $(LC_ALL=C ls -d "$d" 2>&1) in
    *"No such file or directory"*)
      echo "parent directory $d does not exist: No such file or directory (os error 2)" >&2
      exit 1 ;;
  esac
fi
set -C
if [ -e "$t" ]; then (umask 077; : > "$tmp") || exit 1; else : > "$tmp" || exit 1; fi
set +C
if [ -e "$t" ] && [ ! -w "$t" ]; then
  echo "Permission denied (os error 13)" >&2
  rm -f "$tmp"
  exit 1
fi
if cat >> "$tmp"; then
  n=$(wc -c < "$tmp")
  if [ $n -eq "$3" ]; then
    if [ -d "$t" ]; then echo "Is a directory (os error 21)" >&2; rm -f "$tmp"; exit 1; fi
if [ -e "$t" ]; then
  chmod --reference="$t" "$tmp" 2>/dev/null || chmod "$(stat -L -c %a "$t")" "$tmp" || {
    echo "cannot give the new file the mode of $t" >&2
    rm -f "$tmp"
    exit 1
  }
fi
    mv -f "$tmp" "$t" && exit 0
  else
    echo "short write: $n of $3 bytes arrived on stdin" >&2
  fi
fi
rm -f "$tmp"
exit 1
```

### D. REMOVE_TEMP_SCRIPT (`$1` target, `$2` suffix)

```sh
t="$1"
case "$t" in -*) t="./$t" ;; esac
n=0
while [ -L "$t" ] && [ $n -lt 40 ]; do
  l=$(readlink "$t") || exit 1
  case "$l" in
    /*) t="$l" ;;
    *) case "$t" in */*) t="${t%/*}/$l" ;; *) t="$l" ;; esac ;;
  esac
  case "$t" in -*) t="./$t" ;; esac
  n=$((n + 1))
done
if [ -L "$t" ]; then
  echo "Too many levels of symbolic links (os error 40)" >&2
  exit 1
fi
case "$t" in */*) tmp="${t%/*}/.tmp.$2" ;; *) tmp=".tmp.$2" ;; esac
rm -f "$tmp"
```

### E. GIT_SCRIPT (`$1` root)

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

### F. PROBE_SCRIPT (no arguments)

Prints one line per capability the sandbox has (section 4).

```sh
rg --version >/dev/null 2>&1 && echo rg
grep -rnZ -e __cap_probe__ -- /dev/null >/dev/null 2>&1; [ $? -le 1 ] && echo grep_null
[ "$(find /dev/null -maxdepth 0 -printf %y 2>/dev/null)" = c ] && echo find_printf
[ "$(printf abc | dd iflag=skip_bytes,count_bytes skip=1 count=1 status=none 2>/dev/null | base64 -w0 2>/dev/null)" = Yg== ] && echo read_window
true
```

### G. UPLOAD_SCRIPT (`$1` destination directory, `$2` stage file or the empty string)

```sh
d="$1"
s="$2"
if [ -e "$d" ] && [ ! -d "$d" ]; then
  echo "Not a directory (os error 20)" >&2
  [ -z "$s" ] || rm -f -- "$s"
  exit 1
fi
new=
if [ ! -d "$d" ]; then
  mkdir -p -- "$d" || { [ -z "$s" ] || rm -f -- "$s"; exit 1; }
  new=1
fi
umask 0
if cd -- "$d"; then
  if [ -n "$s" ]; then tar -xf "$s"; else tar -xf -; fi && { [ -z "$s" ] || rm -f -- "$s"; exit 0; }
fi
[ -z "$s" ] || rm -f -- "$s"
cd /
[ -z "$new" ] || rm -rf -- "$d"
exit 1
```

Section 5.8. With an empty `$2` the archive is on stdin.

### H. DOWNLOAD_SCRIPT (`$1` path)

```sh
if [ ! -e "$1" ]; then
  echo "No such file or directory (os error 2)" >&2
  exit 1
fi
if [ -d "$1" ]; then
  exec tar -cf - -C "$1" .
fi
exec tar -cf - -C "$(dirname -- "$1")" "./$(basename -- "$1")"
```

### I. STAGE_SCRIPT (`$1` stage file, `$2` bytes already staged, `$3` bytes in this chunk)

```sh
f="$1"
if [ "$2" -eq 0 ]; then
  mkdir -p -- "$(dirname -- "$f")" || exit 1
  set -C
  : > "$f" || exit 1
  set +C
fi
n=$(wc -c < "$f") || exit 1
if [ "$n" -ne "$2" ]; then
  echo "stage is $n bytes, expected $2" >&2
  exit 1
fi
cat >> "$f" || exit 1
n=$(wc -c < "$f")
if [ "$n" -ne $(($2 + $3)) ]; then
  echo "short write: $n of $(($2 + $3)) bytes in the stage" >&2
  exit 1
fi
```

The chunk is on stdin.

### J. DIRS_SCRIPT (`$@` paths)

```sh
for p; do [ -d "$p" ] && printf '%s\0' "$p"; done
```

Prints, NUL-terminated, the arguments that are directories. The exit status is
not read.

## Known deviations from the reference implementation

None currently.

When the reference (`ShellFs`) behaves differently from this spec, the
difference is listed here, and the scenarios carry an `xfail` mark for the
reference driver on exactly those checks, which turns into a plain pass when
the reference is fixed.
