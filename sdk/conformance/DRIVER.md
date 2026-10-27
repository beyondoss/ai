# The driver contract

Every client (the Rust crate, the MCP server, the TypeScript package, the Python
package) ships a small **driver**: a process that reads operations as JSON lines
on stdin and writes results as JSON lines on stdout. A harness feeds the same
scenario files to every driver against every backend and compares the answers.
The behaviour behind each op is specified in
[FILES-OVER-EXEC.md](FILES-OVER-EXEC.md) and in the exec protocol. The reference
driver and harness are in `crates/sandbox/tests/sdk_conformance*.rs`; where this
document is unclear, they are the answer.

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

## 1. Process lifecycle

The harness starts one driver process per (backend, scenario connection) and
talks to it over its stdin and stdout. The driver:

- MUST read its configuration from the environment:

| variable                | meaning                                                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `CONFORMANCE_URL`       | the exec endpoint's URL (required)                                                                                                  |
| `CONFORMANCE_BEARER`    | the bearer token; absent or empty means no `Authorization` header                                                                   |
| `CONFORMANCE_JOBS_ROOT` | where the baseline keeps jobs in the sandbox; default `/tmp/beyond-jobs`                                                            |
| `CONFORMANCE_CAPS`      | optional JSON object overriding probe results: `stdin` (bool), `find_printf` (bool), `search_engine` (`rg`, `grep`, `busybox-grep`) |

- MUST read requests one line at a time, answer each with exactly one line on
  stdout, **in order**, and never have two requests in flight. stdout carries
  only protocol lines; logs go to stderr.
- MUST connect (and probe) on the first request that needs it, and MUST exit
  cleanly when stdin closes.
- MUST apply `CONFORMANCE_CAPS` after probing, so that one sandbox can walk
  every rung of the fallback ladder (`stdin: false` forces the 48 KiB chunked
  write, `search_engine: "grep"` the POSIX rung, `find_printf: false` the
  two-pass listing). A harness that cannot force a rung runs only the scenarios
  whose `requires` the real backend meets.
- MUST NOT retry failed requests or hide an error.

## 2. Envelope

Request, one JSON object per line:

```json
{ "id": 7, "op": "read", "path": "/work/a.txt", "offset": 0, "max": 4096 }
```

`id` is any JSON value the driver echoes. Unknown fields are ignored; a missing
or mistyped required field is `invalid_request` and nothing is sent to the
endpoint. Integers are JSON integers (`-1` and `1.5` are not counts).

Response, exactly one of:

```json
{"id": 7, "ok": {"data_b64": "aGVsbG8="}}
{"id": 7, "error": {"kind": "not_found", "detail": "read /work/a.txt: No such file or directory (os error 2)"}}
```

`ok` is always an object (`{}` when there is nothing to say). `error.kind` is
from the vocabulary below; `detail` is free text for humans.

**Bytes** are base64 (RFC 4648, standard alphabet, padding) in fields ending
`_b64`. Text that is not bytes (names, paths, pattern, `text` of a search hit)
is a JSON string.

## 3. Ops

Layer 1 (transport) and layer 2 (helpers). "Unknown" optional fields are
absent or `null`.

### Meta

| op             | request | result                                                                                       |
| -------------- | ------- | -------------------------------------------------------------------------------------------- |
| `hello`        |         | `{driver, protocol: 1, ops: [names]}`: the ops this driver implements                        |
| `capabilities` |         | `{stdin, stream, jobs: "native"/"baseline"/"none", search_engine, find_printf, read_window}` |

`hello` MUST succeed without reaching the endpoint. `capabilities` reports what
the probes found (after `CONFORMANCE_CAPS`), or the connection error if the
endpoint cannot be used (for example `auth`).

### Exec (layer 1)

| op            | request                                                                                | result                                                                                  |
| ------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `exec`        | `command`, `args` (default `[]`), `cwd?`, `timeout_ms?` (default 120000), `stdin_b64?` | `{exit_code, stdout_b64, stderr_b64}`                                                   |
| `exec_stream` | `command`, `args`, `cwd?`, `timeout_ms?`                                               | `{stdout_b64, stderr_b64}` plus exactly one of `exit_code`, `signal`, `timed_out: true` |

`exec` is the one-shot call (v1.1): a non-zero `exit_code` is a normal result;
a timeout is `exit_code` 124. `exec_stream` is v1.2: output is collected until
the final frame. Without `timeout_ms` the stream has no deadline of its own.
`not_offered` if the endpoint did not prove it streams; the command is not run
as a one-shot instead. (The reference driver puts both streams in `stdout_b64`
when `timeout_ms` is given, because the call it uses then merges them.)

### Jobs (layer 1)

Every job op carries `scope`, a string of 1 to 64 characters of `[0-9a-z-]`
naming one session's jobs; the harness sends a scope unique to the scenario.
`job` is the job id, 1 to 32 characters of `[0-9a-z]`, chosen by the caller.

| op           | request                                             | result                                                                           |
| ------------ | --------------------------------------------------- | -------------------------------------------------------------------------------- |
| `job_start`  | `scope`, `job`, `command`, `args`, `cwd?`, `label?` | `{pgid}`: the job is kept by the sandbox from the start                          |
| `job_list`   | `scope`                                             | `{jobs: [{id, state, code, kept, num, kind, seen, started, ended, label}]}`      |
| `job_read`   | `scope`, `job`, `offset` (0), `max` (131072)        | `{state, code, offset, end, data_b64}`                                           |
| `job_attach` | `scope`, `job`, `offset` (0)                        | `{exit_code, output_b64, gap}`: follows to the job's end; `gap` is bytes dropped |
| `job_kill`   | `scope`, `jobs: [ids]`, `remove?`                   | `{survivors: [{id, detail}]}`: groups still alive after TERM, grace and KILL     |
| `job_remove` | `scope`, `jobs: [ids]`                              | `{}`                                                                             |
| `job_wipe`   | `scope`                                             | `{survivors: [..]}`                                                              |

`state` is `running`, `exited` (the shell exited, its group runs on), `ended`,
`lost` or `gone`; `code` is null unless `exited` or `ended`. An invalid job id
is `invalid_request` without sending anything. A driver whose endpoint offers no
jobs at all answers `not_offered`.

### Files (layer 2)

| op                   | request                                                                                                      | result                                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `stat`               | `path`                                                                                                       | `{exists: false}` or `{exists: true, kind, size, mtime, writable}` (`kind` `file`/`dir`/`other`; `mtime` epoch seconds) |
| `read`               | `path`, `offset?` (0), `max?` (to end of file)                                                               | `{data_b64}`                                                                                                            |
| `write`              | `path`, `data_b64`                                                                                           | `{}`                                                                                                                    |
| `write_if_unchanged` | `path`, `data_b64`, `expected_mtime` (**required**; integer or null for "must not exist")                    | `{}`, or error `conflict`                                                                                               |
| `mkdir`              | `path`                                                                                                       | `{}`                                                                                                                    |
| `list`               | `path`, `include_hidden?` (false), `limit?` (10000)                                                          | `{entries: [{name, kind, size}]}` (order unspecified)                                                                   |
| `search`             | `pattern`, `root`, `literal?`, `ignore_case?`, `glob?`, `glob_exclude?`, `limit?` (100), `before?`, `after?` | `{hits: [{path, line, text, is_match}], truncated, note}`                                                               |
| `remove`             | `path`                                                                                                       | planned: no counterpart in the reference                                                                                |
| `upload_dir`         | `path`, `tar_b64`                                                                                            | **planned, T7**                                                                                                         |
| `download_dir`       | `path`                                                                                                       | **planned, T7**: `{tar_b64}`                                                                                            |

A write that fails its precondition is an error, never `{written: false}`: the
reference returns a boolean, and the driver turns `false` into `conflict`.

## 4. Errors

| kind                                                                      | when                                                          |
| ------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `not_found`, `is_directory`, `not_a_directory`, `permission_denied`, `io` | a command ran and failed (FILES-OVER-EXEC section 3.3)        |
| `invalid_query`                                                           | the search pattern or glob was rejected                       |
| `conflict`                                                                | `write_if_unchanged`: the mtime is not the expected one       |
| `too_large`                                                               | the endpoint's answer exceeded the response cap               |
| `auth`                                                                    | HTTP 401 or 403                                               |
| `timeout`                                                                 | the request deadline passed                                   |
| `transport`                                                               | unreachable, reset, or a stream that broke                    |
| `protocol`                                                                | the answer is not the protocol                                |
| `endpoint`                                                                | any other non-2xx status                                      |
| `not_offered`                                                             | the **endpoint** lacks the capability (no streaming, no jobs) |
| `invalid_request`                                                         | the **request** cannot be formed; nothing was sent            |
| `unsupported`                                                             | the **driver** does not implement the op                      |

`unsupported` is how a driver reports a capability it does not have. It is never
a pass: see section 6.

## 5. Scenarios

A scenario file is `{suite, version, description, scenarios: [...]}` and is
named `<suite>.json`. A scenario:

```json
{
  "id": "files.write.through-symlink",
  "description": "...",
  "requires": {"stdin": true},
  "planned": "T7",
  "connect": {"bearer": "wrong"},
  "setup": [ {step}, ... ],
  "steps": [ {step}, ... ]
}
```

Only `id`, `description` and `steps` are required. IDs are stable, unique and
start with the suite name. A **step** is an op request plus harness fields:

```json
{
  "id": "stat",
  "op": "stat",
  "path": "${ROOT}/f",
  "expect": { "ok": { "size": 5 } },
  "save": { "m": "mtime" }
}
```

`id` (unique in the scenario; a **check** is `<scenario id>/<step id>`), `op`
and the op's own fields go to the driver. The harness fields are not sent:
`expect`, `save`, `poll`, `xfail`, `note`.

### Running a scenario

1. If `planned`, skip it and print `skipped: planned (<value>) <id>`.
2. Evaluate `requires` (below); if unmet, print `skipped: <why>: <id>`.
3. If the driver does not list every op the scenario uses in `hello.ops`, the
   scenario is **unsupported**: its checks are counted as `unsupported`, never
   as passes (except that a step expecting `error.kind` `unsupported` for an op
   the driver lacks is fine).
4. Use the backend's driver, or if `connect` is present a fresh driver whose
   `CONFORMANCE_BEARER` and `CONFORMANCE_URL` are replaced by `connect.bearer`
   (`null` meaning none) and `connect.url`.
5. Make the scenario's directory `${ROOT}` (`mkdir -p` through `exec`), run
   `setup` steps (each must reach the driver and return `ok` or `error`;
   `setup` is not counted), then the `steps` in order, each a check.
6. Afterwards, best effort: `job_wipe` the scope and remove `${ROOT}`.

### Variables

Any string in a step is substituted before sending. `${ROOT}` is an absolute,
unique, empty directory in the sandbox, `${WORK}` its parent, `${SCOPE}` a
unique job scope (`[0-9a-z]`). `save: {"name": "dotted.path"}` stores a value
from the step's `ok` result (array indices are numbers in the path) as
`${name}`. A string that is exactly one variable becomes the saved value with
its type; otherwise the text is interpolated.

`{"$repeat": {"b64": "...", "times": N, "len": L}}` anywhere a base64 string is
wanted stands for the base64 of the decoded chunk repeated `N` times and, if
`len` is given, cut to `L` bytes.

### Derived views

For every field `X_b64` in an `ok` result, the harness also exposes `X_utf8`
(decoded as UTF-8, bad bytes as U+FFFD), `X_len` (byte count) and `X_sha256`
(lowercase hex), so an expectation can speak of text or of a digest.

### Expectations

`expect` is `{"ok": E}` or `{"error": E}` (default `{"ok": {}}`); the response
must have that key. `E` matches the value:

- an **object** matches when every key in `E` matches the same key of the value
  (extra keys in the value are fine; a missing key is `null`);
- an **array** matches when it has the same length and every element matches;
- a scalar matches when equal (numbers compare as numbers);
- an object with a single key starting with `$` is a matcher: `$any` (present),
  `$absent` (null or missing), `$type` (`null boolean number string array object`),
  `$re` (regular expression, unanchored), `$contains` (substring), `$gt`, `$gte`,
  `$lt`, `$lte`, `$oneof` (list of alternatives), `$not`, `$len` (array length
  or string byte length), `$unordered` (an array with the same elements in any
  order), `$includes` (an array containing elements that match each item).

### poll

`"poll": {"timeout_ms": 5000, "interval_ms": 50}` repeats the step until the
expectation matches or the time runs out (then the last mismatch is the
failure). Used for job state.

### requires

All of: `stdin`, `stream`, `find_printf`, `read_window` (booleans, compared with
`capabilities`); `jobs` (`native`, `baseline`, `any` meaning either);
`search_engine` (a value or a list of acceptable values); `auth` (true when the
backend demands a bearer); `not_root` (the sandbox user is not uid 0, so
permission checks mean something); `commands` (a list that `command -v` finds in
the sandbox).

### xfail

`"xfail": {"drivers": ["agent-reference"], "reason": "...", "when": {requires}}`
marks a check the named driver is known to fail because of a reference defect
listed in FILES-OVER-EXEC. For that driver (and only when `when` holds) the
check **must fail** and is counted as a known defect; if it passes, the harness
reports an error so the mark is removed. Every other driver treats the check as
an ordinary check and must pass it.

## 6. Outcomes

Per check: `pass`, `fail`, `skip` (planned or `requires` unmet), `unsupported`
(the driver lacks an op), `known defect` (xfail failed as expected). A run
succeeds only with no `fail`. Counts are reported separately and `skip` and
`unsupported` are never added to `pass`. A driver may declare the ops it knows
it does not implement yet; any other `unsupported` fails the run.

## 7. Backends

A backend is a URL (and optionally a bearer) of an exec endpoint, plus a way for
the harness to make a writable directory in it (`${WORK}`). `execd`, started
anywhere, is the reference backend; any endpoint that speaks v1.1 to v1.3 can be
tested, and `requires` skips what it cannot do. A backend's own probes decide
`stdin`, `stream`, `jobs` and the rest.
