# @beyond.dev/sandbox

**Use any sandbox from TypeScript with one URL.** Run commands, stream them, read and write files atomically, search, and keep
background jobs, over the exec protocol that [`execd`](../../../crates/execd/README.md) speaks (v1.1 to v1.3). No runtime
dependencies: it is `fetch` and web streams, on Node 18+ and Bun.

```ts
import { Sandbox } from "@beyond.dev/sandbox";

const sb = await Sandbox.connect({
  url: process.env.EXEC_URL!,
  token: process.env.EXEC_TOKEN,
});

const { exitCode, stdoutText } = await sb.exec("uname", ["-sr"]); // a non-zero exit is a result, not an error
await sb.files.write("/work/hello.txt", "hello\n"); // atomic, keeps the file's mode
console.log(await sb.files.readText("/work/hello.txt"));

const { hits } = await sb.files.search({
  pattern: "hel+o",
  root: "/work",
  before: 1,
  after: 1,
});
for await (const frame of sb.stream("sh", ["-c", "make"])) {
  if (frame.type === "stdout") process.stdout.write(frame.data); // break out and the sandbox kills the command
}
await sb.jobs.start("my-session", "dev1", {
  command: "npm",
  args: ["run", "dev"],
}); // survives this process
```

**Proven, not claimed.** The package ships a conformance driver (`dist/driver.mjs`) that runs the shared scenarios of
[`sdk/conformance`](../../conformance/README.md) against a real `execd`: 421 checks pass on the plain backend, the same as the
Rust client, with 0 failed and 0 unsupported on all 12 backends (a bearer, no stdin, GNU `grep`, busybox `grep`, a v1.1-only
endpoint, baseline jobs, Alpine with real busybox, Debian under `EXECD_RUN_AS`, a read-only mount, a disk that fills). On Alpine it
passes the 20 checks the Rust client fails. Every file helper runs the byte-for-byte script of
[FILES-OVER-EXEC.md](../../conformance/FILES-OVER-EXEC.md), so it works on busybox as on Debian.

## Three layers, each usable alone

```text
Sandbox    exec, stream, files, jobs, transport        one object over any transport
Files, Jobs                                            built on Transport.run and nothing else
Transport  run, stream, nativeJobs, offers             raw exec; replace it with ssh, docker exec, a vendor SDK, a test double
```

1. **Transport** is the whole interface a sandbox needs:
   `run(command, args, { cwd, timeoutMs, stdin }) -> { exitCode, stdout, stderr }` with bytes in and out. `stream`, `nativeJobs`
   and `offers` are optional additions for endpoints that stream (v1.2) or keep jobs themselves (v1.3).
   `HttpTransport({ url, token })` is the default: `token` is a string, or `() => Promise<string>` which is called again once on
   a `401`, retrying that request once. That is the only retry the package makes.
2. **`Files` and `Jobs`** take any `Transport`: `new Files(transport)`. They never touch anything but `transport.run`, so a helper
   you write has the same powers. `Files` probes the sandbox once (`rg` or GNU `grep` or busybox `grep`; `find -printf`; whether
   stdin arrives; whether `dd`/`base64` can read windows) and walks the fallback ladder: a write without stdin goes in 48 KiB
   base64 chunks, a listing without `-printf` takes two `find` passes.
3. **`Sandbox`** composes both and always exposes raw exec (`sb.exec`, `sb.stream`, `sb.transport`).

### Raw exec is bytes

`sb.exec()` returns `stdout` and `stderr` as `Uint8Array` (`stdoutText` and `stderrText` are the lossy UTF-8 views) and an
`exact` flag. A `tar`, a `gzip` or a `cat` of a binary comes back intact when the endpoint honours `output_base64` (`execd`
does) or streams (v1.2). A plain v1.1 endpoint can only answer text, with invalid UTF-8 already replaced by U+FFFD; the result then
carries that text as UTF-8 and `exact: false`, so the client never passes text off as bytes. The file and job helpers read text
and ask for it (`run(..., { text: true })`); a transport of your own may ignore that, and leaves `exact` unset (meaning exact)
when it has the bytes.

### Files

| call                                                        | notes                                                                                                                                                      |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stat(path)`                                                | `null` if absent; follows symlinks; `mtime` in whole epoch seconds                                                                                         |
| `read(path, { offset, max })`                               | windows of 96 KiB; a missing file is `not_found`, never an empty file                                                                                      |
| `write(path, bytes \| string)`                              | temp file beside the target, `mv -f`; keeps mode; writes through symlinks; refuses a directory                                                             |
| `writeIfUnchanged(path, data, expectedMtime)`               | `null` means "must not exist"; `conflict` otherwise. Two round trips, not a lock                                                                           |
| `mkdir(path)`, `list(path, { includeHidden, limit })`       | `mkdir -p`; entries `{ name, kind, size }`                                                                                                                 |
| `search({ pattern, root, literal, ignoreCase, glob, ... })` | context, limits, long-line windows; `capabilities().searchEngine` says which engine ran, and the engines differ in regex dialect and `.gitignore` handling |

### Jobs

`jobs.start(scope, id, { command, args, cwd, label })`, `list`, `read(scope, id, { offset, max })`, `follow` / `attach`, `kill`,
`remove`, `wipe`. A job is owned by the sandbox: it keeps running when you disconnect, and its last 8 MiB of output is
readable from any offset. Natively (v1.3) when the endpoint offers it; otherwise the baseline, built from `sh`, `setsid`,
`mkfifo`, `dd` and `base64` with the supervisor of the protocol, so it works on every endpoint that can run those.
`jobs.capabilities()` says `native`, `baseline` or `none`. `scope` is 1 to 64 of `[0-9a-z-]`, an id 1 to 32 of `[0-9a-z]`.

### Errors

Every failure is a `SandboxError` with `kind` and `detail`, using the spec's vocabulary: `not_found`, `is_directory`,
`not_a_directory`, `permission_denied`, `io` (a command ran and failed), `invalid_query`, `conflict`, `too_large` (the answer
exceeded the 16 MiB response cap; the command ran), `auth`, `timeout`, `transport`, `protocol`, `endpoint`, `not_offered`,
`invalid_request`. There are subclasses (`NotFoundError`, `AuthError`, ...) and `err.kind` for a switch.

## Bring your own transport

```ts
import { Sandbox, type Transport } from "@beyond.dev/sandbox";

const overSsh: Transport = {
  async run(command, args, { stdin, timeoutMs } = {}) {
    const { code, out, err } = await mySshClient.exec([command, ...args], {
      stdin,
      timeoutMs,
    });
    return { exitCode: code, stdout: out, stderr: err };
  },
};
const sb = new Sandbox(overSsh); // sb.files and sb.jobs (baseline) work as they do over HTTP
```

Report transport failures as `SandboxError`s (`transport`, `timeout`, ...); a command that exits non-zero is a result.
`__tests__/local-transport.test.ts` runs the whole of `Files` and `Jobs` over a transport that spawns processes on the local
machine, with no HTTP.

## Write your own helper

A helper is a function of a `Transport`. Pass variable parts as arguments, never inside a shell string: a path like
`'; rm -rf /` is then one inert argument.

```ts
async function gitStatus(t: Transport, dir: string): Promise<string[]> {
  const r = await t.run("git", ["status", "--porcelain"], { cwd: dir });
  if (r.exitCode !== 0) throw new Error(new TextDecoder().decode(r.stderr));
  return new TextDecoder().decode(r.stdout).split("\n").filter(Boolean);
}
```

When you need a pipeline, send a fixed script and let the variable parts arrive as `$1`, `$2`:
`t.run("sh", ["-c", 'find "$1" -newer "$2" | sort', "sh", dir, marker])`.

## Notes

- `stdout` and `stderr` of the one-shot call are text on the wire (invalid UTF-8 becomes U+FFFD, decided by the endpoint). To
  move arbitrary bytes, encode them (base64) inside the command, or use `stream` (base64 frames, exact).
- `stream` needs an endpoint that proved it streams (a probe, once). If it did not, `stream` throws `not_offered` and the command
  is not run as a one-shot instead.
- `Files.capabilities()` and `Jobs.capabilities()` throw if a probe request fails at the transport level (unreachable, `auth`);
  nothing is cached from a probe that did not run.
- The conformance driver is `node dist/driver.mjs` (env `CONFORMANCE_URL`, `CONFORMANCE_BEARER`, `CONFORMANCE_JOBS_ROOT`,
  `CONFORMANCE_CAPS`); see [DRIVER.md](../../conformance/DRIVER.md).

## Development

```sh
mise run install:ts && mise run build:ts && mise run check:ts && mise run test:ts
```

`test:ts` starts a loopback `execd` (`EXECD_BIN`, else `target/debug/execd`, built with `cargo build -p beyond-execd` if absent).
`src/scripts.ts` is generated from the spec and the protocol's baseline jobs by `node scripts/gen-scripts.mjs`; a test fails when
it and `FILES-OVER-EXEC.md` disagree.
