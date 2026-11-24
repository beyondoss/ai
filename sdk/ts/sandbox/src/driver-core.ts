/**
 * The conformance driver (sdk/conformance/DRIVER.md): reads ops as JSON lines on stdin, answers
 * each with one JSON line on stdout, in order. A thin mapping from the protocol to the public API
 * of this package: layer 1 through `Transport`, layer 2 through `Files` and `Jobs`.
 *
 *   CONFORMANCE_URL       the exec endpoint's URL (required)
 *   CONFORMANCE_BEARER    the bearer token; absent or empty means none
 *   CONFORMANCE_TOKEN_CMD a shell command whose output is the bearer; run again after a 401
 *   CONFORMANCE_CAPS      JSON overriding probes: find_printf, search_engine
 */
import { execFile } from "node:child_process";
import { mkdirSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { extract } from "./bulk.js";
import { fromBase64, toBase64 } from "./bytes.js";
import {
  BulkReject,
  forceRemove,
  materialize,
  withSandboxDir,
} from "./driver-bulk.js";
import {
  DiskFullError,
  JobExistsError,
  JobGoneError,
  SandboxError,
  StdinBusyError,
  StdinClosedError,
  StdinGapError,
} from "./errors.js";
import type { ErrorKind } from "./errors.js";
import type { CapabilityOverrides } from "./files.js";
import { Sandbox } from "./sandbox.js";
import type { SearchEngine } from "./search.js";
import type { JobInfo, Survivor } from "./transport.js";

type Json = Record<string, unknown>;
type DriverError = {
  kind: ErrorKind | "unsupported";
  detail: string;
  /** The job condition, when the error is a finer type of its kind (DRIVER.md, `job_errors`). */
  code?: string;
};

export const OPS = [
  "hello",
  "capabilities",
  "exec",
  "exec_stream",
  "job_start",
  "job_start_held",
  "job_keep",
  "job_seen",
  "job_list",
  "job_read",
  "job_attach",
  "job_kill",
  "job_remove",
  "job_wipe",
  "job_write",
  "job_signal",
  "stat",
  "read",
  "write",
  "write_if_unchanged",
  "mkdir",
  "list",
  "search",
  "upload",
  "download",
  "extract",
] as const;

export interface DriverConfig {
  url: string;
  bearer?: string | undefined;
  /** A shell command whose stdout is the bearer; it takes the place of `bearer`. */
  tokenCmd?: string | undefined;
  caps?: CapabilityOverrides | undefined;
}

/** Thrown inside an op to answer with an error. */
class Reject extends Error {
  readonly err: DriverError;

  constructor(err: DriverError) {
    super(err.detail);
    this.err = err;
  }
}

/** `env`: an object of strings, or nothing. */
function optEnv(req: Json): Record<string, string> | undefined {
  const v = req.env;
  if (v === undefined || v === null) return undefined;
  if (typeof v !== "object" || Array.isArray(v)) {
    throw invalid("`env` must be an object");
  }
  const out: Record<string, string> = {};
  for (const [k, x] of Object.entries(v)) {
    if (typeof x !== "string") throw invalid("`env` must hold strings");
    out[k] = x;
  }
  return out;
}

const invalid = (detail: string): Reject =>
  new Reject({ kind: "invalid_request", detail });

function str(req: Json, key: string): string {
  const v = req[key];
  if (typeof v !== "string") throw invalid(`\`${key}\` must be a string`);
  return v;
}

function optStr(req: Json, key: string): string | undefined {
  const v = req[key];
  if (v === undefined || v === null) return undefined;
  if (typeof v !== "string") throw invalid(`\`${key}\` must be a string`);
  return v;
}

function optCount(req: Json, key: string): number | undefined {
  const v = req[key];
  if (v === undefined || v === null) return undefined;
  if (typeof v !== "number" || !Number.isInteger(v) || v < 0) {
    throw invalid(`\`${key}\` must be a count`);
  }
  return v;
}

function optNumber(req: Json, key: string): number | undefined {
  const v = req[key];
  if (v === undefined || v === null) return undefined;
  if (typeof v !== "number" || !Number.isFinite(v) || v < 0) {
    throw invalid(`\`${key}\` must be a non-negative number`);
  }
  return v;
}

function flag(req: Json, key: string): boolean {
  const v = req[key];
  if (v === undefined || v === null) return false;
  if (typeof v !== "boolean") throw invalid(`\`${key}\` must be true or false`);
  return v;
}

function strings(req: Json, key: string): string[] {
  const v = req[key];
  if (v === undefined || v === null) return [];
  if (!Array.isArray(v)) throw invalid(`\`${key}\` must be a list`);
  return v.map((x) => {
    if (typeof x !== "string") throw invalid(`\`${key}\` must hold strings`);
    return x;
  });
}

function bytes(req: Json, key: string): Uint8Array {
  const b = fromBase64(str(req, key));
  if (!b) throw invalid(`\`${key}\` is not base64`);
  return b;
}

const b64 = toBase64;

const jobJson = (j: JobInfo): Json => ({
  id: j.id,
  state: j.state,
  code: j.code,
  ...(j.signal === undefined ? {} : { signal: j.signal }),
  kept: j.kept,
  num: j.num,
  kind: j.kind,
  seen: j.seen,
  started: j.started,
  ended: j.ended,
  label: j.label,
  mark: j.mark,
  command: j.command,
  args: j.args,
  output_start: j.outputStart,
  output_end: j.outputEnd,
});

export class Driver {
  private readonly cfg: DriverConfig;
  private connected: Promise<Sandbox> | undefined;

  constructor(cfg: DriverConfig) {
    this.cfg = cfg;
  }

  /**
   * Connect (and probe) on the first request that needs it; a failure is that op's answer. The
   * driver goes through `Sandbox`, so its wiring is under test too.
   */
  private connect(): Promise<Sandbox> {
    this.connected ??= Sandbox.connect({
      url: this.cfg.url,
      token: this.cfg.tokenCmd
        ? tokenFromCommand(this.cfg.tokenCmd)
        : this.cfg.bearer || undefined,
      files: { overrides: this.cfg.caps },
    });
    return this.connected;
  }

  async handle(req: Json): Promise<Json> {
    const id = req.id;
    const op = typeof req.op === "string" ? req.op : "";
    try {
      if (!(OPS as readonly string[]).includes(op)) {
        throw new Reject({
          kind: "unsupported",
          detail: `no such op ${JSON.stringify(op)}`,
        });
      }
      if (op === "hello") {
        return {
          id,
          ok: {
            driver: "beyond-sandbox-ts",
            protocol: 1,
            ops: [...OPS],
            features: [
              "frames",
              "unbounded_stream",
              "exit_signal",
              "exec_env",
              "job_streams",
              "job_stdin",
              "job_signal_op",
              "job_errors",
              "job_options",
              "job_marks",
            ],
          },
        };
      }
      return { id, ok: await this.dispatch(op, req, await this.connect()) };
    } catch (e) {
      return { id, error: e instanceof Reject ? e.err : toError(e) };
    }
  }

  private async dispatch(
    op: string,
    req: Json,
    sb: Sandbox,
  ): Promise<Json> {
    const { files, jobs } = sb;
    switch (op) {
      case "capabilities": {
        const c = await files.capabilities();
        return {
          search_engine: c.searchEngine,
          find_printf: c.findPrintf,
          read_window: c.readWindow,
        };
      }
      case "exec": {
        const stdin = req.stdin_b64 === undefined || req.stdin_b64 === null
          ? undefined
          : bytes(req, "stdin_b64");
        const cwd = optStr(req, "cwd");
        const env = optEnv(req);
        const r = await sb.exec(
          str(req, "command"),
          strings(req, "args"),
          {
            timeoutMs: optCount(req, "timeout_ms") ?? 120_000,
            ...(cwd !== undefined ? { cwd } : {}),
            ...(stdin !== undefined ? { stdin } : {}),
            ...(env !== undefined ? { env } : {}),
          },
        );
        return {
          exit_code: r.exitCode,
          stdout_b64: b64(r.stdout),
          stderr_b64: b64(r.stderr),
        };
      }
      case "exec_stream": {
        const command = str(req, "command");
        const args = strings(req, "args");
        const cwd = optStr(req, "cwd");
        const timeoutMs = optCount(req, "timeout_ms");
        const out: Uint8Array[] = [];
        const err: Uint8Array[] = [];
        let end: Json | undefined;
        for await (
          const f of sb.stream(command, args, {
            ...(cwd !== undefined ? { cwd } : {}),
            ...(timeoutMs !== undefined ? { timeoutMs } : {}),
          })
        ) {
          if (f.type === "stdout") out.push(f.data);
          else if (f.type === "stderr") err.push(f.data);
          else if (f.type === "exit") end = { exit_code: f.code };
          else if (f.type === "signal") end = { signal: f.signal };
          else if (f.type === "timed_out") end = { timed_out: true };
        }
        return { stdout_b64: b64(cat(out)), stderr_b64: b64(cat(err)), ...end };
      }
      case "job_start": {
        const cwd = optStr(req, "cwd");
        const label = optStr(req, "label");
        const env = optEnv(req);
        const started = await jobs.start(str(req, "scope"), str(req, "job"), {
          command: str(req, "command"),
          args: strings(req, "args"),
          ...(cwd !== undefined ? { cwd } : {}),
          ...(label !== undefined ? { label } : {}),
          ...(env !== undefined ? { env } : {}),
          ...(flag(req, "stdin_open") ? { stdinOpen: true } : {}),
        });
        return { pgid: started.pgid, mark: started.mark };
      }
      case "job_start_held": {
        // Start held, learn the pgid from the first frame, optionally `keep`, then drop the
        // stream: a job nobody kept dies with it (wire `hold: true`).
        const cwd = optStr(req, "cwd");
        const label = optStr(req, "label");
        const env = optEnv(req);
        const scope = str(req, "scope");
        const job = str(req, "job");
        const stream = jobs.startHeld(scope, job, {
          command: str(req, "command"),
          args: strings(req, "args"),
          ...(cwd !== undefined ? { cwd } : {}),
          ...(label !== undefined ? { label } : {}),
          ...(env !== undefined ? { env } : {}),
          ...(flag(req, "stdin_open") ? { stdinOpen: true } : {}),
        }, heartbeat(req));
        try {
          for await (const f of stream) {
            if (f.type !== "started") continue;
            if (flag(req, "keep")) await jobs.keep(scope, job);
            return { pgid: f.pgid, mark: f.mark };
          }
        } finally {
          await stream.return(undefined);
        }
        throw new Reject({
          kind: "protocol",
          detail: "the held start ended before `started`",
        });
      }
      case "job_keep": {
        const num = optCount(req, "num");
        const kind = optStr(req, "kind");
        const seen = optCount(req, "seen");
        await jobs.keep(str(req, "scope"), str(req, "job"), {
          ...(num !== undefined ? { num } : {}),
          ...(kind !== undefined ? { kind } : {}),
          ...(seen !== undefined ? { seen } : {}),
        });
        return {};
      }
      case "job_seen":
        await jobs.seen(
          str(req, "scope"),
          str(req, "job"),
          optCount(req, "offset") ?? 0,
        );
        return {};
      case "job_list":
        return { jobs: (await jobs.list(str(req, "scope"))).map(jobJson) };
      case "job_read": {
        let c;
        try {
          c = await jobs.read(str(req, "scope"), str(req, "job"), {
            offset: optCount(req, "offset") ?? 0,
            max: optCount(req, "max") ?? 128 * 1024,
          });
        } catch (e) {
          // The wire answers `gone` for a job that is not there; the client says `JobGoneError`.
          // `gone_as_error` asks for the error, as a caller of the client gets it.
          if (e instanceof JobGoneError && !flag(req, "gone_as_error")) {
            return {
              state: "gone",
              code: null,
              offset: 0,
              end: 0,
              data_b64: "",
            };
          }
          throw e;
        }
        return {
          state: c.state,
          code: c.code,
          ...(c.signal === undefined ? {} : { signal: c.signal }),
          offset: c.offset,
          end: c.end,
          data_b64: b64(c.data),
          ...(c.stderrRanges ? { stderr_ranges: c.stderrRanges } : {}),
        };
      }
      case "job_attach": {
        const mark = optStr(req, "mark");
        const output = req.output === undefined || req.output === null
          ? undefined
          : flag(req, "output");
        const a = await jobs.collect(
          str(req, "scope"),
          str(req, "job"),
          optCount(req, "offset") ?? 0,
          {
            ...heartbeat(req),
            ...(mark !== undefined ? { mark } : {}),
            ...(output !== undefined ? { output } : {}),
          },
        );
        // As `exec_stream` ends: `signal` in place of `exit_code` for a death by signal.
        const end = a.signal === undefined
          ? { exit_code: a.exitCode }
          : { signal: a.signal };
        return {
          ...(a.attached ? { attached: a.attached } : {}),
          ...end,
          output_b64: b64(a.output),
          ...(a.stderr.length > 0 ? { stderr_b64: b64(a.stderr) } : {}),
          gap: a.gap,
        };
      }
      case "job_kill": {
        const s = await jobs.kill(str(req, "scope"), strings(req, "jobs"), {
          remove: flag(req, "remove"),
          ...(optCount(req, "grace_ms") !== undefined
            ? { graceMs: optCount(req, "grace_ms") as number }
            : {}),
        });
        return {
          survivors: s.map((x: Survivor) => ({ id: x.id, detail: x.detail })),
        };
      }
      case "job_write": {
        const at = optCount(req, "at");
        if (at === undefined) throw invalid("`at` must be a count");
        const w = await jobs.writeStdin(
          str(req, "scope"),
          str(req, "job"),
          at,
          bytes(req, "data_b64"),
          flag(req, "eof"),
        );
        return { written: w.written, ...(w.closed ? { closed: true } : {}) };
      }
      case "job_signal":
        return {
          sent: await jobs.signal(
            str(req, "scope"),
            strings(req, "jobs"),
            str(req, "signal"),
          ),
        };
      case "job_remove": {
        const raw = req.marks;
        let marks: Record<string, string> | undefined;
        if (raw !== undefined && raw !== null) {
          if (typeof raw !== "object" || Array.isArray(raw)) {
            throw invalid("`marks` must be an object of strings");
          }
          marks = {};
          for (const [k, v] of Object.entries(raw)) {
            if (typeof v !== "string") {
              throw invalid("`marks` must be an object of strings");
            }
            marks[k] = v;
          }
        }
        return {
          ...(await jobs.remove(
            str(req, "scope"),
            strings(req, "jobs"),
            marks !== undefined ? { marks } : {},
          )),
        };
      }
      case "job_wipe":
        return {
          survivors: (await jobs.wipe(str(req, "scope"), {
            ...(optCount(req, "grace_ms") !== undefined
              ? { graceMs: optCount(req, "grace_ms") as number }
              : {}),
          })).map((x) => x.detail),
        };
      case "stat": {
        const s = await files.stat(str(req, "path"));
        return s === null
          ? { exists: false }
          : {
            exists: true,
            kind: s.kind,
            size: s.size,
            mtime: s.mtime,
            writable: s.writable,
          };
      }
      case "read": {
        const offset = optCount(req, "offset");
        const max = optCount(req, "max");
        const data = await files.read(str(req, "path"), {
          ...(offset !== undefined ? { offset } : {}),
          ...(max !== undefined ? { max } : {}),
        });
        return { data_b64: b64(data) };
      }
      case "write":
        await files.write(str(req, "path"), bytes(req, "data_b64"));
        return {};
      case "write_if_unchanged": {
        const path = str(req, "path");
        const data = bytes(req, "data_b64");
        if (!("expected_mtime" in req)) {
          throw invalid(
            "`expected_mtime` is required (null means the file must not exist)",
          );
        }
        await files.writeIfUnchanged(
          path,
          data,
          optNumber(req, "expected_mtime") ?? null,
        );
        return {};
      }
      case "mkdir":
        await files.mkdir(str(req, "path"));
        return {};
      case "list": {
        const limit = optCount(req, "limit");
        const entries = await files.list(str(req, "path"), {
          includeHidden: flag(req, "include_hidden"),
          ...(limit !== undefined ? { limit } : {}),
        });
        return { entries };
      }
      case "search": {
        const glob = optStr(req, "glob");
        const limit = optCount(req, "limit");
        const before = optCount(req, "before");
        const after = optCount(req, "after");
        const r = await files.search({
          pattern: str(req, "pattern"),
          root: str(req, "root"),
          literal: flag(req, "literal"),
          ignoreCase: flag(req, "ignore_case"),
          globExclude: flag(req, "glob_exclude"),
          ...(glob !== undefined ? { glob } : {}),
          ...(limit !== undefined ? { limit } : {}),
          ...(before !== undefined ? { before } : {}),
          ...(after !== undefined ? { after } : {}),
        });
        return {
          hits: r.hits.map((h) => ({
            path: h.path,
            line: h.line,
            text: h.text,
            is_match: h.isMatch,
          })),
          truncated: r.truncated,
          note: r.note,
        };
      }
      case "upload": {
        const dest = str(req, "path");
        if (!Array.isArray(req.entries)) {
          throw invalid("`entries` must be a list");
        }
        const chunk = optCount(req, "chunk_bytes");
        const tmp = mkdtempSync(join(tmpdir(), "bulk-up-"));
        try {
          const src = join(tmp, "src");
          mkdirSync(src);
          materialize(src, req.entries as Json[], tmp);
          const sources = req.sources === undefined || req.sources === null
            ? src
            : strings(req, "sources").map((n) => join(src, n));
          return {
            ...await sb.upload(sources, dest, { chunkBytes: chunk }),
          };
        } finally {
          forceRemove(tmp);
        }
      }
      case "download": {
        const remote = str(req, "path");
        return withSandboxDir(
          req,
          (local) => sb.download(remote, local),
        );
      }
      case "extract": {
        const tar = bytes(req, "tar_b64");
        return withSandboxDir(req, (local) => extract(tar, local));
      }
      default:
        throw new Reject({ kind: "unsupported", detail: op });
    }
  }
}

function cat(chunks: Uint8Array[]): Uint8Array {
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

/** A token function over a shell command: its stdout, trimmed. */
function tokenFromCommand(cmd: string): () => Promise<string> {
  return () =>
    new Promise((resolve, reject) => {
      execFile("sh", ["-c", cmd], (err, stdout, stderr) => {
        if (err) {
          const said = stderr.trim();
          return reject(new Error(said === "" ? err.message : said));
        }
        const token = stdout.trim();
        if (token === "") {
          return reject(new Error("the token command printed nothing"));
        }
        resolve(token);
      });
    });
}

/** `heartbeat_seconds`: the request's own heartbeat (the `Keepalive-Ping-Interval` header). */
function heartbeat(req: Json): { heartbeatSeconds?: number } {
  const v = req.heartbeat_seconds;
  if (v === undefined || v === null) return {};
  if (typeof v !== "number") {
    throw invalid("`heartbeat_seconds` must be a number");
  }
  return { heartbeatSeconds: v };
}

/** The job condition of an error, by its class (DRIVER.md, `job_errors`). */
function codeOf(e: SandboxError): string | undefined {
  if (e instanceof JobGoneError) return "job_gone";
  if (e instanceof JobExistsError) return "job_exists";
  if (e instanceof DiskFullError) return "disk_full";
  if (e instanceof StdinClosedError) {
    return e.reason === "never_opened" ? "stdin_never_opened" : "stdin_closed";
  }
  if (e instanceof StdinGapError) return "stdin_gap";
  if (e instanceof StdinBusyError) return "stdin_busy";
  return undefined;
}

function toError(e: unknown): DriverError {
  if (e instanceof BulkReject) {
    return { kind: e.kind as ErrorKind, detail: e.detail };
  }
  if (e instanceof SandboxError) {
    const code = codeOf(e);
    return {
      kind: e.kind,
      detail: e.detail,
      ...(code === undefined ? {} : { code }),
    };
  }
  return { kind: "io", detail: e instanceof Error ? e.message : String(e) };
}

export function capsFromEnv(raw: string | undefined): CapabilityOverrides {
  if (!raw) return {};
  const v = JSON.parse(raw) as Json;
  const o: CapabilityOverrides = {};
  if (typeof v.find_printf === "boolean") o.findPrintf = v.find_printf;
  if (
    v.search_engine === "rg"
    || v.search_engine === "grep"
    || v.search_engine === "busybox-grep"
  ) {
    o.searchEngine = v.search_engine as SearchEngine;
  }
  return o;
}
