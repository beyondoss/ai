/**
 * The conformance driver (sdk/conformance/DRIVER.md): reads ops as JSON lines on stdin, answers
 * each with one JSON line on stdout, in order. A thin mapping from the protocol to the public API
 * of this package: layer 1 through `Transport`, layer 2 through `Files` and `Jobs`.
 *
 *   CONFORMANCE_URL       the exec endpoint's URL (required)
 *   CONFORMANCE_BEARER    the bearer token; absent or empty means none
 *   CONFORMANCE_JOBS_ROOT where the baseline keeps jobs; default /tmp/beyond-jobs
 *   CONFORMANCE_CAPS      JSON overriding probes: stdin, find_printf, search_engine
 */
import { fromBase64, toBase64 } from "./bytes.js";
import { SandboxError } from "./errors.js";
import type { ErrorKind } from "./errors.js";
import { Files } from "./files.js";
import type { CapabilityOverrides } from "./files.js";
import { HttpTransport } from "./http.js";
import { Jobs } from "./jobs.js";
import type { SearchEngine } from "./search.js";
import type { JobInfo, Survivor } from "./transport.js";

type Json = Record<string, unknown>;
type DriverError = { kind: ErrorKind | "unsupported"; detail: string };

export const OPS = [
  "hello",
  "capabilities",
  "exec",
  "exec_stream",
  "job_start",
  "job_list",
  "job_read",
  "job_attach",
  "job_kill",
  "job_remove",
  "job_wipe",
  "stat",
  "read",
  "write",
  "write_if_unchanged",
  "mkdir",
  "list",
  "search",
] as const;

export interface DriverConfig {
  url: string;
  bearer?: string | undefined;
  jobsRoot?: string | undefined;
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
  kept: j.kept,
  num: j.num,
  kind: j.kind,
  seen: j.seen,
  started: j.started,
  ended: j.ended,
  label: j.label,
});

export class Driver {
  private readonly cfg: DriverConfig;
  private connected:
    | Promise<
      { files: Files; jobs: Jobs; transport: HttpTransport } | DriverError
    >
    | undefined;

  constructor(cfg: DriverConfig) {
    this.cfg = cfg;
  }

  /** Connect and probe on the first request that needs it; a failure is every op's answer. */
  private connect() {
    this.connected ??= (async () => {
      const transport = new HttpTransport({
        url: this.cfg.url,
        token: this.cfg.bearer || undefined,
      });
      const files = new Files(transport, { overrides: this.cfg.caps });
      const jobs = new Jobs(transport, { root: this.cfg.jobsRoot });
      try {
        await files.capabilities();
        await transport.offers();
      } catch (e) {
        return toError(e);
      }
      return { files, jobs, transport };
    })();
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
          ok: { driver: "beyond-sandbox-ts", protocol: 1, ops: [...OPS] },
        };
      }
      const conn = await this.connect();
      if ("kind" in conn) throw new Reject(conn);
      return { id, ok: await this.dispatch(op, req, conn) };
    } catch (e) {
      return { id, error: e instanceof Reject ? e.err : toError(e) };
    }
  }

  private async dispatch(
    op: string,
    req: Json,
    { files, jobs, transport }: {
      files: Files;
      jobs: Jobs;
      transport: HttpTransport;
    },
  ): Promise<Json> {
    switch (op) {
      case "capabilities": {
        const c = await files.capabilities();
        const o = await transport.offers();
        return {
          stdin: c.stdin,
          stream: o.stream,
          output_base64: o.outputBase64,
          // Raw exec returns the command's bytes: over `output_base64`, else over a v1.2 stream.
          exact_exec: o.outputBase64 || o.stream,
          jobs: await jobs.capabilities(),
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
        const r = await transport.run(
          str(req, "command"),
          strings(req, "args"),
          {
            timeoutMs: optCount(req, "timeout_ms") ?? 120_000,
            ...(cwd !== undefined ? { cwd } : {}),
            ...(stdin !== undefined ? { stdin } : {}),
          },
        );
        return {
          exit_code: r.exitCode,
          stdout_b64: b64(r.stdout),
          stderr_b64: b64(r.stderr),
          exact: r.exact ?? true,
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
          const f of transport.stream(command, args, {
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
        const pgid = await jobs.start(str(req, "scope"), str(req, "job"), {
          command: str(req, "command"),
          args: strings(req, "args"),
          ...(cwd !== undefined ? { cwd } : {}),
          ...(label !== undefined ? { label } : {}),
        });
        return { pgid };
      }
      case "job_list":
        return { jobs: (await jobs.list(str(req, "scope"))).map(jobJson) };
      case "job_read": {
        const c = await jobs.read(str(req, "scope"), str(req, "job"), {
          offset: optCount(req, "offset") ?? 0,
          max: optCount(req, "max") ?? 128 * 1024,
        });
        return {
          state: c.state,
          code: c.code,
          offset: c.offset,
          end: c.end,
          data_b64: b64(c.data),
        };
      }
      case "job_attach": {
        const a = await jobs.attach(
          str(req, "scope"),
          str(req, "job"),
          optCount(req, "offset") ?? 0,
        );
        return { exit_code: a.exitCode, output_b64: b64(a.output), gap: a.gap };
      }
      case "job_kill": {
        const s = await jobs.kill(str(req, "scope"), strings(req, "jobs"), {
          remove: flag(req, "remove"),
        });
        return {
          survivors: s.map((x: Survivor) => ({ id: x.id, detail: x.detail })),
        };
      }
      case "job_remove":
        await jobs.remove(str(req, "scope"), strings(req, "jobs"));
        return {};
      case "job_wipe":
        return {
          survivors: (await jobs.wipe(str(req, "scope"))).map((x) => x.detail),
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

function toError(e: unknown): DriverError {
  if (e instanceof SandboxError) return { kind: e.kind, detail: e.detail };
  return { kind: "io", detail: e instanceof Error ? e.message : String(e) };
}

export function capsFromEnv(raw: string | undefined): CapabilityOverrides {
  if (!raw) return {};
  const v = JSON.parse(raw) as Json;
  const o: CapabilityOverrides = {};
  if (typeof v.stdin === "boolean") o.stdin = v.stdin;
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
