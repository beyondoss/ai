import { concat, fromBase64, text, utf8 } from "./bytes.js";
import {
  InvalidRequestError,
  NotOfferedError,
  ProtocolError,
  SandboxError,
} from "./errors.js";
import { describeSurvivors, isJobId } from "./http.js";
import { JOB_SCRIPT, SUPERVISOR } from "./scripts.js";
import type {
  JobChunk,
  JobFrame,
  JobInfo,
  JobSpec,
  JobState,
  Survivor,
  Transport,
} from "./transport.js";

/** Where the baseline keeps jobs in the sandbox. */
export const DEFAULT_JOBS_ROOT = "/tmp/beyond-jobs";
/** At least this much of a job's output is kept (the baseline's cap). */
export const OUTPUT_CAP = 8 * 1024 * 1024;
/** The most output one `read` returns by default. */
export const READ_MAX = 128 * 1024;
const OP_TIMEOUT_MS = 30_000;
const KILL_GRACE_MS = 300;
const POLL_MIN_MS = 250;
const POLL_MAX_MS = 2000;

export type JobsMode = "native" | "baseline" | "none";

export interface JobsOptions {
  /** The baseline's root directory in the sandbox; default `/tmp/beyond-jobs`. */
  root?: string | undefined;
}

export interface JobReadOptions {
  offset?: number;
  max?: number;
}

export interface KillOptions {
  /** Also delete each job whose group is empty afterwards. */
  remove?: boolean;
}

/** What following a job to its end collected. */
export interface JobOutput {
  /** The job's status: its exit code, or 128+N for signal N. */
  exitCode: number;
  output: Uint8Array;
  /** Bytes of output dropped before they could be read. */
  gap: number;
}

function validScope(scope: string): void {
  if (!/^[0-9a-z-]{1,64}$/.test(scope)) {
    throw new InvalidRequestError(
      "a job scope is 1 to 64 characters of [0-9a-z-]",
    );
  }
}

function validId(id: string): void {
  if (typeof id !== "string" || !isJobId(id)) {
    throw new InvalidRequestError(
      `bad job id ${JSON.stringify(id)}: 1 to 32 characters of [0-9a-z]`,
    );
  }
}

function count(name: string, v: number | undefined, dflt: number): number {
  if (v === undefined) return dflt;
  if (!Number.isInteger(v) || v < 0) {
    throw new InvalidRequestError(`\`${name}\` must be a non-negative integer`);
  }
  return v;
}

const oneLine = (label: string): string =>
  Array.from(label, (c) => (/\p{Cc}/u.test(c) ? " " : c)).join("");

/**
 * Layer 2: jobs over ANY {@link Transport}. A job is a command the sandbox keeps: it runs until it
 * exits, is killed or its scope is wiped, and any client can list it, read its output from an
 * offset and kill it. Native (v1.3) when the transport offers it; otherwise the baseline, built
 * from plain `run` calls (`sh`, `setsid`, `mkfifo`, `dd`, `base64`) so it works over every
 * transport and every endpoint.
 */
export class Jobs {
  readonly transport: Transport;
  private readonly root: string;
  private mode: Promise<JobsMode> | undefined;

  constructor(transport: Transport, options: JobsOptions = {}) {
    this.transport = transport;
    this.root = options.root ?? DEFAULT_JOBS_ROOT;
  }

  /** `native`, `baseline`, or `none` (the sandbox lacks a tool the baseline needs). Probed once. */
  capabilities(): Promise<JobsMode> {
    if (!this.mode) {
      const p = this.detect();
      this.mode = p;
      p.catch(() => {
        if (this.mode === p) this.mode = undefined;
      });
    }
    return this.mode;
  }

  private async detect(): Promise<JobsMode> {
    const offers = await this.transport.offers?.();
    if (offers?.nativeJobs && this.transport.nativeJobs) return "native";
    const r = await this.transport.run("sh", this.argv("probe", "probe", []), {
      timeoutMs: OP_TIMEOUT_MS,
      stdin: utf8("stdin-probe\n"),
    });
    const first = text(r.stdout).split("\n")[0]?.trim();
    return r.exitCode === 0 && first === "ok" ? "baseline" : "none";
  }

  private async ready(): Promise<{ mode: "native" | "baseline" }> {
    const mode = await this.capabilities();
    if (mode === "none") {
      throw new NotOfferedError(
        "this sandbox cannot run jobs of its own: the endpoint offers none and a tool the baseline needs is missing",
      );
    }
    return { mode };
  }

  // ---- operations ----------------------------------------------------------------------------

  /** Start a job the sandbox keeps from the start; returns its process group. */
  async start(scope: string, id: string, spec: JobSpec): Promise<number> {
    validScope(scope);
    validId(id);
    const { mode } = await this.ready();
    if (mode === "native") {
      return this.transport.nativeJobs!.start(scope, id, spec);
    }
    const out = await this.op(scope, "start", [
      id,
      "0",
      String(OUTPUT_CAP),
      spec.cwd ?? "",
      oneLine(spec.label ?? ""),
      SUPERVISOR,
      spec.command,
      ...(spec.args ?? []),
    ]);
    const pgid = Number(out.trim());
    if (!/^\d+$/.test(out.trim())) {
      throw new ProtocolError(
        `the sandbox did not report the job's group: ${JSON.stringify(out)}`,
      );
    }
    return pgid;
  }

  /** Every job in the scope. */
  async list(scope: string): Promise<JobInfo[]> {
    validScope(scope);
    const { mode } = await this.ready();
    if (mode === "native") return this.transport.nativeJobs!.list(scope);
    const out = await this.op(scope, "list", []);
    return out.split("\n").flatMap((l) => parseListLine(l) ?? []);
  }

  /** Up to `max` bytes of output from `offset`, and the job's state. An unknown job is `gone`. */
  async read(
    scope: string,
    id: string,
    options: JobReadOptions = {},
  ): Promise<JobChunk> {
    validScope(scope);
    validId(id);
    const offset = count("offset", options.offset, 0);
    const max = count("max", options.max, READ_MAX);
    const { mode } = await this.ready();
    if (mode === "native") {
      return this.transport.nativeJobs!.read(scope, id, offset, max);
    }
    return parseRead(
      await this.op(scope, "read", [id, String(offset), String(max)]),
    );
  }

  /**
   * Follow a job's output from `offset` to its end: frames in order, the last one `exit`. Natively
   * the endpoint streams it; on the baseline this polls `read` (every 250 ms, backing off to 2 s
   * while the job is quiet). Stopping early detaches and never kills the job.
   */
  async *follow(
    scope: string,
    id: string,
    offset = 0,
  ): AsyncGenerator<JobFrame> {
    validScope(scope);
    validId(id);
    count("offset", offset, 0);
    const { mode } = await this.ready();
    if (mode === "native") {
      yield* this.transport.nativeJobs!.attach(scope, id, offset);
      return;
    }
    let at = offset;
    let wait = POLL_MIN_MS;
    let shellTold = false;
    for (;;) {
      const c = await this.read(scope, id, { offset: at });
      if (c.state === "gone") {
        throw new SandboxError("io", "the job is no longer in the sandbox");
      }
      if (c.offset > at) {
        yield { type: "gap", bytes: c.offset - at };
        at = c.offset;
      }
      if (c.data.length > 0) {
        yield { type: "output", data: c.data };
        at += c.data.length;
        wait = POLL_MIN_MS;
        if (at < c.end) continue;
      }
      if (c.state === "exited" && !shellTold) {
        shellTold = true;
        yield { type: "shell_exit", code: c.code ?? 0 };
      } else if (c.state === "ended" && (at >= c.end || c.data.length === 0)) {
        yield { type: "exit", code: c.code ?? 0 };
        return;
      } else if (c.state === "ended") {
        continue;
      } else if (c.state === "lost") {
        throw new SandboxError(
          "io",
          "the job's supervisor in the sandbox is gone, and how it ended was not recorded",
        );
      }
      await new Promise((r) => setTimeout(r, wait));
      wait = Math.min(wait * 2, POLL_MAX_MS);
    }
  }

  /** {@link Jobs.follow} to the end, collected. */
  async attach(scope: string, id: string, offset = 0): Promise<JobOutput> {
    const chunks: Uint8Array[] = [];
    let gap = 0;
    for await (const f of this.follow(scope, id, offset)) {
      if (f.type === "output") chunks.push(f.data);
      else if (f.type === "gap") gap += f.bytes;
      else if (f.type === "exit") {
        return { exitCode: f.code, output: concat(chunks), gap };
      }
    }
    throw new ProtocolError(
      "the job stream ended before the job's final frame",
    );
  }

  /**
   * Stop jobs: SIGTERM to each group, a grace, SIGKILL. A no-op for an ended or unknown job.
   * Returns the processes that survived, per job.
   */
  async kill(
    scope: string,
    ids: string[],
    options: KillOptions = {},
  ): Promise<Survivor[]> {
    validScope(scope);
    ids.forEach(validId);
    const remove = options.remove ?? false;
    const { mode } = await this.ready();
    if (ids.length === 0) return [];
    if (mode === "native") {
      return this.transport.nativeJobs!.kill(scope, ids, remove);
    }
    const out = await this.op(scope, "kill", [
      String(KILL_GRACE_MS),
      remove ? "1" : "0",
      ...ids,
    ]);
    return out.split("\n").flatMap((l): Survivor[] => {
      const f = l.split(" ");
      if (f[0] !== "survived" || f[1] === undefined || !isJobId(f[1])) {
        return [];
      }
      return [{
        id: f[1],
        detail: describeSurvivors((f[2] ?? "").split(",").filter(Boolean)),
      }];
    });
  }

  /** Delete jobs whose process group is empty; a running job is left alone. */
  async remove(scope: string, ids: string[]): Promise<void> {
    validScope(scope);
    ids.forEach(validId);
    const { mode } = await this.ready();
    if (ids.length === 0) return;
    if (mode === "native") return this.transport.nativeJobs!.remove(scope, ids);
    await this.op(scope, "rm", ids);
  }

  /** Kill every job in the scope and delete it. Returns what survived. */
  async wipe(scope: string): Promise<Survivor[]> {
    validScope(scope);
    const { mode } = await this.ready();
    if (mode === "native") return this.transport.nativeJobs!.wipe(scope);
    const out = await this.op(scope, "wipe", [String(KILL_GRACE_MS)]);
    return out.split("\n").flatMap((l): Survivor[] => {
      const rest = l.startsWith("survived - ")
        ? l.slice("survived - ".length)
        : null;
      return rest === null
        ? []
        : [{ detail: describeSurvivors(rest.split(",").filter(Boolean)) }];
    });
  }

  // ---- the baseline --------------------------------------------------------------------------

  private argv(op: string, scope: string, args: string[]): string[] {
    return ["-c", JOB_SCRIPT, "sh", op, this.root, scope, ...args];
  }

  /** One baseline operation: its stdout when it succeeds. */
  private async op(scope: string, op: string, args: string[]): Promise<string> {
    const r = await this.transport.run("sh", this.argv(op, scope, args), {
      timeoutMs: OP_TIMEOUT_MS,
    });
    if (r.exitCode !== 0) {
      throw new SandboxError(
        "io",
        `sandbox job \`${op}\` failed (exit ${r.exitCode}): ${
          text(r.stderr).trim()
        }`,
      );
    }
    return text(r.stdout);
  }
}

function jobState(
  state: string,
  code: string,
): { state: JobState; code: number | null } | null {
  const c = /^-?\d+$/.test(code.trim()) ? Number(code.trim()) : null;
  switch (state) {
    case "running":
    case "lost":
    case "gone":
      return { state, code: null };
    case "exited":
    case "ended":
      return c === null ? null : { state, code: c };
    default:
      return null;
  }
}

/** `read`'s answer: `<state> <code> <offset> <end>`, then the base64 of the bytes. */
export function parseRead(out: string): JobChunk {
  const bad = () =>
    new ProtocolError(
      `unreadable job read: ${JSON.stringify(out.slice(0, 200))}`,
    );
  const nl = out.indexOf("\n");
  const head = (nl < 0 ? out : out.slice(0, nl)).trim();
  if (head === "gone") {
    return {
      state: "gone",
      code: null,
      offset: 0,
      end: 0,
      data: new Uint8Array(0),
    };
  }
  const f = head.split(" ");
  if (f.length !== 4) throw bad();
  const st = jobState(f[0]!, f[1]!);
  if (!st || !/^\d+$/.test(f[2]!) || !/^\d+$/.test(f[3]!)) throw bad();
  const data = fromBase64(nl < 0 ? "" : out.slice(nl + 1).trim());
  if (!data) throw bad();
  return { ...st, offset: Number(f[2]), end: Number(f[3]), data };
}

/** One `list` line: `<id> <state> <code> <num|-> <kind|-> <seen> <started> <ended> <label...>`. */
export function parseListLine(line: string): JobInfo | null {
  const f: string[] = [];
  let rest = line;
  for (let i = 0; i < 8; i++) {
    const sp = rest.indexOf(" ");
    if (sp < 0) {
      f.push(rest);
      rest = "";
      break;
    }
    f.push(rest.slice(0, sp));
    rest = rest.slice(sp + 1);
  }
  if (f.length < 8) return null;
  const [id, state, code, num, kind, seen, started, ended] = f as [
    string,
    string,
    string,
    string,
    string,
    string,
    string,
    string,
  ];
  if (!isJobId(id)) return null;
  const st = jobState(state, code);
  if (!st) return null;
  const n = (s: string) => (/^\d+$/.test(s) ? Number(s) : 0);
  return {
    id,
    state: st.state,
    code: st.code,
    kept: kind !== "-" || num !== "-",
    num: /^\d+$/.test(num) ? Number(num) : null,
    kind: kind !== "-" ? kind : null,
    seen: n(seen),
    started: n(started),
    ended: n(ended),
    label: rest,
  };
}
