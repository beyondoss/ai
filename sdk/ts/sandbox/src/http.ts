import { concat, fromBase64, text, toBase64, utf8 } from "./bytes.js";
import {
  AuthError,
  EndpointError,
  NotOfferedError,
  ProtocolError,
  SandboxError,
  TimeoutError,
  TooLargeError,
  TransportError,
} from "./errors.js";
import type {
  Frame,
  JobChunk,
  JobFrame,
  JobInfo,
  JobSpec,
  JobState,
  NativeJobs,
  Offers,
  RunOptions,
  RunResult,
  Survivor,
  Transport,
} from "./transport.js";

/** The response cap: a larger one-shot answer fails with `too_large` (the command ran). */
export const DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024;

const DEFAULT_TIMEOUT_MS = 120_000;
/** The client's deadline is the command's `timeout_ms` plus this, so the endpoint's verdict wins. */
const DEADLINE_GRACE_MS = 30_000;
/** A stream that goes this long without any frame (a heartbeat is a frame) is presumed dead. */
const FRAME_SILENCE_MS = 30_000;
/** The most a frame line may hold before a newline. */
const MAX_LINE = 1024 * 1024;
/** What a streamed command sends as `timeout_ms`: the connection is its lifetime. */
const STREAM_DEADLINE_MS = 2_147_483_647;
/** The most a timer can hold. */
const MAX_TIMER_MS = 2_147_483_647;
const JOB_OP_TIMEOUT_MS = 30_000;
const KILL_GRACE_MS = 300;
const PROBE_TIMEOUT_MS = 10_000;
/** The v1.2 probe: output, a pause, more output. */
const STREAM_PROBE_SCRIPT =
  "printf a; sleep 0.3 2>/dev/null || sleep 1; printf b";
const STREAM_PROBE_MIN_GAP_MS = 100;

export interface HttpTransportOptions {
  /** The exec endpoint's URL. */
  url: string;
  /**
   * The bearer token, or a function that returns one. A function is called when the transport
   * needs a token (cached after) and once more, to refresh, when the endpoint answers 401: that
   * request is retried once. That is the only retry this transport ever makes.
   */
  token?: string | (() => Promise<string>) | undefined;
  /** Extra headers sent with every request. */
  headers?: Record<string, string> | undefined;
  /** Response cap for one-shot answers; default 16 MiB. */
  maxResponseBytes?: number | undefined;
  /** A `fetch` to use instead of the global one. */
  fetch?: typeof fetch | undefined;
  /**
   * How long past a command's own `timeoutMs` the request may take before this client gives up
   * with `timeout`; default 30 s, so the endpoint's own verdict normally wins.
   */
  deadlineGraceMs?: number | undefined;
}

interface Exchange {
  res: Response;
  /** Release the exchange: stop the deadline and, unless the body is done, abandon it. */
  close(): void;
  timedOut(): boolean;
}

interface Body {
  command: string;
  args: string[];
  cwd?: string | undefined;
  timeoutMs: number;
  stdin?: Uint8Array | undefined;
  stream?: boolean | undefined;
  job?: Record<string, unknown> | undefined;
}

/** The default transport: the exec protocol (v1.1 to v1.3) over HTTP `POST`. */
export class HttpTransport implements Transport {
  readonly url: string;
  readonly nativeJobs: NativeJobs;
  private readonly tokenSource: string | (() => Promise<string>) | undefined;
  private tokenValue: string | undefined;
  private readonly headers: Record<string, string>;
  private readonly cap: number;
  private readonly grace: number;
  private readonly fetchFn: typeof fetch;
  private offered: Promise<Offers> | undefined;

  constructor(options: HttpTransportOptions) {
    this.url = options.url;
    this.tokenSource = options.token;
    this.tokenValue = typeof options.token === "string"
      ? options.token
      : undefined;
    this.headers = { ...options.headers };
    this.cap = options.maxResponseBytes ?? DEFAULT_MAX_RESPONSE_BYTES;
    this.grace = options.deadlineGraceMs ?? DEADLINE_GRACE_MS;
    this.fetchFn = options.fetch ?? ((...a) => globalThis.fetch(...a));
    this.nativeJobs = new HttpNativeJobs(this);
  }

  // ---- layer 1: one-shot ---------------------------------------------------------------------

  async run(
    command: string,
    args: string[],
    options: RunOptions = {},
  ): Promise<RunResult> {
    const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const ex = await this.post({ command, args, ...options, timeoutMs });
    try {
      return this.envelope(await this.readEnvelope(ex));
    } finally {
      ex.close();
    }
  }

  // ---- layer 1: stream (v1.2) ----------------------------------------------------------------

  async *stream(
    command: string,
    args: string[],
    options: RunOptions = {},
  ): AsyncGenerator<Frame> {
    const offers = await this.offers();
    if (!offers.stream) {
      throw new NotOfferedError(
        `exec endpoint ${this.url} did not prove it streams (protocol v1.1); the command was not run`,
      );
    }
    const timeoutMs = options.timeoutMs ?? STREAM_DEADLINE_MS;
    const ex = await this.post({
      command,
      args,
      ...options,
      timeoutMs,
      stream: true,
    });
    try {
      if (!isNdjson(ex.res)) {
        // A fleet mid-rollout answered with one object after all.
        const r = this.envelope(await this.readEnvelope(ex));
        if (r.stdout.length) yield { type: "stdout", data: r.stdout };
        if (r.stderr.length) yield { type: "stderr", data: r.stderr };
        yield { type: "exit", code: r.exitCode };
        return;
      }
      for await (const line of this.lines(ex)) {
        for (const f of decodeFrameLine(line, this.url)) {
          if (f === "heartbeat") continue;
          if (f.type === "error") {
            throw new TransportError(
              `exec endpoint ${this.url}: the endpoint failed mid-command: ${f.message} (the command may have been killed)`,
            );
          }
          yield f;
          if (
            f.type === "exit" || f.type === "signal" || f.type === "timed_out"
          ) return;
        }
      }
      throw new TransportError(
        `exec endpoint ${this.url}: the stream ended before the command's final frame (the command may have been killed)`,
      );
    } finally {
      ex.close();
    }
  }

  // ---- probes --------------------------------------------------------------------------------

  /** What the endpoint offers (v1.2 streaming, v1.3 jobs), probed once. A failed probe is retried. */
  offers(): Promise<Offers> {
    if (!this.offered) {
      const p = (async () => ({
        stream: await this.probeStream(),
        nativeJobs: await this.probeJobs(),
      }))();
      this.offered = p;
      p.catch(() => {
        if (this.offered === p) this.offered = undefined;
      });
    }
    return this.offered;
  }

  private async probeStream(): Promise<boolean> {
    let ex: Exchange;
    try {
      ex = await this.post({
        command: "sh",
        args: ["-c", STREAM_PROBE_SCRIPT],
        timeoutMs: PROBE_TIMEOUT_MS,
        stream: true,
      });
    } catch (e) {
      // A front that refuses `stream` has answered: it does not stream.
      if (e instanceof EndpointError && e.status === 400) return false;
      throw e;
    }
    try {
      if (!isNdjson(ex.res)) return false;
      let frames = 0;
      let out = "";
      let firstAt: number | undefined;
      let gap: number | undefined;
      let end: Frame | undefined;
      for await (const line of this.lines(ex)) {
        for (const f of decodeFrameLine(line, this.url)) {
          if (f === "heartbeat") continue;
          if (f.type === "error") {
            throw new TransportError(`exec endpoint ${this.url}: ${f.message}`);
          }
          if (f.type === "stdout" || f.type === "stderr") {
            frames++;
            firstAt ??= performance.now();
            if (f.type === "stdout") out += text(f.data);
          } else if (
            f.type === "exit" || f.type === "signal" || f.type === "timed_out"
          ) {
            end = f;
            if (firstAt !== undefined) gap = performance.now() - firstAt;
          }
        }
        if (end) break;
      }
      return (
        frames >= 2
        && out === "ab"
        && end?.type === "exit"
        && end.code === 0
        && gap !== undefined
        && gap >= STREAM_PROBE_MIN_GAP_MS
      );
    } finally {
      ex.close();
    }
  }

  private async probeJobs(): Promise<boolean> {
    let ex: Exchange;
    try {
      ex = await this.post({
        command: "sh",
        args: ["-c", "printf exec-v1.2"],
        timeoutMs: PROBE_TIMEOUT_MS,
        job: { op: "probe" },
      });
    } catch (e) {
      if (e instanceof EndpointError && e.status === 400) return false;
      throw e;
    }
    try {
      const { body, over } = await this.readCapped(ex);
      if (over) return false;
      try {
        const v: unknown = JSON.parse(text(body));
        return isRecord(v) && v.jobs === "v1.3";
      } catch {
        return false;
      }
    } finally {
      ex.close();
    }
  }

  // ---- the wire ------------------------------------------------------------------------------

  /** Authorization plus configured headers. */
  private async requestHeaders(
    refresh: boolean,
  ): Promise<Record<string, string>> {
    const h: Record<string, string> = {
      "content-type": "application/json",
      ...this.headers,
    };
    if (
      typeof this.tokenSource === "function"
      && (refresh || this.tokenValue === undefined)
    ) {
      this.tokenValue = await this.tokenSource();
    }
    if (this.tokenValue) h.authorization = `Bearer ${this.tokenValue}`;
    return h;
  }

  /** POST one request. Resolves once the status is known to be a success. */
  async post(body: Body): Promise<Exchange> {
    const payload: Record<string, unknown> = {
      command: body.command,
      args: body.args,
      timeout_ms: body.timeoutMs,
    };
    if (body.cwd !== undefined) payload.cwd = body.cwd;
    if (body.stdin !== undefined) payload.stdin_base64 = toBase64(body.stdin);
    if (body.stream) payload.stream = true;
    if (body.job !== undefined) payload.job = body.job;
    const json = JSON.stringify(payload);
    const deadline = Math.min(body.timeoutMs + this.grace, MAX_TIMER_MS);

    let refreshed = false;
    for (;;) {
      const ex = await this.send(json, deadline, refreshed);
      if (ex.res.ok) return ex;
      const status = ex.res.status;
      let detail: string;
      let over = false;
      try {
        const r = await this.readCapped(ex);
        detail = excerpt(r.body);
        over = r.over;
      } finally {
        ex.close();
      }
      if (
        status === 401 && typeof this.tokenSource === "function" && !refreshed
      ) {
        refreshed = true;
        continue;
      }
      void over;
      const msg = `exec endpoint ${this.url} returned ${status}: ${detail}`;
      if (status === 401 || status === 403) throw new AuthError(msg);
      if (status === 422 && detail.includes("cap")) {
        throw new TooLargeError(msg);
      }
      throw new EndpointError(msg, status);
    }
  }

  private async send(
    json: string,
    deadlineMs: number,
    refresh: boolean,
  ): Promise<Exchange> {
    const headers = await this.requestHeaders(refresh);
    const ctl = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      ctl.abort();
    }, deadlineMs);
    unref(timer);
    let res: Response;
    try {
      res = await this.fetchFn(this.url, {
        method: "POST",
        headers,
        body: json,
        signal: ctl.signal,
      });
    } catch (e) {
      clearTimeout(timer);
      throw timedOut
        ? new TimeoutError(`exec endpoint ${this.url} timed out`, { cause: e })
        : new TransportError(
          `exec endpoint ${this.url} unreachable: ${errorText(e)}`,
          {
            cause: e,
          },
        );
    }
    return {
      res,
      timedOut: () => timedOut,
      close: () => {
        clearTimeout(timer);
        if (!ctl.signal.aborted) ctl.abort();
        res.body?.cancel().catch(() => {});
      },
    };
  }

  /** Read the body incrementally and stop at the cap. `over` means the bytes are only a prefix. */
  async readCapped(ex: Exchange): Promise<{ body: Uint8Array; over: boolean }> {
    const declared = Number(ex.res.headers.get("content-length") ?? 0);
    if (declared > this.cap) return { body: new Uint8Array(0), over: true };
    const reader = ex.res.body?.getReader();
    if (!reader) return { body: new Uint8Array(0), over: false };
    const chunks: Uint8Array[] = [];
    let n = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value.length > this.cap - n) {
          await reader.cancel().catch(() => {});
          return { body: concat(chunks), over: true };
        }
        chunks.push(value);
        n += value.length;
      }
    } catch (e) {
      throw this.bodyError(ex, e);
    }
    return { body: concat(chunks), over: false };
  }

  private bodyError(ex: Exchange, e: unknown): SandboxError {
    if (e instanceof SandboxError) return e;
    return ex.timedOut()
      ? new TimeoutError(`exec endpoint ${this.url} timed out`, { cause: e })
      : new TransportError(`exec endpoint ${this.url}: ${errorText(e)}`, {
        cause: e,
      });
  }

  /** Read and parse a v1.1 body: one JSON object, capped. */
  async readEnvelope(ex: Exchange): Promise<Record<string, unknown>> {
    const { body, over } = await this.readCapped(ex);
    if (over) {
      throw new TooLargeError(
        `exec endpoint ${this.url}: the response exceeded the ${this.cap}-byte cap and was discarded; the command ran, but its output is too large to return. Narrow it (for example pipe through \`head\`) rather than re-running it unchanged`,
      );
    }
    return this.parseObject(body);
  }

  parseObject(body: Uint8Array): Record<string, unknown> {
    let v: unknown;
    try {
      v = JSON.parse(text(body));
    } catch (e) {
      throw new ProtocolError(
        `exec endpoint ${this.url} returned unparseable JSON (${
          errorText(e)
        }): ${excerpt(body)}`,
      );
    }
    if (!isRecord(v)) {
      throw new ProtocolError(
        `exec endpoint ${this.url} returned unparseable JSON (not an object): ${
          excerpt(body)
        }`,
      );
    }
    return v;
  }

  private envelope(v: Record<string, unknown>): RunResult {
    const code = v.exit_code;
    if (typeof code !== "number" || !Number.isInteger(code)) {
      throw new ProtocolError(
        `exec endpoint ${this.url} returned unparseable JSON (no integer exit_code)`,
      );
    }
    const str = (x: unknown): string => (typeof x === "string" ? x : "");
    return {
      exitCode: code,
      stdout: utf8(str(v.stdout)),
      stderr: utf8(str(v.stderr)),
    };
  }

  /** The lines of a streamed body, each without its newline. Presumes a dead connection after 30 s of silence. */
  async *lines(ex: Exchange): AsyncGenerator<Uint8Array> {
    const reader = ex.res.body?.getReader();
    if (!reader) return;
    let buf: Uint8Array = new Uint8Array(0);
    let scanned = 0;
    try {
      for (;;) {
        const nl = buf.indexOf(0x0a, scanned);
        if (nl >= 0) {
          const line = buf.slice(0, nl);
          buf = buf.slice(nl + 1);
          scanned = 0;
          yield line;
          continue;
        }
        scanned = buf.length;
        if (buf.length > MAX_LINE) {
          throw new TransportError(
            `exec endpoint ${this.url}: a frame exceeded ${MAX_LINE} bytes without a newline`,
          );
        }
        let silence: ReturnType<typeof setTimeout> | undefined;
        let next: ReadableStreamReadResult<Uint8Array>;
        try {
          next = await Promise.race([
            reader.read(),
            new Promise<never>((_, reject) => {
              silence = setTimeout(
                () =>
                  reject(
                    new TransportError(
                      `exec endpoint ${this.url}: no frame for ${
                        FRAME_SILENCE_MS / 1000
                      } s; the endpoint stopped sending heartbeats, so the connection is presumed dead`,
                    ),
                  ),
                FRAME_SILENCE_MS,
              );
              unref(silence);
            }),
          ]);
        } catch (e) {
          throw this.bodyError(ex, e);
        } finally {
          clearTimeout(silence);
        }
        if (next.done) {
          if (buf.length > 0) {
            throw new TransportError(
              `exec endpoint ${this.url}: the stream ended in the middle of a frame`,
            );
          }
          return;
        }
        buf = buf.length === 0
          ? new Uint8Array(next.value)
          : concat([buf, next.value]);
      }
    } finally {
      reader.cancel().catch(() => {});
    }
  }

  /** One job operation that is not a stream: its JSON answer. */
  async jobCall(
    job: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const ex = await this.post({
      command: "true",
      args: [],
      timeoutMs: JOB_OP_TIMEOUT_MS,
      job,
    });
    try {
      const { body, over } = await this.readCapped(ex);
      if (over) {
        throw new TooLargeError("a job answer exceeded the response cap");
      }
      return this.parseObject(body);
    } finally {
      ex.close();
    }
  }
}

/** Let the process exit while a deadline timer is pending (Node and Bun; a no-op elsewhere). */
function unref(timer: unknown): void {
  (timer as { unref?: () => void }).unref?.();
}

function isNdjson(res: Response): boolean {
  const ct = res.headers.get("content-type") ?? "";
  return (ct.split(";")[0] ?? "").trim().toLowerCase()
    === "application/x-ndjson";
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function errorText(e: unknown): string {
  if (e instanceof Error) {
    const cause = e.cause instanceof Error ? ` (${e.cause.message})` : "";
    return `${e.message}${cause}`;
  }
  return String(e);
}

function excerpt(body: Uint8Array): string {
  return Array.from(text(body).trim()).slice(0, 400).join("");
}

type Decoded = Frame | { type: "error"; message: string } | "heartbeat";

function b64Field(field: string, s: unknown, url: string): Uint8Array {
  const b = typeof s === "string" ? fromBase64(s) : null;
  if (!b) {
    throw new TransportError(
      `exec endpoint ${url}: \`${field}\` is not standard base64`,
    );
  }
  return b;
}

/**
 * One line of a v1.2 stream. A frame names exactly one field; one that names more is split rather
 * than truncated (output first, then the final field), and one with no known field is a heartbeat.
 */
export function decodeFrameLine(line: Uint8Array, url = "endpoint"): Decoded[] {
  let s = text(line);
  if (s.endsWith("\r")) s = s.slice(0, -1);
  if (s.trim() === "") return ["heartbeat"];
  let v: unknown;
  try {
    v = JSON.parse(s);
  } catch (e) {
    throw new TransportError(
      `exec endpoint ${url}: unparseable frame (${errorText(e)}): ${
        s.slice(0, 200)
      }`,
    );
  }
  if (!isRecord(v)) {
    throw new TransportError(
      `exec endpoint ${url}: unparseable frame: ${s.slice(0, 200)}`,
    );
  }
  const out: Decoded[] = [];
  if (v.stdout !== undefined && v.stdout !== null) {
    out.push({ type: "stdout", data: b64Field("stdout", v.stdout, url) });
  }
  if (v.stderr !== undefined && v.stderr !== null) {
    out.push({ type: "stderr", data: b64Field("stderr", v.stderr, url) });
  }
  if (typeof v.shell_exit === "number") {
    out.push({ type: "shell_exit", code: v.shell_exit });
  }
  if (typeof v.error === "string") {
    out.push({ type: "error", message: v.error });
  } else if (typeof v.exit_code === "number") {
    out.push({ type: "exit", code: v.exit_code });
  } else if (typeof v.signal === "number") {
    out.push({ type: "signal", signal: v.signal });
  } else if (v.timed_out === true) out.push({ type: "timed_out" });
  if (out.length === 0) out.push("heartbeat");
  return out;
}

// ---- native jobs (v1.3) ----------------------------------------------------------------------

function jobState(
  state: unknown,
  code: unknown,
): { state: JobState; code: number | null } | null {
  const c = typeof code === "number" ? code : null;
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

function badJob(what: string, v: unknown): ProtocolError {
  return new ProtocolError(
    `unreadable job ${what}: ${JSON.stringify(v)?.slice(0, 200)}`,
  );
}

function count(v: unknown): number {
  return typeof v === "number" && Number.isInteger(v) && v >= 0 ? v : 0;
}

/** Describe the live pids `kill` or `wipe` could not stop. */
export function describeSurvivors(pids: string[]): string {
  if (pids.length === 0) return "a process in the sandbox";
  if (pids.length === 1) return `process ${pids[0]} in the sandbox`;
  return `processes ${pids.join(", ")} in the sandbox`;
}

function parseSurvivors(v: Record<string, unknown>): Survivor[] {
  const list = Array.isArray(v.survivors) ? v.survivors : [];
  return list.map((s): Survivor => {
    const rec = isRecord(s) ? s : {};
    const pids = Array.isArray(rec.pids)
      ? rec.pids.filter((p): p is number => typeof p === "number").map(String)
      : [];
    const out: Survivor = { detail: describeSurvivors(pids) };
    if (typeof rec.id === "string") out.id = rec.id;
    return out;
  });
}

class HttpNativeJobs implements NativeJobs {
  private readonly t: HttpTransport;

  constructor(t: HttpTransport) {
    this.t = t;
  }

  private job(op: string, scope: string, extra: Record<string, unknown> = {}) {
    return { op, scope, ...extra };
  }

  async start(scope: string, id: string, spec: JobSpec): Promise<number> {
    const job = this.job("start", scope, {
      id,
      hold: false,
      label: spec.label ?? "",
    });
    const ex = await this.t.post({
      command: spec.command,
      args: spec.args ?? [],
      cwd: spec.cwd,
      timeoutMs: JOB_OP_TIMEOUT_MS,
      job,
    });
    try {
      const { body } = await this.t.readCapped(ex);
      let started: unknown;
      try {
        const v: unknown = JSON.parse(text(body));
        started = isRecord(v) ? v.started : undefined;
      } catch {
        started = undefined;
      }
      if (typeof started !== "number") {
        throw new ProtocolError(
          `the endpoint did not report the job's start: ${excerpt(body)}`,
        );
      }
      return started;
    } finally {
      ex.close();
    }
  }

  async list(scope: string): Promise<JobInfo[]> {
    const v = await this.t.jobCall(this.job("list", scope));
    if (!Array.isArray(v.jobs)) throw badJob("list", v);
    const out: JobInfo[] = [];
    for (const j of v.jobs) {
      if (!isRecord(j) || typeof j.id !== "string" || !isJobId(j.id)) continue;
      const st = jobState(j.state, j.code);
      if (!st) continue;
      out.push({
        id: j.id,
        state: st.state,
        code: st.code,
        kept: j.kept === true,
        num: typeof j.num === "number" ? j.num : null,
        kind: typeof j.kind === "string" ? j.kind : null,
        seen: count(j.seen),
        started: count(j.started),
        ended: count(j.ended),
        label: typeof j.label === "string" ? j.label : "",
      });
    }
    return out;
  }

  async read(
    scope: string,
    id: string,
    offset: number,
    max: number,
  ): Promise<JobChunk> {
    const v = await this.t.jobCall(
      this.job("read", scope, { id, offset, max }),
    );
    const st = jobState(v.state, v.code);
    const data = fromBase64(typeof v.data === "string" ? v.data : "");
    if (!st || !data) throw badJob("read", v);
    return { ...st, offset: count(v.offset), end: count(v.end), data };
  }

  async *attach(
    scope: string,
    id: string,
    offset: number,
  ): AsyncGenerator<JobFrame> {
    const ex = await this.t.post({
      command: "true",
      args: [],
      timeoutMs: STREAM_DEADLINE_MS,
      stream: true,
      job: this.job("attach", scope, { id, offset }),
    });
    try {
      for await (const line of this.t.lines(ex)) {
        const f = decodeJobFrame(line);
        if (f === "heartbeat") continue;
        yield f;
        if (f.type === "exit") return;
      }
      throw new TransportError(
        "job stream: the stream ended before the job's final frame",
      );
    } finally {
      ex.close();
    }
  }

  async kill(
    scope: string,
    ids: string[],
    remove: boolean,
  ): Promise<Survivor[]> {
    if (ids.length === 0) return [];
    const v = await this.t.jobCall(
      this.job("kill", scope, { ids, remove, grace_ms: KILL_GRACE_MS }),
    );
    return parseSurvivors(v);
  }

  async remove(scope: string, ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    await this.t.jobCall(this.job("remove", scope, { ids }));
  }

  async wipe(scope: string): Promise<Survivor[]> {
    return parseSurvivors(
      await this.t.jobCall(
        this.job("wipe", scope, { grace_ms: KILL_GRACE_MS }),
      ),
    );
  }
}

/** A job id is 1 to 32 characters of `[0-9a-z]`. */
export function isJobId(s: string): boolean {
  return /^[0-9a-z]{1,32}$/.test(s);
}

/** One line of a v1.3 job stream: v1.2's frames plus `started` and `gap`. */
export function decodeJobFrame(line: Uint8Array): JobFrame | "heartbeat" {
  let s = text(line);
  if (s.endsWith("\r")) s = s.slice(0, -1);
  if (s.trim() === "") return "heartbeat";
  let v: unknown;
  try {
    v = JSON.parse(s);
  } catch (e) {
    throw new TransportError(
      `job stream: unparseable job frame (${errorText(e)}): ${s.slice(0, 200)}`,
    );
  }
  if (!isRecord(v)) {
    throw new TransportError(
      `job stream: unparseable job frame: ${s.slice(0, 200)}`,
    );
  }
  if (typeof v.error === "string") {
    throw new TransportError(`job stream: ${v.error}`);
  }
  if (typeof v.started === "number") {
    return { type: "started", pgid: v.started };
  }
  const out = typeof v.stdout === "string"
    ? v.stdout
    : typeof v.stderr === "string"
    ? v.stderr
    : null;
  if (out !== null) {
    const data = fromBase64(out);
    if (!data) {
      throw new TransportError("job stream: output is not standard base64");
    }
    return { type: "output", data };
  }
  if (typeof v.gap === "number") return { type: "gap", bytes: v.gap };
  if (typeof v.shell_exit === "number") {
    return { type: "shell_exit", code: v.shell_exit };
  }
  if (typeof v.exit_code === "number") {
    return { type: "exit", code: v.exit_code };
  }
  if (typeof v.signal === "number") {
    return { type: "exit", code: 128 + v.signal };
  }
  if (v.timed_out === true) {
    throw new TransportError("job stream: the endpoint timed the job out");
  }
  return "heartbeat";
}
