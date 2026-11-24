import { concat, fromBase64, text, toBase64 } from "./bytes.js";
import {
  AuthError,
  DiskFullError,
  EndpointError,
  InvalidRequestError,
  JobExistsError,
  JobGoneError,
  NotOfferedError,
  ProtocolError,
  SandboxError,
  StdinBusyError,
  StdinClosedError,
  StdinGapError,
  TimeoutError,
  TooLargeError,
  TransportError,
} from "./errors.js";
import {
  type AttachOptions,
  DEFAULT_KILL_GRACE_MS,
  type Frame,
  HEARTBEAT_MAX_S,
  HEARTBEAT_MIN_S,
  isJobId,
  JOB_SIGNALS,
  type JobChunk,
  type JobFrame,
  type JobInfo,
  type JobOps,
  type JobSpec,
  type JobStarted,
  type JobState,
  type KeepFields,
  type RemoveOptions,
  type RemoveResult,
  type ReqOptions,
  type RunOptions,
  type RunResult,
  silenceMs,
  type StdinWritten,
  type Survivor,
  type Transport,
  WRITE_MAX,
} from "./transport.js";

/** The response cap: a larger one-shot answer fails with `too_large` (the command ran). */
export const DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024;

const DEFAULT_TIMEOUT_MS = 120_000;
/** The client's deadline is the command's `timeout_ms` plus this, so the endpoint's verdict wins. */
const DEADLINE_GRACE_MS = 30_000;
/** The most a frame line may hold before a newline. */
const MAX_LINE = 1024 * 1024;
/** What a streamed command sends as `timeout_ms`: the connection is its lifetime. */
const STREAM_DEADLINE_MS = 2_147_483_647;
/** The most a timer can hold. */
const MAX_TIMER_MS = 2_147_483_647;
const JOB_OP_TIMEOUT_MS = 30_000;
/** Room for the JSON around two base64 strings. */
const ENVELOPE_SLACK = 4096;

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
  /** The caller's {@link AbortSignal} fired (not the client's own deadline). */
  aborted(): boolean;
}

interface Body {
  command: string;
  args: string[];
  cwd?: string | undefined;
  timeoutMs: number;
  stdin?: Uint8Array | undefined;
  stream?: boolean | undefined;
  job?: Record<string, unknown> | undefined;
  env?: Record<string, string> | undefined;
  /** Sent as `Keepalive-Ping-Interval` (whole seconds, 1 to 240). */
  heartbeatSeconds?: number | undefined;
}

/** The default transport: the exec protocol over HTTP `POST`. */
export class HttpTransport implements Transport {
  readonly url: string;
  readonly jobs: JobOps;
  private readonly tokenSource: string | (() => Promise<string>) | undefined;
  private tokenValue: string | undefined;
  private readonly headers: Record<string, string>;
  private readonly cap: number;
  private readonly grace: number;
  private readonly fetchFn: typeof fetch;
  /** An `env` was proved applied by this endpoint. */
  private envProved = false;

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
    this.jobs = new HttpJobs(this);
  }

  // ---- layer 1: one-shot ---------------------------------------------------------------------

  async run(
    command: string,
    args: string[],
    options: RunOptions = {},
  ): Promise<RunResult> {
    const { signal, timeoutMs: t, ...rest } = options;
    const timeoutMs = t ?? DEFAULT_TIMEOUT_MS;
    await this.proveEnv(rest.env);
    const ex = await this.post({ command, args, ...rest, timeoutMs }, signal);
    try {
      return decodeOutput(
        this.parseObject(await this.readBounded(ex, bodyCap(this.cap))),
        this.url,
        this.cap,
      );
    } finally {
      ex.close();
    }
  }

  // ---- layer 1: stream -----------------------------------------------------------------------

  async *stream(
    command: string,
    args: string[],
    options: RunOptions = {},
  ): AsyncGenerator<Frame> {
    const { signal, timeoutMs: t, ...rest } = options;
    const timeoutMs = t ?? STREAM_DEADLINE_MS;
    await this.proveEnv(rest.env);
    const ex = await this.post({
      command,
      args,
      ...rest,
      timeoutMs,
      stream: true,
    }, signal);
    try {
      if (!isNdjson(ex.res)) {
        throw new ProtocolError(
          `exec endpoint ${this.url} answered a stream request with something that is not a stream`,
        );
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
      this.tokenValue = await this.mint(this.tokenSource);
    }
    if (this.tokenValue) h.authorization = `Bearer ${this.tokenValue}`;
    return h;
  }

  /** A token from the source. One that throws, or gives no token, cannot authorise the call: `auth`. */
  private async mint(source: () => Promise<string>): Promise<string> {
    let token: unknown;
    try {
      token = await source();
    } catch (e) {
      if (e instanceof SandboxError) throw e;
      throw new AuthError(`token source failed: ${errorText(e)}`, { cause: e });
    }
    if (typeof token !== "string" || token.trim() === "") {
      throw new AuthError("token source failed: it gave no token");
    }
    return token;
  }

  /**
   * An endpoint that predates `env` ignores it and says nothing, so before the first `env` sent to
   * one, prove it applies one (a command that prints a variable only `env` could have set) and fail
   * closed otherwise: running the command without its environment is a silent wrong answer.
   */
  async proveEnv(env: Record<string, string> | undefined): Promise<void> {
    if (!env || Object.keys(env).length === 0 || this.envProved) return;
    const ex = await this.post({
      command: "sh",
      args: ["-c", "printf %s \"$BEYOND_ENV_PROBE\""],
      timeoutMs: 10_000,
      env: { BEYOND_ENV_PROBE: "1" },
    });
    let r: RunResult;
    try {
      r = decodeOutput(
        this.parseObject(await this.readBounded(ex, bodyCap(this.cap))),
        this.url,
        this.cap,
      );
    } finally {
      ex.close();
    }
    if (r.exitCode !== 0 || text(r.stdout) !== "1") {
      throw new NotOfferedError(
        `exec endpoint ${this.url} predates per-command env: it ran the command without the environment it was given`,
      );
    }
    this.envProved = true;
  }

  /** POST one request. Resolves once the status is known to be a success. */
  async post(body: Body, signal?: AbortSignal): Promise<Exchange> {
    checkHeartbeat(body.heartbeatSeconds);
    const payload: Record<string, unknown> = {
      command: body.command,
      args: body.args,
      timeout_ms: body.timeoutMs,
    };
    if (body.cwd !== undefined) payload.cwd = body.cwd;
    if (body.stdin !== undefined) payload.stdin = toBase64(body.stdin);
    if (body.stream) payload.stream = true;
    if (body.job !== undefined) payload.job = body.job;
    if (body.env !== undefined && Object.keys(body.env).length > 0) {
      payload.env = body.env;
    }
    const json = JSON.stringify(payload);
    const deadline = Math.min(body.timeoutMs + this.grace, MAX_TIMER_MS);

    let refreshed = false;
    for (;;) {
      const ex = await this.send(
        json,
        deadline,
        refreshed,
        signal,
        body.heartbeatSeconds,
      );
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
      const msg = `exec endpoint ${this.url} returned ${status}: ${detail}`;
      if (status === 401 || status === 403) throw new AuthError(msg);
      // 422 + over-cap: the body may be empty (Content-Length over the cap) or truncated, so the
      // excerpt can lack the word "cap" — `over` is then the client's own verdict that the answer
      // was too large (the command ran).
      if (status === 422 && (over || detail.includes("cap"))) {
        throw new TooLargeError(msg);
      }
      throw new EndpointError(msg, status);
    }
  }

  private async send(
    json: string,
    deadlineMs: number,
    refresh: boolean,
    signal?: AbortSignal,
    heartbeatSeconds?: number,
  ): Promise<Exchange> {
    if (signal?.aborted) {
      throw new TransportError(`exec endpoint ${this.url}: aborted`);
    }
    const headers = await this.requestHeaders(refresh);
    if (heartbeatSeconds !== undefined) {
      headers["keepalive-ping-interval"] = String(heartbeatSeconds);
    }
    // Abort during token mint / header build never reaches the listener below: re-check.
    if (signal?.aborted) {
      throw new TransportError(`exec endpoint ${this.url}: aborted`);
    }
    const ctl = new AbortController();
    let timedOut = false;
    let callerAborted = false;
    let res: Response | undefined;
    const onCallerAbort = () => {
      callerAborted = true;
      ctl.abort();
      // After headers arrive, aborting fetch's signal does not stop the body: cancel it.
      res?.body?.cancel().catch(() => {});
    };
    signal?.addEventListener("abort", onCallerAbort, { once: true });
    const timer = setTimeout(() => {
      timedOut = true;
      ctl.abort();
    }, deadlineMs);
    unref(timer);
    const detach = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onCallerAbort);
    };
    try {
      res = await this.fetchFn(this.url, {
        method: "POST",
        headers,
        body: json,
        signal: ctl.signal,
      });
    } catch (e) {
      detach();
      if (timedOut) {
        throw new TimeoutError(`exec endpoint ${this.url} timed out`, {
          cause: e,
        });
      }
      if (callerAborted) {
        throw new TransportError(`exec endpoint ${this.url}: aborted`, {
          cause: e,
        });
      }
      throw new TransportError(
        `exec endpoint ${this.url} unreachable: ${errorText(e)}`,
        { cause: e },
      );
    }
    const response = res;
    return {
      res: response,
      timedOut: () => timedOut,
      aborted: () => callerAborted,
      close: () => {
        detach();
        if (!ctl.signal.aborted) ctl.abort();
        response.body?.cancel().catch(() => {});
      },
    };
  }

  /** Read the body incrementally and stop at the cap. `over` means the bytes are only a prefix. */
  async readCapped(
    ex: Exchange,
    cap: number = this.cap,
  ): Promise<{ body: Uint8Array; over: boolean }> {
    const declared = Number(ex.res.headers.get("content-length") ?? 0);
    if (declared > cap) return { body: new Uint8Array(0), over: true };
    const reader = ex.res.body?.getReader();
    if (!reader) return { body: new Uint8Array(0), over: false };
    const chunks: Uint8Array[] = [];
    let n = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value.length > cap - n) {
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
    if (ex.timedOut()) {
      return new TimeoutError(`exec endpoint ${this.url} timed out`, {
        cause: e,
      });
    }
    if (ex.aborted()) {
      return new TransportError(`exec endpoint ${this.url}: aborted`, {
        cause: e,
      });
    }
    return new TransportError(`exec endpoint ${this.url}: ${errorText(e)}`, {
      cause: e,
    });
  }

  /** The whole body, at most `bound` bytes of it; more is the cap's error (the command ran). */
  private async readBounded(ex: Exchange, bound: number): Promise<Uint8Array> {
    const { body, over } = await this.readCapped(ex, bound);
    if (over) throw tooLarge(this.url, this.cap);
    return body;
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

  /**
   * The lines of a streamed body, each without its newline. Presumes a dead connection after
   * {@link silenceMs} of silence: 30 s, or longer when the request chose a slower heartbeat.
   */
  async *lines(
    ex: Exchange,
    heartbeatSeconds?: number,
  ): AsyncGenerator<Uint8Array> {
    const silenceLimit = silenceMs(heartbeatSeconds);
    const reader = ex.res.body?.getReader();
    if (!reader) return;
    // `start` is the first unread byte: we copy each line once and only compact
    // the consumed prefix when appending the next network chunk (not on every NL).
    let buf: Uint8Array = new Uint8Array(0);
    let start = 0;
    let scanned = 0;
    try {
      for (;;) {
        const nl = buf.indexOf(0x0a, scanned);
        if (nl >= 0) {
          const line = buf.slice(start, nl);
          start = nl + 1;
          scanned = start;
          yield line;
          continue;
        }
        scanned = buf.length;
        if (buf.length - start > MAX_LINE) {
          throw new ProtocolError(
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
                        silenceLimit / 1000
                      } s; the endpoint stopped sending heartbeats, so the connection is presumed dead`,
                    ),
                  ),
                silenceLimit,
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
          if (start < buf.length) {
            throw new TransportError(
              `exec endpoint ${this.url}: the stream ended in the middle of a frame`,
            );
          }
          return;
        }
        if (start > 0) {
          const rem = buf.subarray(start);
          const merged = new Uint8Array(rem.length + next.value.length);
          merged.set(rem, 0);
          merged.set(next.value, rem.length);
          buf = merged;
          scanned = rem.length;
          start = 0;
        } else {
          buf = buf.length === 0
            ? new Uint8Array(next.value)
            : concat([buf, next.value]);
        }
      }
    } finally {
      reader.cancel().catch(() => {});
    }
  }

  /** One job operation that is not a stream: its JSON answer. */
  async jobCall(
    job: Record<string, unknown>,
    req: ReqOptions = {},
  ): Promise<Record<string, unknown>> {
    const ex = await this.post({
      command: "true",
      args: [],
      timeoutMs: JOB_OP_TIMEOUT_MS,
      job,
      heartbeatSeconds: req.heartbeatSeconds,
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

/**
 * Up to 400 code points of a body for error detail. Only a UTF-8 prefix is
 * decoded (enough for 400 4-byte characters), so a multi-MiB error body cannot
 * force a full-body decode + `Array.from`.
 */
export function excerpt(body: Uint8Array): string {
  const prefix = body.subarray(0, Math.min(body.length, 1600));
  let out = "";
  let n = 0;
  for (const ch of text(prefix).trim()) {
    if (n >= 400) break;
    out += ch;
    n++;
  }
  return out;
}

/** The error for an answer past the response cap: the command ran. */
function tooLarge(url: string, cap: number): TooLargeError {
  return new TooLargeError(
    `exec endpoint ${url}: the response exceeded the ${cap}-byte cap and was discarded; the command ran, but its output is too large to return. Narrow it (for example pipe through \`head\`) rather than re-running it unchanged`,
  );
}

/**
 * The most body a client reads for a one-shot answer under response cap `cap`: the base64 of `cap`
 * decoded bytes plus the envelope. The cap itself is on the decoded bytes.
 */
export function bodyCap(cap: number): number {
  return Math.ceil(cap / 3) * 4 + ENVELOPE_SLACK;
}

/**
 * A one-shot answer: `exit_code`, and `stdout` and `stderr` as standard base64 (the empty string
 * for no output). A field that is missing or not base64 is a protocol error, never read as empty.
 * The cap is on the decoded bytes of both streams together.
 */
export function decodeOutput(
  v: Record<string, unknown>,
  url: string,
  cap: number,
): RunResult {
  const code = v.exit_code;
  if (typeof code !== "number" || !Number.isInteger(code)) {
    throw new ProtocolError(
      `exec endpoint ${url} returned unparseable JSON (no integer exit_code)`,
    );
  }
  const stdout = b64Output("stdout", v.stdout, url);
  const stderr = b64Output("stderr", v.stderr, url);
  if (stdout.length + stderr.length > cap) throw tooLarge(url, cap);
  return { exitCode: code, stdout, stderr };
}

function b64Output(field: string, s: unknown, url: string): Uint8Array {
  const b = typeof s === "string" ? fromBase64(s) : null;
  if (!b) {
    throw new ProtocolError(
      `exec endpoint ${url}: \`${field}\` is not standard base64`,
    );
  }
  return b;
}

type Decoded = Frame | { type: "error"; message: string } | "heartbeat";

function b64Field(field: string, s: unknown, url: string): Uint8Array {
  const b = typeof s === "string" ? fromBase64(s) : null;
  if (!b) {
    throw new ProtocolError(
      `exec endpoint ${url}: \`${field}\` is not standard base64`,
    );
  }
  return b;
}

/**
 * One line of a stream. A frame names exactly one field; one that names more is split rather
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
    throw new ProtocolError(
      `exec endpoint ${url}: unparseable frame (${errorText(e)}): ${
        s.slice(0, 200)
      }`,
    );
  }
  if (!isRecord(v)) {
    throw new ProtocolError(
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
  if (typeof v.shell_exit === "number" && Number.isInteger(v.shell_exit)) {
    out.push({ type: "shell_exit", code: v.shell_exit });
  }
  if (typeof v.error === "string") {
    out.push({ type: "error", message: v.error });
  } else if (typeof v.exit_code === "number" && Number.isInteger(v.exit_code)) {
    out.push({ type: "exit", code: v.exit_code });
  } else if (typeof v.signal === "number" && Number.isInteger(v.signal)) {
    out.push({ type: "signal", signal: v.signal });
  } else if (v.timed_out === true) out.push({ type: "timed_out" });
  if (out.length === 0) out.push("heartbeat");
  return out;
}

// ---- jobs ----------------------------------------------------------------------

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

/** A job answer's `signal`, kept only where the state has a code to go with it. */
function jobSignal(
  v: Record<string, unknown>,
  st: { state: JobState },
): { signal: number } | Record<string, never> {
  return (st.state === "exited" || st.state === "ended")
      && typeof v.signal === "number" && Number.isInteger(v.signal)
    ? { signal: v.signal }
    : {};
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

/** A request's heartbeat is whole seconds, 1 to 240 (the endpoint would otherwise silently use its own). */
function checkHeartbeat(seconds: number | undefined): void {
  if (seconds === undefined) return;
  if (
    !Number.isInteger(seconds) || seconds < HEARTBEAT_MIN_S
    || seconds > HEARTBEAT_MAX_S
  ) {
    throw new InvalidRequestError(
      `heartbeatSeconds must be a whole number from ${HEARTBEAT_MIN_S} to ${HEARTBEAT_MAX_S}, got ${seconds}`,
    );
  }
}

/** The `written` an endpoint's 409 body carries, when its excerpt still has it. */
function writtenIn(detail: string): number | undefined {
  const m = /"written"\s*:\s*(\d+)/.exec(detail);
  return m ? Number(m[1]) : undefined;
}

/**
 * A job op's refusal, as the typed error of its status. Matches the reference client
 * (`crates/sandbox/src/exec_endpoint/jobs_native.rs`, `write_stdin`): 404 no such job; stdin 409
 * `has no stdin` (never opened), 409 `gap` (a write ahead of the stream), any other 409 (closed);
 * 429 too many waiting writes. Each is a subclass of what this client threw before, so a catch of
 * `NotFoundError`, `ConflictError` or `IoError` keeps working.
 */
function jobFailure(
  e: unknown,
  id: string,
  op: "start" | "attach" | "write" | "other",
): unknown {
  if (!(e instanceof EndpointError)) return e;
  const { status, detail } = e;
  const cause = { cause: e };
  if (status === 404 && op !== "start") {
    return new JobGoneError(`job ${id}: ${detail}`, id, cause);
  }
  if (op === "start" && status === 409) {
    return new JobExistsError(`job ${id} exists: ${detail}`, id, cause);
  }
  if (op === "start" && status === 507) {
    return new DiskFullError(detail, cause);
  }
  if (op === "write") {
    if (status === 409 && detail.includes("has no stdin")) {
      return new StdinClosedError(
        `job stdin: ${detail}`,
        "never_opened",
        undefined,
        cause,
      );
    }
    if (status === 409 && detail.includes("\"gap\"")) {
      return new StdinGapError(
        `job stdin: ${detail}`,
        writtenIn(detail),
        cause,
      );
    }
    if (status === 409) {
      return new StdinClosedError(
        `job stdin: ${detail}`,
        "closed",
        writtenIn(detail),
        cause,
      );
    }
    if (status === 429) {
      return new StdinBusyError(`job stdin: ${detail}`, cause);
    }
    if (status === 413) return new InvalidRequestError(detail, cause);
  }
  return e;
}

class HttpJobs implements JobOps {
  private readonly t: HttpTransport;

  constructor(t: HttpTransport) {
    this.t = t;
  }

  private job(op: string, scope: string, extra: Record<string, unknown> = {}) {
    return { op, scope, ...extra };
  }

  /** One non-stream job op: its JSON answer, a refusal typed. */
  private async call(
    id: string,
    kind: "attach" | "write" | "other",
    job: Record<string, unknown>,
    req?: ReqOptions,
  ): Promise<Record<string, unknown>> {
    try {
      return await this.t.jobCall(job, req);
    } catch (e) {
      throw jobFailure(e, id, kind);
    }
  }

  private startBody(
    scope: string,
    id: string,
    spec: JobSpec,
    hold: boolean,
    req: ReqOptions,
  ): Body {
    return {
      command: spec.command,
      args: spec.args ?? [],
      cwd: spec.cwd,
      timeoutMs: hold ? STREAM_DEADLINE_MS : JOB_OP_TIMEOUT_MS,
      ...(hold ? { stream: true } : {}),
      job: this.job("start", scope, {
        id,
        hold,
        label: spec.label ?? "",
        stdin_open: spec.stdinOpen === true,
      }),
      env: spec.env,
      heartbeatSeconds: req.heartbeatSeconds,
    };
  }

  async start(
    scope: string,
    id: string,
    spec: JobSpec,
    req: ReqOptions = {},
  ): Promise<JobStarted> {
    await this.t.proveEnv(spec.env);
    let ex: Exchange;
    try {
      ex = await this.t.post(this.startBody(scope, id, spec, false, req));
    } catch (e) {
      throw jobFailure(e, id, "start");
    }
    try {
      const { body, over } = await this.t.readCapped(ex);
      if (over) {
        throw new TooLargeError("a job answer exceeded the response cap");
      }
      let started: unknown;
      let mark: unknown;
      let echoed = false;
      try {
        const v: unknown = JSON.parse(text(body));
        started = isRecord(v) ? v.started : undefined;
        mark = isRecord(v) ? v.mark : undefined;
        echoed = isRecord(v) && v.stdin === true;
      } catch {
        started = undefined;
      }
      if (spec.stdinOpen === true && !echoed) {
        // Started without its stdin: stop it, so nothing is left running that reads EOF.
        await this.kill(scope, [id], {
          remove: true,
          graceMs: DEFAULT_KILL_GRACE_MS,
        }).catch(() => {});
        throw new NotOfferedError(
          "the endpoint predates job stdin: it ran the job without it",
        );
      }
      if (typeof started !== "number") {
        throw new ProtocolError(
          `the endpoint did not report the job's start: ${excerpt(body)}`,
        );
      }
      if (typeof mark !== "string" || mark === "") {
        throw new NotOfferedError(
          "the endpoint predates job marks: it did not name the job's incarnation",
        );
      }
      return { pgid: started, mark };
    } finally {
      ex.close();
    }
  }

  async *startHeld(
    scope: string,
    id: string,
    spec: JobSpec,
    req: ReqOptions = {},
  ): AsyncGenerator<JobFrame> {
    await this.t.proveEnv(spec.env);
    let ex: Exchange;
    try {
      ex = await this.t.post(this.startBody(scope, id, spec, true, req));
    } catch (e) {
      throw jobFailure(e, id, "start");
    }
    try {
      if (!isNdjson(ex.res)) {
        throw new ProtocolError(
          `exec endpoint ${this.t.url} answered a job stream with something that is not a stream`,
        );
      }
      let first = true;
      for await (const line of this.t.lines(ex, req.heartbeatSeconds)) {
        const f = decodeJobFrame(line);
        if (f === "heartbeat") continue;
        if (first) {
          first = false;
          if (f.type !== "started") {
            throw new ProtocolError(
              "the endpoint's held start did not begin with `started`",
            );
          }
          if (f.mark === "") {
            throw new NotOfferedError(
              "the endpoint predates job marks: it did not name the job's incarnation",
            );
          }
          if (spec.stdinOpen === true && f.stdin !== true) {
            // Closing the stream kills the held job (nothing was kept).
            throw new NotOfferedError(
              "the endpoint predates job stdin: it ran the job without it",
            );
          }
        }
        yield f;
        if (f.type === "exit" || f.type === "signal") return;
      }
      throw new TransportError(
        "job stream: the stream ended before the job's final frame",
      );
    } finally {
      ex.close();
    }
  }

  async list(scope: string, req?: ReqOptions): Promise<JobInfo[]> {
    const v = await this.t.jobCall(this.job("list", scope), req);
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
        ...jobSignal(j, st),
        kept: j.kept === true,
        num: typeof j.num === "number" ? j.num : null,
        kind: typeof j.kind === "string" ? j.kind : null,
        seen: count(j.seen),
        started: count(j.started),
        ended: count(j.ended),
        label: typeof j.label === "string" ? j.label : "",
        mark: typeof j.mark === "string" ? j.mark : "",
        command: typeof j.command === "string" ? j.command : "",
        args: Array.isArray(j.args)
          ? j.args.filter((a): a is string => typeof a === "string")
          : [],
        outputStart: count(j.outputStart),
        outputEnd: count(j.outputEnd),
      });
    }
    return out;
  }

  async read(
    scope: string,
    id: string,
    offset: number,
    max: number,
    req?: ReqOptions,
  ): Promise<JobChunk> {
    const v = await this.t.jobCall(
      this.job("read", scope, { id, offset, max }),
      req,
    );
    const st = jobState(v.state, v.code);
    const data = typeof v.data === "string" ? fromBase64(v.data) : null;
    if (!st || data === null) throw badJob("read", v);
    let stderrRanges: [number, number][] | undefined;
    if (v.stderr_ranges !== undefined && v.stderr_ranges !== null) {
      const r = v.stderr_ranges;
      if (
        !Array.isArray(r)
        || !r.every((p) =>
          Array.isArray(p) && p.length === 2
          && p.every((n) =>
            typeof n === "number" && Number.isInteger(n) && n >= 0
          )
        )
      ) throw badJob("read", v);
      stderrRanges = r as [number, number][];
    }
    return {
      ...st,
      ...jobSignal(v, st),
      offset: count(v.offset),
      end: count(v.end),
      data,
      ...(stderrRanges ? { stderrRanges } : {}),
    };
  }

  async *attach(
    scope: string,
    id: string,
    offset: number,
    req: AttachOptions = {},
  ): AsyncGenerator<JobFrame> {
    let ex: Exchange;
    try {
      ex = await this.t.post({
        command: "true",
        args: [],
        timeoutMs: STREAM_DEADLINE_MS,
        stream: true,
        job: this.job("attach", scope, {
          id,
          offset,
          ...(req.mark !== undefined ? { mark: req.mark } : {}),
          ...(req.output !== undefined ? { output: req.output } : {}),
        }),
        heartbeatSeconds: req.heartbeatSeconds,
      });
    } catch (e) {
      throw jobFailure(e, id, "attach");
    }
    try {
      if (!isNdjson(ex.res)) {
        throw new ProtocolError(
          `exec endpoint ${this.t.url} answered a job stream with something that is not a stream`,
        );
      }
      let first = true;
      for await (const line of this.t.lines(ex, req.heartbeatSeconds)) {
        const f = decodeJobFrame(line);
        if (f === "heartbeat") continue;
        if (first) {
          first = false;
          if (f.type !== "attached") {
            throw new NotOfferedError(
              "the endpoint predates job marks: its attach did not begin with `attached`",
            );
          }
        }
        yield f;
        if (f.type === "exit" || f.type === "signal") return;
      }
      throw new TransportError(
        "job stream: the stream ended before the job's final frame",
      );
    } finally {
      ex.close();
    }
  }

  async keep(
    scope: string,
    id: string,
    fields: KeepFields,
    req?: ReqOptions,
  ): Promise<void> {
    await this.call(
      id,
      "other",
      this.job("keep", scope, {
        id,
        ...(fields.num !== undefined ? { num: fields.num } : {}),
        ...(fields.kind !== undefined ? { kind: fields.kind } : {}),
        ...(fields.seen !== undefined ? { seen: fields.seen } : {}),
      }),
      req,
    );
  }

  async seen(
    scope: string,
    id: string,
    offset: number,
    req?: ReqOptions,
  ): Promise<void> {
    await this.call(id, "other", this.job("seen", scope, { id, offset }), req);
  }

  async kill(
    scope: string,
    ids: string[],
    fields: { remove: boolean; graceMs: number },
    req?: ReqOptions,
  ): Promise<Survivor[]> {
    if (ids.length === 0) return [];
    const v = await this.t.jobCall(
      this.job("kill", scope, {
        ids,
        remove: fields.remove,
        grace_ms: fields.graceMs,
      }),
      req,
    );
    return parseSurvivors(v);
  }

  async remove(
    scope: string,
    ids: string[],
    req?: RemoveOptions,
  ): Promise<RemoveResult> {
    if (ids.length === 0) return { running: [], mismatch: [] };
    const v = await this.t.jobCall(
      this.job("remove", scope, {
        ids,
        ...(req?.marks !== undefined ? { marks: req.marks } : {}),
      }),
      req,
    );
    const strings = (a: unknown): string[] | null =>
      Array.isArray(a) && a.every((x) => typeof x === "string")
        ? (a as string[])
        : null;
    const running = strings(v.running);
    const mismatch = strings(v.mismatch);
    if (!running || !mismatch) {
      throw new NotOfferedError(
        "the endpoint predates job marks: remove did not report what it skipped",
      );
    }
    return { running, mismatch };
  }

  async writeStdin(
    scope: string,
    id: string,
    at: number,
    data: Uint8Array,
    eof: boolean,
    req?: ReqOptions,
  ): Promise<StdinWritten> {
    if (data.length > WRITE_MAX) {
      throw new InvalidRequestError(
        `a write is at most ${WRITE_MAX} bytes, got ${data.length}`,
      );
    }
    const v = await this.call(
      id,
      "write",
      this.job("write", scope, { id, at, eof, data: toBase64(data) }),
      req,
    );
    if (typeof v.written !== "number") throw badJob("write", v);
    return { written: v.written, closed: v.closed === true };
  }

  async signal(
    scope: string,
    ids: string[],
    signal: string,
    req?: ReqOptions,
  ): Promise<string[]> {
    if (!JOB_SIGNALS.includes(signal)) {
      throw new InvalidRequestError(
        `unsupported signal ${JSON.stringify(signal)}; one of ${
          JOB_SIGNALS.join(", ")
        }`,
      );
    }
    if (ids.length === 0) return [];
    const v = await this.t.jobCall(
      this.job("signal", scope, { ids, signal }),
      req,
    );
    if (!Array.isArray(v.sent)) throw badJob("signal", v);
    return v.sent.filter((i): i is string => typeof i === "string");
  }

  async wipe(
    scope: string,
    fields: { graceMs: number },
    req?: ReqOptions,
  ): Promise<Survivor[]> {
    return parseSurvivors(
      await this.t.jobCall(
        this.job("wipe", scope, { grace_ms: fields.graceMs }),
        req,
      ),
    );
  }
}

/** One line of a job stream: the stream frames plus `started` and `gap`. */
export function decodeJobFrame(line: Uint8Array): JobFrame | "heartbeat" {
  let s = text(line);
  if (s.endsWith("\r")) s = s.slice(0, -1);
  if (s.trim() === "") return "heartbeat";
  let v: unknown;
  try {
    v = JSON.parse(s);
  } catch (e) {
    throw new ProtocolError(
      `job stream: unparseable job frame (${errorText(e)}): ${s.slice(0, 200)}`,
    );
  }
  if (!isRecord(v)) {
    throw new ProtocolError(
      `job stream: unparseable job frame: ${s.slice(0, 200)}`,
    );
  }
  if (typeof v.error === "string") {
    throw new TransportError(`job stream: ${v.error}`);
  }
  if (typeof v.started === "number") {
    return {
      type: "started",
      pgid: v.started,
      mark: typeof v.mark === "string" ? v.mark : "",
      stdin: v.stdin === true,
    };
  }
  if (isRecord(v.attached)) {
    const a = v.attached;
    if (typeof a.mark !== "string" || typeof a.from !== "number") {
      throw new ProtocolError(
        `job stream: unreadable attached frame: ${s.slice(0, 200)}`,
      );
    }
    return { type: "attached", mark: a.mark, from: a.from };
  }
  const stream = typeof v.stdout === "string"
    ? "stdout"
    : typeof v.stderr === "string"
    ? "stderr"
    : null;
  if (stream !== null) {
    const data = fromBase64(String(v[stream]));
    if (!data) {
      throw new ProtocolError("job stream: output is not standard base64");
    }
    return { type: "output", data, stream };
  }
  if (typeof v.gap === "number") return { type: "gap", bytes: v.gap };
  if (typeof v.shell_exit === "number" && Number.isInteger(v.shell_exit)) {
    return { type: "shell_exit", code: v.shell_exit };
  }
  if (typeof v.exit_code === "number" && Number.isInteger(v.exit_code)) {
    return { type: "exit", code: v.exit_code };
  }
  if (typeof v.signal === "number" && Number.isInteger(v.signal)) {
    return { type: "signal", signal: v.signal };
  }
  if (v.timed_out === true) {
    throw new TransportError("job stream: the endpoint timed the job out");
  }
  return "heartbeat";
}
