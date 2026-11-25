/**
 * Layer 1: raw exec behind a small interface a user can replace (ssh, docker exec, a vendor SDK, a
 * test double). The HTTP implementation ({@link HttpTransport}) is the default, not the definition.
 * Everything in layer 2 ({@link Files}, {@link Jobs}) is built on this interface and nothing else.
 */

/** Options of one exec call. */
export interface RunOptions {
  /** Working directory in the sandbox. Helpers send none. */
  cwd?: string;
  /** The endpoint's deadline for the command, in milliseconds. Default 120000. */
  timeoutMs?: number;
  /** Bytes fed to the command's stdin, then closed. */
  stdin?: Uint8Array;
  /**
   * The command's own environment, over the endpoint's. An endpoint that predates it is proved
   * once and refused (`not_offered`): the command is never run without it.
   */
  env?: Record<string, string>;
  /**
   * Abort the request. For a stream, closes the connection so the sandbox kills the command (same
   * as stopping iteration). Reported as `transport` with detail mentioning abort.
   */
  signal?: AbortSignal;
}

/**
 * What a command did: its exit code and the bytes it wrote. A command killed by signal S has
 * `exitCode` 128+S, a command that hit its timeout has `exitCode` 124 with stderr ending
 * `command exceeded its timeout`, and a command that cannot be started has `exitCode` 127.
 * None of these is an error.
 */
export interface RunResult {
  exitCode: number;
  /** The bytes the command wrote to stdout. */
  stdout: Uint8Array;
  /** The bytes the command wrote to stderr. */
  stderr: Uint8Array;
}

/** One frame of a streamed command. Heartbeats are not frames. */
export type Frame =
  | { type: "stdout"; data: Uint8Array }
  | { type: "stderr"; data: Uint8Array }
  /** Not final: the command exited with `code` but processes of its group run on. */
  | { type: "shell_exit"; code: number }
  /** Final: the command exited. */
  | { type: "exit"; code: number }
  /** Final: the command was killed by `signal`. */
  | { type: "signal"; signal: number }
  /** Final: the endpoint killed the command at `timeoutMs`. */
  | { type: "timed_out" };

/** The state of a job as the sandbox reports it. */
export type JobState = "running" | "exited" | "ended" | "lost" | "gone";

/** A terminal's size: both are integers from 1 to 1024. */
export interface PtySize {
  cols: number;
  rows: number;
}

/** A terminal job's size and whether its terminal is still held (`open`: false once its keeper is gone). */
export interface PtyInfo extends PtySize {
  open: boolean;
}

/** One job, as `list` reports it. */
export interface JobInfo {
  id: string;
  state: JobState;
  /** The exit code (128+N when signal N killed the shell); null unless `exited` or `ended`. */
  code: number | null;
  /** The signal that killed the shell; absent for a shell that exited, whatever its code. */
  signal?: number;
  kept: boolean;
  num: number | null;
  kind: string | null;
  /** The offset of the output a person has been shown. */
  seen: number;
  /** Epoch seconds. */
  started: number;
  /** Epoch seconds when it ended; 0 while it runs. */
  ended: number;
  label: string;
  /** Present on a terminal job only. */
  pty?: PtyInfo;
  /** Which incarnation of `id` this is (see {@link JobStarted.mark}). */
  mark: string;
  command: string;
  args: string[];
  /** The offset of the oldest byte of output the sandbox holds. */
  outputStart: number;
  /** The offset after the newest byte of output so far. */
  outputEnd: number;
}

/** What `start` answers: the process group and the mark of this incarnation of the id. */
export interface JobStarted {
  pgid: number;
  /** Names this incarnation: a later job that reuses the id has another mark. */
  mark: string;
}

/** Options of a held job (`startHeld`, `follow`). */
export interface HeldOptions extends ReqOptions {
  /**
   * Called when the handle has failed to reopen its stream 12 times in a row, and again at each
   * further failure. A status, not an end: the handle keeps trying until the job ends, the endpoint
   * refuses for good, or the caller stops iterating.
   */
  onReconnecting?: (status: { attempts: number; sinceMs: number }) => void;
}

/** Options of `follow` and `collect`. */
export interface AttachOptions extends HeldOptions {
  /** Attach only if the job is this incarnation; otherwise the job is gone (`JobGoneError`). */
  mark?: string;
  /** `false`: a status-only attach, which sends no output and no gap (default `true`). */
  output?: boolean;
}

/** Options of `remove`. */
export interface RemoveOptions extends ReqOptions {
  /** Per id, the mark of the incarnation to remove; a job with another mark is left. */
  marks?: Record<string, string>;
}

/** What `remove` skipped: jobs still running, and jobs whose mark was not the one named. */
export interface RemoveResult {
  running: string[];
  mismatch: string[];
}

/** One `read`: the state, where the data starts, where the output ends so far, and the bytes. */
export interface JobChunk {
  state: JobState;
  code: number | null;
  /** The signal that killed the shell; absent for a shell that exited, whatever its code. */
  signal?: number;
  /** Where `data` starts: later than asked when old output was dropped. */
  offset: number;
  /** The end of the job's output so far, whatever `max` was. */
  end: number;
  data: Uint8Array;
  /** Which bytes of `data` came from stderr, as absolute `[from, len]` offsets; absent when none. */
  stderrRanges?: [number, number][];
  /** Present on a terminal job only. A terminal's output is one stream: no `stderrRanges`. */
  pty?: PtyInfo;
}

/** What a job stream says: output, a gap, the shell's exit, and the final status. */
export type JobFrame =
  | {
    type: "started";
    pgid: number;
    mark: string;
    stdin?: boolean;
    pty?: PtySize;
  }
  /** The first frame of an attach: the job's mark and the offset the stream starts at. */
  | { type: "attached"; mark: string; from: number }
  /** `stream` is the pipe the bytes came from; concatenated, the pieces are the combined output. */
  | { type: "output"; data: Uint8Array; stream: "stdout" | "stderr" }
  /** `bytes` of output from the requested offset are no longer kept. */
  | { type: "gap"; bytes: number }
  | { type: "shell_exit"; code: number }
  /** Final: the shell exited with `code`. */
  | { type: "exit"; code: number }
  /** Final: signal `signal` killed the shell (its status is 128+`signal`). */
  | { type: "signal"; signal: number };

/** One request of a held job (see {@link JobOps.open}). */
export type JobOpen =
  & { scope: string; id: string; heartbeatSeconds?: number; silenceMs: number }
  & (
    | { op: "start"; spec: JobSpec; hold: boolean }
    | { op: "attach"; offset: number; mark: string; output?: boolean }
  );

/** The answer to one {@link JobOpen}. */
export interface JobOpened {
  /** The HTTP status. */
  status: number;
  /** The server's `Retry-After`, whole seconds; 0 when it sent none. */
  retryAfterS: number;
  /** What a refusal said, for the error that reports it. */
  detail: string;
  /**
   * The stream's lines, each without its newline, when the status is a success. It throws
   * `TransportError` or `TimeoutError` when the stream is cut or silent for `silenceMs`, and
   * `ProtocolError` for a line it cannot hold.
   */
  lines: AsyncIterable<Uint8Array>;
  /** Release the connection. */
  close(): void;
}

/** What `start` takes: the same command, arguments and directory as `run`, plus a label. */
export interface JobSpec {
  command: string;
  args?: string[];
  cwd?: string;
  label?: string;
  /** The command's own environment (see {@link RunOptions.env}). */
  env?: Record<string, string>;
  /** Keep the job's stdin open for `writeStdin`; default: empty (EOF at once). */
  stdinOpen?: boolean;
  /**
   * Run the command on a pseudo-terminal of this size (wire `pty`; cols and rows are integers from 1
   * to 1024). The job's stdin is open, its output is one stream (`stdout` only) and `writeStdin`
   * takes no `eof`. An endpoint that does not echo `pty` is {@link NotOfferedError}.
   */
  pty?: PtySize;
  /**
   * The start's idempotency key (wire `idempotency_key`): 1 to 128 characters of `[0-9A-Za-z._:-]`. A start that
   * carries the key of an earlier start of the same command, arguments, directory, environment, stdin and hold
   * answers that job and runs nothing; the same key with another command is `409`. Omitted, `Jobs.start` and
   * `Jobs.startHeld` generate one per call, so a start whose answer was lost can be sent again without running
   * twice; pass your own to make a re-run of a whole program return its job. Level 3 sends exactly what is given.
   */
  idempotencyKey?: string;
}

/** The most bytes one `writeStdin` carries. */
export const WRITE_MAX = 128 * 1024;

/** The signals a job can be sent: names only. */
export const JOB_SIGNALS = [
  "SIGHUP",
  "SIGINT",
  "SIGQUIT",
  "SIGUSR1",
  "SIGUSR2",
  "SIGTERM",
  "SIGKILL",
];

/** What a `writeStdin` left: bytes of the stdin stream applied so far, and whether it is closed. */
export interface StdinWritten {
  written: number;
  closed: boolean;
}

/** A process that survived a `kill`. */
export interface Survivor {
  id?: string;
  detail: string;
}

/** A job id is 1 to 64 characters of `[0-9a-z_-]`. */
export function isJobId(s: string): boolean {
  return /^[0-9a-z_-]{1,64}$/.test(s);
}

/** The SIGTERM grace a `kill` or `wipe` gives a job before SIGKILL when the caller names none. */
export const DEFAULT_KILL_GRACE_MS = 300;

/** The silence limit of a stream whose heartbeat the client did not choose (the endpoint beats every 5 s). */
const DEFAULT_SILENCE_MS = 30_000;

/** The fewest and most whole seconds a request may ask `Keepalive-Ping-Interval` for. */
export const HEARTBEAT_MIN_S = 1;
export const HEARTBEAT_MAX_S = 240;

/**
 * How long a stream may go without any frame (a heartbeat is a frame) before it is presumed dead,
 * given the heartbeat the request asked for (`undefined`: the endpoint's own, 5 s). Two beats and
 * ten seconds of slack, never under today's 30 s, so a stream that asked for a 60 s beat is not
 * killed by its own client. A pure function of its argument: it moves into the client kernel
 * unchanged (B1).
 */
export function silenceMs(heartbeatSeconds?: number): number {
  if (heartbeatSeconds === undefined) return DEFAULT_SILENCE_MS;
  return Math.max(DEFAULT_SILENCE_MS, heartbeatSeconds * 2000 + 10_000);
}

/** Options that shape one request, whatever the op. */
export interface ReqOptions {
  /**
   * The heartbeat this request's stream runs at, whole seconds 1 to 240: sent as the
   * `Keepalive-Ping-Interval` request header. Absent: the endpoint's own (5 s).
   */
  heartbeatSeconds?: number;
}

/**
 * What `keep` records. Wire names. (`notify` is left out on purpose: it carries a bearer token the
 * callback is authenticated with, which an SDK handle must not hold; the agent uses it through its
 * own client.)
 */
export interface KeepFields {
  /** The job's number in the agent's listing; absent: nobody's numbered job, only kept. */
  num?: number;
  kind?: string;
  /** The offset of the output a person has been shown. */
  seen?: number;
}

/**
 * The sandbox's jobs: commands it keeps. `scope` is the opaque name of one session's jobs; `id` is
 * 1 to 64 characters of `[0-9a-z_-]`. Every wire op is here under its wire name, with every field it
 * takes. An op never reconnects and never retries.
 */
export interface JobOps {
  /**
   * One request of a held job, answered with the status the endpoint gave: a held start (`hold: true`,
   * the stream's first frame is `started`; if it closes before the job's final frame and no `keep`
   * arrived, the sandbox kills the job's group) or an attach (closing never kills). It never retries
   * and never throws for a refusal: the status is the answer, and `JobHandle` (the client kernel)
   * decides what it means. It throws only when no answer came (`TransportError`, `TimeoutError`).
   */
  open(open: JobOpen): Promise<JobOpened>;
  list(scope: string, req?: ReqOptions): Promise<JobInfo[]>;
  /** Up to `max` bytes from `offset`. An unknown or removed job answers state `gone`. */
  read(
    scope: string,
    id: string,
    offset: number,
    max: number,
    req?: ReqOptions,
  ): Promise<JobChunk>;
  /** Hand a held job to the sandbox for good, recording `num`, `kind` and `seen`. */
  keep(
    scope: string,
    id: string,
    fields: KeepFields,
    req?: ReqOptions,
  ): Promise<void>;
  /** Record the offset of the output a person has been shown (wire op `seen`). */
  seen(
    scope: string,
    id: string,
    offset: number,
    req?: ReqOptions,
  ): Promise<void>;
  kill(
    scope: string,
    ids: string[],
    fields: { remove: boolean; graceMs: number },
    req?: ReqOptions,
  ): Promise<Survivor[]>;
  remove(
    scope: string,
    ids: string[],
    req?: RemoveOptions,
  ): Promise<RemoveResult>;
  wipe(
    scope: string,
    fields: { graceMs: number },
    req?: ReqOptions,
  ): Promise<Survivor[]>;
  /** Positional write to the job's stdin (see the spec's "Job stdin"). */
  writeStdin(
    scope: string,
    id: string,
    at: number,
    data: Uint8Array,
    eof: boolean,
    req?: ReqOptions,
  ): Promise<StdinWritten>;
  /** Set a terminal job's size (wire op `resize`); absolute, so a retry is idempotent. */
  resize(
    scope: string,
    id: string,
    cols: number,
    rows: number,
    req?: ReqOptions,
  ): Promise<PtySize>;
  /** Send a named signal to each running job's group; the ids that were signalled. */
  signal(
    scope: string,
    ids: string[],
    signal: string,
    req?: ReqOptions,
  ): Promise<string[]>;
}

/**
 * The layer-1 interface: one-shot exec, streaming exec and jobs. Everything in layer 2 works over
 * it.
 *
 * A transport reports failures as {@link SandboxError}s with the kinds of FILES-OVER-EXEC.md
 * section 3.2 (`transport`, `timeout`, `auth`, `too_large`, `endpoint`, `protocol`). A command that
 * ran and exited non-zero is a result, never an error.
 */
export interface Transport {
  /** Run a command to completion; its output is bytes. */
  run(
    command: string,
    args: string[],
    options?: RunOptions,
  ): Promise<RunResult>;

  /**
   * Stream a command: frames in order, the last one final (`exit`, `signal` or `timed_out`). The
   * command's lifetime is the iteration: when the consumer stops (`break`, `return()`, an
   * exception) the transport MUST close the stream so the sandbox kills the command. A stream that
   * ends without its final frame throws `transport`.
   */
  stream(
    command: string,
    args: string[],
    options?: RunOptions,
  ): AsyncIterable<Frame>;

  /** The sandbox's jobs. */
  readonly jobs: JobOps;
}
