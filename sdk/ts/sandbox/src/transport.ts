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
}

/** What a job stream says: output, a gap, the shell's exit, and the final status. */
export type JobFrame =
  | { type: "started"; pgid: number; stdin?: boolean }
  /** `stream` is the pipe the bytes came from; concatenated, the pieces are the combined output. */
  | { type: "output"; data: Uint8Array; stream: "stdout" | "stderr" }
  /** `bytes` of output from the requested offset are no longer kept. */
  | { type: "gap"; bytes: number }
  | { type: "shell_exit"; code: number }
  /** Final: the shell exited with `code`. */
  | { type: "exit"; code: number }
  /** Final: signal `signal` killed the shell (its status is 128+`signal`). */
  | { type: "signal"; signal: number };

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

/** A job id is 1 to 32 characters of `[0-9a-z]`. */
export function isJobId(s: string): boolean {
  return /^[0-9a-z]{1,32}$/.test(s);
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
 * 1 to 32 characters of `[0-9a-z]`. Every wire op is here under its wire name, with every field it
 * takes. An op never reconnects and never retries.
 */
export interface JobOps {
  /** Start a job that the sandbox keeps from the start (`hold: false`); its process group. */
  start(
    scope: string,
    id: string,
    spec: JobSpec,
    req?: ReqOptions,
  ): Promise<number>;
  /**
   * Start a job held by this stream (`hold: true`): the first frame is `started`, and if the stream
   * closes before the job's final frame and no `keep` arrived, the sandbox kills the job's group.
   */
  startHeld(
    scope: string,
    id: string,
    spec: JobSpec,
    req?: ReqOptions,
  ): AsyncIterable<JobFrame>;
  list(scope: string, req?: ReqOptions): Promise<JobInfo[]>;
  /** Up to `max` bytes from `offset`. An unknown or removed job answers state `gone`. */
  read(
    scope: string,
    id: string,
    offset: number,
    max: number,
    req?: ReqOptions,
  ): Promise<JobChunk>;
  /** Follow a job's output from `offset`; the last frame is `exit`. Closing the iterator detaches. */
  attach(
    scope: string,
    id: string,
    offset: number,
    req?: ReqOptions,
  ): AsyncIterable<JobFrame>;
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
  remove(scope: string, ids: string[], req?: ReqOptions): Promise<void>;
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
