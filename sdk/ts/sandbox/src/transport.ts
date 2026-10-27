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
  /** Bytes fed to the command's stdin, then closed (exec v1.1 `stdin_base64`). */
  stdin?: Uint8Array;
  /**
   * The caller reads the output as text (a helper parsing a listing, or base64 it asked the shell
   * for), so it does not need the command's exact bytes: a transport may answer the plain way and
   * skip what exact bytes cost. Never needed for correctness; a transport may ignore it.
   */
  text?: boolean;
}

/**
 * What a command did. Over the one-shot protocol (v1.1) a command killed by signal S has
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
  /**
   * Whether `stdout` and `stderr` are exactly what the command wrote. `false` only when the
   * transport could get nothing but text from the target (a plain v1.1 endpoint, which already
   * replaced invalid UTF-8 with U+FFFD before this client saw it) or was told the caller reads text
   * ({@link RunOptions.text}); the bytes are then the UTF-8 of that text. Absent means exact: a
   * transport that has the bytes (ssh, a vendor SDK) need not set it.
   */
  exact?: boolean;
}

/** One frame of a streamed command (exec v1.2). Heartbeats are not frames. */
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
  /** The exit code; null unless `exited` or `ended`. */
  code: number | null;
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
  /** Where `data` starts: later than asked when old output was dropped. */
  offset: number;
  /** The end of the job's output so far, whatever `max` was. */
  end: number;
  data: Uint8Array;
}

/** What a job stream says: output, a gap, the shell's exit, and the final status. */
export type JobFrame =
  | { type: "started"; pgid: number }
  | { type: "output"; data: Uint8Array }
  /** `bytes` of output from the requested offset are no longer kept. */
  | { type: "gap"; bytes: number }
  | { type: "shell_exit"; code: number }
  /** Final: the job's status (the exit code, or 128+N for signal N). */
  | { type: "exit"; code: number };

/** What `start` takes: the same command, arguments and directory as `run`, plus a label. */
export interface JobSpec {
  command: string;
  args?: string[];
  cwd?: string;
  label?: string;
}

/** A process that survived a `kill`. */
export interface Survivor {
  id?: string;
  detail: string;
}

/**
 * Native sandbox jobs (exec v1.3), for a transport whose endpoint keeps jobs its own way. A
 * transport without it gets the baseline: {@link Jobs} builds jobs from plain `run` calls.
 * `scope` is the opaque name of one session's jobs; `id` is 1 to 32 characters of `[0-9a-z]`.
 */
export interface NativeJobs {
  /** Start a job that the sandbox keeps from the start; its process group. */
  start(scope: string, id: string, spec: JobSpec): Promise<number>;
  list(scope: string): Promise<JobInfo[]>;
  read(
    scope: string,
    id: string,
    offset: number,
    max: number,
  ): Promise<JobChunk>;
  /** Follow a job's output from `offset`; the last frame is `exit`. Closing the iterator detaches. */
  attach(scope: string, id: string, offset: number): AsyncIterable<JobFrame>;
  kill(scope: string, ids: string[], remove: boolean): Promise<Survivor[]>;
  remove(scope: string, ids: string[]): Promise<void>;
  wipe(scope: string): Promise<Survivor[]>;
}

/** What a transport offers beyond the one-shot call. */
export interface Offers {
  /** The endpoint proved it streams incrementally (exec v1.2). */
  stream: boolean;
  /** The endpoint keeps jobs natively (exec v1.3). */
  nativeJobs: boolean;
  /** The endpoint answers a one-shot call with the command's bytes (`output_base64`, exec v1.1). */
  outputBase64: boolean;
}

/**
 * The layer-1 interface. Implement `run` and you have a transport; everything in layer 2 works over
 * it. `stream`, `nativeJobs` and `offers` are optional: a transport without them is a v1.1 one.
 *
 * A transport reports failures as {@link SandboxError}s with the kinds of FILES-OVER-EXEC.md
 * section 3.2 (`transport`, `timeout`, `auth`, `too_large`, `endpoint`, `protocol`). A command that
 * ran and exited non-zero is a result, never an error.
 */
export interface Transport {
  run(
    command: string,
    args: string[],
    options?: RunOptions,
  ): Promise<RunResult>;

  /**
   * Stream a command (v1.2): frames in order, the last one final (`exit`, `signal` or
   * `timed_out`). The command's lifetime is the iteration: when the consumer stops (`break`,
   * `return()`, an exception) the transport MUST close the stream so the sandbox kills the command.
   * Throws `not_offered` when the endpoint did not prove it streams; the command is never run as a
   * one-shot instead. A stream that ends without its final frame throws `transport`.
   */
  stream?(
    command: string,
    args: string[],
    options?: RunOptions,
  ): AsyncIterable<Frame>;

  /** Native jobs (v1.3), when {@link Transport.offers} says the endpoint has them. */
  readonly nativeJobs?: NativeJobs;

  /**
   * What the endpoint offers, probed once and cached. A probe that could not reach the endpoint
   * throws and is not cached. Absent means neither stream nor native jobs.
   */
  offers?(): Promise<Offers>;
}
