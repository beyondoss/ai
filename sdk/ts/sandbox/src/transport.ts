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
}

/** What a job stream says: output, a gap, the shell's exit, and the final status. */
export type JobFrame =
  | { type: "started"; pgid: number }
  | { type: "output"; data: Uint8Array }
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

/**
 * The sandbox's jobs: commands it keeps. `scope` is the opaque name of one session's jobs; `id` is
 * 1 to 32 characters of `[0-9a-z]`.
 */
export interface JobOps {
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
