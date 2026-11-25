import { concat } from "./bytes.js";
import { InvalidRequestError, JobGoneError, ProtocolError } from "./errors.js";
import { DEFAULT_KILL_GRACE_MS, isJobId } from "./transport.js";
import type {
  AttachOptions,
  JobChunk,
  JobFrame,
  JobInfo,
  JobSpec,
  JobStarted,
  KeepFields,
  RemoveOptions,
  RemoveResult,
  ReqOptions,
  StdinWritten,
  Survivor,
  Transport,
} from "./transport.js";

/** The most output one `read` returns by default. */
export const READ_MAX = 128 * 1024;

export interface JobReadOptions extends ReqOptions {
  offset?: number;
  max?: number;
}

export interface KillOptions extends ReqOptions {
  /** Also delete each job whose group is empty afterwards. */
  remove?: boolean;
  /** Milliseconds between SIGTERM and SIGKILL (wire `grace_ms`); default 300. */
  graceMs?: number;
}

export interface WipeOptions extends ReqOptions {
  /** Milliseconds between SIGTERM and SIGKILL (wire `grace_ms`); default 300. */
  graceMs?: number;
}

/** What following a job to its end collected. */
export interface JobOutput {
  /** The job's status: its exit code, or 128+N for signal N. */
  exitCode: number;
  /** The signal that killed the shell; absent for a shell that exited, whatever its code. */
  signal?: number;
  output: Uint8Array;
  /** The bytes of `output` that came on stderr frames. */
  stderr: Uint8Array;
  /** Bytes of output dropped before they could be read. */
  gap: number;
  /** The job's mark and where the stream started (the `attached` frame). */
  attached?: { mark: string; from: number };
}

function validScope(scope: string): void {
  if (typeof scope !== "string" || !/^[0-9a-z-]{1,64}$/.test(scope)) {
    throw new InvalidRequestError(
      "a job scope is 1 to 64 characters of [0-9a-z-]",
    );
  }
}

function validId(id: string): void {
  if (typeof id !== "string" || !isJobId(id)) {
    throw new InvalidRequestError(
      `bad job id ${JSON.stringify(id)}: 1 to 64 characters of [0-9a-z_-]`,
    );
  }
}

function validKey(key: string): void {
  if (typeof key !== "string" || !/^[0-9A-Za-z._:-]{1,128}$/.test(key)) {
    throw new InvalidRequestError(
      `bad idempotency key ${
        JSON.stringify(key)
      }: 1 to 128 characters of [0-9A-Za-z._:-]`,
    );
  }
}

/** The spec with its idempotency key: the caller's, checked, else a fresh one (one per call, before the first send). */
async function keyed(spec: JobSpec): Promise<JobSpec> {
  if (spec.idempotencyKey !== undefined) {
    validKey(spec.idempotencyKey);
    return spec;
  }
  const c = (globalThis as { crypto?: Crypto }).crypto;
  const key = c?.randomUUID
    ? c.randomUUID()
    : (await import("node:crypto")).randomUUID();
  return { ...spec, idempotencyKey: key };
}

function count(name: string, v: number | undefined, dflt: number): number {
  if (v === undefined) return dflt;
  if (!Number.isInteger(v) || v < 0) {
    throw new InvalidRequestError(`\`${name}\` must be a non-negative integer`);
  }
  return v;
}

/**
 * Layer 2: jobs over ANY {@link Transport}. A job is a command the sandbox keeps: it runs until it
 * exits, is killed or its scope is wiped, and any client can list it, read its output from an
 * offset and kill it. The sandbox keeps them; this validates and forwards.
 */
export class Jobs {
  readonly transport: Transport;

  constructor(transport: Transport) {
    this.transport = transport;
  }

  /**
   * Start a job the sandbox keeps from the start; returns its process group and mark. An id that exists is
   * {@link JobExistsError}; a full disk is {@link DiskFullError}. The start carries an idempotency key
   * (`spec.idempotencyKey`, or one generated for this call): sent again, it answers the same job and runs
   * nothing, and the key of another command or id is {@link KeyConflictError} (`409`). An endpoint that does
   * not echo the key is {@link NotOfferedError}: the job it started is stopped.
   */
  async start(
    scope: string,
    id: string,
    spec: JobSpec,
    req: ReqOptions = {},
  ): Promise<JobStarted> {
    validScope(scope);
    validId(id);
    return this.transport.jobs.start(scope, id, await keyed(spec), req);
  }

  /**
   * Start a job held by the returned stream (wire `hold: true`): frames in order, the first
   * `started`, the last `exit` or `signal`. If the stream stops before the last frame and the job
   * was not {@link Jobs.keep}t, the sandbox kills its group.
   */
  async *startHeld(
    scope: string,
    id: string,
    spec: JobSpec,
    req: ReqOptions = {},
  ): AsyncGenerator<JobFrame> {
    validScope(scope);
    validId(id);
    yield* this.transport.jobs.startHeld(scope, id, await keyed(spec), req);
  }

  /** Every job in the scope. */
  async list(scope: string, req: ReqOptions = {}): Promise<JobInfo[]> {
    validScope(scope);
    return this.transport.jobs.list(scope, req);
  }

  /**
   * Up to `max` bytes of output from `offset`, and the job's state. A job that does not exist or
   * was removed (wire state `gone`) is {@link JobGoneError}; level 3 (`transport.jobs.read`) keeps
   * the wire's `gone` state.
   */
  async read(
    scope: string,
    id: string,
    options: JobReadOptions = {},
  ): Promise<JobChunk> {
    validScope(scope);
    validId(id);
    const offset = count("offset", options.offset, 0);
    const max = count("max", options.max, READ_MAX);
    const chunk = await this.transport.jobs.read(scope, id, offset, max, {
      ...(options.heartbeatSeconds !== undefined
        ? { heartbeatSeconds: options.heartbeatSeconds }
        : {}),
    });
    if (chunk.state === "gone") {
      throw new JobGoneError(
        `job ${id} is gone: it does not exist or was removed`,
        id,
      );
    }
    return chunk;
  }

  /**
   * Follow a job's output from `offset` to its end: frames in order, the last one `exit` or
   * `signal`. Stopping early detaches and never kills the job. (The transport wire call is
   * {@link Transport.jobs}.attach; this surface names the stream `follow` so it is not confused
   * with {@link Jobs.collect}.) A job that does not exist is {@link JobGoneError}. A stream that
   * asked for a `heartbeatSeconds` is presumed dead only after silence longer than two beats.
   */
  async *follow(
    scope: string,
    id: string,
    offset = 0,
    req: AttachOptions = {},
  ): AsyncGenerator<JobFrame> {
    validScope(scope);
    validId(id);
    count("offset", offset, 0);
    yield* this.transport.jobs.attach(scope, id, offset, req);
  }

  /** {@link Jobs.follow} to the end, collected into one {@link JobOutput}. */
  async collect(
    scope: string,
    id: string,
    offset = 0,
    req: AttachOptions = {},
  ): Promise<JobOutput> {
    const chunks: Uint8Array[] = [];
    const errs: Uint8Array[] = [];
    let gap = 0;
    let attached: { mark: string; from: number } | undefined;
    for await (const f of this.follow(scope, id, offset, req)) {
      if (f.type === "attached") attached = { mark: f.mark, from: f.from };
      else if (f.type === "output") {
        chunks.push(f.data);
        if (f.stream === "stderr") errs.push(f.data);
      } else if (f.type === "gap") gap += f.bytes;
      else if (f.type === "exit") {
        return {
          exitCode: f.code,
          output: concat(chunks),
          stderr: concat(errs),
          gap,
          ...(attached ? { attached } : {}),
        };
      } else if (f.type === "signal") {
        return {
          exitCode: 128 + f.signal,
          signal: f.signal,
          output: concat(chunks),
          stderr: concat(errs),
          gap,
          ...(attached ? { attached } : {}),
        };
      }
    }
    throw new ProtocolError(
      "the job stream ended before the job's final frame",
    );
  }

  /**
   * Hand a job to the sandbox for good (wire `keep`): after it, a closed {@link Jobs.startHeld}
   * stream kills nothing. Records `num`, `kind` and `seen` for `list`. An unknown job is
   * {@link JobGoneError}.
   */
  async keep(
    scope: string,
    id: string,
    fields: KeepFields = {},
    req: ReqOptions = {},
  ): Promise<void> {
    validScope(scope);
    validId(id);
    count("num", fields.num, 0);
    count("seen", fields.seen, 0);
    return this.transport.jobs.keep(scope, id, fields, req);
  }

  /** Record the offset of the output a person has been shown (wire `seen`). Unknown job: {@link JobGoneError}. */
  async seen(
    scope: string,
    id: string,
    offset: number,
    req: ReqOptions = {},
  ): Promise<void> {
    validScope(scope);
    validId(id);
    count("offset", offset, 0);
    return this.transport.jobs.seen(scope, id, offset, req);
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
    if (ids.length === 0) return [];
    const { remove, graceMs, ...req } = options;
    return this.transport.jobs.kill(scope, ids, {
      remove: remove ?? false,
      graceMs: count("graceMs", graceMs, DEFAULT_KILL_GRACE_MS),
    }, req);
  }

  /**
   * Delete jobs whose process group is empty. A running job is left and reported in `running`; a job
   * whose mark is not the one `marks` names for its id is left and reported in `mismatch`.
   */
  async remove(
    scope: string,
    ids: string[],
    req: RemoveOptions = {},
  ): Promise<RemoveResult> {
    validScope(scope);
    ids.forEach(validId);
    if (ids.length === 0) return { running: [], mismatch: [] };
    return this.transport.jobs.remove(scope, ids, req);
  }

  /**
   * Write `data` at position `at` of the job's stdin (the job must have started with `stdinOpen`);
   * `eof` closes it after. Positional: a retry (same `at`) applies nothing twice and a write ahead of
   * the stream waits for the bytes before it. All of these are `conflict`, told apart by class:
   * {@link StdinClosedError} (closed, or never opened: retrying cannot succeed), {@link StdinGapError}
   * (the bytes before it never came) and {@link StdinBusyError} (16 writes already wait: retry the
   * same write later). No such job is {@link JobGoneError} (`not_found`); over 128 KiB is
   * `invalid_request`.
   */
  async writeStdin(
    scope: string,
    id: string,
    at: number,
    data: Uint8Array,
    eof = false,
    req: ReqOptions = {},
  ): Promise<StdinWritten> {
    validScope(scope);
    validId(id);
    count("at", at, 0);
    return this.transport.jobs.writeStdin(scope, id, at, data, eof, req);
  }

  /**
   * Send a signal (`SIGHUP SIGINT SIGQUIT SIGUSR1 SIGUSR2 SIGTERM SIGKILL`) to each running job's
   * group, with no grace and no wait. The ids that were running.
   */
  async signal(
    scope: string,
    ids: string[],
    signal: string,
    req: ReqOptions = {},
  ): Promise<string[]> {
    validScope(scope);
    ids.forEach(validId);
    return this.transport.jobs.signal(scope, ids, signal, req);
  }

  /** Kill every job in the scope and delete it. Returns what survived. */
  async wipe(scope: string, options: WipeOptions = {}): Promise<Survivor[]> {
    validScope(scope);
    const { graceMs, ...req } = options;
    return this.transport.jobs.wipe(scope, {
      graceMs: count("graceMs", graceMs, DEFAULT_KILL_GRACE_MS),
    }, req);
  }
}
