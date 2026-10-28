import { concat } from "./bytes.js";
import { InvalidRequestError, ProtocolError } from "./errors.js";
import { isJobId } from "./http.js";
import type {
  JobChunk,
  JobFrame,
  JobInfo,
  JobSpec,
  Survivor,
  Transport,
} from "./transport.js";

/** The most output one `read` returns by default. */
export const READ_MAX = 128 * 1024;

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
  /** The signal that killed the shell; absent for a shell that exited, whatever its code. */
  signal?: number;
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

  /** Start a job the sandbox keeps from the start; returns its process group. */
  async start(scope: string, id: string, spec: JobSpec): Promise<number> {
    validScope(scope);
    validId(id);
    return this.transport.jobs.start(scope, id, spec);
  }

  /** Every job in the scope. */
  async list(scope: string): Promise<JobInfo[]> {
    validScope(scope);
    return this.transport.jobs.list(scope);
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
    return this.transport.jobs.read(scope, id, offset, max);
  }

  /**
   * Follow a job's output from `offset` to its end: frames in order, the last one `exit`. Stopping
   * early detaches and never kills the job.
   */
  async *follow(
    scope: string,
    id: string,
    offset = 0,
  ): AsyncGenerator<JobFrame> {
    validScope(scope);
    validId(id);
    count("offset", offset, 0);
    yield* this.transport.jobs.attach(scope, id, offset);
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
      } else if (f.type === "signal") {
        return {
          exitCode: 128 + f.signal,
          signal: f.signal,
          output: concat(chunks),
          gap,
        };
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
    if (ids.length === 0) return [];
    return this.transport.jobs.kill(scope, ids, options.remove ?? false);
  }

  /** Delete jobs whose process group is empty; a running job is left alone. */
  async remove(scope: string, ids: string[]): Promise<void> {
    validScope(scope);
    ids.forEach(validId);
    if (ids.length === 0) return;
    return this.transport.jobs.remove(scope, ids);
  }

  /** Kill every job in the scope and delete it. Returns what survived. */
  async wipe(scope: string): Promise<Survivor[]> {
    validScope(scope);
    return this.transport.jobs.wipe(scope);
  }
}
