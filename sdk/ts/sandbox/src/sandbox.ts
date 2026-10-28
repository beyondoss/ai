import { download, upload } from "./bulk.js";
import type { Transfer, UploadOptions } from "./bulk.js";
import { text, utf8 } from "./bytes.js";
import { Files } from "./files.js";
import type { FilesOptions } from "./files.js";
import { HttpTransport } from "./http.js";
import type { HttpTransportOptions } from "./http.js";
import { Jobs } from "./jobs.js";
import type { Frame, RunOptions, RunResult, Transport } from "./transport.js";

/**
 * What {@link Sandbox.exec} returns: the command's output as **bytes**, plus the same decoded as
 * UTF-8 text (invalid sequences as U+FFFD: lossy by construction, use the bytes when they matter).
 */
export interface ExecResult extends RunResult {
  stdoutText: string;
  stderrText: string;
}

/** Options of {@link Sandbox.exec} and {@link Sandbox.stream}; `stdin` may be text. */
export interface ExecOptions extends Omit<RunOptions, "stdin"> {
  stdin?: Uint8Array | string;
}

export type ConnectOptions =
  & (HttpTransportOptions | { transport: Transport })
  & {
    files?: FilesOptions;
  };

/**
 * Layer 3: one object with everything, over any {@link Transport}. Raw exec is always there
 * ({@link Sandbox.exec}, {@link Sandbox.stream}, {@link Sandbox.transport}), so you are never
 * limited to what the helpers offer.
 */
export class Sandbox {
  readonly transport: Transport;
  readonly files: Files;
  readonly jobs: Jobs;

  constructor(
    transport: Transport,
    options: { files?: FilesOptions | undefined } = {},
  ) {
    this.transport = transport;
    this.files = new Files(transport, options.files);
    this.jobs = new Jobs(transport);
  }

  /**
   * Connect to an exec endpoint (`{url, token}`) or wrap your own transport (`{transport}`), and
   * probe the sandbox once (one round trip) so a bad URL or token fails here, not on first use.
   */
  static async connect(options: ConnectOptions): Promise<Sandbox> {
    const transport = "transport" in options
      ? options.transport
      : new HttpTransport(options);
    const sandbox = new Sandbox(transport, { files: options.files });
    await sandbox.files.capabilities();
    return sandbox;
  }

  /**
   * Copy local files into the sandbox directory `remoteDir` as a tar stream (created if missing).
   * `sources` is a directory (its contents land directly in `remoteDir`) or a list of paths (each
   * under its base name). Modes and symlinks are kept; nothing is followed.
   */
  upload(
    sources: string | string[],
    remoteDir: string,
    options?: UploadOptions,
  ): Promise<Transfer> {
    return upload(this.transport, sources, remoteDir, options);
  }

  /**
   * Copy a sandbox directory's contents (or one file) into the local directory `localDir`
   * (created if missing). The sandbox is not trusted: nothing is written outside `localDir` or
   * through a link.
   */
  download(remotePath: string, localDir: string): Promise<Transfer> {
    return download(this.transport, remotePath, localDir);
  }

  /** Run a command to completion. A non-zero exit is a result, not an error. */
  async exec(
    command: string,
    args: string[] = [],
    options: ExecOptions = {},
  ): Promise<ExecResult> {
    const { stdin, ...rest } = options;
    const r = await this.transport.run(command, args, {
      ...rest,
      ...(stdin !== undefined
        ? { stdin: typeof stdin === "string" ? utf8(stdin) : stdin }
        : {}),
    });
    return {
      ...r,
      stdoutText: text(r.stdout),
      stderrText: text(r.stderr),
    };
  }

  /**
   * Stream a command. Stop iterating and the command is killed.
   */
  stream(
    command: string,
    args: string[] = [],
    options: RunOptions = {},
  ): AsyncIterable<Frame> {
    return this.transport.stream(command, args, options);
  }
}
