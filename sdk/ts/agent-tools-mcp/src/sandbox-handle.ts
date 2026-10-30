/**
 * Eve-shaped sandbox handle for tools that already run *inside* the guest.
 * Same author API as `ctx.getSandbox()` (`run`, `writeTextFile`, `resolvePath`, …).
 */
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

export type SandboxRunResult = {
  exitCode: number;
  stdout: string;
  stderr: string;
};

export type SandboxRunOptions = {
  command: string;
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  signal?: AbortSignal;
};

export type SandboxSpawnOptions = SandboxRunOptions & {
  stdin?: boolean;
};

export type SandboxProcess = {
  stdout: AsyncIterable<string>;
  stderr: AsyncIterable<string>;
  stdin?: import("node:stream").Writable | undefined;
  exitCode: Promise<number>;
  kill: (signal?: NodeJS.Signals) => void;
};

export type SandboxHandle = {
  resolvePath: (relative: string) => string;
  writeTextFile: (opts: { path: string; content: string }) => Promise<void>;
  readTextFile: (opts: { path: string }) => Promise<string>;
  run: (opts: SandboxRunOptions) => Promise<SandboxRunResult>;
  spawn: (opts: SandboxSpawnOptions) => Promise<SandboxProcess>;
  /** Eve: stop compute; in-guest no-op (process is the guest). */
  stop: () => Promise<void>;
  /** Eve: delete sandbox; in-guest no-op. */
  delete: () => Promise<void>;
};

export function createLocalSandboxHandle(
  workspaceRoot: string,
  abortSignal?: AbortSignal,
): SandboxHandle {
  const root = path.resolve(workspaceRoot);

  const resolvePath = (relative: string): string => {
    const resolved = path.resolve(root, relative);
    if (resolved !== root && !resolved.startsWith(root + path.sep)) {
      throw new Error(`path escapes workspace: ${relative}`);
    }
    return resolved;
  };

  return {
    resolvePath,
    async writeTextFile({ path: rel, content }) {
      const abs = resolvePath(rel);
      await mkdir(path.dirname(abs), { recursive: true });
      await writeFile(abs, content, "utf8");
    },
    async readTextFile({ path: rel }) {
      return readFile(resolvePath(rel), "utf8");
    },
    async run({ command, cwd, env, signal }) {
      const sig = signal ?? abortSignal;
      return new Promise<SandboxRunResult>((resolve, reject) => {
        const child = spawn("/bin/bash", ["-c", command], {
          cwd: cwd ? resolvePath(cwd) : root,
          env: {
            ...process.env,
            ...env,
            AI_AGENT: env?.AI_AGENT ?? process.env.AI_AGENT ?? "beyond",
          },
          signal: sig,
        });
        let stdout = "";
        let stderr = "";
        child.stdout?.on("data", (c: Buffer) => {
          stdout += c.toString();
        });
        child.stderr?.on("data", (c: Buffer) => {
          stderr += c.toString();
        });
        child.on("error", reject);
        child.on("close", (code) => {
          resolve({ exitCode: code ?? 1, stdout, stderr });
        });
      });
    },
    async spawn({ command, cwd, env, signal, stdin: wantStdin }) {
      const sig = signal ?? abortSignal;
      const child = spawn("/bin/bash", ["-c", command], {
        cwd: cwd ? resolvePath(cwd) : root,
        env: {
          ...process.env,
          ...env,
          AI_AGENT: env?.AI_AGENT ?? process.env.AI_AGENT ?? "beyond",
        },
        signal: sig,
        stdio: wantStdin
          ? ["pipe", "pipe", "pipe"]
          : ["ignore", "pipe", "pipe"],
      });

      // Attach the readers now. Flowing mode drains the pipes even when the
      // caller only waits on exitCode, and the iterator yields lines (no
      // trailing newline), not raw chunks.
      const proc: SandboxProcess = {
        stdout: readLines(child.stdout),
        stderr: readLines(child.stderr),
        exitCode: new Promise((resolve, reject) => {
          child.on("error", reject);
          child.on("close", (code) => resolve(code ?? 1));
        }),
        kill: (s) => {
          child.kill(s);
        },
      };
      if (wantStdin && child.stdin) proc.stdin = child.stdin;
      return proc;
    },
    async stop() {},
    async delete() {},
  };
}

/** Split a child pipe into lines. The `'data'` listener is attached immediately so an unread stream cannot fill the pipe and stall `exitCode`. */
function readLines(
  stream: NodeJS.ReadableStream | null,
): AsyncIterable<string> {
  const lines: string[] = [];
  let pending = "";
  let done = !stream;
  let notify: (() => void) | null = null;
  const wake = () => {
    notify?.();
    notify = null;
  };
  const push = (line: string) => {
    lines.push(line.endsWith("\r") ? line.slice(0, -1) : line);
    wake();
  };
  if (stream) {
    stream.on("data", (chunk: Buffer | string) => {
      pending += String(chunk);
      for (;;) {
        const nl = pending.indexOf("\n");
        if (nl < 0) break;
        push(pending.slice(0, nl));
        pending = pending.slice(nl + 1);
      }
    });
    const finish = () => {
      if (pending.length) push(pending);
      pending = "";
      done = true;
      wake();
    };
    stream.on("end", finish);
    stream.on("error", finish);
  }

  return {
    [Symbol.asyncIterator]() {
      return {
        async next(): Promise<IteratorResult<string>> {
          for (;;) {
            const line = lines.shift();
            if (line !== undefined) return { value: line, done: false };
            if (done) return { value: undefined, done: true };
            await new Promise<void>((resolve) => {
              notify = resolve;
            });
          }
        },
      };
    },
  };
}
