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
  stdin?: import("node:stream").Writable;
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
        stdio: wantStdin ? ["pipe", "pipe", "pipe"] : ["ignore", "pipe", "pipe"],
      });

      async function* lines(
        stream: NodeJS.ReadableStream | null,
      ): AsyncIterable<string> {
        if (!stream) return;
        for await (const chunk of stream) {
          yield String(chunk);
        }
      }

      return {
        stdout: lines(child.stdout),
        stderr: lines(child.stderr),
        stdin: wantStdin ? (child.stdin ?? undefined) : undefined,
        exitCode: new Promise((resolve, reject) => {
          child.on("error", reject);
          child.on("close", (code) => resolve(code ?? 1));
        }),
        kill: (s) => {
          child.kill(s);
        },
      };
    },
    async stop() {},
    async delete() {},
  };
}
