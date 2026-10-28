import { spawn, spawnSync } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { TestProject } from "vitest/node";

export interface ExecdInfo {
  url: string;
  token: string;
  root: string;
}

declare module "vitest" {
  export interface ProvidedContext {
    execd: ExecdInfo | null;
  }
}

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");

function freePort(): Promise<number> {
  return new Promise((res, rej) => {
    const s = createServer();
    s.listen(0, "127.0.0.1", () => {
      const { port } = s.address() as { port: number };
      s.close(() => res(port));
    });
    s.on("error", rej);
  });
}

/**
 * The host's cargo build lock: `$BEYOND_BUILD_LOCK`, else `$HOME/.cache/beyond-locks/rust-build.lock`.
 * Present only when its directory exists and `flock` is on PATH (CI has neither).
 */
function buildLock(): string | null {
  const lock = process.env.BEYOND_BUILD_LOCK
    ?? (process.env.HOME
      ? join(process.env.HOME, ".cache/beyond-locks/rust-build.lock")
      : undefined);
  if (!lock || !existsSync(dirname(lock))) return null;
  const onPath = (process.env.PATH ?? "")
    .split(":")
    .some((d) => d && existsSync(join(d, "flock")));
  return onPath && !heldByAncestor(lock) ? lock : null;
}

/**
 * Whether an ancestor already runs `flock ... LOCK` (`flock -o LOCK mise run test:ts`, say).
 * Taking the lock again would wait on the process that holds it, forever.
 */
function heldByAncestor(lock: string): boolean {
  let pid = process.pid;
  while (pid > 1) {
    let stat: string;
    try {
      stat = readFileSync(`/proc/${pid}/stat`, "utf8");
    } catch {
      return false;
    }
    // Field 4 of /proc/PID/stat is the parent pid; it follows the parenthesised command name.
    const ppid = Number(stat.slice(stat.lastIndexOf(")") + 2).split(" ")[1]);
    if (!(ppid > 0)) return false;
    try {
      const argv = readFileSync(`/proc/${ppid}/cmdline`, "utf8").split("\0");
      if (argv[0]?.endsWith("flock") && argv.includes(lock)) return true;
    } catch {
      // The parent is gone or not readable; keep walking up.
    }
    pid = ppid;
  }
  return false;
}

/** `$EXECD_BIN`, else the repository's debug build, building it (`cargo build -p beyond-execd`) if absent. */
function execdBinary(): string | null {
  if (process.env.EXECD_BIN) return process.env.EXECD_BIN;
  const bin = join(
    process.env.CARGO_TARGET_DIR ?? join(repo, "target"),
    "debug",
    "execd",
  );
  if (!existsSync(bin)) {
    const lock = buildLock();
    const build = ["cargo", "build", "-p", "beyond-execd"];
    const [prog = "cargo", ...args] = lock
      ? ["flock", "-o", lock, ...build]
      : build;
    const r = spawnSync(prog, args, {
      cwd: repo,
      stdio: "inherit",
    });
    if (r.status !== 0) return null;
  }
  return existsSync(bin) ? bin : null;
}

/**
 * Starts one loopback `execd` for the smoke tests. Without a binary (no Rust toolchain) the tests
 * that need it skip themselves, loudly.
 */
export default async function setup(
  project: TestProject,
): Promise<() => Promise<void>> {
  const bin = execdBinary();
  if (!bin) {
    console.warn(
      "execd smoke tests skipped: no execd binary (set EXECD_BIN or run `cargo build -p beyond-execd`)",
    );
    project.provide("execd", null);
    return async () => {};
  }
  const port = await freePort();
  const root = mkdtempSync(join(tmpdir(), "sandbox-ts-"));
  const token = "smoke-test-token";
  const child: ChildProcess = spawn(bin, [], {
    env: {
      PATH: process.env.PATH ?? "",
      EXECD_LISTEN: `127.0.0.1:${port}`,
      EXECD_ROOT: root,
      EXECD_BEARER: token,
      EXECD_LOG: "warn",
    },
    stdio: ["ignore", "inherit", "inherit"],
  });
  const url = `http://127.0.0.1:${port}/exec`;
  for (let i = 0; i < 200; i++) {
    try {
      if ((await fetch(`http://127.0.0.1:${port}/livez`)).ok) break;
    } catch {
      await new Promise((r) => setTimeout(r, 50));
    }
    if (child.exitCode !== null) {
      throw new Error(`execd exited with ${child.exitCode}`);
    }
  }
  project.provide("execd", { url, token, root });
  return async () => {
    child.kill("SIGTERM");
    await new Promise((r) => child.once("exit", r));
    rmSync(root, { recursive: true, force: true });
  };
}
