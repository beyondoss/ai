import { spawn, spawnSync } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
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

/** `$EXECD_BIN`, else the repository's debug build, building it (`cargo build -p beyond-execd`) if absent. */
function execdBinary(): string | null {
  if (process.env.EXECD_BIN) return process.env.EXECD_BIN;
  const bin = join(
    process.env.CARGO_TARGET_DIR ?? join(repo, "target"),
    "debug",
    "execd",
  );
  if (!existsSync(bin)) {
    const r = spawnSync("cargo", ["build", "-p", "beyond-execd"], {
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
