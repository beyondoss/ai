import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { Files, SandboxError } from "../src/index.js";
import type {
  Frame,
  JobOps,
  RunOptions,
  RunResult,
  Transport,
} from "../src/index.js";

const refuse = (what: string) => (): never => {
  throw new SandboxError("not_offered", `this transport has no ${what}`);
};

/**
 * Bring your own transport: raw exec on this machine, no HTTP anywhere. The file helpers work over
 * it because they are built on `run` alone. This one has no streaming and no jobs, so it throws
 * `not_offered` from them: the transport's own business, not a capability the client probes.
 */
class LocalTransport implements Transport {
  calls = 0;
  readonly jobs: JobOps = {
    start: refuse("jobs"),
    list: refuse("jobs"),
    read: refuse("jobs"),
    attach: refuse("jobs"),
    kill: refuse("jobs"),
    remove: refuse("jobs"),
    wipe: refuse("jobs"),
  };

  stream(): AsyncIterable<Frame> {
    return refuse("stream")();
  }

  run(
    command: string,
    args: string[],
    options: RunOptions = {},
  ): Promise<RunResult> {
    this.calls++;
    return new Promise((resolve, reject) => {
      const child = spawn(command, args, {
        cwd: options.cwd,
        stdio: ["pipe", "pipe", "pipe"],
        detached: true,
      });
      const out: Buffer[] = [];
      const err: Buffer[] = [];
      child.stdout.on("data", (c: Buffer) => out.push(c));
      child.stderr.on("data", (c: Buffer) => err.push(c));
      child.on("error", (e: NodeJS.ErrnoException) =>
        e.code === "ENOENT"
          ? resolve({
            exitCode: 127,
            stdout: new Uint8Array(),
            stderr: new TextEncoder().encode(`${command}: not found`),
          })
          : reject(e));
      child.on("close", (code, signal) =>
        resolve({
          exitCode: code
            ?? (signal ? 128 + (signal === "SIGKILL" ? 9 : 15) : 1),
          stdout: new Uint8Array(Buffer.concat(out)),
          stderr: new Uint8Array(Buffer.concat(err)),
        }));
      child.stdin.on("error", () => {});
      child.stdin.end(options.stdin ? Buffer.from(options.stdin) : undefined);
    });
  }
}

const dir = mkdtempSync(join(tmpdir(), "sandbox-ts-local-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

describe("a user's own transport", () => {
  it("runs the file helpers over it", async () => {
    const t = new LocalTransport();
    const files = new Files(t);
    expect((await files.capabilities()).readWindow).toBe(true);
    await files.write(`${dir}/a.txt`, "hello\nworld needle\n");
    expect(await files.readText(`${dir}/a.txt`)).toBe("hello\nworld needle\n");
    expect(await files.stat(`${dir}/a.txt`)).toMatchObject({
      kind: "file",
      size: 19,
    });
    expect((await files.list(dir)).map((e) => e.name)).toEqual(["a.txt"]);
    const hit = (await files.search({ pattern: "needle", root: dir })).hits[0];
    expect(hit).toMatchObject({ line: 2, text: "world needle", isMatch: true });
    await expect(files.read(`${dir}/nope`)).rejects.toMatchObject({
      kind: "not_found",
    });
  });

  it("says a missing parent directory in plain words, with no temp file or script line", async () => {
    const files = new Files(new LocalTransport());
    const missing = `${dir}/no/such/dir`;
    const err = await files.write(`${missing}/f`, "x").then(
      () => null,
      (e) => e,
    );
    expect(err).toMatchObject({ kind: "not_found" });
    expect(err.detail).toBe(
      `write ${missing}/f: parent directory ${missing} does not exist: No such file or directory (os error 2)`,
    );
    expect(err.detail).not.toMatch(/\.tmp\.|sh: \d+/);
    expect((await files.list(dir)).map((e) => e.name)).not.toContain("no");
  });

  it("writes without a probe; the first read probes once, then one command per window", async () => {
    const t = new LocalTransport();
    const files = new Files(t);
    await files.write(`${dir}/b.txt`, "x");
    expect(t.calls).toBe(1); // write needs nothing and never probes
    await files.readText(`${dir}/b.txt`);
    expect(t.calls).toBe(3); // probe + one read
  });

  it("writes content larger than one read window through stdin, in one command", async () => {
    const t = new LocalTransport();
    const files = new Files(t);
    const data = new Uint8Array(100_000).map((_, i) => i % 251);
    await files.write(`${dir}/big`, data);
    expect(t.calls).toBe(1); // write alone; no probe
    expect(
      Buffer.from(await files.read(`${dir}/big`)).equals(Buffer.from(data)),
    ).toBe(true);
  });

  it("refuses to write through a transport that drops stdin: the size check fails the write", async () => {
    class DropsStdin extends LocalTransport {
      override run(c: string, a: string[], o: RunOptions = {}) {
        const { stdin: _drop, ...rest } = o;
        return super.run(c, a, rest);
      }
    }
    await new Files(new LocalTransport()).write(`${dir}/precious`, "keep me");
    await expect(new Files(new DropsStdin()).write(`${dir}/precious`, "new"))
      .rejects.toMatchObject({
        kind: "io",
        detail: expect.stringContaining("short write"),
      });
    const plain = new Files(new LocalTransport());
    expect(await plain.readText(`${dir}/precious`)).toBe("keep me");
    expect(
      (await plain.list(dir)).map((e) => e.name).filter((n) =>
        n.startsWith("precious")
      ),
    ).toEqual(["precious"]);
  });
});

describe("round trips of the first file operation on a fresh sandbox", () => {
  it("writes with one command; a first read probes once then reads", async () => {
    await new Files(new LocalTransport()).write(`${dir}/rt`, "x");
    const writeT = new LocalTransport();
    await new Files(writeT).write(`${dir}/rt`, "y");
    expect(writeT.calls).toBe(1);
    const readT = new LocalTransport();
    await new Files(readT).read(`${dir}/rt`);
    expect(readT.calls).toBe(2); // probe + read
  });
});
