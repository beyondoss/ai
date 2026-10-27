import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { Files, Jobs, Sandbox } from "../src/index.js";
import type { RunOptions, RunResult, Transport } from "../src/index.js";

/**
 * Bring your own transport: raw exec on this machine, no HTTP anywhere. Everything in layers 2
 * and 3 works over it because it is built on `run` alone.
 */
class LocalTransport implements Transport {
  calls = 0;

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
const dec = (b: Uint8Array) => new TextDecoder().decode(b);

describe("a user's own transport", () => {
  it("runs the file helpers over it", async () => {
    const t = new LocalTransport();
    const files = new Files(t);
    expect((await files.capabilities()).stdin).toBe(true);
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

  it("does not stream or keep jobs natively, and says so", async () => {
    const sb = new Sandbox(new LocalTransport());
    expect(() => sb.stream("true")).toThrowError(/does not stream/);
    expect(await sb.jobs.capabilities()).toBe("baseline");
  });

  it("runs the baseline jobs over it", async () => {
    const jobs = new Jobs(new LocalTransport(), { root: `${dir}/jobs` });
    const scope = "local-scope";
    try {
      await jobs.start(scope, "j1", {
        command: "sh",
        args: ["-c", "echo out; echo err >&2; exit 2"],
      });
      const done = await jobs.attach(scope, "j1");
      expect(done.exitCode).toBe(2);
      expect(dec(done.output).split("\n").sort()).toEqual(["", "err", "out"]);
    } finally {
      await jobs.wipe(scope);
    }
  });

  it("falls back to chunked writes when stdin is not delivered, and refuses to write through a lie", async () => {
    class NoStdin extends LocalTransport {
      override run(c: string, a: string[], o: RunOptions = {}) {
        const { stdin: _drop, ...rest } = o;
        return super.run(c, a, rest);
      }
    }
    const files = new Files(new NoStdin());
    expect((await files.capabilities()).stdin).toBe(false);
    const data = new Uint8Array(100_000).map((_, i) => i % 251);
    await files.write(`${dir}/chunked`, data);
    expect(
      Buffer.from(await files.read(`${dir}/chunked`)).equals(Buffer.from(data)),
    ).toBe(true);

    // A transport that passed the probe once and then drops stdin: the size check fails the write.
    let probes = 0;
    class Flaky extends LocalTransport {
      override run(c: string, a: string[], o: RunOptions = {}) {
        if (c === "cat" && probes++ === 0) return super.run(c, a, o);
        const { stdin: _drop, ...rest } = o;
        return super.run(c, a, rest);
      }
    }
    await new Files(new LocalTransport()).write(`${dir}/precious`, "keep me");
    const flaky = new Files(new Flaky());
    expect((await flaky.capabilities()).stdin).toBe(true);
    await expect(flaky.write(`${dir}/precious`, "new content")).rejects
      .toMatchObject({
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
