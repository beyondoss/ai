import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { Kernel } from "../src/kernel/kernel.js";
import { KERNEL_WASM_BASE64 } from "../src/kernel/wasm.js";

const crates = join(__dirname, "../../../../crates");

describe("the inlined kernel is the one the sources build", () => {
  it("kernel.src.sha256 is the hash of the sources: a source changed without crates/job-client-ffi/build.sh", () => {
    // Skipped from a published package, which has no crates beside it; CI runs it in the repo.
    expect(existsSync(crates)).toBe(true);
    const rs = (dir: string) =>
      readdirSync(join(crates, dir, "src")).filter((f) => f.endsWith(".rs"))
        .map((f) => `${dir}/src/${f}`);
    const files = [
      "job-client-ffi/Cargo.toml",
      "job-client/Cargo.toml",
      ...[...rs("job-client-ffi"), ...rs("job-client")].sort(),
    ];
    const h = createHash("sha256");
    for (const f of files) h.update(readFileSync(join(crates, f)));
    const pinned = readFileSync(
      join(__dirname, "../src/kernel/kernel.src.sha256"),
      "utf8",
    ).trim();
    expect(h.digest("hex")).toBe(pinned);
  });

  it("is a module with no imports, in the size the package carries", () => {
    const bytes = Buffer.from(KERNEL_WASM_BASE64, "base64");
    const m = new WebAssembly.Module(bytes);
    expect(WebAssembly.Module.imports(m)).toEqual([]);
    expect(bytes.length).toBeLessThan(64 * 1024);
  });
});

describe("the kernel's ABI from TypeScript", () => {
  it("classifies one table for every status", async () => {
    const k = await Kernel.load();
    expect(k.classify("attach", 200)).toEqual({ t: "ok" });
    expect(k.classify("attach", 429)).toEqual({ t: "retry" });
    expect(k.classify("attach", 404)).toEqual({
      t: "terminal",
      fault: "job_gone",
    });
    expect(k.classify("start", 409)).toEqual({
      t: "terminal",
      fault: "job_exists",
    });
    expect(k.classify("start", 507)).toEqual({
      t: "terminal",
      fault: "disk_full",
    });
    expect(k.classify("attach", 401)).toEqual({ t: "terminal", fault: "auth" });
    expect(k.classify("remove", 404)).toEqual({ t: "ok" });
  });

  it("the silence limit follows the heartbeat", async () => {
    const k = await Kernel.load();
    expect([0, 1, 10, 11, 60, 240].map((b) => k.silenceMs(b))).toEqual([
      30_000,
      30_000,
      30_000,
      32_000,
      130_000,
      490_000,
    ]);
  });

  it("a state resumed from a cursor reopens at its offset and mark; a bad cursor is refused", async () => {
    const k = await Kernel.load();
    const s = k.stateForAttach("00ff1a", 4096);
    const cursor = k.cursorOf(s, "sess-1", "a1b2c3");
    expect(cursor).toMatch(
      /^j1\.sess-1\.a1b2c3\.00ff1a\.0000000000001000\.[0-9a-f]{4}$/,
    );
    const r = k.step(k.stateForCursor(cursor), { t: "go" }, 0, 0);
    expect(r.actions).toEqual([{
      t: "openAttach",
      offset: 4096,
      mark: "00ff1a",
      silenceMs: 30_000,
    }]);
    expect(() => k.stateForCursor(cursor.slice(0, -1))).toThrow(/refused/);
    expect(() => k.stateForAttach("not hex", 0)).toThrow(/refused/);
  });

  it("leaks nothing across 20 000 calls", async () => {
    const k = await Kernel.load();
    const s = k.stateForAttach("00ff1a", 0);
    const before =
      (k as unknown as { x: { memory: WebAssembly.Memory } }).x.memory.buffer
        .byteLength;
    for (let i = 0; i < 20_000; i++) {
      k.step(s, { t: "frame", frame: { t: "heartbeat" } }, i, i);
      k.classify("attach", 429);
    }
    const after =
      (k as unknown as { x: { memory: WebAssembly.Memory } }).x.memory.buffer
        .byteLength;
    expect(after).toBe(before);
  });
});
