import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpTransport } from "../src/http.js";
import { Jobs } from "../src/jobs.js";
import { markServer } from "./held-server.js";

// The same module with one constant of the kernel's table changed: 429 becomes 428 (`i32.const`, LEB128).
// The retry of a busy answer in `held.test.ts` ("waits out a busy answer") goes red here, so the
// behaviour of the handle comes from the WASM and from nothing the shell holds.
vi.mock("../src/kernel/wasm.js", async (original) => {
  const real = await original<typeof import("../src/kernel/wasm.js")>();
  const b = Buffer.from(real.KERNEL_WASM_BASE64, "base64");
  let patched = 0;
  for (let i = 0; i + 2 < b.length; i++) {
    if (b[i] === 0x41 && b[i + 1] === 0xad && b[i + 2] === 0x03) {
      b[i + 1] = 0xac;
      patched++;
    }
  }
  if (patched === 0) {
    throw new Error("the kernel has no 429 to break: update this test");
  }
  return { KERNEL_WASM_BASE64: b.toString("base64") };
});

const closers: Array<() => Promise<void>> = [];
afterEach(async () => {
  while (closers.length) await closers.pop()!();
});

describe("a deliberately broken kernel", () => {
  it("is not the kernel the TypeScript path runs: 429 is no longer waited out", async () => {
    const s = await markServer({
      output: "abc",
      refuse: [[429, { "retry-after": "0" }]],
    });
    closers.push(s.close);
    const jobs = new Jobs(new HttpTransport({ url: s.url }));
    const run = async () => {
      for await (const _ of jobs.follow("s", "aa", 0)) { /* drain */ }
    };
    // With the real kernel this resolves after a second attach (held.test.ts).
    await expect(run()).rejects.toMatchObject({ status: 429 });
    expect(s.attaches).toHaveLength(1);
  });
});
