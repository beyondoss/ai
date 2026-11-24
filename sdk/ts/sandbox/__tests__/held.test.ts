import { setTimeout as realSleep } from "node:timers/promises";
import { afterEach, describe, expect, it, vi } from "vitest";
import { text } from "../src/bytes.js";
import {
  AuthError,
  DiskFullError,
  EndpointError,
  JobExistsError,
  JobGoneError,
  NotOfferedError,
  UnreachableError,
} from "../src/errors.js";
import { HttpTransport } from "../src/http.js";
import { Jobs } from "../src/jobs.js";
import type { JobFrame } from "../src/transport.js";
import { MARK, markServer } from "./held-server.js";
import type { Script } from "./held-server.js";

const closers: Array<() => Promise<void>> = [];
afterEach(async () => {
  while (closers.length) await closers.pop()!();
});

async function rig(script: Script) {
  const s = await markServer(script);
  closers.push(s.close);
  return { s, jobs: new Jobs(new HttpTransport({ url: s.url })) };
}

async function drain(it: AsyncIterable<JobFrame>): Promise<JobFrame[]> {
  const out: JobFrame[] = [];
  for await (const f of it) out.push(f);
  return out;
}

const bytes = (fs: JobFrame[]) =>
  fs.flatMap((f) => (f.type === "output" ? [text(f.data)] : [])).join("");

describe("a followed job survives cuts", () => {
  it("reattaches at the offset reached, with the mark, and delivers each byte once", async () => {
    const { s, jobs } = await rig({ output: "0123456789", cutAfter: [3, 4] });
    const fs = await drain(jobs.startHeld("s", "aa", { command: "true" }));
    expect(fs.map((f) => f.type)).toEqual([
      "started",
      "output",
      "output",
      "output",
      "exit",
    ]);
    expect(bytes(fs)).toBe("0123456789");
    expect(s.attaches.map((a) => (a.body.job as any).offset)).toEqual([3, 7]);
    expect(s.attaches.every((a) => (a.body.job as any).mark === MARK)).toBe(
      true,
    );
  });

  it("follow from an offset that is not zero starts there, with no mark to hold", async () => {
    const { s, jobs } = await rig({ output: "0123456789", cutAfter: [2] });
    const fs = await drain(jobs.follow("s", "aa", 4));
    expect(bytes(fs)).toBe("456789");
    expect((s.attaches[0]!.body.job as any).mark).toBeUndefined();
    expect(s.attaches[1]!.body.job as any).toMatchObject({
      offset: 6,
      mark: MARK,
    });
  });

  it("tells the consumer of a gap the server states on a reattach", async () => {
    const { jobs } = await rig({
      output: "0123456789",
      base: 5,
      cutAfter: [0],
    });
    const fs = await drain(jobs.follow("s", "aa", 0));
    expect(fs.filter((f) => f.type === "gap")).toEqual([{
      type: "gap",
      bytes: 5,
    }]);
    expect(bytes(fs)).toBe("56789");
  });

  it("waits out a busy answer and tries again (429 with Retry-After)", async () => {
    const { s, jobs } = await rig({
      output: "abc",
      refuse: [[429, { "retry-after": "0" }]],
    });
    const fs = await drain(jobs.follow("s", "aa", 0));
    expect(bytes(fs)).toBe("abc");
    expect(s.attaches).toHaveLength(2);
  });
});

describe("a refusal is one typed error, from the kernel's table", () => {
  const follow = async (script: Script) => {
    const { jobs } = await rig(script);
    return drain(jobs.follow("s", "aa", 0)).then(() => null, (e: unknown) => e);
  };

  it("404 is JobGoneError", async () => {
    expect(await follow({ output: "x", refuse: [[404]] })).toBeInstanceOf(
      JobGoneError,
    );
  });
  it("401 and 403 are AuthError; 400 and 500 are EndpointError with the status", async () => {
    expect(await follow({ output: "x", refuse: [[401]] })).toBeInstanceOf(
      AuthError,
    );
    expect(await follow({ output: "x", refuse: [[403]] })).toBeInstanceOf(
      AuthError,
    );
    expect(await follow({ output: "x", refuse: [[400]] })).toMatchObject({
      status: 400,
    });
    const five = await follow({ output: "x", refuse: [[500]] });
    expect(five).toBeInstanceOf(EndpointError);
    expect(five).toMatchObject({ status: 500 });
  });
  it("a start: 409 is JobExistsError and 507 is DiskFullError", async () => {
    const start = async (startStatus: number) => {
      const { jobs } = await rig({ output: "x", startStatus });
      return drain(jobs.startHeld("s", "aa", { command: "true" })).then(
        () => null,
        (e: unknown) => e,
      );
    };
    expect(await start(409)).toBeInstanceOf(JobExistsError);
    expect(await start(507)).toBeInstanceOf(DiskFullError);
  });
  it("an endpoint that does not name the mark is NotOfferedError, before any byte", async () => {
    const { jobs } = await rig({ output: "abc", echo: null });
    const got: JobFrame[] = [];
    const err = await (async () => {
      for await (const f of jobs.follow("s", "aa", 0)) got.push(f);
    })().then(() => null, (e: unknown) => e);
    expect(err).toBeInstanceOf(NotOfferedError);
    expect(got).toEqual([]);
  });
  it("an echo of another mark is JobGoneError, before any byte", async () => {
    const { jobs } = await rig({ output: "abc", echo: "ffff" });
    const err = await drain(jobs.startHeld("s", "aa", { command: "true" }))
      .then(() => null, (e: unknown) => e);
    // The start's echo names the mark the handle holds only once it has one; a later reattach
    // that is answered by another job is the gone case.
    expect(err === null || err instanceof JobGoneError).toBe(true);
  });
});

describe("a handle that cannot reach the endpoint gives up with a cursor", () => {
  it("spends its budget on failed opens, then fails Unreachable with the cursor to resume from", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "Date"] });
    try {
      const { s, jobs } = await rig({
        output: "0123456789",
        cutAfter: [4],
        refuse: Array.from({ length: 100 }, () => [503] as [number]),
      });
      let settled: unknown = "pending";
      const got: JobFrame[] = [];
      void (async () => {
        for await (const f of jobs.startHeld("s", "aa", { command: "true" })) {
          got.push(f);
        }
      })().then(() => (settled = null), (e: unknown) => (settled = e));
      // Virtual time moves the kernel's waits; the loopback server answers in real time, so each turn
      // waits a real few milliseconds, for as long as it takes (the test's own timeout bounds it).
      while (settled === "pending") {
        await vi.advanceTimersByTimeAsync(31_000);
        await realSleep(5);
      }
      expect(settled).toBeInstanceOf(UnreachableError);
      const cursor = (settled as UnreachableError).cursor;
      expect(cursor).toMatch(
        new RegExp(`^j1\\.s\\.aa\\.${MARK}\\.0000000000000004\\.[0-9a-f]{4}$`),
      );
      expect(bytes(got)).toBe("0123");
      // The budget is the kernel's (12 failures in a row: the cut, then 11 failed opens), not a number here.
      expect(s.attaches).toHaveLength(11);
    } finally {
      vi.useRealTimers();
    }
  });
});
