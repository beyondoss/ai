import { afterEach, describe, expect, it } from "vitest";
import { text, utf8 } from "../src/bytes.js";
import {
  ConflictError,
  DiskFullError,
  IoError,
  JobExistsError,
  JobGoneError,
  KeyConflictError,
  NotFoundError,
  StdinBusyError,
  StdinClosedError,
  StdinGapError,
} from "../src/errors.js";
import { decodeJobFrame, HttpTransport } from "../src/http.js";
import { Jobs } from "../src/jobs.js";
import { silenceMs } from "../src/transport.js";
import { fakeEndpoint, json } from "./fake-endpoint.js";

const closers: Array<() => Promise<void>> = [];
afterEach(async () => {
  while (closers.length) await closers.pop()!();
});
const b64 = (s: string) => Buffer.from(s).toString("base64");
const line = (v: unknown) => utf8(JSON.stringify(v));

describe("job frames keep their stream", () => {
  it("decodes stdout and stderr frames apart, and the stdin echo", () => {
    expect(decodeJobFrame(line({ stdout: b64("a") }))).toMatchObject({
      type: "output",
      stream: "stdout",
    });
    const e = decodeJobFrame(line({ stderr: b64("b") }));
    expect(e).toMatchObject({ type: "output", stream: "stderr" });
    expect(decodeJobFrame(line({ started: 7, stdin: true }))).toEqual({
      type: "started",
      pgid: 7,
      stdin: true,
    });
    expect(decodeJobFrame(line({ started: 7 }))).toMatchObject({
      stdin: false,
    });
  });
});

describe("job ops over HTTP", () => {
  async function jobs(answer: (body: any) => [number, unknown]) {
    const seen: any[] = [];
    const e = await fakeEndpoint((r, res) => {
      const b = r.body as any;
      seen.push(b);
      const [status, v] = answer(b);
      json(res, status, v);
    });
    closers.push(e.close);
    return { t: new HttpTransport({ url: e.url }), seen };
  }

  it("reads stderr_ranges", async () => {
    const { t } = await jobs(() => [200, {
      state: "ended",
      code: 0,
      offset: 0,
      end: 9,
      data: b64("AAABBBCCC"),
      stderr_ranges: [[3, 3]],
    }]);
    const c = await t.jobs.read("s", "aa", 0, 100);
    expect(c.stderrRanges).toEqual([[3, 3]]);
    expect(text(c.data)).toBe("AAABBBCCC");
  });

  it("writes positionally and maps the errors", async () => {
    const { t, seen } = await jobs((b) =>
      b.job.at === 0
        ? [200, { written: 3 }]
        : b.job.at === 1
        ? [409, { error: "stdin is closed", written: 3 }]
        : [404, { error: "no such job" }]
    );
    expect(await t.jobs.writeStdin("s", "aa", 0, utf8("abc"), true)).toEqual({
      written: 3,
      closed: false,
    });
    expect(seen[0].job).toMatchObject({ op: "write", at: 0, eof: true });
    const kind = async (p: Promise<unknown>) =>
      p.then(() => "ok", (e) => e.kind);
    expect(await kind(t.jobs.writeStdin("s", "aa", 1, utf8("x"), false))).toBe(
      "conflict",
    );
    expect(await kind(t.jobs.writeStdin("s", "aa", 2, utf8("x"), false))).toBe(
      "not_found",
    );
    expect(
      await kind(
        t.jobs.writeStdin("s", "aa", 0, new Uint8Array(128 * 1024 + 1), false),
      ),
    ).toBe("invalid_request");
  });

  it("signals by name only", async () => {
    const { t, seen } = await jobs(() => [200, { sent: ["aa"] }]);
    expect(await t.jobs.signal("s", ["aa"], "SIGINT")).toEqual(["aa"]);
    expect(seen[0].job).toMatchObject({ op: "signal", signal: "SIGINT" });
    await expect(t.jobs.signal("s", ["aa"], "SIGSTOP")).rejects.toMatchObject({
      kind: "invalid_request",
    });
  });

  it("fails closed when an endpoint ignores env", async () => {
    const { t } = await jobs(
      () => [200, { exit_code: 0, stdout: "", stderr: "" }],
    );
    await expect(t.run("true", [], { env: { A: "1" } })).rejects.toMatchObject({
      kind: "not_offered",
    });
  });

  it("sends an idempotency key: one per start call, the caller's when given, checked", async () => {
    const { t, seen } = await jobs(() => [200, { started: 5 }]);
    const jobsApi = new Jobs(t);
    await jobsApi.start("s", "aa", { command: "true" });
    await jobsApi.start("s", "ab", { command: "true" });
    const [first, second] = seen.map((b) => b.job.idempotency_key);
    expect(first).not.toEqual(second);
    for (const k of [first, second]) {
      expect(k).toMatch(/^[0-9A-Za-z._:-]{1,128}$/);
    }
    await jobsApi.start("s", "ac", {
      command: "true",
      idempotencyKey: "deploy-2026-11-24",
    });
    expect(seen[2].job.idempotency_key).toBe("deploy-2026-11-24");
    for (const bad of ["", "a b", "k/1", "k".repeat(129)]) {
      await expect(
        jobsApi.start("s", "ad", { command: "true", idempotencyKey: bad }),
      ).rejects.toMatchObject({ kind: "invalid_request" });
    }
    expect(seen).toHaveLength(3);
  });

  it("kills a job whose stdin_open was ignored", async () => {
    const { t, seen } = await jobs((b) =>
      b.job.op === "start" ? [200, { started: 5 }] : [200, { survivors: [] }]
    );
    await expect(
      t.jobs.start("s", "aa", { command: "cat", stdinOpen: true }),
    ).rejects.toMatchObject({ kind: "not_offered" });
    expect(seen.map((b) => b.job.op)).toEqual(["start", "kill"]);
  });
});

describe("typed job errors (each a subclass of today's kind)", () => {
  async function failing(status: number, body: unknown) {
    const e = await fakeEndpoint((_r, res) => json(res, status, body));
    closers.push(e.close);
    return new HttpTransport({ url: e.url });
  }
  const fail = (p: Promise<unknown>) => p.then(() => undefined, (e) => e);

  it("start: a 409 key_conflict is KeyConflictError, a ConflictError that is not JobExistsError", async () => {
    const e = await fail(
      (await failing(409, {
        error: "the key was used for a different command or id",
        code: "key_conflict",
      })).jobs.start("s", "aa", { command: "true" }),
    );
    expect(e).toBeInstanceOf(KeyConflictError);
    expect(e).toBeInstanceOf(ConflictError);
    expect(e).not.toBeInstanceOf(JobExistsError);
  });

  it("start: 409 is JobExistsError, 507 is DiskFullError", async () => {
    const exists = await fail(
      (await failing(409, { error: "the job exists" })).jobs.start("s", "aa", {
        command: "true",
      }),
    );
    expect(exists).toBeInstanceOf(JobExistsError);
    expect(exists).toBeInstanceOf(ConflictError);
    expect(exists).toMatchObject({ kind: "conflict", id: "aa" });
    const full = await fail(
      (await failing(507, { error: "the sandbox's disk is full" })).jobs.start(
        "s",
        "aa",
        { command: "true" },
      ),
    );
    expect(full).toBeInstanceOf(DiskFullError);
    expect(full).toBeInstanceOf(IoError);
    expect(full).toMatchObject({ kind: "io" });
  });

  it("attach 404 and read `gone` are the same JobGoneError (a NotFoundError)", async () => {
    const t = await failing(404, { error: "no such job" });
    const gone = await fail(
      (async () => {
        for await (const _ of t.jobs.attach("s", "aa", 0)) { /* none */ }
      })(),
    );
    expect(gone).toBeInstanceOf(JobGoneError);
    expect(gone).toBeInstanceOf(NotFoundError);
    expect(gone).toMatchObject({ kind: "not_found", id: "aa" });
    const read = await failing(200, {
      state: "gone",
      code: null,
      offset: 0,
      end: 0,
      data: "",
    });
    // Level 3 keeps the wire's `gone`; level 2 says JobGoneError.
    expect(await read.jobs.read("s", "aa", 0, 10)).toMatchObject({
      state: "gone",
    });
    const sb = new Jobs(read);
    expect(await fail(sb.read("s", "aa"))).toBeInstanceOf(JobGoneError);
  });

  it("stdin: told apart as the reference client tells them (jobs_native.rs write_stdin)", async () => {
    const w = (status: number, body: unknown) =>
      failing(status, body).then((t) =>
        fail(t.jobs.writeStdin("s", "aa", 0, utf8("x"), false))
      );
    const never = await w(409, { error: "the job has no stdin" });
    expect(never).toBeInstanceOf(StdinClosedError);
    expect(never).toMatchObject({ reason: "never_opened", kind: "conflict" });
    const closed = await w(409, { error: "stdin is closed", written: 7 });
    expect(closed).toBeInstanceOf(StdinClosedError);
    expect(closed).toMatchObject({ reason: "closed", written: 7 });
    const gap = await w(409, { error: "gap", written: 2 });
    expect(gap).toBeInstanceOf(StdinGapError);
    expect(gap).toMatchObject({ written: 2 });
    const busy = await w(429, { error: "too many writes wait" });
    expect(busy).toBeInstanceOf(StdinBusyError);
    for (const e of [never, closed, gap, busy]) {
      expect(e).toBeInstanceOf(ConflictError);
    }
    expect(busy).not.toBeInstanceOf(StdinClosedError);
    const none = await w(404, { error: "no such job" });
    expect(none).toBeInstanceOf(JobGoneError);
  });
});

describe("the wire options, under their wire names", () => {
  async function recorder(answer: unknown = {}) {
    const e = await fakeEndpoint((_r, res) => json(res, 200, answer));
    closers.push(e.close);
    return { t: new HttpTransport({ url: e.url }), e };
  }

  it("keep, seen and kill/wipe grace_ms", async () => {
    const { t, e } = await recorder({ survivors: [] });
    await t.jobs.keep("s", "aa", { num: 3, kind: "background", seen: 120 });
    await t.jobs.seen("s", "aa", 9);
    await t.jobs.kill("s", ["aa"], { remove: true, graceMs: 2500 });
    await t.jobs.wipe("s", { graceMs: 7 });
    expect(e.requests.map((r) => r.body.job)).toEqual([
      {
        op: "keep",
        scope: "s",
        id: "aa",
        num: 3,
        kind: "background",
        seen: 120,
      },
      { op: "seen", scope: "s", id: "aa", offset: 9 },
      { op: "kill", scope: "s", ids: ["aa"], remove: true, grace_ms: 2500 },
      { op: "wipe", scope: "s", grace_ms: 7 },
    ]);
    // Layer 2: the default is 300 ms, and a named grace goes through.
    const jobs = new Jobs(t);
    await jobs.kill("s", ["aa"]);
    await jobs.kill("s", ["aa"], { graceMs: 50 });
    await jobs.wipe("s", { graceMs: 20 });
    const graces = e.requests.slice(4).map((r) => (r.body.job as any).grace_ms);
    expect(graces).toEqual([300, 50, 20]);
  });

  it("a held start sends hold: true with stream, and the heartbeat header", async () => {
    const e = await fakeEndpoint((_r, res) => {
      res.writeHead(200, { "content-type": "application/x-ndjson" });
      res.end("{\"started\":42}\n{\"exit_code\":0}\n");
    });
    closers.push(e.close);
    const t = new HttpTransport({ url: e.url });
    const frames = [];
    for await (
      const f of t.jobs.startHeld("s", "aa", { command: "true" }, {
        heartbeatSeconds: 60,
      })
    ) frames.push(f);
    expect(frames.map((f) => f.type)).toEqual(["started", "exit"]);
    expect(e.requests[0]!.body).toMatchObject({
      stream: true,
      job: { op: "start", hold: true },
    });
    expect(e.requests[0]!.headers["keepalive-ping-interval"]).toBe("60");
  });

  it("Keepalive-Ping-Interval is 1 to 240 whole seconds, checked before anything is sent", async () => {
    const { t, e } = await recorder({ jobs: [] });
    await t.jobs.list("s", { heartbeatSeconds: 240 });
    expect(e.requests[0]!.headers["keepalive-ping-interval"]).toBe("240");
    await t.jobs.list("s");
    expect(e.requests[1]!.headers["keepalive-ping-interval"]).toBeUndefined();
    for (const bad of [0, 241, 1.5, -1, Number.NaN]) {
      await expect(t.jobs.list("s", { heartbeatSeconds: bad })).rejects
        .toMatchObject({ kind: "invalid_request" });
    }
    expect(e.requests).toHaveLength(2);
  });

  it("the silence limit follows the heartbeat the request chose", () => {
    expect(silenceMs()).toBe(30_000);
    expect(silenceMs(1)).toBe(30_000);
    expect(silenceMs(5)).toBe(30_000);
    // A 60 s beat would be killed by today's 30 s limit; two beats and slack hold.
    expect(silenceMs(60)).toBe(130_000);
    expect(silenceMs(240)).toBe(490_000);
  });
});
