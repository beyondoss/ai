import { afterEach, describe, expect, it } from "vitest";
import { text, utf8 } from "../src/bytes.js";
import { decodeJobFrame, HttpTransport } from "../src/http.js";
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
    expect(decodeJobFrame(line({ started: 7 }))).toMatchObject({ stdin: false });
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
    const { t } = await jobs(() => [200, { exit_code: 0, stdout: "", stderr: "" }]);
    await expect(t.run("true", [], { env: { A: "1" } })).rejects.toMatchObject({
      kind: "not_offered",
    });
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
