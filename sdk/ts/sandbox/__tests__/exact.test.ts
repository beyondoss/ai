import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { fromBase64, text, toBase64, utf8 } from "../src/bytes.js";
import {
  bodyCap,
  decodeFrameLine,
  decodeOutput,
  HttpTransport,
  OneShot,
  OUTPUT_PROBE_BYTES,
  OUTPUT_PROBE_SCRIPT,
  probeHonoured,
} from "../src/http.js";
import type { RunResult } from "../src/transport.js";
import { fakeEndpoint, json, ok } from "./fake-endpoint.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(
  readFileSync(
    join(
      here,
      "../../../../crates/sandbox/tests/fixtures/exec/v1.1-output.json",
    ),
    "utf8",
  ),
) as {
  probe: { script: string; bytes_base64: string };
  probes: Array<
    {
      name: string;
      answer: unknown;
      supported?: boolean;
      error?: boolean;
    }
  >;
  answers: Array<{ name: string; answer: unknown; expect: Expect }>;
  streams: Array<{ name: string; lines: string[]; expect: Expect }>;
};

interface Expect {
  invalid?: boolean;
  exit_code?: number;
  stdout_base64?: string;
  stderr_base64?: string;
  exact?: boolean;
}

function check(name: string, got: () => RunResult, want: Expect): void {
  if (want.invalid) {
    expect(got, `${name}: a conforming client reports an error`).toThrow();
    return;
  }
  const r = got();
  expect(r.exitCode, `${name}: exit code`).toBe(want.exit_code);
  expect(toBase64(r.stdout), `${name}: stdout`).toBe(want.stdout_base64);
  expect(toBase64(r.stderr), `${name}: stderr`).toBe(want.stderr_base64);
  expect(r.exact, `${name}: exact`).toBe(want.exact);
}

describe("the output_base64 vectors (crates/sandbox/tests/fixtures/exec/v1.1-output.json)", () => {
  it("the probe is the bytes the vectors name", () => {
    expect(OUTPUT_PROBE_SCRIPT).toBe(fixture.probe.script);
    expect(toBase64(OUTPUT_PROBE_BYTES)).toBe(fixture.probe.bytes_base64);
  });

  for (const c of fixture.probes) {
    it(`probe: ${c.name}`, () => {
      const v = c.answer;
      const obj = typeof v === "object" && v !== null && !Array.isArray(v)
        ? v as Record<string, unknown>
        : null;
      if (c.error) {
        // Not a v1.1 answer: parsing it, or its missing exit code, is an error.
        expect(() => {
          if (!obj) throw new Error("not an object");
          decodeOutput(obj, "http://x", 1 << 20);
        }).toThrow();
      } else {
        expect(probeHonoured(obj!)).toBe(c.supported);
      }
    });
  }

  for (const c of fixture.answers) {
    it(`answer: ${c.name}`, () => {
      check(
        c.name,
        () =>
          decodeOutput(
            c.answer as Record<string, unknown>,
            "http://x",
            1 << 20,
          ),
        c.expect,
      );
    });
  }

  for (const c of fixture.streams) {
    it(`stream: ${c.name}`, () => {
      check(
        c.name,
        () => {
          const fold = new OneShot();
          for (const line of c.lines) {
            for (const f of decodeFrameLine(utf8(line))) {
              if (f === "heartbeat") continue;
              if (f.type === "error") throw new Error(f.message);
              const done = fold.feed(f);
              if (done) return done;
            }
          }
          throw new Error("the body ended without a final frame");
        },
        c.expect,
      );
    });
  }

  it("the cap is on the decoded bytes of both streams together", () => {
    const a = { exit_code: 0, stdout_base64: "AAAA", stderr_base64: "AAAA" };
    expect(decodeOutput(a, "u", 6).exact).toBe(true);
    expect(() => decodeOutput(a, "u", 5)).toThrowError(/the command ran/);
  });

  it("the body bound holds the base64 of a cap's worth of output", () => {
    expect(bodyCap(16 * 1024 * 1024)).toBeGreaterThanOrEqual(22_369_620);
    expect(bodyCap(3)).toBe(4 + 4096);
  });
});

const closers: Array<() => Promise<void>> = [];
afterEach(async () => {
  while (closers.length) await closers.pop()!();
});
async function endpoint(...a: Parameters<typeof fakeEndpoint>) {
  const e = await fakeEndpoint(...a);
  closers.push(e.close);
  return e;
}

const ALL = Uint8Array.from({ length: 256 }, (_, i) => i);
const b64 = (s: Uint8Array | string) =>
  toBase64(typeof s === "string" ? utf8(s) : s);

/** An endpoint that honours output_base64: answers the probe and every call with base64. */
const honours = (bytes: Uint8Array) =>
(
  seen: { body: Record<string, unknown> },
  res: import("node:http").ServerResponse,
) => {
  if (seen.body.output_base64 !== true) return json(res, 200, ok("text"));
  const probe = String((seen.body.args as string[])[1]).includes("377");
  json(res, 200, {
    exit_code: 0,
    stdout_base64: probe ? b64(OUTPUT_PROBE_BYTES) : b64(bytes),
    stderr_base64: "",
  });
};

describe("HttpTransport.run returns bytes", () => {
  it("sends output_base64 and returns every byte value exactly, probing once", async () => {
    const e = await endpoint(honours(ALL));
    const t = new HttpTransport({ url: e.url });
    const r = await t.run("gen", []);
    expect(r.exact).toBe(true);
    expect(Array.from(r.stdout)).toEqual(Array.from(ALL));
    await t.run("gen", []);
    // probe, call, call: the probe is once per connection.
    expect(e.requests.map((x) => x.body.output_base64)).toEqual([
      true,
      true,
      true,
    ]);
    expect(e.requests[0]!.body.args).toEqual(["-c", OUTPUT_PROBE_SCRIPT]);
  });

  it("text: true skips the probe and answers the plain way", async () => {
    const e = await endpoint((_, res) => json(res, 200, ok("plain")));
    const r = await new HttpTransport({ url: e.url }).run("x", [], {
      text: true,
    });
    expect(text(r.stdout)).toBe("plain");
    expect(r.exact).toBe(false);
    expect(e.requests).toHaveLength(1);
    expect(e.requests[0]!.body.output_base64).toBeUndefined();
  });

  it("a plain v1.1 endpoint answers text, and the result says it is not exact", async () => {
    const e = await endpoint((_, res) => json(res, 200, ok("café �", 4, "e")));
    const r = await new HttpTransport({ url: e.url }).run("x", []);
    expect(r.exact).toBe(false);
    expect(r.exitCode).toBe(4);
    expect(text(r.stdout)).toBe("café �");
    expect(text(r.stderr)).toBe("e");
  });

  it("a probe whose request failed is an error, never 'not supported'", async () => {
    const e = await endpoint((_, res) => json(res, 500, {}));
    const t = new HttpTransport({ url: e.url });
    expect(await t.run("x", []).catch((x: { kind: string }) => x.kind)).toBe(
      "endpoint",
    );
    expect(e.requests).toHaveLength(1); // the call did not run as text instead
    expect(await t.run("x", []).catch((x: { kind: string }) => x.kind)).toBe(
      "endpoint",
    );
    expect(e.requests).toHaveLength(2); // and the probe was asked again
  });

  it("a strict front that answers the field with 400 has answered: not supported", async () => {
    const e = await endpoint((seen, res) => {
      if (seen.body.output_base64 || seen.body.stream) {
        res.writeHead(400);
        res.end("unknown field");
        return;
      }
      json(res, 200, ok("hi"));
    });
    const r = await new HttpTransport({ url: e.url }).run("x", []);
    expect(text(r.stdout)).toBe("hi");
    expect(r.exact).toBe(false);
  });

  it("an endpoint that only streams is exact over the stream", async () => {
    const e = await endpoint((seen, res) => {
      if (seen.body.output_base64) return json(res, 200, ok(""));
      if (seen.body.job) return json(res, 200, ok(""));
      res.writeHead(200, { "content-type": "application/x-ndjson" });
      const args = String((seen.body.args as string[])[1]);
      if (args.startsWith("printf a;")) {
        res.write(JSON.stringify({ stdout: b64("a") }) + "\n");
        setTimeout(
          () =>
            res.end(
              JSON.stringify({ stdout: b64("b") }) + "\n{\"exit_code\":0}\n",
            ),
          200,
        );
        return;
      }
      res.write(JSON.stringify({ stdout: b64(ALL) }) + "\n");
      res.write(
        JSON.stringify({ stderr: b64(new Uint8Array([255, 0])) }) + "\n",
      );
      res.end("{\"signal\":9}\n");
    });
    const r = await new HttpTransport({ url: e.url }).run("gen", []);
    expect(r.exact).toBe(true);
    expect(r.exitCode).toBe(137);
    expect(Array.from(r.stdout)).toEqual(Array.from(ALL));
    expect(Array.from(r.stderr)).toEqual([255, 0]);
  });

  it("a call with stdin on an endpoint without output_base64 is text, not exact", async () => {
    const e = await endpoint((_, res) => json(res, 200, ok("echo")));
    const r = await new HttpTransport({ url: e.url }).run("cat", [], {
      stdin: new Uint8Array([255]),
    });
    expect(r.exact).toBe(false);
  });

  it("output past the cap is too_large and says the command ran", async () => {
    const e = await endpoint((seen, res) => {
      if (seen.body.output_base64 !== true) return json(res, 200, ok(""));
      const probe = String((seen.body.args as string[])[1]).includes("377");
      json(res, 200, {
        exit_code: 0,
        stdout_base64: probe
          ? b64(OUTPUT_PROBE_BYTES)
          : b64(new Uint8Array(600)),
        stderr_base64: probe ? "" : b64(new Uint8Array(600)),
      });
    });
    const t = new HttpTransport({ url: e.url, maxResponseBytes: 1000 });
    await expect(t.run("x", [])).rejects.toMatchObject({
      kind: "too_large",
      detail: expect.stringContaining("the command ran"),
    });
  });

  it("fromBase64 is strict about padding and alphabet", () => {
    expect(fromBase64("/wA")).toBeNull();
    expect(fromBase64("_-8=")).toBeNull();
    expect(fromBase64("/wAK")).not.toBeNull();
  });
});
