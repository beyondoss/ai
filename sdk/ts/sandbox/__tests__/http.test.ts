import { afterEach, describe, expect, it } from "vitest";
import { text, utf8 } from "../src/bytes.js";
import { decodeFrameLine, HttpTransport } from "../src/http.js";
import type { Frame } from "../src/transport.js";
import { fakeEndpoint, json, ok } from "./fake-endpoint.js";

const closers: Array<() => Promise<void>> = [];
afterEach(async () => {
  while (closers.length) await closers.pop()!();
});
async function endpoint(...a: Parameters<typeof fakeEndpoint>) {
  const e = await fakeEndpoint(...a);
  closers.push(e.close);
  return e;
}

const kindOf = async (p: Promise<unknown>): Promise<string> => {
  try {
    await p;
    return "no error";
  } catch (e) {
    return (e as { kind?: string }).kind ?? `not a SandboxError: ${String(e)}`;
  }
};

describe("HttpTransport.run", () => {
  it("posts the request envelope with the bearer and returns bytes", async () => {
    const e = await endpoint((_, res) => json(res, 200, ok("héllo", 3, "err")));
    const t = new HttpTransport({
      url: e.url,
      token: "sekret",
      headers: { "x-tenant": "acme" },
    });
    const r = await t.run("echo", ["a", "b"], {
      cwd: "/w",
      timeoutMs: 5000,
      stdin: utf8("in"),
    });
    expect(r.exitCode).toBe(3);
    expect(text(r.stdout)).toBe("héllo");
    expect(text(r.stderr)).toBe("err");
    const [seen] = e.requests;
    expect(seen!.headers.authorization).toBe("Bearer sekret");
    expect(seen!.headers["x-tenant"]).toBe("acme");
    expect(seen!.body).toEqual({
      command: "echo",
      args: ["a", "b"],
      cwd: "/w",
      timeout_ms: 5000,
      stdin: "aW4=",
    });
  });

  it("sends no Authorization without a token, and no cwd or stdin unless given", async () => {
    const e = await endpoint((_, res) => json(res, 200, ok()));
    await new HttpTransport({ url: e.url }).run("true", []);
    expect(e.requests[0]!.headers.authorization).toBeUndefined();
    expect(e.requests[0]!.body).toEqual({
      command: "true",
      args: [],
      timeout_ms: 120000,
    });
  });

  it("refreshes a function token once on 401 and retries that request once", async () => {
    let calls = 0;
    const e = await endpoint((seen, res) =>
      seen.headers.authorization === "Bearer t2"
        ? json(res, 200, ok("fine"))
        : json(res, 401, {})
    );
    const t = new HttpTransport({
      url: e.url,
      token: async () => `t${++calls}`,
    });
    expect(text((await t.run("x", [])).stdout)).toBe("fine");
    expect(calls).toBe(2);
    expect(e.requests.map((r) => r.headers.authorization)).toEqual([
      "Bearer t1",
      "Bearer t2",
    ]);
    // The refreshed token is kept: no second refresh for the next request.
    await t.run("x", []);
    expect(calls).toBe(2);
  });

  it("fails with auth when the refreshed token is refused too, after exactly one retry", async () => {
    const e = await endpoint((_, res) => json(res, 401, {}));
    const t = new HttpTransport({ url: e.url, token: async () => "nope" });
    expect(await kindOf(t.run("x", []))).toBe("auth");
    expect(e.requests).toHaveLength(2);
  });

  it("never retries with a string token, or on 403", async () => {
    const e = await endpoint((_, res) => json(res, 401, {}));
    expect(
      await kindOf(new HttpTransport({ url: e.url, token: "x" }).run("x", [])),
    ).toBe("auth");
    expect(e.requests).toHaveLength(1);
    const f = await endpoint((_, res) => json(res, 403, {}));
    let calls = 0;
    const t = new HttpTransport({
      url: f.url,
      token: async () => `t${++calls}`,
    });
    expect(await kindOf(t.run("x", []))).toBe("auth");
    expect(f.requests).toHaveLength(1);
  });

  it("maps statuses and bodies to the spec's kinds", async () => {
    const cases: Array<[number, string, string]> = [
      [500, "boom", "endpoint"],
      [404, "", "endpoint"],
      [
        422,
        "the command ran, but its output exceeded the 5-byte response cap",
        "too_large",
      ],
      [422, "something else", "endpoint"],
      [400, "bad", "endpoint"],
    ];
    for (const [status, body, kind] of cases) {
      const f = await endpoint((_, res) => {
        res.writeHead(status);
        res.end(body);
      });
      expect(
        await kindOf(new HttpTransport({ url: f.url }).run("x", [])),
        `${status} ${body}`,
      ).toBe(kind);
    }
  });

  it("carries the status on an endpoint error", async () => {
    const f = await endpoint((_, res) => {
      res.writeHead(503);
      res.end("down");
    });
    await expect(new HttpTransport({ url: f.url }).run("x", [])).rejects
      .toMatchObject({
        kind: "endpoint",
        status: 503,
        detail: expect.stringContaining("returned 503: down"),
      });
  });

  it("answers protocol for a 200 that is not the envelope", async () => {
    for (
      const body of [
        "<html>",
        "[]",
        "{\"stdout\":\"x\"}",
        "null",
        "{\"exit_code\":0}",
        "{\"exit_code\":0,\"stdout\":\"***\",\"stderr\":\"\"}",
        "{\"exit_code\":0,\"stdout\":\"\"}",
      ]
    ) {
      const f = await endpoint((_, res) => {
        res.writeHead(200);
        res.end(body);
      });
      expect(await kindOf(new HttpTransport({ url: f.url }).run("x", [])), body)
        .toBe("protocol");
    }
  });

  it("answers transport when nothing listens", async () => {
    const f = await fakeEndpoint(() => {});
    await f.close();
    expect(await kindOf(new HttpTransport({ url: f.url }).run("x", []))).toBe(
      "transport",
    );
  });

  it("answers timeout when the request deadline passes", async () => {
    const e = await endpoint(() => {
      /* never answers */
    });
    const t = new HttpTransport({ url: e.url, deadlineGraceMs: 50 });
    expect(await kindOf(t.run("x", [], { timeoutMs: 50 }))).toBe(
      "timeout",
    );
  });

  it("reads incrementally and errors over the cap instead of returning a partial result", async () => {
    let wrote = 0;
    const e = await endpoint((_, res) => {
      res.writeHead(200, { "content-type": "application/json" });
      const piece = "x".repeat(1000);
      const timer = setInterval(() => {
        wrote += piece.length;
        if (!res.write(piece) || wrote > 50_000_000) clearInterval(timer);
      }, 0);
      res.on("close", () => clearInterval(timer));
    });
    const t = new HttpTransport({ url: e.url, maxResponseBytes: 4096 });
    await expect(t.run("x", [])).rejects.toMatchObject({
      kind: "too_large",
      detail: expect.stringContaining("the command ran"),
    });
    expect(wrote).toBeLessThan(5_000_000); // stopped reading long before the end
  });

  it("refuses a declared length over the cap before reading", async () => {
    const e = await endpoint((_, res) => {
      res.writeHead(200, { "content-length": "100000" });
      res.write("{");
    });
    expect(
      await kindOf(
        new HttpTransport({ url: e.url, maxResponseBytes: 10 }).run("x", []),
      ),
    ).toBe(
      "too_large",
    );
  });
});

describe("HttpTransport.run bytes", () => {
  it("returns the exact bytes of both streams, however they look as text", async () => {
    const stdout = Uint8Array.of(0xff, 0x00, 0x0a, 0xc3);
    const stderr = Uint8Array.of(0xfe);
    const e = await endpoint((_, res) => json(res, 200, ok(stdout, 0, stderr)));
    const r = await new HttpTransport({ url: e.url }).run("x", []);
    expect(Array.from(r.stdout)).toEqual(Array.from(stdout));
    expect(Array.from(r.stderr)).toEqual(Array.from(stderr));
    expect(Object.keys(r).sort()).toEqual(["exitCode", "stderr", "stdout"]);
  });

  it("reads no output as empty strings", async () => {
    const e = await endpoint((_, res) => json(res, 200, ok()));
    const r = await new HttpTransport({ url: e.url }).run("true", []);
    expect(r.stdout).toHaveLength(0);
    expect(r.stderr).toHaveLength(0);
  });

  it("counts the cap on decoded bytes of both streams together", async () => {
    const e = await endpoint((_, res) =>
      json(res, 200, ok(new Uint8Array(60), 0, new Uint8Array(50)))
    );
    const at = (n: number) =>
      new HttpTransport({ url: e.url, maxResponseBytes: n }).run("x", []);
    expect((await at(110)).stdout).toHaveLength(60);
    expect(await kindOf(at(109))).toBe("too_large");
  });

  it("is a protocol error when an output field is not standard base64, never empty", async () => {
    const e = await endpoint((_, res) =>
      json(res, 200, { exit_code: 0, stdout: "***", stderr: "" })
    );
    await expect(new HttpTransport({ url: e.url }).run("x", [])).rejects
      .toMatchObject({
        kind: "protocol",
        detail: expect.stringContaining("base64"),
      });
  });

  it("never sends the removed request fields", async () => {
    const e = await endpoint((_, res) => json(res, 200, ok()));
    await new HttpTransport({ url: e.url }).run("x", [], { stdin: utf8("i") });
    expect(Object.keys(e.requests[0]!.body).sort()).toEqual([
      "args",
      "command",
      "stdin",
      "timeout_ms",
    ]);
  });
});

const b64 = (s: string) => Buffer.from(s).toString("base64");

/** An endpoint that streams: answers every stream request by running `script`. */
async function streamingEndpoint(
  script: (res: import("node:http").ServerResponse) => void,
) {
  return endpoint((_seen, res, req) => {
    res.writeHead(200, { "content-type": "application/x-ndjson" });
    req.on("close", () => closed.push(Date.now()));
    script(res);
  });
}
const closed: number[] = [];

describe("HttpTransport.stream", () => {
  it("yields frames in order", async () => {
    const e = await streamingEndpoint((res) => {
      res.write(JSON.stringify({ stdout: b64("out") }) + "\n");
      res.write("{}\n");
      res.write(
        JSON.stringify({ stderr: b64("err"), shell_exit: 4, future_field: 1 })
          + "\n",
      );
      res.end("{\"exit_code\":4}\n");
    });
    const t = new HttpTransport({ url: e.url });
    const frames: Frame[] = [];
    for await (const f of t.stream("x", [])) frames.push(f);
    expect(frames.map((f) => f.type)).toEqual([
      "stdout",
      "stderr",
      "shell_exit",
      "exit",
    ]);
    expect(e.requests).toHaveLength(1);
    expect(e.requests[0]!.body.stream).toBe(true);
  });

  it("is a protocol error when the endpoint answers a stream request with one object", async () => {
    const e = await endpoint((_, res) => json(res, 200, ok("x")));
    const run = async () => {
      for await (const f of new HttpTransport({ url: e.url }).stream("x", [])) {
        void f;
      }
    };
    expect(await kindOf(run())).toBe("protocol");
  });

  it("closes the connection when the consumer stops", async () => {
    closed.length = 0;
    const e = await streamingEndpoint((res) => {
      res.write(JSON.stringify({ stdout: b64("first") }) + "\n");
      // never finishes
    });
    const t = new HttpTransport({ url: e.url });
    const start = Date.now();
    for await (const f of t.stream("sleep", ["99"])) {
      expect(f.type).toBe("stdout");
      break;
    }
    for (let i = 0; i < 50 && closed.length === 0; i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(closed.length).toBe(1);
    expect(closed[0]! - start).toBeLessThan(2000);
  });

  it("fails with transport when the stream ends without its final frame", async () => {
    const e = await streamingEndpoint((res) => {
      res.write(JSON.stringify({ stdout: b64("x") }) + "\n");
      res.end();
    });
    const t = new HttpTransport({ url: e.url });
    const run = async () => {
      for await (const f of t.stream("x", [])) void f;
    };
    expect(await kindOf(run())).toBe("transport");
  });

  it("fails with transport on an error frame", async () => {
    const e = await streamingEndpoint((res) =>
      res.end("{\"error\":\"lost the sandbox\"}\n")
    );
    const t = new HttpTransport({ url: e.url });
    const run = async () => {
      for await (const f of t.stream("x", [])) void f;
    };
    expect(await kindOf(run())).toBe("transport");
  });

  it("fails with protocol on a line that is not a frame", async () => {
    for (
      const line of [
        "not json",
        "[1]",
        JSON.stringify({ stdout: "!!not base64!!" }),
      ]
    ) {
      const e = await streamingEndpoint((res) => res.end(line + "\n"));
      const t = new HttpTransport({ url: e.url });
      const run = async () => {
        for await (const f of t.stream("x", [])) void f;
      };
      expect(await kindOf(run()), line).toBe("protocol");
    }
  });

  it("reports signal and timed_out as final frames", async () => {
    for (
      const [line, type] of [
        ["{\"signal\":9}", "signal"],
        ["{\"timed_out\":true}", "timed_out"],
      ] as const
    ) {
      const e = await streamingEndpoint((res) => res.end(line + "\n"));
      const frames: Frame[] = [];
      for await (const f of new HttpTransport({ url: e.url }).stream("x", [])) {
        frames.push(f);
      }
      expect(frames.at(-1)!.type).toBe(type);
    }
  });
});

describe("decodeFrameLine", () => {
  const dec = (s: string) => decodeFrameLine(utf8(s));
  it("splits a sloppy frame that names more than one field: output first, then the final one", () => {
    const out = dec(`{"stdout":"${b64("a")}","exit_code":2}`);
    expect(out.map((f) => (f === "heartbeat" ? f : f.type))).toEqual([
      "stdout",
      "exit",
    ]);
  });
  it("reads {} and a blank line as heartbeats, and tolerates a CR", () => {
    expect(dec("{}")).toEqual(["heartbeat"]);
    expect(dec("   ")).toEqual(["heartbeat"]);
    expect(dec("{\"exit_code\":0}\r")).toEqual([{ type: "exit", code: 0 }]);
  });
  it("requires standard base64", () => {
    expect(() => dec("{\"stdout\":\"***\"}")).toThrow(/base64/);
  });
});
