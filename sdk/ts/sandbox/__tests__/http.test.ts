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
      text: true,
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
      stdin_base64: "aW4=",
    });
  });

  it("sends no Authorization without a token, and no cwd or stdin unless given", async () => {
    const e = await endpoint((_, res) => json(res, 200, ok()));
    await new HttpTransport({ url: e.url }).run("true", [], { text: true });
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
    expect(text((await t.run("x", [], { text: true })).stdout)).toBe("fine");
    expect(calls).toBe(2);
    expect(e.requests.map((r) => r.headers.authorization)).toEqual([
      "Bearer t1",
      "Bearer t2",
    ]);
    // The refreshed token is kept: no second refresh for the next request.
    await t.run("x", [], { text: true });
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
    for (const body of ["<html>", "[]", "{\"stdout\":\"x\"}", "null"]) {
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
    expect(await kindOf(t.run("x", [], { timeoutMs: 50, text: true }))).toBe("timeout");
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

describe("probes", () => {
  it("a v1.1 endpoint offers neither stream nor jobs, and stream() is not_offered without running the command", async () => {
    const e = await endpoint((seen, res) =>
      json(res, 200, ok(seen.body.job ? "exec-v1.2" : "a"))
    );
    const t = new HttpTransport({ url: e.url });
    expect(await t.offers()).toEqual({
      stream: false,
      nativeJobs: false,
      outputBase64: false,
    });
    const before = e.requests.length;
    const it = t.stream("echo", ["hi"]);
    expect(await kindOf(it.next())).toBe("not_offered");
    expect(e.requests.length).toBe(before);
  });

  it("a front that refuses the extensions with 400 has answered: no offers, and it is cached", async () => {
    const e = await endpoint((_, res) => {
      res.writeHead(400);
      res.end("unknown field");
    });
    const t = new HttpTransport({ url: e.url });
    expect(await t.offers()).toEqual({
      stream: false,
      nativeJobs: false,
      outputBase64: false,
    });
    const n = e.requests.length;
    await t.offers();
    expect(e.requests.length).toBe(n);
  });

  it("a probe that could not reach the endpoint throws and is retried next time", async () => {
    const f = await fakeEndpoint(() => {});
    await f.close();
    const t = new HttpTransport({ url: f.url });
    expect(await kindOf(t.offers())).toBe("transport");
    const g = await endpoint((seen, res) =>
      json(res, 200, ok(seen.body.job ? "exec-v1.2" : ""))
    );
    const t2 = new HttpTransport({ url: g.url });
    expect((await t2.offers()).nativeJobs).toBe(false);
  });

  it("native jobs are offered only for exactly {jobs: v1.3}", async () => {
    for (
      const [answer, want] of [
        [{ jobs: "v1.3" }, true],
        [{ jobs: "v1.3", notify: "v1.4" }, true],
        [{ jobs: "v1.4" }, false],
        [{ jobs: true }, false],
        [{ notify: "v1.4" }, false],
        [{ exit_code: 0, stdout: "exec-v1.2", stderr: "" }, false],
      ] as const
    ) {
      const e = await endpoint((seen, res) =>
        seen.body.job ? json(res, 200, answer) : json(res, 200, ok(""))
      );
      expect(
        (await new HttpTransport({ url: e.url }).offers()).nativeJobs,
        JSON.stringify(answer),
      ).toBe(want);
    }
  });
});

const b64 = (s: string) => Buffer.from(s).toString("base64");

/** An endpoint that streams: answers the probe like execd does, then runs `script` for a real stream. */
async function streamingEndpoint(
  script: (res: import("node:http").ServerResponse) => void,
) {
  return endpoint((seen, res, req) => {
    if (seen.body.output_base64) {
      json(res, 200, ok(""));
      return;
    }
    const probe = Array.isArray(seen.body.args)
      && String(seen.body.args[1]).startsWith("printf a;");
    if (seen.body.job) {
      json(res, 200, {});
      return;
    }
    res.writeHead(200, { "content-type": "application/x-ndjson" });
    if (probe) {
      res.write(JSON.stringify({ stdout: b64("a") }) + "\n");
      setTimeout(
        () =>
          res.end(
            JSON.stringify({ stdout: b64("b") }) + "\n" + "{\"exit_code\":0}\n",
          ),
        200,
      );
      return;
    }
    req.on("close", () => closed.push(Date.now()));
    script(res);
  });
}
const closed: number[] = [];

describe("HttpTransport.stream", () => {
  it("proves streaming with the spec's probe, then yields frames in order", async () => {
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
    expect((await t.offers()).stream).toBe(true);
    const req = e.requests.find((r) =>
      r.body.stream && !String(r.body.args).includes("printf a")
    )!;
    expect(req.body.stream).toBe(true);
  });

  it("refuses an endpoint that buffers: all frames at once is not streaming", async () => {
    const e = await endpoint((seen, res) => {
      if (seen.body.job || seen.body.output_base64) {
        return json(res, 200, ok(""));
      }
      res.writeHead(200, { "content-type": "application/x-ndjson" });
      res.end(
        JSON.stringify({ stdout: b64("a") })
          + "\n"
          + JSON.stringify({ stdout: b64("b") })
          + "\n"
          + "{\"exit_code\":0}\n",
      );
    });
    expect((await new HttpTransport({ url: e.url }).offers()).stream).toBe(
      false,
    );
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

  it("fails with transport on an error frame, and on a line that is not a frame", async () => {
    for (const line of ["{\"error\":\"lost the sandbox\"}", "not json"]) {
      const e = await streamingEndpoint((res) => res.end(line + "\n"));
      const t = new HttpTransport({ url: e.url });
      const run = async () => {
        for await (const f of t.stream("x", [])) void f;
      };
      expect(await kindOf(run()), line).toBe("transport");
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
