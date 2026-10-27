import { describe, expect, inject, it } from "vitest";
import { Driver, OPS } from "../src/driver-core.js";

const execd = inject("execd");

const b64 = (s: string) => Buffer.from(s).toString("base64");
const ask = (d: Driver, req: Record<string, unknown>) => d.handle(req);

describe("driver without an endpoint", () => {
  const dead = new Driver({ url: "http://127.0.0.1:1/exec" });

  it("answers hello without reaching the endpoint, and echoes the id", async () => {
    const r = await ask(dead, { id: { any: ["json"] }, op: "hello" });
    expect(r).toEqual({
      id: { any: ["json"] },
      ok: { driver: "beyond-sandbox-ts", protocol: 1, ops: [...OPS] },
    });
  });

  it("answers unsupported for an op it does not know, never a silent pass", async () => {
    for (const op of ["remove", "upload_dir", "download_dir", "nonsense", ""]) {
      const r = await ask(dead, { id: 1, op });
      expect(r, op).toMatchObject({ id: 1, error: { kind: "unsupported" } });
      expect(OPS as readonly string[]).not.toContain(op);
    }
    expect(await ask(dead, { id: 2 })).toMatchObject({
      error: { kind: "unsupported" },
    });
  });

  it("reports the connection error from every other op, capabilities included", async () => {
    for (const op of ["capabilities", "exec", "stat", "job_list"]) {
      const r = await ask(dead, {
        id: 1,
        op,
        command: "true",
        path: "/",
        scope: "s",
      });
      expect(r, op).toMatchObject({ error: { kind: "transport" } });
    }
  });
});

(execd ? describe : describe.skip)("driver against execd", () => {
  const driver = () => new Driver({ url: execd!.url, bearer: execd!.token });

  it("answers capabilities, honoring CONFORMANCE_CAPS overrides", async () => {
    const r = await ask(driver(), { id: 1, op: "capabilities" });
    expect(r).toMatchObject({
      ok: {
        find_printf: true,
        read_window: true,
      },
    });
    const forced = new Driver({
      url: execd!.url,
      bearer: execd!.token,
      caps: { findPrintf: false, searchEngine: "busybox-grep" },
    });
    expect(await ask(forced, { id: 1, op: "capabilities" })).toMatchObject({
      ok: { find_printf: false, search_engine: "busybox-grep" },
    });
  });

  it("reports bytes without an exactness flag and no probe-derived protocol capabilities", async () => {
    const r = await ask(driver(), { id: 1, op: "capabilities" });
    expect(Object.keys((r as { ok: object }).ok).sort()).toEqual([
      "find_printf",
      "read_window",
      "search_engine",
    ]);
  });

  it("maps a wrong bearer to auth for every op", async () => {
    const d = new Driver({ url: execd!.url, bearer: "wrong" });
    expect(await ask(d, { id: 1, op: "capabilities" })).toMatchObject({
      error: { kind: "auth" },
    });
    expect(await ask(d, { id: 2, op: "exec", command: "true" })).toMatchObject({
      error: { kind: "auth" },
    });
  });

  it("rejects what cannot form a request as invalid_request, sending nothing", async () => {
    const d = driver();
    const bad: Array<Record<string, unknown>> = [
      { op: "stat" },
      { op: "stat", path: 5 },
      { op: "read", path: "/x", offset: -1 },
      { op: "read", path: "/x", max: 1.5 },
      { op: "write", path: "/x", data_b64: "***" },
      { op: "write_if_unchanged", path: "/x", data_b64: "" },
      {
        op: "write_if_unchanged",
        path: "/x",
        data_b64: "",
        expected_mtime: "1",
      },
      { op: "exec", command: "true", args: "no" },
      { op: "exec", command: "true", args: [1] },
      { op: "list", path: "/", include_hidden: "yes" },
      { op: "job_start", scope: "s", job: "BAD", command: "true" },
      { op: "job_start", scope: "s", job: "x".repeat(33), command: "true" },
      { op: "job_kill", scope: "s", jobs: ["a/b"] },
      { op: "job_remove", scope: "s", jobs: ["A"] },
    ];
    for (const req of bad) {
      expect(await ask(d, { id: 1, ...req }), JSON.stringify(req))
        .toMatchObject({
          error: { kind: "invalid_request" },
        });
    }
  });

  it("runs ops end to end in the contract's JSON", async () => {
    const d = driver();
    const dir = `${execd!.root}/driver-${process.pid}`;
    expect(await ask(d, { id: 1, op: "mkdir", path: dir })).toEqual({
      id: 1,
      ok: {},
    });
    expect(
      await ask(d, {
        id: 2,
        op: "write",
        path: `${dir}/a`,
        data_b64: b64("hi\n"),
      }),
    ).toEqual({ id: 2, ok: {} });
    expect(await ask(d, { id: 3, op: "read", path: `${dir}/a` })).toEqual({
      id: 3,
      ok: { data_b64: b64("hi\n") },
    });
    const st = await ask(d, { id: 4, op: "stat", path: `${dir}/a` });
    expect(st).toMatchObject({
      ok: { exists: true, kind: "file", size: 3, writable: true },
    });
    expect(await ask(d, { id: 5, op: "stat", path: `${dir}/none` })).toEqual({
      id: 5,
      ok: { exists: false },
    });
    expect(
      await ask(d, {
        id: 6,
        op: "write_if_unchanged",
        path: `${dir}/a`,
        data_b64: "",
        expected_mtime: null,
      }),
    ).toMatchObject({ error: { kind: "conflict" } });
    expect(await ask(d, { id: 7, op: "read", path: `${dir}/none` }))
      .toMatchObject({
        error: { kind: "not_found" },
      });
    const ex = await ask(d, {
      id: 8,
      op: "exec",
      command: "sh",
      args: ["-c", "cat; exit 2"],
      stdin_b64: b64("x"),
    });
    expect(ex).toEqual({
      id: 8,
      ok: {
        exit_code: 2,
        stdout_b64: b64("x"),
        stderr_b64: "",
      },
    });
    const es = await ask(d, {
      id: 9,
      op: "exec_stream",
      command: "sh",
      args: ["-c", "echo o; echo e >&2"],
    });
    expect(es).toEqual({
      id: 9,
      ok: { stdout_b64: b64("o\n"), stderr_b64: b64("e\n"), exit_code: 0 },
    });
    const hit = await ask(d, {
      id: 10,
      op: "search",
      pattern: "hi",
      root: dir,
    });
    expect(hit).toMatchObject({
      ok: {
        hits: [{ path: `${dir}/a`, line: 1, text: "hi", is_match: true }],
        truncated: false,
      },
    });
  });
});
