import { describe, expect, it } from "vitest";
import { utf8 } from "../src/bytes.js";
import { Files, Jobs, NotFoundError, TransportError } from "../src/index.js";
import type { RunOptions, RunResult, Transport } from "../src/index.js";

const res = (exitCode: number, stdout = "", stderr = ""): RunResult => ({
  exitCode,
  stdout: utf8(stdout),
  stderr: utf8(stderr),
});

/** A transport whose answers a test scripts per command line; every call is recorded. */
function scripted(
  answer: (command: string, args: string[], o: RunOptions) => RunResult | Error,
) {
  const calls: Array<{ command: string; args: string[]; options: RunOptions }> =
    [];
  const t: Transport = {
    async run(command, args, options = {}) {
      calls.push({ command, args, options });
      const a = answer(command, args, options);
      if (a instanceof Error) throw a;
      return a;
    },
  };
  return { t, calls };
}

const gnu = (c: string, _a: string[], o: RunOptions): RunResult => {
  switch (c) {
    case "rg":
      return res(127, "", "rg: not found");
    case "grep":
      return res(1);
    case "find":
      return res(0, "c");
    case "cat":
      return res(0, new TextDecoder().decode(o.stdin));
    default:
      return res(0, "Yg==\n"); // the read-window probe
  }
};

describe("Files capability probes", () => {
  it("reads GNU tools as stdin, find -printf, read window and GNU grep", async () => {
    const { t } = scripted(gnu);
    expect(await new Files(t).capabilities()).toEqual({
      stdin: true,
      findPrintf: true,
      readWindow: true,
      rg: false,
      grepNull: true,
      searchEngine: "grep",
    });
  });

  it("sends the spec's probe commands with a 10 s timeout", async () => {
    const { t, calls } = scripted(gnu);
    await new Files(t).capabilities();
    const sorted = calls.map((
      c,
    ) => [c.command, ...c.args.filter((a) => !a.startsWith("printf abc"))]);
    expect(sorted).toContainEqual(["rg", "--version"]);
    expect(sorted).toContainEqual([
      "grep",
      "-rnZ",
      "-e",
      "__cap_probe__",
      "--",
      "/dev/null",
    ]);
    expect(sorted).toContainEqual([
      "find",
      "/dev/null",
      "-maxdepth",
      "0",
      "-printf",
      "%y",
    ]);
    expect(calls.find((c) => c.command === "cat")!.options.stdin).toEqual(
      utf8("stdin-probe"),
    );
    expect(calls.every((c) => c.options.timeoutMs === 10_000)).toBe(true);
    expect(calls).toHaveLength(5);
  });

  it("reads busybox as no -printf (usage, exit 1), no grep -Z, no stdin", async () => {
    const { t } = scripted((c) => {
      switch (c) {
        case "rg":
          return res(127);
        case "grep":
          return res(2, "", "grep: unrecognized option: Z");
        case "find":
          return res(
            1,
            "",
            "BusyBox v1.36.1 multi-call binary.\nUsage: find ...",
          );
        case "cat":
          return res(0, ""); // a v1 endpoint drops stdin
        default:
          return res(1);
      }
    });
    expect(await new Files(t).capabilities()).toEqual({
      stdin: false,
      findPrintf: false,
      readWindow: false,
      rg: false,
      grepNull: false,
      searchEngine: "busybox-grep",
    });
  });

  it("proves stdin by an echo: exit 0 with the wrong bytes is no stdin", async () => {
    const { t } = scripted((
      c,
      a,
      o,
    ) => (c === "cat" ? res(0, "stdin-probe\n") : gnu(c, a, o)));
    expect((await new Files(t).capabilities()).stdin).toBe(false);
  });

  it("prefers rg, and applies overrides after probing", async () => {
    const { t } = scripted((
      c,
      a,
      o,
    ) => (c === "rg" ? res(0, "ripgrep 14") : gnu(c, a, o)));
    expect((await new Files(t).capabilities()).searchEngine).toBe("rg");
    const forced = new Files(t, {
      overrides: {
        stdin: false,
        findPrintf: false,
        searchEngine: "busybox-grep",
      },
    });
    expect(await forced.capabilities()).toMatchObject({
      stdin: false,
      findPrintf: false,
      rg: false,
      grepNull: false,
      searchEngine: "busybox-grep",
    });
  });

  it("treats a probe the transport could not even start as absent", async () => {
    const { t } = scripted((
      c,
      a,
      o,
    ) => (c === "rg" ? new NotFoundError("rg: not found") : gnu(c, a, o)));
    expect((await new Files(t).capabilities()).rg).toBe(false);
  });

  it("reports a probe that failed at the transport level, caches nothing, and tries again", async () => {
    let down = true;
    const { t, calls } = scripted((c, a, o) =>
      down && c === "find"
        ? new TransportError("connection refused")
        : gnu(c, a, o)
    );
    const files = new Files(t);
    await expect(files.stat("/x")).rejects.toMatchObject({ kind: "transport" });
    await expect(files.read("/x")).rejects.toMatchObject({ kind: "transport" });
    down = false;
    const before = calls.length;
    expect((await files.capabilities()).findPrintf).toBe(true);
    expect(calls.length).toBeGreaterThan(before);
    const after = calls.length;
    await files.capabilities(); // now cached
    expect(calls.length).toBe(after);
  });

  it("refuses to read when the read window does not work, instead of reading an empty file", async () => {
    const { t, calls } = scripted((
      c,
      a,
      o,
    ) => (c === "sh" ? res(0, "") : gnu(c, a, o)));
    await expect(new Files(t).read("/f")).rejects.toMatchObject({
      kind: "io",
      detail: expect.stringContaining("so files cannot be read"),
    });
    expect(calls.filter((c) => c.args[1]?.startsWith("if [ ! -e")))
      .toHaveLength(0);
  });
});

describe("Jobs capability", () => {
  it("is none when the sandbox lacks a tool, and then every operation is not_offered", async () => {
    const { t, calls } = scripted(() => res(0, "missing setsid\n"));
    const jobs = new Jobs(t);
    expect(await jobs.capabilities()).toBe("none");
    await expect(jobs.list("scope")).rejects.toMatchObject({
      kind: "not_offered",
    });
    expect(calls).toHaveLength(1); // the probe only
    expect(calls[0]!.options.stdin).toEqual(utf8("stdin-probe\n"));
  });

  it("is baseline when the probe prints ok, native only when the transport offers it", async () => {
    const { t } = scripted(() => res(0, "ok\n"));
    expect(await new Jobs(t).capabilities()).toBe("baseline");
    const native: Transport = {
      run: async () => res(1),
      offers: async () => ({ stream: false, nativeJobs: true }),
      nativeJobs: {} as never,
    };
    expect(await new Jobs(native).capabilities()).toBe("native");
  });

  it("validates scope and id before sending anything", async () => {
    const { t, calls } = scripted(() => res(0, "ok\n"));
    const jobs = new Jobs(t);
    for (const id of ["", "A", "a b", "a/b", "x".repeat(33), "é"]) {
      await expect(jobs.start("s", id, { command: "true" })).rejects
        .toMatchObject({
          kind: "invalid_request",
        });
    }
    for (const scope of ["", "S", "a/b", "..", "x".repeat(65)]) {
      await expect(jobs.list(scope)).rejects.toMatchObject({
        kind: "invalid_request",
      });
    }
    expect(calls).toHaveLength(0);
  });
});
