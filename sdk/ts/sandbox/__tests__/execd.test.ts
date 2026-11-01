import { randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, inject, it } from "vitest";
import { Sandbox } from "../src/index.js";
import type { Frame } from "../src/index.js";
import { HttpTransport } from "../src/index.js";

const execd = inject("execd");
/** Writes every byte value, 0x00 to 0xFF, to stdout. */
const ALL_256 =
  "i=0; while [ $i -lt 256 ]; do printf \"\\\\$(printf %03o $i)\"; i=$((i+1)); done";
const d = execd ? describe : describe.skip;

let n = 0;
const scratch = () => `${execd!.root}/t${process.pid}-${++n}`;
const dec = (b: Uint8Array) => new TextDecoder().decode(b);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function connect(): Promise<Sandbox> {
  return Sandbox.connect({ url: execd!.url, token: execd!.token });
}

d("against a loopback execd", () => {
  it("raw exec returns every byte value exactly, on both streams", async () => {
    const sb = await connect();
    const r = await sb.exec("sh", [
      "-c",
      `${ALL_256}; ${ALL_256} >&2; exit 3`,
    ]);
    const all = Array.from({ length: 256 }, (_, i) => i);
    expect(r.exitCode).toBe(3);
    expect(Array.from(r.stdout)).toEqual(all);
    expect(Array.from(r.stderr)).toEqual(all);
    // And with stdin: binary in, binary out.
    const echoed = await sb.exec("cat", [], { stdin: Uint8Array.from(all) });
    expect(Array.from(echoed.stdout)).toEqual(all);
  });

  it("runs a command, with stdin, and reports a non-zero exit as a result", async () => {
    const sb = await connect();
    const r = await sb.exec("sh", ["-c", "cat; echo err >&2; exit 3"], {
      stdin: "hello",
    });
    expect([r.exitCode, r.stdoutText, r.stderrText]).toEqual([
      3,
      "hello",
      "err\n",
    ]);
    const nf = await sb.exec("no-such-command-xyz");
    expect(nf.exitCode).toBe(127);
  });

  it("connect probes and fails with auth on a wrong or missing token", async () => {
    await expect(
      Sandbox.connect({ url: execd!.url, token: "wrong" }),
    ).rejects.toMatchObject({ kind: "auth" });
    await expect(Sandbox.connect({ url: execd!.url })).rejects.toMatchObject({
      kind: "auth",
    });
  });

  it("errors over the response cap and says the command ran", async () => {
    const sb = await Sandbox.connect({
      url: execd!.url,
      token: execd!.token,
      maxResponseBytes: 2000,
    });
    await expect(
      sb.exec("sh", ["-c", "head -c 5000 /dev/zero | tr '\\0' x"]),
    ).rejects.toMatchObject({
      kind: "too_large",
    });
  });

  it("streams frames, and stopping the iteration kills the command's whole group", async () => {
    const sb = await connect();
    const dir = scratch();
    await sb.files.mkdir(dir);
    const marker = join(dir, "alive");
    // A grandchild keeps touching a marker; killing the stream must stop it.
    // The first write is synchronous so "started" implies the marker exists.
    const script =
      `date +%s%N > "${marker}"; while :; do date +%s%N > "${marker}"; sleep 0.1; done & echo started; wait`;
    const frames: Frame[] = [];
    for await (const f of sb.stream("sh", ["-c", script])) {
      frames.push(f);
      if (f.type === "stdout") break;
    }
    expect(dec((frames[0] as { data: Uint8Array }).data)).toBe("started\n");
    expect(existsSync(marker)).toBe(true);
    await sleep(1500); // execd kills the group at once on EOF; give it a margin
    const before = dec(await sb.files.read(marker));
    await sleep(600);
    expect(dec(await sb.files.read(marker))).toBe(before);
  });

  it("reports exit, signal and timeout as the final frame, and heartbeats are not output", async () => {
    const sb = await connect();
    const last = async (cmd: string, o = {}) => {
      let f: Frame | undefined;
      for await (f of sb.stream("sh", ["-c", cmd], o));
      return f;
    };
    expect(await last("exit 7")).toEqual({ type: "exit", code: 7 });
    expect(await last("kill -9 $$")).toEqual({ type: "signal", signal: 9 });
    expect(await last("sleep 30", { timeoutMs: 300 })).toEqual({
      type: "timed_out",
    });
  });

  describe("files", () => {
    it("round-trips bytes exactly, across read windows, with offset and max", async () => {
      const sb = await connect();
      const dir = scratch();
      await sb.files.mkdir(dir);
      const data = randomBytes(300_000); // three windows of 96 KiB
      await sb.files.write(`${dir}/bin`, data);
      expect(Buffer.from(await sb.files.read(`${dir}/bin`)).equals(data)).toBe(
        true,
      );
      expect(
        Buffer.from(
          await sb.files.read(`${dir}/bin`, { offset: 98_000, max: 200_000 }),
        ).equals(
          data.subarray(98_000, 298_000),
        ),
      ).toBe(true);
      expect(await sb.files.read(`${dir}/bin`, { offset: 400_000 }))
        .toHaveLength(0);
      expect(await sb.files.read(`${dir}/bin`, { max: 0 })).toHaveLength(0);
      const st = await sb.files.stat(`${dir}/bin`);
      expect(st).toMatchObject({ kind: "file", size: 300_000, writable: true });
    });

    it("reports the spec's error kinds", async () => {
      const sb = await connect();
      const dir = scratch();
      await sb.files.mkdir(dir);
      await sb.files.write(`${dir}/f`, "x");
      const kind = async (p: Promise<unknown>) =>
        p.then(
          () => "ok",
          (e: { kind: string }) => e.kind,
        );
      expect(await kind(sb.files.read(`${dir}/missing`))).toBe("not_found");
      expect(await kind(sb.files.read(dir))).toBe("is_directory");
      expect(await kind(sb.files.write(`${dir}/nodir/x`, "y"))).toBe(
        "not_found",
      );
      expect(await kind(sb.files.write(dir, "y"))).toBe("is_directory");
      expect(await kind(sb.files.mkdir(`${dir}/f/sub`))).toBe(
        "not_a_directory",
      );
      expect(await kind(sb.files.search({ pattern: "(", root: dir }))).toBe(
        "invalid_query",
      );
      expect(await kind(sb.files.search({ pattern: "x", root: `${dir}/nope` })))
        .toBe("not_found");
      expect(await sb.files.stat(`${dir}/missing`)).toBeNull();
    });

    it("guards writes with the whole-second mtime", async () => {
      const sb = await connect();
      const dir = scratch();
      await sb.files.mkdir(dir);
      await sb.files.writeIfUnchanged(`${dir}/g`, "one", null); // must not exist
      await expect(sb.files.writeIfUnchanged(`${dir}/g`, "two", null)).rejects
        .toMatchObject({
          kind: "conflict",
        });
      const m = (await sb.files.stat(`${dir}/g`))!.mtime;
      await expect(sb.files.writeIfUnchanged(`${dir}/g`, "two", m - 5)).rejects
        .toMatchObject({
          kind: "conflict",
        });
      await sb.files.writeIfUnchanged(`${dir}/g`, "two", m);
      expect(await sb.files.readText(`${dir}/g`)).toBe("two");
    });

    it("lists with hidden names and limits, and searches with context and limits", async () => {
      const sb = await connect();
      const dir = scratch();
      await sb.files.mkdir(`${dir}/sub`);
      await sb.files.write(
        `${dir}/a.txt`,
        "alpha\nbeta needle\ngamma\ndelta\nneedle again\n",
      );
      await sb.files.write(`${dir}/.hidden`, "needle hidden\n");
      await sb.files.write(`${dir}/sub/b.txt`, "needle in sub\n");
      const names = async (o = {}) =>
        (await sb.files.list(dir, o)).map((e) => e.name).sort();
      expect(await names()).toEqual(["a.txt", "sub"]);
      expect(await names({ includeHidden: true })).toEqual([
        ".hidden",
        "a.txt",
        "sub",
      ]);
      expect(await names({ limit: 1 })).toHaveLength(1);
      expect((await sb.files.list(dir)).find((e) => e.name === "sub")?.kind)
        .toBe("dir");
      expect(await sb.files.list(`${dir}/a.txt`)).toEqual([]);

      const r = await sb.files.search({
        pattern: "needle",
        root: dir,
        literal: true,
        before: 1,
        after: 1,
      });
      expect(r.truncated).toBe(false);
      const key = (h: { path: string; line: number; isMatch: boolean }) =>
        `${h.path.slice(dir.length)}:${h.line}${h.isMatch ? "m" : "c"}`;
      expect(r.hits.map(key)).toEqual([
        "/.hidden:1m",
        "/a.txt:1c",
        "/a.txt:2m",
        "/a.txt:3c",
        "/a.txt:4c",
        "/a.txt:5m",
        "/sub/b.txt:1m",
      ]);
      const lim = await sb.files.search({
        pattern: "needle",
        root: dir,
        limit: 2,
      });
      expect(lim.truncated).toBe(true);
      expect(lim.hits.filter((h) => h.isMatch)).toHaveLength(2);
      const g = await sb.files.search({
        pattern: "needle",
        root: dir,
        glob: "*.txt",
        globExclude: true,
      });
      expect(g.hits.map(key)).toEqual(["/.hidden:1m"]);
    });
  });

  {
    describe("jobs", () => {
      it("starts, reads from an offset, follows to the end, lists and removes", async () => {
        const sb = await connect();
        const scope = `ts-jobs-${process.pid}`;
        try {
          await sb.jobs.start(scope, "job1", {
            command: "sh",
            args: ["-c", "echo one; sleep 0.3; echo two; exit 5"],
            label: "demo",
          });
          const att = await sb.jobs.collect(scope, "job1");
          expect(att).toMatchObject({ exitCode: 5, gap: 0 });
          expect(dec(att.output)).toBe("one\ntwo\n");
          // deprecated alias still works
          expect(await sb.jobs.attach(scope, "job1")).toMatchObject({
            exitCode: 5,
            gap: 0,
          });
          const c = await sb.jobs.read(scope, "job1", { offset: 4 });
          expect(c).toMatchObject({
            state: "ended",
            code: 5,
            offset: 4,
            end: 8,
          });
          expect(dec(c.data)).toBe("two\n");
          const [j] = await sb.jobs.list(scope);
          expect(j).toMatchObject({
            id: "job1",
            state: "ended",
            code: 5,
            label: "demo",
          });
          await sb.jobs.remove(scope, ["job1"]);
          expect(await sb.jobs.list(scope)).toEqual([]);
          expect((await sb.jobs.read(scope, "job1")).state).toBe("gone");
        } finally {
          await sb.jobs.wipe(scope);
        }
      });

      it("tells a job killed by a signal from one that exited 137", async () => {
        const sb = await connect();
        const scope = `ts-signal-${process.pid}`;
        try {
          await sb.jobs.start(scope, "sig", {
            command: "sh",
            args: ["-c", "kill -9 $$"],
          });
          await sb.jobs.start(scope, "exit", {
            command: "sh",
            args: ["-c", "exit 137"],
          });
          for (let i = 0; i < 100; i++) {
            const states = await sb.jobs.list(scope);
            if (
              states.length === 2 && states.every((j) => j.state === "ended")
            ) break;
            await sleep(50);
          }
          const killed = await sb.jobs.read(scope, "sig");
          const exited = await sb.jobs.read(scope, "exit");
          expect(killed).toMatchObject({
            state: "ended",
            code: 137,
            signal: 9,
          });
          expect(exited).toMatchObject({ state: "ended", code: 137 });
          expect("signal" in exited).toBe(false);
          const listed = Object.fromEntries(
            (await sb.jobs.list(scope)).map((j) => [j.id, j]),
          );
          expect(listed.sig).toMatchObject({ code: 137, signal: 9 });
          expect(listed.exit !== undefined && "signal" in listed.exit).toBe(
            false,
          );
        } finally {
          await sb.jobs.wipe(scope);
        }
      });

      it("kills the whole group, and refuses a bad id before sending anything", async () => {
        const sb = await connect();
        const scope = `ts-kill-${process.pid}`;
        try {
          await sb.jobs.start(scope, "sleeper", {
            command: "sh",
            args: ["-c", "sleep 60 & sleep 60"],
          });
          expect((await sb.jobs.read(scope, "sleeper")).state).toBe("running");
          expect(await sb.jobs.kill(scope, ["sleeper"])).toEqual([]);
          for (
            let i = 0;
            i < 50
            && (await sb.jobs.read(scope, "sleeper")).state === "running";
            i++
          ) {
            await sleep(50);
          }
          const after = await sb.jobs.read(scope, "sleeper");
          expect(after.state).toBe("ended");
          expect(after.code).toBe(143);
          expect(after.signal).toBe(15);
          await expect(sb.jobs.start(scope, "BAD ID", { command: "true" }))
            .rejects.toMatchObject({
              kind: "invalid_request",
            });
          await expect(sb.jobs.start("Bad/Scope", "ok", { command: "true" }))
            .rejects.toMatchObject(
              { kind: "invalid_request" },
            );
        } finally {
          await sb.jobs.wipe(scope);
        }
      });
    });
  }

  it("exposes the raw layer on the Sandbox, and Files/Jobs work over the transport alone", async () => {
    const sb = await connect();
    const r = await sb.transport.run("echo", ["raw"]);
    expect(dec(r.stdout)).toBe("raw\n");
    const dir = scratch();
    await sb.files.mkdir(dir);
    // A user's own helper, on raw exec only.
    const wc = await sb.exec("sh", ["-c", "printf \"a b c\" | wc -w"], {
      cwd: dir,
    });
    expect(wc.stdoutText.trim()).toBe("3");
    expect(sb.transport).toBeInstanceOf(HttpTransport);
  });
});
