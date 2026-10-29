import {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { download, extract, upload } from "../src/bulk.js";
import { SandboxError } from "../src/errors.js";
import { HttpTransport } from "../src/http.js";
import {
  DOWNLOAD_SCRIPT,
  STAGE_SCRIPT,
  UPLOAD_SCRIPT,
} from "../src/scripts.js";
import {
  END_BLOCKS,
  padLen,
  T_DIR,
  T_FILE,
  TarParser,
  writeHeader,
} from "../src/tar.js";
import type { TarEntry, TarSink } from "../src/tar.js";
import type {
  Frame,
  JobOps,
  RunOptions,
  RunResult,
  Transport,
} from "../src/transport.js";
import { fakeEndpoint, json, ok } from "./fake-endpoint.js";

const enc = new TextEncoder();
const dirs: string[] = [];
function tmp(): string {
  const d = mkdtempSync(join(tmpdir(), "bulk-test-"));
  dirs.push(d);
  return d;
}
afterEach(() => {
  while (dirs.length) rmSync(dirs.pop()!, { recursive: true, force: true });
});

// ---- hand-built archives ----------------------------------------------------------------------

type M = [
  name: string,
  kind: string,
  data: string | Uint8Array,
  link: string,
  mode: number,
];

function rawTar(members: M[], opts: { end?: boolean; corrupt?: boolean } = {}) {
  const parts: Uint8Array[] = [];
  for (const [name, kind, data, link, mode] of members) {
    const d = typeof data === "string" ? enc.encode(data) : data;
    const h = new Uint8Array(512);
    h.set(enc.encode(name));
    h.set(enc.encode(mode.toString(8).padStart(7, "0")), 100);
    h.set(enc.encode("0000000"), 108);
    h.set(enc.encode("0000000"), 116);
    h.set(enc.encode(d.length.toString(8).padStart(11, "0")), 124);
    h.set(enc.encode((1500000000).toString(8).padStart(11, "0")), 136);
    h.fill(0x20, 148, 156);
    h[156] = kind.charCodeAt(0);
    h.set(enc.encode(link), 157);
    h.set(enc.encode("ustar"), 257);
    h[263] = 0x30;
    h[264] = 0x30;
    let sum = 0;
    for (const b of h) sum += b;
    h.set(enc.encode(sum.toString(8).padStart(6, "0")), 148);
    h[154] = 0;
    h[155] = 0x20;
    if (opts.corrupt) h[0]! ^= 1;
    parts.push(h, d, new Uint8Array(padLen(d.length)));
  }
  if (opts.end !== false) parts.push(new Uint8Array(1024));
  return Buffer.concat(parts);
}
const file = (
  name: string,
  data = "x",
  mode = 0o644,
): M => [name, "0", data, "", mode];

function entries(): {
  sink: TarSink;
  seen: Array<{ e: TarEntry; data: Buffer }>;
} {
  const seen: Array<{ e: TarEntry; data: Buffer }> = [];
  return {
    seen,
    sink: {
      begin: (e) => seen.push({ e, data: Buffer.alloc(0) }),
      data: (b) => {
        const s = seen[seen.length - 1]!;
        s.data = Buffer.concat([s.data, b]);
      },
      end: () => undefined,
    },
  };
}

describe("tar", () => {
  const entry = (
    name: string,
    kind: number,
    size: number,
    link = "",
  ): TarEntry => ({
    name: enc.encode(name),
    kind,
    mode: 0o640,
    size,
    mtime: 1_700_000_000,
    link: enc.encode(link),
  });

  it("reads back what it writes at every chunk size", () => {
    const long = "d".repeat(150);
    const tgt = "t".repeat(130);
    const parts: Uint8Array[] = [];
    const add = (e: TarEntry, data = "") => {
      parts.push(
        writeHeader(e),
        enc.encode(data),
        new Uint8Array(padLen(data.length)),
      );
    };
    add(entry("a.txt", T_FILE, 5), "hello");
    add(entry("sub", T_DIR, 0));
    add(entry(`sub/${long}`, T_FILE, 3), "xyz");
    add(entry("l", 50, 0, tgt));
    add(entry("e", T_FILE, 0));
    parts.push(END_BLOCKS);
    const a = Buffer.concat(parts);
    for (const step of [1, 7, 511, 512, 513, 100_000]) {
      const p = new TarParser();
      const { sink, seen } = entries();
      for (let i = 0; i < a.length; i += step) {
        p.feed(a.subarray(i, i + step), sink);
      }
      p.finish();
      expect(seen.map((s) => Buffer.from(s.e.name).toString())).toEqual([
        "a.txt",
        "sub/",
        `sub/${long}`,
        "l",
        "e",
      ]);
      expect(seen[0]!.data.toString()).toBe("hello");
      expect(seen[2]!.data.toString()).toBe("xyz");
      expect(Buffer.from(seen[3]!.e.link).toString()).toBe(tgt);
      expect(seen[0]!.e.mode).toBe(0o640);
      expect(seen[0]!.e.mtime).toBe(1_700_000_000);
    }
  });

  it("writes base-256 sizes and reads them", () => {
    const big = 9 * 1024 ** 3;
    const h = writeHeader(entry("big", T_FILE, big));
    expect(h[124]).toBe(0x80);
    const p = new TarParser();
    const { sink, seen } = entries();
    p.feed(h, sink);
    expect(seen[0]!.e.size).toBe(big);
  });

  it("refuses a bad checksum and an archive without its end", () => {
    const { sink } = entries();
    const p = new TarParser();
    p.feed(rawTar([file("a")], { end: false }), sink);
    expect(() => p.finish()).toThrow(/end-of-archive/);
    expect(() =>
      new TarParser().feed(rawTar([file("a")], { corrupt: true }), sink)
    )
      .toThrow(/checksum/);
  });

  it("applies pax path and size, and ustar prefix", () => {
    const rec = (k: string, v: string) => {
      const body = ` ${k}=${v}\n`;
      let len = body.length + 1;
      while (String(len).length + body.length !== len) {
        len = String(len).length + body.length;
      }
      return `${len}${body}`;
    };
    const recs = rec("path", "pax/name") + rec("size", "3");
    const x = rawTar([["PaxHeader", "x", recs, "", 0]], { end: false });
    const f = rawTar([["short", "0", "abc", "", 0o600]], { end: false });
    const p = new TarParser();
    const { sink, seen } = entries();
    p.feed(Buffer.concat([x, f, new Uint8Array(1024)]), sink);
    p.finish();
    expect(Buffer.from(seen[0]!.e.name).toString()).toBe("pax/name");
    expect(seen[0]!.data.toString()).toBe("abc");

    const u = Buffer.from(rawTar([file("file.txt", "deep")]));
    u.set(enc.encode("pre/fix"), 345);
    u.fill(0x20, 148, 156);
    let sum = 0;
    for (let i = 0; i < 512; i++) sum += u[i]!;
    u.set(enc.encode(sum.toString(8).padStart(6, "0")), 148);
    u[154] = 0;
    const p2 = new TarParser();
    const s2 = entries();
    p2.feed(u, s2.sink);
    expect(Buffer.from(s2.seen[0]!.e.name).toString()).toBe("pre/fix/file.txt");
  });

  it("applies pax g defaults to later members and refuses a non-safe pax size", () => {
    const rec = (k: string, v: string) => {
      const body = ` ${k}=${v}\n`;
      let len = body.length + 1;
      while (String(len).length + body.length !== len) {
        len = String(len).length + body.length;
      }
      return `${len}${body}`;
    };
    const g = rawTar(
      [["./PaxHeaders.", "g", rec("path", "from-g"), "", 0]],
      { end: false },
    );
    const f = rawTar([["short", "0", "x", "", 0o600]], { end: false });
    const p = new TarParser();
    const { sink, seen } = entries();
    p.feed(Buffer.concat([g, f, new Uint8Array(1024)]), sink);
    p.finish();
    expect(Buffer.from(seen[0]!.e.name).toString()).toBe("from-g");

    const bad = rawTar(
      [["PaxHeader", "x", rec("size", "1e20"), "", 0]],
      { end: false },
    );
    expect(() => new TarParser().feed(bad, entries().sink)).toThrow(/pax size/);
  });
});

// ---- the extractor's rules --------------------------------------------------------------------

describe("extraction rules", () => {
  const run = (
    t: Uint8Array,
    prefill?: (local: string, outside: string) => void,
  ) => {
    const root = tmp();
    const local = join(root, "local");
    const outside = join(root, "outside");
    mkdirSync(local);
    mkdirSync(outside);
    prefill?.(local, outside);
    let err: unknown;
    let res;
    try {
      res = extract(t, local);
    } catch (e) {
      err = e;
    }
    expect(readdirSync(root).sort(), "nothing beside local and outside")
      .toEqual([
        "local",
        "outside",
      ]);
    expect(readdirSync(outside), "outside stays empty").toEqual([]);
    return { err: err as SandboxError | undefined, res, local };
  };
  const refused = (
    t: Uint8Array,
    why: RegExp,
    pre?: Parameters<typeof run>[1],
  ) => {
    const { err } = run(t, pre);
    expect(err?.kind).toBe("protocol");
    expect(err?.detail).toMatch(/^unsafe archive member /);
    expect(err?.detail).toMatch(why);
  };

  it("refuses .. and absolute names", () => {
    refused(rawTar([file("../escape.txt")]), /\.\./);
    refused(rawTar([file("a/b/../../../escape.txt")]), /\.\./);
    refused(rawTar([file("a/..")]), /\.\./);
    refused(rawTar([["../d/", "5", "", "", 0o755]]), /\.\./);
    refused(rawTar([file("/tmp/bulk-ts-abs.txt")]), /absolute/);
    expect(existsSync("/tmp/bulk-ts-abs.txt")).toBe(false);
  });

  it("never writes through a link", () => {
    refused(
      rawTar([["up", "2", "", "..", 0o777], file("up/escape.txt")]),
      /not a directory/,
    );
    refused(
      rawTar([file("link/f.txt")]),
      /not a directory/,
      (l, o) => symlinkSync(o, join(l, "link")),
    );
    const r = run(
      rawTar([["link/", "5", "", "", 0o755], file("link/f.txt", "safe")]),
      (l, o) => symlinkSync(o, join(l, "link")),
    );
    expect(r.err).toBeUndefined();
    expect(lstatSync(join(r.local, "link")).isDirectory()).toBe(true);
    const r2 = run(
      rawTar([file("link", "now a file")]),
      (l, o) => symlinkSync(join(o, "t"), join(l, "link")),
    );
    expect(readFileSync(join(r2.local, "link"), "utf8")).toBe("now a file");
  });

  it("creates symlinks as given and hard links only to files it wrote", () => {
    const r = run(
      rawTar([["abs", "2", "", "/etc/passwd", 0o777], [
        "up",
        "2",
        "",
        "../../..",
        0o777,
      ]]),
    );
    expect(readlinkSync(join(r.local, "abs"))).toBe("/etc/passwd");
    const h = run(
      rawTar([file("a", "data"), ["b", "1", "", "a", 0o644], [
        "./c",
        "1",
        "",
        "./a",
        0o644,
      ]]),
    );
    expect(readFileSync(join(h.local, "c"), "utf8")).toBe("data");
    refused(
      rawTar([["h", "1", "", "../../../etc/passwd", 0o644]]),
      /link target/,
    );
    refused(rawTar([["h", "1", "", "/etc/passwd", 0o644]]), /absolute/);
    refused(rawTar([["h", "1", "", "never", 0o644]]), /already wrote/);
    refused(
      rawTar([["s", "2", "", "/etc/passwd", 0o777], [
        "h",
        "1",
        "",
        "s",
        0o644,
      ]]),
      /already wrote/,
    );
  });

  it("reports unsupported types as io and malformed archives as protocol", () => {
    expect(run(rawTar([["dev", "3", "", "", 0o644]])).err).toMatchObject({
      kind: "io",
      detail: expect.stringMatching(/unsupported member type/),
    });
    expect(run(rawTar([file("a")], { corrupt: true })).err?.kind).toBe(
      "protocol",
    );
    expect(run(rawTar([file("a")], { end: false })).err?.kind).toBe("protocol");
    expect(
      run(Buffer.from("this is not a tar archive at all".repeat(40))).err?.kind,
    ).toBe("protocol");
  });

  it("normalizes names, masks set-id and applies directory modes last", () => {
    const r = run(rawTar([
      ["./", "5", "", "", 0o700],
      ["a/./b//c.txt", "0", "x", "", 0o4755],
      ["d/", "5", "", "", 0o555],
      ["d/f", "0", "f", "", 0o444],
    ]));
    expect(r.err).toBeUndefined();
    expect(lstatSync(join(r.local, "a/b/c.txt")).mode & 0o7777).toBe(0o755);
    expect(lstatSync(join(r.local, "d")).mode & 0o777).toBe(0o555);
    expect(lstatSync(join(r.local, "d/f")).mtimeMs / 1000).toBe(1500000000);
    chmodSync(join(r.local, "d"), 0o755);
  });

  it("will not replace a directory by a file", () => {
    const { err } = run(rawTar([file("d")]), (l) => mkdirSync(join(l, "d")));
    expect(err?.kind).toBe("is_directory");
  });
});

// ---- upload -----------------------------------------------------------------------------------

interface Call {
  command: string;
  args: string[];
  stdin: Buffer | undefined;
}

/** A transport double that records every call and answers by `reply`. */
class Recorder implements Transport {
  calls: Call[] = [];
  jobs = {} as JobOps;
  private reply: (c: Call) => RunResult;
  constructor(reply: (c: Call) => RunResult = () => okResult()) {
    this.reply = reply;
  }
  run(command: string, args: string[], o: RunOptions = {}): Promise<RunResult> {
    const c = {
      command,
      args,
      stdin: o.stdin ? Buffer.from(o.stdin) : undefined,
    };
    this.calls.push(c);
    return Promise.resolve(this.reply(c));
  }
  stream(): AsyncIterable<Frame> {
    throw new Error("no stream");
  }
}
const okResult = (): RunResult => ({
  exitCode: 0,
  stdout: new Uint8Array(),
  stderr: new Uint8Array(),
});

function tree(): string {
  const d = tmp();
  writeFileSync(join(d, "a.txt"), "alpha");
  mkdirSync(join(d, "sub"));
  writeFileSync(join(d, "sub", "b.bin"), Buffer.alloc(3000, 7));
  symlinkSync("a.txt", join(d, "link"));
  return d;
}

describe("upload", () => {
  it("is one command with the archive on stdin when it fits", async () => {
    const t = new Recorder();
    const r = await upload(t, tree(), "/w/dst/");
    expect(r).toEqual({ files: 2, dirs: 1, links: 1, bytes: 3005 });
    expect(t.calls).toHaveLength(1);
    const c = t.calls[0]!;
    expect(c.command).toBe("sh");
    expect(c.args).toEqual(["-c", UPLOAD_SCRIPT, "sh", "/w/dst", ""]);
    // Parses back to the tree, ends with two zero blocks.
    const p = new TarParser();
    const { sink, seen } = entries();
    p.feed(c.stdin!, sink);
    p.finish();
    expect(seen.map((s) => Buffer.from(s.e.name).toString())).toEqual([
      "a.txt",
      "link",
      "sub/",
      "sub/b.bin",
    ]);
  });

  it("stages, appends at checked offsets, extracts and never sends an empty chunk", async () => {
    const t = new Recorder();
    await upload(t, tree(), "/w/dst", { chunkBytes: 2048 });
    const n = t.calls.length;
    expect(n).toBeGreaterThan(3);
    const stage = t.calls[0]!.args[3]!;
    expect(stage).toMatch(/^\/w\/dst\.upload\.[0-9a-f]{16}$/);
    let offset = 0;
    const got: Buffer[] = [];
    for (const c of t.calls.slice(0, -1)) {
      expect(c.args.slice(0, 3)).toEqual(["-c", STAGE_SCRIPT, "sh"]);
      expect(c.args[3]).toBe(stage);
      expect(c.args[4]).toBe(String(offset));
      expect(c.args[5]).toBe(String(c.stdin!.length));
      expect(c.stdin!.length).toBeGreaterThan(0);
      expect(c.stdin!.length).toBeLessThanOrEqual(2048);
      offset += c.stdin!.length;
      got.push(c.stdin!);
    }
    expect(t.calls[n - 1]).toMatchObject({
      command: "sh",
      args: ["-c", UPLOAD_SCRIPT, "sh", "/w/dst", stage],
    });
    expect(t.calls[n - 1]!.stdin).toBeUndefined();
    const p = new TarParser();
    p.feed(Buffer.concat(got), entries().sink);
    p.finish();
  });

  it("removes the stage when a command fails, and reports it classified", async () => {
    const t = new Recorder((c) =>
      c.args[1] === STAGE_SCRIPT && c.args[4] === "2048"
        ? {
          exitCode: 1,
          stdout: new Uint8Array(),
          stderr: enc.encode("cat: write error: No space left on device\n"),
        }
        : okResult()
    );
    const e = await upload(t, tree(), "/w/dst", { chunkBytes: 2048 }).catch((
      x,
    ) => x as SandboxError);
    expect(e).toMatchObject({
      kind: "io",
      detail: "upload /w/dst: cat: write error: No space left on device",
    });
    const last = t.calls[t.calls.length - 1]!;
    expect(last.command).toBe("rm");
    expect(last.args).toEqual(["-f", "--", t.calls[0]!.args[3]]);
  });

  it("reports a destination that is a file as not_a_directory", async () => {
    const t = new Recorder(() => ({
      exitCode: 1,
      stdout: new Uint8Array(),
      stderr: enc.encode("Not a directory (os error 20)\n"),
    }));
    expect(await upload(t, tree(), "/f").catch((x) => x)).toMatchObject({
      kind: "not_a_directory",
    });
  });

  it("refuses bad requests before sending anything", async () => {
    const t = new Recorder();
    for (const dest of ["rel/dir", "/", "//"]) {
      expect(await upload(t, tree(), dest).catch((x) => x)).toMatchObject({
        kind: "invalid_request",
      });
    }
    const d = tree();
    const fifo = join(d, "pipe");
    (await import("node:child_process")).execFileSync("mkfifo", [fifo]);
    expect(await upload(t, d, "/w/x").catch((x) => x)).toMatchObject({
      kind: "invalid_request",
    });
    expect(await upload(t, join(d, "nope"), "/w/x").catch((x) => x))
      .toMatchObject({ kind: "not_found" });
    expect(await upload(t, join(d, "a.txt"), "/w/x").catch((x) => x))
      .toMatchObject({ kind: "not_a_directory" });
    expect(t.calls).toEqual([]);
  });

  it("uploads a list of paths by base name", async () => {
    const d = tree();
    const t = new Recorder();
    await upload(t, [join(d, "a.txt"), join(d, "sub")], "/w/dst");
    const p = new TarParser();
    const { sink, seen } = entries();
    p.feed(t.calls[0]!.stdin!, sink);
    expect(seen.map((s) => Buffer.from(s.e.name).toString())).toEqual([
      "a.txt",
      "sub/",
      "sub/b.bin",
    ]);
  });

  it("builds chunks as it goes: a large file is never held whole", async () => {
    const d = tmp();
    const f = join(d, "big");
    writeFileSync(f, Buffer.alloc(12 * 1024 * 1024, 1));
    const sizes: number[] = [];
    const t = new Recorder((c) => {
      if (c.stdin) sizes.push(c.stdin.length);
      return okResult();
    });
    await upload(t, d, "/w/dst", { chunkBytes: 1024 * 1024 });
    expect(Math.max(...sizes)).toBeLessThanOrEqual(1024 * 1024);
    expect(sizes.length).toBeGreaterThanOrEqual(12);
  });

  it("sends the wire shape the spec states over HTTP", async () => {
    const e = await fakeEndpoint((_, res) => json(res, 200, ok("", 0, "")));
    try {
      const t = new HttpTransport({ url: e.url });
      await upload(t, tree(), "/w/dst");
      expect(e.requests).toHaveLength(1);
      const body = e.requests[0]!.body;
      expect(body.command).toBe("sh");
      expect(body.args).toEqual(["-c", UPLOAD_SCRIPT, "sh", "/w/dst", ""]);
      expect(body.timeout_ms).toBe(600_000);
      expect(typeof body.stdin).toBe("string");
      expect(Buffer.from(body.stdin as string, "base64").length % 512).toBe(0);
    } finally {
      await e.close();
    }
  });
});

// ---- download ---------------------------------------------------------------------------------

/** A transport whose stream plays `frames`, one byte of stdout at a time. */
class Streamer implements Transport {
  jobs = {} as JobOps;
  closed = false;
  args: string[] = [];
  private archive: Uint8Array;
  private exit: number;
  private stderr: string;
  constructor(archive: Uint8Array, exit = 0, stderr = "") {
    this.archive = archive;
    this.exit = exit;
    this.stderr = stderr;
  }
  run(): Promise<RunResult> {
    throw new Error("download streams");
  }
  stream(_c: string, args: string[]): AsyncIterable<Frame> {
    this.args = args;
    const self = this;
    return (async function*() {
      try {
        for (let i = 0; i < self.archive.length; i++) {
          yield { type: "stdout", data: self.archive.subarray(i, i + 1) };
        }
        if (self.stderr) {
          yield { type: "stderr", data: enc.encode(self.stderr) };
        }
        yield { type: "exit", code: self.exit };
      } finally {
        self.closed = true;
      }
    })();
  }
}

describe("download", () => {
  it("extracts a stream of one-byte frames", async () => {
    const t = new Streamer(rawTar([
      ["d/", "5", "", "", 0o750],
      file("d/f.txt", "hello", 0o600),
      ["d/l", "2", "", "f.txt", 0o777],
    ]));
    const local = join(tmp(), "out");
    const r = await download(t, "/w/src", local);
    expect(r).toEqual({ files: 1, dirs: 1, links: 1, bytes: 5 });
    expect(t.args).toEqual(["-c", DOWNLOAD_SCRIPT, "sh", "/w/src"]);
    expect(readFileSync(join(local, "d/f.txt"), "utf8")).toBe("hello");
    expect(lstatSync(join(local, "d")).mode & 0o777).toBe(0o750);
  });

  it("reports a failed command, classified, whatever it streamed", async () => {
    const t = new Streamer(
      new Uint8Array(),
      1,
      "No such file or directory (os error 2)\n",
    );
    const e = await download(t, "/nope", join(tmp(), "o")).catch((x) => x);
    expect(e).toMatchObject({
      kind: "not_found",
      detail: "download /nope: No such file or directory (os error 2)",
    });
  });

  it("closes the stream at an unsafe member and writes nothing outside", async () => {
    const root = tmp();
    const t = new Streamer(rawTar([file("ok.txt"), file("../escape.txt")]));
    const e = await download(t, "/w/src", join(root, "local")).catch((x) => x);
    expect(e).toMatchObject({ kind: "protocol" });
    expect(t.closed).toBe(true);
    expect(readdirSync(root)).toEqual(["local"]);
  });

  it("refuses an absolute path and a truncated archive", async () => {
    const t = new Streamer(new Uint8Array());
    expect(await download(t, "rel", join(tmp(), "o")).catch((x) => x))
      .toMatchObject({ kind: "invalid_request" });
    const cut = new Streamer(
      rawTar([file("a", "x".repeat(2000))], { end: false }).subarray(0, 700),
    );
    expect(await download(cut, "/w", join(tmp(), "o")).catch((x) => x))
      .toMatchObject({ kind: "protocol" });
  });

  it("keeps only BULK_STDERR_BYTES of stderr", async () => {
    const t = new Streamer(
      new Uint8Array(),
      1,
      `first line\n${"x".repeat(200_000)}`,
    );
    const e =
      (await download(t, "/w", join(tmp(), "o")).catch((x) =>
        x
      )) as SandboxError;
    expect(e.detail).toBe("download /w: first line");
  });
});
