/**
 * The bulk-copy ops of the conformance driver (DRIVER.md, "Trees"): build a local tree from a
 * request, run the client on it, and report the tree an extraction left.
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Transfer } from "./bulk.js";
import { fromBase64 } from "./bytes.js";
import { SandboxError } from "./errors.js";

type Json = Record<string, unknown>;

export class BulkReject extends Error {
  readonly kind: string;
  readonly detail: string;

  constructor(kind: string, detail: string) {
    super(detail);
    this.kind = kind;
    this.detail = detail;
  }
}

const invalid = (d: string) => new BulkReject("invalid_request", d);

function field(e: Json, key: string): string {
  const v = e[key];
  if (typeof v !== "string") throw invalid(`\`${key}\` must be a string`);
  return v;
}

const io = (e: unknown): never => {
  if (e instanceof SandboxError || e instanceof BulkReject) throw e;
  throw new BulkReject("io", e instanceof Error ? e.message : String(e));
};

/** Build the local tree a request describes. Directory modes are applied last. */
export function materialize(root: string, entries: Json[], outside: string) {
  const dirs: Array<[string, number]> = [];
  const times: Array<[string, number]> = [];
  try {
    for (const e of entries) {
      const rel = field(e, "path");
      const path = join(root, rel);
      // The directories this creates are 0755 whatever the umask.
      const parent = join(path, "..");
      const missing: string[] = [];
      for (let p = parent; p !== root && !fs.existsSync(p); p = join(p, "..")) {
        missing.push(p);
      }
      fs.mkdirSync(parent, { recursive: true });
      for (const m of missing) dirs.push([m, 0o755]);
      const modeStr = e.mode;
      let mode = 0o644;
      if (typeof modeStr === "string") {
        if (!/^[0-7]+$/.test(modeStr)) throw invalid(`bad mode ${modeStr}`);
        mode = parseInt(modeStr, 8);
      }
      const type = typeof e.type === "string" ? e.type : "file";
      if (type === "dir") {
        fs.mkdirSync(path, { recursive: true });
        const at = dirs.findIndex(([p]) => p === path);
        if (at >= 0) dirs.splice(at, 1);
        dirs.push([path, typeof modeStr === "string" ? mode : 0o755]);
      } else if (type === "symlink") {
        fs.symlinkSync(
          field(e, "target").replaceAll("${OUTSIDE}", outside),
          path,
        );
      } else if (type === "fifo") {
        try {
          execFileSync("mkfifo", [path]);
        } catch {
          throw new BulkReject("io", "mkfifo failed");
        }
      } else {
        const fd = fs.openSync(path, "w");
        try {
          if (typeof e.pattern_b64 === "string") {
            const pat = fromBase64(e.pattern_b64);
            if (!pat) throw invalid("pattern_b64");
            let left = typeof e.size === "number" ? e.size : 0;
            const step = Math.max(
              pat.length,
              Math.floor((1 << 20) / pat.length) * pat.length,
            );
            const block = Buffer.alloc(step);
            for (let i = 0; i < step; i++) block[i] = pat[i % pat.length]!;
            while (left > 0) {
              const n = Math.min(left, step);
              fs.writeSync(fd, block, 0, n);
              left -= n;
            }
          } else if (typeof e.data_b64 === "string") {
            const d = fromBase64(e.data_b64);
            if (!d) throw invalid("data_b64");
            fs.writeSync(fd, d);
          }
        } finally {
          fs.closeSync(fd);
        }
        fs.chmodSync(path, mode);
        if (typeof e.mtime === "number") times.push([path, e.mtime]);
      }
    }
    for (const [p, t] of times) fs.utimesSync(p, t, t);
    dirs.sort((a, b) => b[0].split("/").length - a[0].split("/").length);
    for (const [p, m] of dirs) fs.chmodSync(p, m);
  } catch (e) {
    io(e);
  }
}

const modeStr = (m: number) => (m & 0o7777).toString(8).padStart(4, "0");

/** What is under `root`, sorted by path. */
export function describe(root: string): Json[] {
  const out: Json[] = [];
  const walk = (dir: string) => {
    for (const name of fs.readdirSync(dir).sort()) {
      const p = join(dir, name);
      const rel = p.slice(root.length + 1);
      const st = fs.lstatSync(p);
      if (st.isSymbolicLink()) {
        out.push({ path: rel, type: "symlink", target: fs.readlinkSync(p) });
      } else if (st.isDirectory()) {
        out.push({
          path: rel,
          type: "dir",
          mode: modeStr(st.mode),
          mtime: Math.floor(st.mtimeMs / 1000),
        });
        // A mode-000 directory is listed, not entered.
        if ((st.mode & 0o500) === 0o500) walk(p);
      } else {
        const h = createHash("sha256");
        const fd = fs.openSync(p, "r");
        const buf = Buffer.allocUnsafe(1 << 20);
        try {
          for (;;) {
            const n = fs.readSync(fd, buf, 0, buf.length, null);
            if (n === 0) break;
            h.update(buf.subarray(0, n));
          }
        } finally {
          fs.closeSync(fd);
        }
        out.push({
          path: rel,
          type: "file",
          mode: modeStr(st.mode),
          size: st.size,
          sha256: h.digest("hex"),
          mtime: Math.floor(st.mtimeMs / 1000),
        });
      }
    }
  };
  walk(root);
  return out;
}

export function forceRemove(dir: string) {
  const chmodAll = (p: string) => {
    try {
      const st = fs.lstatSync(p);
      if (!st.isDirectory()) return;
      fs.chmodSync(p, 0o700);
      for (const n of fs.readdirSync(p)) chmodAll(join(p, n));
    } catch {
      // best effort
    }
  };
  chmodAll(dir);
  fs.rmSync(dir, { recursive: true, force: true });
}

/**
 * Run a local extraction with the sentinel layout: the target is `<tmp>/local`, `<tmp>/outside`
 * is where a `${OUTSIDE}` link points, and anything else that appears in `<tmp>` is a write
 * outside the target, which fails the op.
 */
export async function withSandboxDir(
  req: Json,
  run: (local: string) => Promise<Transfer> | Transfer,
): Promise<Json> {
  const tmp = fs.mkdtempSync(join(tmpdir(), "bulk-"));
  try {
    const local = join(tmp, "local");
    const outside = join(tmp, "outside");
    fs.mkdirSync(local);
    fs.mkdirSync(outside);
    if (Array.isArray(req.prefill)) {
      materialize(local, req.prefill as Json[], outside);
    }
    let result: Transfer | undefined;
    let failure: unknown;
    try {
      result = await run(local);
    } catch (e) {
      failure = e;
    }
    for (const name of fs.readdirSync(tmp)) {
      if (name !== "local" && name !== "outside") {
        throw new BulkReject(
          "io",
          `the extraction wrote outside its directory: ${name}`,
        );
      }
    }
    if (fs.readdirSync(outside).length > 0) {
      throw new BulkReject(
        "io",
        "the extraction wrote through a link into `outside`",
      );
    }
    if (failure !== undefined) throw failure;
    return { ...result!, entries: describe(local) };
  } finally {
    forceRemove(tmp);
  }
}
