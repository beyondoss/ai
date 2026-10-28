/**
 * Bulk copy: `upload` and `download` of whole trees as a tar stream over raw exec
 * (FILES-OVER-EXEC.md section 5.8).
 *
 * Layer 2 like {@link Files}: built only on {@link Transport}. The archive is written and read by
 * this package (`tar.ts`), so the client's machine needs no `tar`; the sandbox needs `sh` and a
 * `tar`, GNU or busybox. The local side of a download is untrusted input and is extracted by rules
 * that never write through a link.
 */
import * as fs from "node:fs";
import * as fsp from "node:fs/promises";
import { basename } from "node:path";
import { text, utf8 } from "./bytes.js";
import {
  classifyFailure,
  InvalidRequestError,
  SandboxError,
} from "./errors.js";
import type { ErrorKind } from "./errors.js";
import { DOWNLOAD_SCRIPT, STAGE_SCRIPT, UPLOAD_SCRIPT } from "./scripts.js";
import {
  archiveError,
  BLOCK,
  END_BLOCKS,
  padLen,
  T_DIR,
  T_FILE,
  T_HARD,
  T_SYMLINK,
  TarParser,
  writeHeader,
} from "./tar.js";
import type { TarEntry, TarSink } from "./tar.js";
import type { Transport } from "./transport.js";

/**
 * Raw tar bytes per upload request. 16 MiB is 21.3 MB of base64: under the 64 MiB an `execd`
 * accepts, and equal to the one-shot response cap. A call holds about two chunks.
 */
export const BULK_CHUNK_BYTES = 16 * 1024 * 1024;
/** The deadline of the extraction command and of the download stream: a hung `tar` ends. */
export const BULK_TIMEOUT_MS = 600_000;
/** The stderr a download keeps for its error detail. */
export const BULK_STDERR_BYTES = 64 * 1024;
/** The deadline of a stage append and of a cleanup, the helpers' usual. */
const STAGE_TIMEOUT_MS = 120_000;

/** What a copy moved: files, directories (not the root), links, and bytes of file content. */
export interface Transfer {
  files: number;
  dirs: number;
  links: number;
  bytes: number;
}

export interface UploadOptions {
  /** Raw tar bytes per request (at least 512); default {@link BULK_CHUNK_BYTES}. */
  chunkBytes?: number | undefined;
}

const sh = (script: string, args: string[]): string[] => [
  "-c",
  script,
  "sh",
  ...args,
];

function remotePath(what: string, p: string, notRoot: boolean): void {
  if (!p.startsWith("/")) {
    throw new InvalidRequestError(
      `${what} must be absolute: ${JSON.stringify(p)}`,
    );
  }
  if (notRoot && p.replace(/\/+$/, "") === "") {
    throw new InvalidRequestError(`${what} must not be /`);
  }
  if (p.includes("\0")) throw new InvalidRequestError(`${what} has a NUL`);
}

/** A local filesystem failure in the spec's vocabulary, from the OS error code. */
export function localError(path: string, e: unknown): SandboxError {
  if (e instanceof SandboxError) return e;
  const err = e as NodeJS.ErrnoException;
  const msg = err instanceof Error ? err.message : String(e);
  let kind: ErrorKind = "io";
  switch (err?.code) {
    case "ENOENT":
      kind = "not_found";
      break;
    case "EACCES":
    case "EPERM":
      kind = "permission_denied";
      break;
    case "ENOTDIR":
      kind = "not_a_directory";
      break;
    case "EISDIR":
      kind = "is_directory";
      break;
  }
  return new SandboxError(kind, `${path}: ${msg}`, { cause: e });
}

// ---- upload -----------------------------------------------------------------------------------

const strict = new TextDecoder("utf-8", { fatal: true });

function validUtf8(b: Uint8Array): boolean {
  try {
    strict.decode(b);
    return true;
  } catch {
    return false;
  }
}

const SLASH = Buffer.from("/");

/**
 * The archive of `sources`, in chunks of `chunk` bytes (the last one may be shorter, or empty).
 * An async generator: the tree is read as the consumer asks for chunks, so a call holds a chunk or
 * two and not the archive.
 */
async function* build(
  sources: string | string[],
  chunk: number,
  counts: Transfer,
): AsyncGenerator<Uint8Array, void, void> {
  let buf = Buffer.allocUnsafe(chunk);
  let pos = 0;

  async function* put(bytes: Uint8Array): AsyncGenerator<Uint8Array> {
    let o = 0;
    while (o < bytes.length) {
      const n = Math.min(chunk - pos, bytes.length - o);
      buf.set(bytes.subarray(o, o + n), pos);
      pos += n;
      o += n;
      if (pos === chunk) {
        yield buf;
        buf = Buffer.allocUnsafe(chunk);
        pos = 0;
      }
    }
  }

  async function* add(name: Buffer, path: Buffer): AsyncGenerator<Uint8Array> {
    const shown = path.toString();
    if (!validUtf8(name)) {
      throw new InvalidRequestError(`${shown}: the name is not valid UTF-8`);
    }
    let st: fs.Stats;
    try {
      st = await fsp.lstat(path);
    } catch (e) {
      throw localError(shown, e);
    }
    const e: TarEntry = {
      name,
      kind: T_FILE,
      mode: st.mode & 0o777,
      size: 0,
      mtime: Math.max(0, Math.floor(st.mtimeMs / 1000)),
      link: new Uint8Array(),
    };
    if (st.isDirectory()) {
      e.kind = T_DIR;
      yield* put(writeHeader(e));
      counts.dirs++;
      let names: Buffer[];
      try {
        names = await fsp.readdir(path, { encoding: "buffer" });
      } catch (err) {
        throw localError(shown, err);
      }
      names.sort(Buffer.compare);
      for (const n of names) {
        const child = name.length === 0 ? n : Buffer.concat([name, SLASH, n]);
        yield* add(child, Buffer.concat([path, SLASH, n]));
      }
    } else if (st.isSymbolicLink()) {
      e.kind = T_SYMLINK;
      e.mode = 0o777;
      try {
        e.link = await fsp.readlink(path, { encoding: "buffer" });
      } catch (err) {
        throw localError(shown, err);
      }
      yield* put(writeHeader(e));
      counts.links++;
    } else if (st.isFile()) {
      e.size = st.size;
      yield* put(writeHeader(e));
      let fh: fsp.FileHandle;
      try {
        fh = await fsp.open(path, "r");
      } catch (err) {
        throw localError(shown, err);
      }
      try {
        let left = e.size;
        while (left > 0) {
          const want = Math.min(left, 64 * 1024, chunk - pos);
          let got: number;
          try {
            got = (await fh.read(buf, pos, want, null)).bytesRead;
          } catch (err) {
            throw localError(shown, err);
          }
          if (got === 0) {
            throw new SandboxError(
              "io",
              `${shown}: changed while being archived`,
            );
          }
          pos += got;
          left -= got;
          counts.bytes += got;
          if (pos === chunk) {
            yield buf;
            buf = Buffer.allocUnsafe(chunk);
            pos = 0;
          }
        }
      } finally {
        await fh.close();
      }
      yield* put(new Uint8Array(padLen(e.size)));
      counts.files++;
    } else {
      const what = st.isFIFO()
        ? "a FIFO"
        : st.isSocket()
        ? "a socket"
        : "a device";
      throw new InvalidRequestError(`${shown}: ${what} cannot be uploaded`);
    }
  }

  if (typeof sources === "string") {
    let st: fs.Stats;
    try {
      st = await fsp.stat(sources);
    } catch (e) {
      throw localError(sources, e);
    }
    if (!st.isDirectory()) {
      throw new SandboxError(
        "not_a_directory",
        `${sources}: Not a directory (os error 20)`,
      );
    }
    let names: Buffer[];
    try {
      names = await fsp.readdir(Buffer.from(sources), { encoding: "buffer" });
    } catch (e) {
      throw localError(sources, e);
    }
    names.sort(Buffer.compare);
    for (const n of names) {
      yield* add(n, Buffer.concat([Buffer.from(sources), SLASH, n]));
    }
  } else {
    for (const p of sources) {
      const base = basename(p);
      if (base === "" || base === "/") {
        throw new InvalidRequestError(`${p}: no base name`);
      }
      yield* add(Buffer.from(base), Buffer.from(p));
    }
  }
  yield* put(END_BLOCKS);
  yield buf.subarray(0, pos);
}

function suffix(): string {
  const b = new Uint8Array(8);
  globalThis.crypto.getRandomValues(b);
  return [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
}

async function runChecked(
  t: Transport,
  what: string,
  dest: string,
  script: string,
  args: string[],
  timeoutMs: number,
  stdin?: Uint8Array,
): Promise<void> {
  const r = await t.run("sh", sh(script, args), {
    timeoutMs,
    ...(stdin !== undefined ? { stdin } : {}),
  });
  if (r.exitCode !== 0) {
    throw classifyFailure(`${what} ${dest}`, text(r.stderr));
  }
}

/**
 * Upload `sources` into the sandbox directory `remoteDir` (created if missing). `sources` is a
 * local directory (its contents land directly in `remoteDir`) or a list of paths (each lands
 * under its base name).
 */
export async function upload(
  transport: Transport,
  sources: string | string[],
  remoteDir: string,
  options: UploadOptions = {},
): Promise<Transfer> {
  remotePath("remote_dir", remoteDir, true);
  const dest = remoteDir.replace(/\/+$/, "");
  const chunk = Math.max(options.chunkBytes ?? BULK_CHUNK_BYTES, BLOCK);
  const counts: Transfer = { files: 0, dirs: 0, links: 0, bytes: 0 };
  const it = build(sources, chunk, counts);
  try {
    let cur = await it.next();
    if (cur.done) throw new SandboxError("io", "upload aborted");
    let next = await it.next();
    if (next.done) {
      // The whole archive is one chunk: one command with it on stdin.
      await runChecked(
        transport,
        "upload",
        dest,
        UPLOAD_SCRIPT,
        [dest, ""],
        BULK_TIMEOUT_MS,
        cur.value,
      );
      return counts;
    }
    const stage = `${dest}.upload.${suffix()}`;
    try {
      let offset = 0;
      for (;;) {
        const bytes = cur.value as Uint8Array;
        const last = next.done === true;
        if (bytes.length > 0) {
          await runChecked(
            transport,
            "upload",
            dest,
            STAGE_SCRIPT,
            [stage, String(offset), String(bytes.length)],
            STAGE_TIMEOUT_MS,
            bytes,
          );
          offset += bytes.length;
        }
        if (last) {
          await runChecked(transport, "upload", dest, UPLOAD_SCRIPT, [
            dest,
            stage,
          ], BULK_TIMEOUT_MS);
          return counts;
        }
        cur = next;
        next = await it.next();
      }
    } catch (e) {
      // The compensating action: the scripts remove the stage on the failures they see.
      await transport
        .run("rm", ["-f", "--", stage], { timeoutMs: STAGE_TIMEOUT_MS })
        .catch(() => undefined);
      throw e;
    }
  } finally {
    await it.return();
  }
}

// ---- extraction -------------------------------------------------------------------------------

/** Escape control characters the way a debug print would, for messages. */
function shown(name: Uint8Array): string {
  return text(name).replace(/[\u0000-\u001f\u007f\\"]/g, (c) => {
    switch (c) {
      case "\n":
        return "\\n";
      case "\t":
        return "\\t";
      case "\r":
        return "\\r";
      case "\\":
        return "\\\\";
      case "\"":
        return "\\\"";
      default:
        return `\\u{${c.charCodeAt(0).toString(16)}}`;
    }
  });
}

function unsafeMember(name: Uint8Array, why: string): SandboxError {
  return archiveError(`unsafe archive member ${shown(name)}: ${why}`);
}

/** A name or link target read the way rule 1 reads it: the components below the root. */
function components(name: Uint8Array): string[] | string {
  if (name.includes(0)) return "it has a NUL";
  if (name[0] === 0x2f) return "it is absolute";
  let s: string;
  try {
    s = strict.decode(name);
  } catch {
    return "it is not valid UTF-8";
  }
  const out: string[] = [];
  for (const c of s.split("/")) {
    if (c === "" || c === ".") continue;
    if (c === "..") return "it has a \"..\" component";
    out.push(c);
  }
  return out;
}

interface Open {
  fd: number;
  path: string;
  mode: number;
  mtime: number;
}

/** Extracts an untrusted archive under a directory by the rules of section 5.8. */
export class Extractor implements TarSink {
  readonly counts: Transfer = { files: 0, dirs: 0, links: 0, bytes: 0 };
  private current: Open | undefined;
  private dirs: Array<
    { depth: number; path: string; mode: number; mtime: number }
  > = [];
  private files = new Set<string>();

  private readonly root: string;

  constructor(root: string) {
    this.root = root;
  }

  /** Make every directory above the last of `comps` a real directory; the member's path. */
  private parents(name: Uint8Array, comps: string[]): string {
    let p = this.root;
    for (const c of comps.slice(0, -1)) {
      p = `${p}/${c}`;
      let st: fs.Stats | undefined;
      try {
        st = fs.lstatSync(p);
      } catch (e) {
        if ((e as NodeJS.ErrnoException).code !== "ENOENT") {
          throw localError(p, e);
        }
      }
      if (st === undefined) {
        try {
          fs.mkdirSync(p, { mode: 0o755 });
          fs.chmodSync(p, 0o755);
        } catch (e) {
          throw localError(p, e);
        }
      } else if (!st.isDirectory()) {
        throw unsafeMember(
          name,
          "a parent is not a directory (a link or a file)",
        );
      }
    }
    return `${p}/${comps[comps.length - 1]!}`;
  }

  /** Remove a non-directory in the way of a member; true when a directory is there. */
  private clear(path: string): boolean {
    let st: fs.Stats;
    try {
      st = fs.lstatSync(path);
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") return false;
      throw localError(path, e);
    }
    if (st.isDirectory()) return true;
    try {
      fs.unlinkSync(path);
    } catch (e) {
      throw localError(path, e);
    }
    return false;
  }

  private isDirectory(path: string): SandboxError {
    return new SandboxError(
      "is_directory",
      `${path}: Is a directory (os error 21)`,
    );
  }

  begin(e: TarEntry): void {
    const comps = components(e.name);
    if (typeof comps === "string") throw unsafeMember(e.name, comps);
    if (comps.length === 0) {
      // The root: sets nothing on the directory the caller chose.
      if (e.kind === T_DIR) return;
      throw unsafeMember(e.name, "it names the root");
    }
    const key = comps.join("/");
    switch (e.kind) {
      case T_DIR: {
        const path = this.parents(e.name, comps);
        if (!this.clear(path)) {
          try {
            fs.mkdirSync(path, { mode: 0o700 });
          } catch (err) {
            throw localError(path, err);
          }
        }
        this.dirs.push({
          depth: comps.length,
          path,
          mode: e.mode,
          mtime: e.mtime,
        });
        this.counts.dirs++;
        break;
      }
      case T_FILE:
      case 0x37: {
        const path = this.parents(e.name, comps);
        if (this.clear(path)) throw this.isDirectory(path);
        let fd: number;
        try {
          // "wx" is O_CREAT|O_EXCL, which never follows a link at the final component.
          fd = fs.openSync(path, "wx", 0o600);
        } catch (err) {
          throw localError(path, err);
        }
        this.files.add(key);
        this.counts.files++;
        this.current = { fd, path, mode: e.mode, mtime: e.mtime };
        break;
      }
      case T_SYMLINK: {
        const path = this.parents(e.name, comps);
        if (this.clear(path)) throw this.isDirectory(path);
        try {
          fs.symlinkSync(Buffer.from(e.link), path);
        } catch (err) {
          throw localError(path, err);
        }
        this.counts.links++;
        break;
      }
      case T_HARD: {
        const target = components(e.link);
        if (typeof target === "string") {
          throw unsafeMember(e.name, `its link target is unsafe: ${target}`);
        }
        const tkey = target.join("/");
        if (target.length === 0 || !this.files.has(tkey)) {
          throw unsafeMember(
            e.name,
            "a hard link must name a file this archive already wrote",
          );
        }
        const path = this.parents(e.name, comps);
        if (this.clear(path)) throw this.isDirectory(path);
        const src = `${this.root}/${tkey}`;
        let ok = false;
        try {
          ok = fs.lstatSync(src).isFile();
        } catch {
          ok = false;
        }
        if (!ok) throw unsafeMember(e.name, "its link target is not a file");
        try {
          fs.linkSync(src, path);
        } catch (err) {
          throw localError(path, err);
        }
        this.files.add(key);
        this.counts.links++;
        break;
      }
      default:
        throw new SandboxError(
          "io",
          `${text(e.name)}: unsupported member type '${
            String.fromCharCode(e.kind)
          }'`,
        );
    }
  }

  data(bytes: Uint8Array): void {
    const cur = this.current;
    if (!cur) return;
    let o = 0;
    try {
      while (o < bytes.length) {
        o += fs.writeSync(cur.fd, bytes, o, bytes.length - o);
      }
    } catch (e) {
      throw localError(cur.path, e);
    }
    this.counts.bytes += bytes.length;
  }

  end(): void {
    const cur = this.current;
    if (!cur) return;
    this.current = undefined;
    try {
      fs.fchmodSync(cur.fd, cur.mode & 0o777);
      fs.futimesSync(cur.fd, cur.mtime, cur.mtime);
    } catch (e) {
      throw localError(cur.path, e);
    } finally {
      fs.closeSync(cur.fd);
    }
  }

  /** Release an open file after a failure. */
  abort(): void {
    if (this.current) {
      try {
        fs.closeSync(this.current.fd);
      } catch {
        // already closed
      }
      this.current = undefined;
    }
  }

  /** The directories' modes and mtimes, deepest first. */
  finish(): Transfer {
    this.dirs.sort((a, b) => b.depth - a.depth);
    for (const d of this.dirs) {
      try {
        fs.chmodSync(d.path, d.mode & 0o777);
        fs.utimesSync(d.path, d.mtime, d.mtime);
      } catch (e) {
        throw localError(d.path, e);
      }
    }
    this.dirs = [];
    return this.counts;
  }
}

function makeDir(dir: string): void {
  try {
    fs.mkdirSync(dir, { recursive: true });
  } catch (e) {
    throw localError(dir, e);
  }
}

/**
 * Extract an archive already in memory under `localDir` (created if missing): the local half of a
 * download.
 */
export function extract(archive: Uint8Array, localDir: string): Transfer {
  makeDir(localDir);
  const parser = new TarParser();
  const ex = new Extractor(localDir);
  try {
    parser.feed(archive, ex);
    parser.finish();
    return ex.finish();
  } finally {
    ex.abort();
  }
}

// ---- download ---------------------------------------------------------------------------------

/**
 * Download `remotePath` (a directory's contents, or one file) into `localDir`, created if missing.
 * The archive is streamed through the extractor and never held whole.
 */
export async function download(
  transport: Transport,
  remotePathArg: string,
  localDir: string,
): Promise<Transfer> {
  remotePath("remote_path", remotePathArg, false);
  makeDir(localDir);
  const parser = new TarParser();
  const ex = new Extractor(localDir);
  const err: Uint8Array[] = [];
  let errLen = 0;
  let code: number | undefined;
  try {
    for await (
      const f of transport.stream("sh", sh(DOWNLOAD_SCRIPT, [remotePathArg]), {
        timeoutMs: BULK_TIMEOUT_MS,
      })
    ) {
      if (f.type === "stdout") {
        // A throw leaves the loop, which closes the stream and kills the command.
        parser.feed(f.data, ex);
      } else if (f.type === "stderr") {
        const room = BULK_STDERR_BYTES - errLen;
        if (room > 0) {
          const part = f.data.subarray(0, room);
          err.push(part);
          errLen += part.length;
        }
      } else if (f.type === "exit") code = f.code;
      else if (f.type === "signal") code = 128 + f.signal;
      else if (f.type === "timed_out") {
        code = 124;
        err.push(utf8("command exceeded its timeout\n"));
      }
    }
  } finally {
    ex.abort();
  }
  if (code === undefined || code !== 0) {
    const stderr = text(Buffer.concat(err));
    throw classifyFailure(`download ${remotePathArg}`, stderr);
  }
  parser.finish();
  return ex.finish();
}
