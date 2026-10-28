/**
 * The tar subset of FILES-OVER-EXEC.md section 5.8: header writing for upload and a push parser for
 * download. Written by hand (no dependency): the parser is *pushed* chunks as stream frames arrive,
 * so nothing sits between the network and the extractor.
 */
import { concat } from "./bytes.js";
import { SandboxError } from "./errors.js";

export const BLOCK = 512;

/** Any bytes, whatever buffer backs them. */
export type Bytes = Uint8Array<ArrayBufferLike>;

/** A member of an archive. `name` and `link` are bytes, as the archive has them. */
export interface TarEntry {
  /** The name as the archive gives it (prefix, `L` and pax already applied). */
  name: Bytes;
  /** The type flag as a character code (`0` is 48). */
  kind: number;
  mode: number;
  size: number;
  mtime: number;
  link: Bytes;
}

/** What a parser tells its consumer. */
export interface TarSink {
  begin(e: TarEntry): void;
  data(bytes: Uint8Array): void;
  end(): void;
}

const ch = (c: string): number => c.charCodeAt(0);
export const T_FILE = ch("0");
export const T_HARD = ch("1");
export const T_SYMLINK = ch("2");
export const T_DIR = ch("5");
const T_CONTIG = ch("7");

export const archiveError = (msg: string): SandboxError =>
  new SandboxError("protocol", msg);

function putOctal(h: Uint8Array, off: number, len: number, value: number) {
  const w = len - 1;
  const s = value.toString(8).padStart(w, "0").slice(-w);
  for (let i = 0; i < w; i++) h[off + i] = s.charCodeAt(i);
  h[off + w] = 0;
}

function rawHeader(
  name: Uint8Array,
  kind: number,
  mode: number,
  size: number,
  mtime: number,
  link: Uint8Array,
): Uint8Array {
  const h = new Uint8Array(BLOCK);
  h.set(name.subarray(0, 100), 0);
  putOctal(h, 100, 8, mode & 0o7777);
  putOctal(h, 108, 8, 0);
  putOctal(h, 116, 8, 0);
  if (size >= 8 ** 11) {
    // Base-256: the high bit of the first byte, then the number big-endian.
    h[124] = 0x80;
    let v = size;
    for (let i = 135; i >= 125; i--) {
      h[i] = v % 256;
      v = Math.floor(v / 256);
    }
  } else {
    putOctal(h, 124, 12, size);
  }
  putOctal(h, 136, 12, mtime);
  h[156] = kind;
  h.set(link.subarray(0, 100), 157);
  h.set([0x75, 0x73, 0x74, 0x61, 0x72, 0], 257); // "ustar\0"
  h[263] = 0x30;
  h[264] = 0x30; // "00"
  h.fill(0x20, 148, 156);
  let sum = 0;
  for (const b of h) sum += b;
  const c = sum.toString(8).padStart(6, "0");
  for (let i = 0; i < 6; i++) h[148 + i] = c.charCodeAt(i);
  h[154] = 0;
  h[155] = 0x20;
  return h;
}

function extension(kind: string, text: Uint8Array, out: Uint8Array[]) {
  const size = text.length + 1;
  out.push(
    rawHeader(
      new TextEncoder().encode("././@LongLink"),
      ch(kind),
      0,
      size,
      0,
      new Uint8Array(),
    ),
  );
  const body = new Uint8Array(size + padLen(size));
  body.set(text);
  out.push(body);
}

/** Bytes of zero padding after `size` bytes of content. */
export function padLen(size: number): number {
  return (BLOCK - (size % BLOCK)) % BLOCK;
}

/**
 * The header of one member (preceded by its `L`/`K` members when the name or target is over 100
 * bytes). `name` is relative with no trailing slash; a directory gets one.
 */
export function writeHeader(e: TarEntry): Uint8Array {
  const parts: Uint8Array[] = [];
  let name = e.name;
  if (e.kind === T_DIR) {
    name = new Uint8Array(name.length + 1);
    name.set(e.name);
    name[name.length - 1] = 0x2f;
  }
  if (name.length > 100) extension("L", name, parts);
  if (e.link.length > 100) extension("K", e.link, parts);
  parts.push(rawHeader(name, e.kind, e.mode & 0o777, e.size, e.mtime, e.link));
  return parts.length === 1 ? parts[0]! : concat(parts);
}

/** The end of the archive: two zero blocks. */
export const END_BLOCKS = new Uint8Array(2 * BLOCK);

function strField(f: Uint8Array): Uint8Array {
  const z = f.indexOf(0);
  return z < 0 ? f : f.subarray(0, z);
}

function number(f: Uint8Array, what: string): number {
  if ((f[0]! & 0x80) !== 0) {
    if (f[0] === 0xff) throw archiveError(`negative ${what} in a tar header`);
    let v = f[0]! & 0x7f;
    for (let i = 1; i < f.length; i++) {
      v = v * 256 + f[i]!;
      if (!Number.isSafeInteger(v)) {
        throw archiveError(`${what} overflows in a tar header`);
      }
    }
    return v;
  }
  let v = 0;
  let seen = false;
  for (const b of f) {
    if (b >= 0x30 && b <= 0x37) {
      seen = true;
      v = v * 8 + (b - 0x30);
      if (!Number.isSafeInteger(v)) {
        throw archiveError(`${what} overflows in a tar header`);
      }
    } else if ((b === 0x20 || b === 0) && !seen) {
      continue;
    } else if (b === 0x20 || b === 0) {
      break;
    } else {
      throw archiveError(`bad ${what} in a tar header`);
    }
  }
  return v;
}

function checksumOk(h: Uint8Array): boolean {
  let want: number;
  try {
    want = number(h.subarray(148, 156), "checksum");
  } catch {
    return false;
  }
  let unsigned = 0;
  let signed = 0;
  for (let i = 0; i < h.length; i++) {
    const b = i >= 148 && i < 156 ? 0x20 : h[i]!;
    unsigned += b;
    signed += b > 127 ? b - 256 : b;
  }
  return want === unsigned || want === signed;
}

/** The longest meta member (long name, long link, pax block) the reader holds. */
const META_MAX = 1 << 20;

type State =
  | { t: "header" }
  | { t: "data"; left: number; pad: number }
  | { t: "meta"; kind: number; left: number; pad: number; chunks: Uint8Array[] }
  | { t: "pad"; n: number }
  | { t: "done" };

interface Pax {
  path?: Bytes;
  link?: Bytes;
  size?: number;
}

/** An incremental reader of the archive subset: `feed` it chunks as they arrive. */
export class TarParser {
  private hdr = new Uint8Array(BLOCK);
  private hlen = 0;
  private state: State = { t: "header" };
  private zeroBlocks = 0;
  private longName: Bytes | undefined;
  private longLink: Bytes | undefined;
  private pax: Pax = {};

  feed(input: Uint8Array, sink: TarSink): void {
    let pos = 0;
    for (;;) {
      const s = this.state;
      if (s.t === "done") return;
      if (s.t === "header") {
        const take = Math.min(BLOCK - this.hlen, input.length - pos);
        this.hdr.set(input.subarray(pos, pos + take), this.hlen);
        this.hlen += take;
        pos += take;
        if (this.hlen < BLOCK) return;
        this.hlen = 0;
        this.header(this.hdr, sink);
      } else if (s.t === "data") {
        if (s.left > 0) {
          const take = Math.min(s.left, input.length - pos);
          if (take === 0) return;
          sink.data(input.subarray(pos, pos + take));
          pos += take;
          s.left -= take;
          if (s.left > 0) return;
        }
        sink.end();
        this.state = { t: "pad", n: s.pad };
      } else if (s.t === "meta") {
        const take = Math.min(s.left, input.length - pos);
        if (take > 0) s.chunks.push(input.slice(pos, pos + take));
        pos += take;
        s.left -= take;
        if (s.left > 0) return;
        this.meta(s.kind, concat(s.chunks));
        this.state = { t: "pad", n: s.pad };
      } else {
        const take = Math.min(s.n, input.length - pos);
        pos += take;
        s.n -= take;
        if (s.n > 0) return;
        this.state = { t: "header" };
      }
    }
  }

  /** The input is over: it must have ended at the end-of-archive block. */
  finish(): void {
    const s = this.state;
    if (s.t === "done") return;
    if (s.t === "header" && this.hlen === 0 && this.zeroBlocks > 0) return;
    throw archiveError("the archive ended before its end-of-archive block");
  }

  private header(h: Uint8Array, sink: TarSink): void {
    if (h.every((b) => b === 0)) {
      this.zeroBlocks++;
      if (this.zeroBlocks >= 2) this.state = { t: "done" };
      return;
    }
    this.zeroBlocks = 0;
    if (!checksumOk(h)) throw archiveError("bad tar header checksum");
    const kind = h[156] === 0 ? T_FILE : h[156]!;
    let size = number(h.subarray(124, 136), "size");
    const pad = (n: number) => padLen(n);
    if (
      kind === ch("L") || kind === ch("K") || kind === ch("x")
      || kind === ch("g")
    ) {
      if (size > META_MAX) {
        throw archiveError(`a ${size}-byte tar extension header`);
      }
      this.state = { t: "meta", kind, left: size, pad: pad(size), chunks: [] };
      return;
    }
    let name: Bytes = strField(h.subarray(0, 100)).slice();
    // POSIX ustar only: a GNU header ("ustar  \0") keeps other fields where `prefix` is.
    const magic = h.subarray(257, 263);
    const ustar = magic[0] === 0x75 && magic[1] === 0x73 && magic[2] === 0x74
      && magic[3] === 0x61 && magic[4] === 0x72 && magic[5] === 0;
    if (ustar) {
      const prefix = strField(h.subarray(345, 500));
      if (prefix.length > 0) name = concat([prefix, Uint8Array.of(0x2f), name]);
    }
    let link: Bytes = strField(h.subarray(157, 257)).slice();
    if (this.longName) name = this.longName;
    if (this.longLink) link = this.longLink;
    this.longName = undefined;
    this.longLink = undefined;
    const pax = this.pax;
    this.pax = {};
    if (pax.path) name = pax.path;
    if (pax.link) link = pax.link;
    if (pax.size !== undefined) size = pax.size;
    const entry: TarEntry = {
      name,
      kind,
      mode: number(h.subarray(100, 108), "mode"),
      size,
      mtime: safeNumber(h.subarray(136, 148)),
      link,
    };
    // Only regular files carry content; a link or directory with a size has none.
    const content = kind === T_FILE || kind === T_CONTIG ? size : 0;
    sink.begin(entry);
    this.state = { t: "data", left: content, pad: pad(content) };
  }

  private meta(kind: number, buf: Uint8Array): void {
    if (kind === ch("L")) this.longName = strField(buf).slice();
    else if (kind === ch("K")) this.longLink = strField(buf).slice();
    else if (kind === ch("x")) {
      // `LEN KEY=VALUE\n`, LEN counting the whole record.
      let rest = buf;
      while (rest.length > 0) {
        const sp = rest.indexOf(0x20);
        if (sp < 0) throw archiveError("bad pax record");
        const lenStr = new TextDecoder().decode(rest.subarray(0, sp));
        const len = /^\d+$/.test(lenStr) ? Number(lenStr) : NaN;
        if (!Number.isInteger(len)) throw archiveError("bad pax record length");
        if (len <= sp + 1 || len > rest.length) {
          throw archiveError("bad pax record length");
        }
        let rec = rest.subarray(sp + 1, len);
        if (rec[rec.length - 1] === 0x0a) rec = rec.subarray(0, rec.length - 1);
        const eq = rec.indexOf(0x3d);
        if (eq >= 0) {
          const k = new TextDecoder().decode(rec.subarray(0, eq));
          const v = rec.slice(eq + 1);
          if (k === "path") this.pax.path = v;
          else if (k === "linkpath") this.pax.link = v;
          else if (k === "size") {
            const n = Number(new TextDecoder().decode(v));
            if (Number.isInteger(n) && n >= 0) this.pax.size = n;
          }
        }
        rest = rest.subarray(len);
      }
    }
    // `g`: global records are not used.
  }
}

function safeNumber(f: Uint8Array): number {
  try {
    return number(f, "mtime");
  } catch {
    return 0;
  }
}
