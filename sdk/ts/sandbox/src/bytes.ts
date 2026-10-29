const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: false });

/** UTF-8 encode. */
export function utf8(s: string): Uint8Array {
  return encoder.encode(s);
}

/** UTF-8 decode, bad bytes as U+FFFD (the endpoint's own rule for the text it sends). */
export function text(b: Uint8Array): string {
  return decoder.decode(b);
}

/** Standard base64 (RFC 4648 section 4) with padding. */
export function toBase64(bytes: Uint8Array): string {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength)
      .toString("base64");
  }
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(s);
}

/** Whitespace as JavaScript's `\s` matches it (WhiteSpace and LineTerminator). */
function isSpace(c: number): boolean {
  if (c < 128) return c === 32 || (c >= 9 && c <= 13);
  return (
    c === 0xa0
    || c === 0x1680
    || (c >= 0x2000 && c <= 0x200a)
    || c === 0x2028
    || c === 0x2029
    || c === 0x202f
    || c === 0x205f
    || c === 0x3000
    || c === 0xfeff
  );
}

/** Removes every whitespace char in one linear pass. Returns `s` itself when there is none. */
function stripSpace(s: string): string {
  let out = "";
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    if (isSpace(s.charCodeAt(i))) {
      out += s.slice(start, i);
      start = i + 1;
    }
  }
  return start === 0 ? s : out + s.slice(start);
}

/**
 * Strict standard base64 with padding; whitespace is ignored. Returns null if it is not base64.
 * Decodes with the platform, then re-encodes and compares: that rejects anything the decoder
 * silently skips (bad characters, wrong padding, bad length) and non-canonical trailing bits,
 * as the Rust client does. Both calls are native and linear.
 *
 * Measured (Node 22): a JS alphabet/trailing-bits pre-check + single decode was *slower* than
 * this native decode+re-encode pair on multi-MiB inputs — keep the re-encode on Node.
 */
export function fromBase64(s: string): Uint8Array | null {
  const clean = stripSpace(s);
  if (typeof Buffer !== "undefined") {
    const b = Buffer.from(clean, "base64");
    if (b.toString("base64") !== clean) return null;
    return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  }
  let bin: string;
  try {
    bin = atob(clean);
  } catch {
    return null;
  }
  if (btoa(bin) !== clean) return null;
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Concatenate chunks into one array. */
export function concat(chunks: Uint8Array[]): Uint8Array {
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}
