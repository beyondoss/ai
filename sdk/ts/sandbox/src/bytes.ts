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

const B64_RE =
  /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;

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

/** Strict standard base64 with padding; whitespace is ignored. Returns null if it is not base64. */
export function fromBase64(s: string): Uint8Array | null {
  const clean = s.replace(/\s+/g, "");
  if (!B64_RE.test(clean)) return null;
  if (typeof Buffer !== "undefined") {
    const b = Buffer.from(clean, "base64");
    return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  }
  const bin = atob(clean);
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
