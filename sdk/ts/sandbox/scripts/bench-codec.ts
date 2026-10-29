#!/usr/bin/env npx tsx
/**
 * Before/after microbenches for the HTTP codec hot path.
 *
 * Legacy implementations are inlined (the pre-fix code). Current production
 * code is imported from src/.
 *
 *   cd sdk/ts/sandbox && npx --yes tsx scripts/bench-codec.ts
 */
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import { fromBase64, toBase64 } from "../src/bytes.js";
import { excerpt as newExcerpt } from "../src/http.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");

// ---- legacy (pre-fix) implementations -------------------------------------------------------

function legacyIsSpace(c: number): boolean {
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

function legacyStripSpace(s: string): string {
  let out = "";
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    if (legacyIsSpace(s.charCodeAt(i))) {
      out += s.slice(start, i);
      start = i + 1;
    }
  }
  return start === 0 ? s : out + s.slice(start);
}

/** Pre-fix: platform decode + re-encode compare. */
function legacyFromBase64(s: string): Uint8Array | null {
  const clean = legacyStripSpace(s);
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

const decoder = new TextDecoder("utf-8", { fatal: false });
function text(b: Uint8Array): string {
  return decoder.decode(b);
}

function legacyExcerpt(body: Uint8Array): string {
  return Array.from(text(body).trim()).slice(0, 400).join("");
}

function concat(chunks: Uint8Array[]): Uint8Array {
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

/** Pre-fix lines(): slice line + slice remainder every newline. */
function legacySplitLines(chunks: Uint8Array[]): Uint8Array[] {
  const lines: Uint8Array[] = [];
  let buf = new Uint8Array(0);
  let scanned = 0;
  for (const value of chunks) {
    buf = buf.length === 0 ? new Uint8Array(value) : concat([buf, value]);
    for (;;) {
      const nl = buf.indexOf(0x0a, scanned);
      if (nl < 0) {
        scanned = buf.length;
        break;
      }
      lines.push(buf.slice(0, nl));
      buf = buf.slice(nl + 1);
      scanned = 0;
    }
  }
  return lines;
}

/** Fixed lines(): copy each line once; compact prefix only when appending. */
function newSplitLines(chunks: Uint8Array[]): Uint8Array[] {
  const lines: Uint8Array[] = [];
  let buf = new Uint8Array(0);
  let start = 0;
  let scanned = 0;
  for (const value of chunks) {
    if (start > 0) {
      const rem = buf.subarray(start);
      const next = new Uint8Array(rem.length + value.length);
      next.set(rem, 0);
      next.set(value, rem.length);
      buf = next;
      scanned = rem.length;
      start = 0;
    } else {
      buf = buf.length === 0 ? new Uint8Array(value) : concat([buf, value]);
    }
    for (;;) {
      const nl = buf.indexOf(0x0a, scanned);
      if (nl < 0) {
        scanned = buf.length;
        break;
      }
      lines.push(buf.slice(start, nl));
      start = nl + 1;
      scanned = start;
    }
  }
  return lines;
}

// ---- helpers ---------------------------------------------------------------------------------

function makePayload(decodedBytes: number) {
  const bytes = new Uint8Array(decodedBytes);
  for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 31 + (i >> 8)) & 0xff;
  return { bytes, b64: toBase64(bytes) };
}

function median(xs: number[]): number {
  const a = [...xs].sort((x, y) => x - y);
  return a[Math.floor(a.length / 2)]!;
}

function bench(fn: () => void, { rounds = 7, warmup = 2 } = {}) {
  for (let i = 0; i < warmup; i++) fn();
  const times: number[] = [];
  for (let i = 0; i < rounds; i++) {
    const t0 = performance.now();
    fn();
    times.push(performance.now() - t0);
  }
  return median(times);
}

function fmt(ms: number): string {
  return ms < 1 ? `${(ms * 1000).toFixed(0)}µs` : `${ms.toFixed(2)}ms`;
}

function speedup(before: number, after: number): string {
  return `${(before / after).toFixed(2)}×`;
}

// ---- run -------------------------------------------------------------------------------------

const results: Array<{
  suite: string;
  size: number;
  legacyMs: number;
  newMs: number;
  speedup: string;
  note?: string;
  productionMs?: number;
}> = [];

/**
 * Candidate that was tried and rejected: JS alphabet + trailing-bits check, then
 * one Buffer.from (no re-encode). Kept here so the bench proves the legacy path wins.
 */
const B64_VAL = (() => {
  const t = new Int8Array(128).fill(-1);
  for (let i = 0; i < 26; i++) {
    t[65 + i] = i;
    t[97 + i] = 26 + i;
  }
  for (let i = 0; i < 10; i++) t[48 + i] = 52 + i;
  t[43] = 62;
  t[47] = 63;
  return t;
})();

function candidateValidateThenDecode(s: string): Uint8Array | null {
  const clean = legacyStripSpace(s);
  const n = clean.length;
  if (n === 0) return new Uint8Array(0);
  if ((n & 3) !== 0) return null;
  let pads = 0;
  if (clean.charCodeAt(n - 1) === 0x3d) {
    pads = 1;
    if (clean.charCodeAt(n - 2) === 0x3d) pads = 2;
  }
  const dataChars = n - pads;
  if (dataChars < 2) return null;
  for (let i = 0; i < dataChars; i++) {
    const c = clean.charCodeAt(i);
    if (c >= 128 || B64_VAL[c]! < 0) return null;
  }
  for (let i = dataChars; i < n; i++) {
    if (clean.charCodeAt(i) !== 0x3d) return null;
  }
  if (pads === 1 && (B64_VAL[clean.charCodeAt(n - 2)]! & 3) !== 0) return null;
  if (pads === 2 && (B64_VAL[clean.charCodeAt(n - 3)]! & 15) !== 0) return null;
  const b = Buffer.from(clean, "base64");
  return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
}

for (
  const size of [256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 12 * 1024 * 1024]
) {
  const { b64 } = makePayload(size);
  const a = legacyFromBase64(b64);
  const b = fromBase64(b64);
  const c = candidateValidateThenDecode(b64);
  if (
    !a || !b || !c || a.length !== b.length || !a.every((v, i) => v === b[i])
  ) {
    throw new Error(`decode mismatch at ${size}`);
  }
  if (
    fromBase64("AB==") !== null || candidateValidateThenDecode("AB==") !== null
  ) {
    throw new Error("canonical reject failed");
  }

  const legacyMs = bench(() => {
    legacyFromBase64(b64);
  });
  const candidateMs = bench(() => {
    candidateValidateThenDecode(b64);
  });
  const productionMs = bench(() => {
    fromBase64(b64);
  });
  results.push({
    suite: "fromBase64",
    size,
    legacyMs,
    newMs: candidateMs,
    speedup: speedup(legacyMs, candidateMs),
    note: "candidate=JS-validate+decode (rejected); production≡legacy",
    productionMs,
  });
  console.log(
    `fromBase64 ${size}B: legacy/production ${
      fmt(legacyMs)
    } vs JS-validate candidate ${fmt(candidateMs)} (${
      speedup(legacyMs, candidateMs)
    }; production ${fmt(productionMs)})`,
  );
}

{
  const body = new Uint8Array(4 * 1024 * 1024);
  body.fill(0x41);
  const legacyMs = bench(() => legacyExcerpt(body), { rounds: 5, warmup: 1 });
  const newMs = bench(() => newExcerpt(body), { rounds: 5, warmup: 1 });
  if (legacyExcerpt(body).length !== 400 || newExcerpt(body).length !== 400) {
    throw new Error("excerpt length mismatch");
  }
  results.push({
    suite: "excerpt",
    size: body.length,
    legacyMs,
    newMs,
    speedup: speedup(legacyMs, newMs),
  });
  console.log(
    `excerpt 4MiB: legacy ${fmt(legacyMs)} → new ${fmt(newMs)} (${
      speedup(legacyMs, newMs)
    })`,
  );
}

{
  // One buffer with many lines: legacy copies the shrinking remainder on every NL.
  const line = Buffer.from(`{"stdout":"${"A".repeat(64)}"}\n`);
  const big = Buffer.allocUnsafe(line.length * 4096);
  for (let i = 0; i < 4096; i++) line.copy(big, i * line.length);
  const frames = [new Uint8Array(big.buffer, big.byteOffset, big.byteLength)];
  const legacyMs = bench(() => legacySplitLines(frames));
  const newMs = bench(() => newSplitLines(frames));
  if (legacySplitLines(frames).length !== newSplitLines(frames).length) {
    throw new Error("line count mismatch");
  }
  results.push({
    suite: "splitLines",
    size: 4096,
    legacyMs,
    newMs,
    speedup: speedup(legacyMs, newMs),
  });
  console.log(
    `splitLines 4096 lines/1 chunk: legacy ${fmt(legacyMs)} → new ${
      fmt(newMs)
    } (${speedup(legacyMs, newMs)})`,
  );
}

const out = {
  when: new Date().toISOString(),
  node: process.version,
  results,
};
const jsonPath = join(root, "bench-results.json");
writeFileSync(jsonPath, JSON.stringify(out, null, 2) + "\n");
console.log(`\nWrote ${jsonPath}`);
