import { describe, expect, it } from "vitest";
import { fromBase64, toBase64 } from "../src/bytes.js";

describe("fromBase64", () => {
  it("decodes a 16 MB base64 string (no call-stack overflow)", () => {
    const n = 12 * 1024 * 1024;
    const bytes = new Uint8Array(n);
    for (let i = 0; i < n; i++) bytes[i] = (i * 31 + (i >> 8)) & 0xff;
    const b64 = toBase64(bytes);
    expect(b64.length).toBe(16 * 1024 * 1024);
    const out = fromBase64(b64);
    expect(out).not.toBeNull();
    expect(out!.length).toBe(n);
    expect(out!.every((v, i) => v === bytes[i])).toBe(true);
  });

  it("decodes valid input, with padding and with whitespace", () => {
    expect(fromBase64("")).toEqual(new Uint8Array());
    expect(fromBase64("AAAA")).toEqual(new Uint8Array([0, 0, 0]));
    expect(fromBase64("AA==")).toEqual(new Uint8Array([0]));
    expect(fromBase64("AAA=")).toEqual(new Uint8Array([0, 0]));
    expect(fromBase64(" AA\n==\t")).toEqual(new Uint8Array([0]));
    expect(fromBase64(" AAAA ")).toEqual(new Uint8Array([0, 0, 0]));
  });

  it("rejects invalid input with null", () => {
    expect(fromBase64("AAA")).toBeNull(); // length not a multiple of 4
    expect(fromBase64("AB=")).toBeNull(); // padding must complete a quad
    expect(fromBase64("A===")).toBeNull(); // too much padding
    expect(fromBase64("====")).toBeNull();
    expect(fromBase64("AA==AAAA")).toBeNull(); // padding not at the end
    expect(fromBase64("AAA=AAAA")).toBeNull();
    expect(fromBase64("AA-_")).toBeNull(); // url-safe alphabet is not standard
    expect(fromBase64("AAA!")).toBeNull();
    expect(fromBase64("AAAé")).toBeNull();
    expect(fromBase64("AAAA\u0000")).toBeNull();
  });

  it("rejects invalid input of 16 MB quickly without overflowing", () => {
    const big = "A".repeat(16 * 1024 * 1024 - 4) + "A!AA";
    expect(fromBase64(big)).toBeNull();
  });
});
