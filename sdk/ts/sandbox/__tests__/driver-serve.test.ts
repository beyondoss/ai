import { describe, expect, it } from "vitest";
import { Driver } from "../src/driver-core.js";
import { serve } from "../src/driver-serve.js";

/** Feeds `chunks` to the driver's loop, and returns each response line parsed. */
async function answers(chunks: Uint8Array[]): Promise<unknown[]> {
  const out: string[] = [];
  async function* input() {
    yield* chunks;
  }
  const driver = new Driver({ url: "http://127.0.0.1:1/exec" });
  await serve(input(), { write: (s) => out.push(s) }, driver);
  return out.join("").split("\n").filter((l) => l !== "").map((l) =>
    JSON.parse(l)
  );
}

const enc = (s: string) => new TextEncoder().encode(s);

describe("driver line protocol", () => {
  it("keeps U+2028 and U+2029 inside a request as one request, one answer", async () => {
    const r = await answers([
      enc("{\"id\":\"a b c\",\"op\":\"hello\"}\n{\"id\":2,\"op\":\"hello\"}\n"),
    ]);
    expect(r).toMatchObject([
      { id: "a b c", ok: { driver: "beyond-sandbox-ts" } },
      { id: 2, ok: { driver: "beyond-sandbox-ts" } },
    ]);
  });

  it("joins a multi-byte character split across chunks", async () => {
    const bytes = enc("{\"id\":\"x y\",\"op\":\"hello\"}\n");
    const cut = bytes.indexOf(0xe2) + 1; // inside the 3-byte U+2028
    const r = await answers([bytes.slice(0, cut), bytes.slice(cut)]);
    expect(r).toMatchObject([{
      id: "x y",
      ok: { driver: "beyond-sandbox-ts" },
    }]);
  });

  it("accepts CRLF line endings and an unterminated last line", async () => {
    const r = await answers([
      enc("{\"id\":1,\"op\":\"hello\"}\r\n\r\n{\"id\":3,\"op\":\"hello\"}"),
    ]);
    expect(r).toMatchObject([{ id: 1 }, { id: 3 }]);
  });
});
