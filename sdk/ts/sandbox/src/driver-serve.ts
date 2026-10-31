import type { Driver } from "./driver-core.js";

type Json = Record<string, unknown>;

/**
 * Answers each JSON line on `input` with one JSON line on `output`, in order. Lines are split on LF
 * alone: `node:readline` also splits on U+2028 and U+2029 (from Node 24 on), and a request may carry
 * those inside a string, which would turn one request into two bad ones and shift every later answer.
 */
export async function serve(
  input: AsyncIterable<Uint8Array | string>,
  output: { write(chunk: string): unknown },
  driver: Driver,
): Promise<void> {
  // One request at a time, answered in order: `for await` does not read ahead of the loop body.
  for await (const line of lines(input)) {
    if (line.trim() === "") continue;
    let req: Json;
    try {
      const v: unknown = JSON.parse(line);
      if (typeof v !== "object" || v === null || Array.isArray(v)) {
        throw new Error("not an object");
      }
      req = v as Json;
    } catch (e) {
      output.write(
        JSON.stringify({
          id: null,
          error: {
            kind: "invalid_request",
            detail: `bad request line: ${String(e)}`,
          },
        }) + "\n",
      );
      continue;
    }
    output.write(JSON.stringify(await driver.handle(req)) + "\n");
  }
}

/** The lines of a UTF-8 stream, split on LF; a CR before the LF is dropped, as readline's CRLF handling did. */
async function* lines(
  input: AsyncIterable<Uint8Array | string>,
): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let buf = "";
  for await (const chunk of input) {
    buf += typeof chunk === "string"
      ? chunk
      : decoder.decode(chunk, { stream: true });
    let start = 0;
    let nl: number;
    while ((nl = buf.indexOf("\n", start)) !== -1) {
      yield buf.slice(start, nl).replace(/\r$/, "");
      start = nl + 1;
    }
    buf = buf.slice(start);
  }
  buf += decoder.decode();
  if (buf !== "") yield buf.replace(/\r$/, "");
}
