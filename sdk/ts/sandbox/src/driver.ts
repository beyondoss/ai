#!/usr/bin/env node
// The conformance driver executable: see driver-core.ts and sdk/conformance/DRIVER.md.
import { createInterface } from "node:readline";
import { capsFromEnv, Driver } from "./driver-core.js";

type Json = Record<string, unknown>;

async function main(): Promise<void> {
  const url = process.env.CONFORMANCE_URL;
  if (!url) {
    process.stderr.write("CONFORMANCE_URL is required\n");
    process.exit(2);
  }
  const driver = new Driver({
    url,
    bearer: process.env.CONFORMANCE_BEARER,
    caps: capsFromEnv(process.env.CONFORMANCE_CAPS),
  });
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
  // One request at a time, answered in order: `for await` does not read ahead of the loop body.
  for await (const line of rl) {
    if (line.trim() === "") continue;
    let req: Json;
    try {
      const v: unknown = JSON.parse(line);
      if (typeof v !== "object" || v === null || Array.isArray(v)) {
        throw new Error("not an object");
      }
      req = v as Json;
    } catch (e) {
      process.stdout.write(
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
    process.stdout.write(JSON.stringify(await driver.handle(req)) + "\n");
  }
  // stdin closed: leave without waiting for idle keep-alive sockets.
  process.exit(0);
}

void main();
