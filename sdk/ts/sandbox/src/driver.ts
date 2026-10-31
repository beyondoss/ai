#!/usr/bin/env node
// The conformance driver executable: see driver-core.ts and sdk/conformance/DRIVER.md.
import { capsFromEnv, Driver } from "./driver-core.js";
import { serve } from "./driver-serve.js";

async function main(): Promise<void> {
  const url = process.env.CONFORMANCE_URL;
  if (!url) {
    process.stderr.write("CONFORMANCE_URL is required\n");
    process.exit(2);
  }
  const driver = new Driver({
    url,
    bearer: process.env.CONFORMANCE_BEARER,
    tokenCmd: process.env.CONFORMANCE_TOKEN_CMD,
    caps: capsFromEnv(process.env.CONFORMANCE_CAPS),
  });
  await serve(process.stdin, process.stdout, driver);
  // stdin closed: leave without waiting for idle keep-alive sockets.
  process.exit(0);
}

void main();
