import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";

export interface Seen {
  headers: IncomingMessage["headers"];
  body: Record<string, unknown>;
}

export type Handler = (
  seen: Seen,
  res: ServerResponse,
  req: IncomingMessage,
) => void;

/** A loopback HTTP server whose answers a test scripts. */
export async function fakeEndpoint(handler: Handler): Promise<{
  url: string;
  requests: Seen[];
  close(): Promise<void>;
}> {
  const requests: Seen[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      let body: Record<string, unknown> = {};
      try {
        body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<
          string,
          unknown
        >;
      } catch {
        // not JSON
      }
      const seen = { headers: req.headers, body };
      requests.push(seen);
      handler(seen, res, req);
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}/exec`,
    requests,
    close: () =>
      new Promise<void>((r) => {
        server.closeAllConnections();
        server.close(() => r());
      }),
  };
}

export const json = (res: ServerResponse, status: number, v: unknown): void => {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(v));
};

const b64 = (s: string | Uint8Array) => Buffer.from(s).toString("base64");

/** A one-shot answer: the text given is what the command wrote; the envelope carries it as base64. */
export const ok = (
  stdout: string | Uint8Array = "",
  exit_code = 0,
  stderr: string | Uint8Array = "",
): unknown => ({
  exit_code,
  stdout: b64(stdout),
  stderr: b64(stderr),
});
