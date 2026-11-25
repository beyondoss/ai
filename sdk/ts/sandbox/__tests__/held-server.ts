import type { ServerResponse } from "node:http";
import { fakeEndpoint } from "./fake-endpoint.js";
import type { Seen } from "./fake-endpoint.js";

export const MARK = "00ff00ff00ff00ff1a";
const b64 = (s: string) => Buffer.from(s).toString("base64");

export interface Script {
  /** The job's whole output; `exit` follows it. */
  output: string;
  /** Statuses to answer the next attaches with, before any stream. */
  refuse?: Array<[number, Record<string, string>?]>;
  /** The first stream sends this many bytes of output and then drops the connection. */
  cutAfter?: number[];
  /** Echo this mark instead of the held one (`null`: none). */
  echo?: string | null;
  /** Echo this mark on a reattach only: another incarnation answers it. */
  reattachEcho?: string;
  /** Bytes of the front of the output the server no longer keeps. */
  base?: number;
  /** End with `{signal}` (a kill) instead of `{exit_code: 0}`. */
  signal?: number;
  /** The shell's exit, sent before the final frame. */
  shellExit?: number;
  /** A held start's answer. */
  startStatus?: number;
}

/** A fake job endpoint that echoes marks as `crates/execd` will: `started{mark}` and `attached{mark, from}`. */
export async function markServer(script: Script) {
  const cuts = [...(script.cutAfter ?? [])];
  const refuse = [...(script.refuse ?? [])];
  const attaches: Seen[] = [];
  const starts: Seen[] = [];
  const e = await fakeEndpoint((seen, res) => {
    const job = seen.body.job as Record<string, unknown>;
    if (job.op === "start") {
      starts.push(seen);
      if (script.startStatus !== undefined) {
        res.writeHead(script.startStatus, {
          "content-type": "application/json",
        });
        res.end(JSON.stringify({ error: "refused" }));
        return;
      }
      stream(res, 0, true);
      return;
    }
    attaches.push(seen);
    const r = refuse.shift();
    if (r !== undefined) {
      res.writeHead(r[0], {
        "content-type": "application/json",
        ...(r[1] ?? {}),
      });
      res.end(JSON.stringify({ error: "refused" }));
      return;
    }
    if (job.mark !== undefined && job.mark !== MARK) {
      res.writeHead(404, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: "no such job" }));
      return;
    }
    stream(res, Number(job.offset), false);
  });

  function stream(res: ServerResponse, from: number, start: boolean): void {
    res.writeHead(200, { "content-type": "application/x-ndjson" });
    const base = script.base ?? 0;
    const at = Math.max(from, base);
    const mark = !start && script.reattachEcho !== undefined
      ? script.reattachEcho
      : script.echo === undefined
      ? MARK
      : script.echo;
    if (start) {
      res.write(
        JSON.stringify({ started: 42, ...(mark === null ? {} : { mark }) })
          + "\n",
      );
    } else if (mark !== null) {
      res.write(JSON.stringify({ attached: { mark, from: at } }) + "\n");
    }
    const cut = cuts.shift();
    const body = script.output.slice(at);
    const piece = cut === undefined ? body : body.slice(0, cut);
    if (cut !== undefined && cut < body.length) {
      // Flush what was written, then drop the connection mid-stream.
      const line = piece === ""
        ? "\n"
        : JSON.stringify({ stdout: b64(piece) }) + "\n";
      res.write(line, () => setTimeout(() => res.socket?.destroy(), 20));
      return;
    }
    if (piece !== "") res.write(JSON.stringify({ stdout: b64(piece) }) + "\n");
    if (script.shellExit !== undefined) {
      res.write(JSON.stringify({ shell_exit: script.shellExit }) + "\n");
    }
    res.end(
      JSON.stringify(
        script.signal === undefined
          ? { exit_code: 0 }
          : { signal: script.signal },
      ) + "\n",
    );
  }
  return { ...e, attaches, starts };
}
