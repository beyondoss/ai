import {
  AuthError,
  DiskFullError,
  EndpointError,
  InvalidRequestError,
  JobExistsError,
  JobGoneError,
  NotOfferedError,
  ProtocolError,
  TimeoutError,
  TransportError,
} from "./errors.js";
import type { SandboxError } from "./errors.js";
import { decodeJobLine } from "./http.js";
import type { JobLine } from "./http.js";
import { Kernel } from "./kernel/kernel.js";
import type { Fault, KEvent, KState } from "./kernel/kernel.js";
import type {
  AttachOptions,
  JobFrame,
  JobOpened,
  JobOps,
  JobSpec,
} from "./transport.js";

/**
 * A held job, followed by the client kernel (`crates/job-client`, proven in Verus, run as WASM).
 *
 * This file is the shell: it makes the request the kernel names, reads the answer, turns it into an
 * event, and does what the kernel's actions say. It holds no status table, no offset sum, no retry
 * counter and no time constant; every one of those is the kernel's. The only decision a line here
 * makes is which fact an exception or a frame is (a cut stream is `broke`; a frame past the line
 * cap is `bad`).
 */
export type HeldInit =
  | { flow: "start"; scope: string; id: string; spec: JobSpec }
  | {
    flow: "attach";
    scope: string;
    id: string;
    offset: number;
    mark?: string;
  };

const randomDraw = (): number => Math.floor(Math.random() * 0x1_0000_0000);

const randomKey = (): bigint => {
  const b = new BigUint64Array(1);
  globalThis.crypto.getRandomValues(b);
  return b[0] ?? 0n;
};

/** The error class of each of the kernel's faults. One to one. */
function faultError(
  fault: Fault,
  status: number,
  detail: string,
  id: string,
): SandboxError {
  const said = detail === "" ? `status ${status}` : detail;
  switch (fault) {
    case "job_gone":
      return new JobGoneError(`job ${id}: ${said}`, id);
    case "job_exists":
      return new JobExistsError(`job ${id} exists: ${said}`, id);
    case "disk_full":
      return new DiskFullError(said);
    case "no_pty":
      return new EndpointError(said, status);
    case "deadline":
      return new TimeoutError(`job ${id}: the deadline passed`);
    case "not_offered":
      return new NotOfferedError(
        "the endpoint did not name the job's mark: it predates marks, so a cursor cannot be trusted to name one job",
      );
    case "protocol":
      return status === 0
        ? new ProtocolError(said)
        : new EndpointError(said, status);
    case "auth":
      return new AuthError(said);
  }
}

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    (t as { unref?: () => void }).unref?.();
  });

/** Frames in order, `started` first for a start and the last `exit` or `signal`. Closing detaches. */
export async function* held(
  ops: JobOps,
  init: HeldInit,
  req: AttachOptions = {},
): AsyncGenerator<JobFrame, void, void> {
  const k = await Kernel.load();
  const policy = {
    beatS: req.heartbeatSeconds ?? 0,
    watch: req.output === false,
  };
  let st: KState;
  try {
    st = init.flow === "start"
      ? k.stateForStart(randomKey(), policy)
      : k.stateForAttach(req.mark ?? "", init.offset, policy);
  } catch (e) {
    // The kernel refuses a mark that is not 1 to 32 hex digits: nothing was sent.
    throw new InvalidRequestError("a job mark is 1 to 32 hex digits", {
      cause: e,
    });
  }
  let conn: JobOpened | undefined;
  let lines: AsyncIterator<Uint8Array> | undefined;
  let line: JobLine | undefined;
  let refused = "";
  let ev: KEvent | undefined = { t: "go" };
  /** The first `attached` frame is the consumer's; a reattach's is the kernel's. */
  let told = init.flow === "start";

  /** Send `open` and report its answer as an event. */
  const send = async (open: Parameters<JobOps["open"]>[0]): Promise<KEvent> => {
    try {
      conn = await ops.open(open);
    } catch (e) {
      if (e instanceof TransportError || e instanceof TimeoutError) {
        return { t: "broke" };
      }
      throw e;
    }
    lines = conn.lines[Symbol.asyncIterator]();
    refused = conn.detail;
    return {
      t: "answered",
      status: conn.status,
      retryAfterS: conn.retryAfterS,
    };
  };

  /** The next frame of the open stream, as an event. */
  const read = async (): Promise<KEvent> => {
    try {
      const n = await lines?.next();
      if (n === undefined || n.done) return { t: "broke" };
      line = decodeJobLine(n.value);
    } catch (e) {
      if (e instanceof ProtocolError) {
        line = { frame: { t: "bad" }, note: e.detail };
      } else if (e instanceof TransportError || e instanceof TimeoutError) {
        return { t: "broke" };
      } else {
        throw e;
      }
    }
    return { t: "frame", frame: line.frame };
  };

  try {
    for (;;) {
      const r = k.step(st, ev ?? (await read()), Date.now(), randomDraw());
      const wasFrame = ev === undefined;
      st = r.state;
      ev = undefined;
      for (const a of r.actions) {
        switch (a.t) {
          case "start":
            if (init.flow === "start") {
              ev = await send({
                op: "start",
                scope: init.scope,
                id: init.id,
                spec: init.spec,
                silenceMs: a.silenceMs,
                ...(req.heartbeatSeconds === undefined
                  ? {}
                  : { heartbeatSeconds: req.heartbeatSeconds }),
              });
            }
            break;
          case "openAttach":
            ev = await send({
              op: "attach",
              scope: init.scope,
              id: init.id,
              offset: a.offset,
              mark: a.mark,
              silenceMs: a.silenceMs,
              ...(req.output === undefined ? {} : { output: req.output }),
              ...(req.heartbeatSeconds === undefined
                ? {}
                : { heartbeatSeconds: req.heartbeatSeconds }),
            });
            break;
          case "reconnecting":
            req.onReconnecting?.({
              attempts: a.attempts,
              sinceMs: Date.now() - a.sinceMs,
            });
            break;
          case "closeStream":
            conn?.close();
            conn = undefined;
            lines = undefined;
            break;
          case "deliver":
            if (a.dropped > 0) yield { type: "gap", bytes: a.dropped };
            if (line?.data !== undefined && line.stream !== undefined) {
              yield { type: "output", data: line.data, stream: line.stream };
            }
            break;
          case "deliverExit":
            if (a.dropped > 0) yield { type: "gap", bytes: a.dropped };
            yield a.signal === 0
              ? { type: "exit", code: a.code }
              : { type: "signal", signal: a.signal };
            ev = { t: "exitTaken" };
            break;
          case "sleep":
            await sleep(a.untilMs - Date.now());
            ev = { t: "tick" };
            break;
          case "fail":
            throw faultError(
              a.fault,
              a.status,
              a.fault === "protocol" && wasFrame ? (line?.note ?? "") : refused,
              init.id,
            );
          case "finish":
          case "detached":
            return;
          case "remove":
            // The handle asks for a remove only when told to remove on exit; this shell never is.
            throw new ProtocolError(
              "the kernel asked for a remove that was not requested",
            );
        }
      }
      if (wasFrame && r.actions.length === 0 && line !== undefined) {
        const f = line;
        const fr = f.frame;
        if (fr.t === "started") {
          if (
            init.flow === "start" && init.spec.stdinOpen === true
            && f.stdin !== true
          ) {
            throw new NotOfferedError(
              "the endpoint predates job stdin: it ran the job without it",
            );
          }
          yield {
            type: "started",
            pgid: f.pgid ?? 0,
            mark: fr.mark,
            ...(f.stdin === true ? { stdin: true } : {}),
          };
        } else if (fr.t === "attached" && !told) {
          told = true;
          yield { type: "attached", mark: fr.mark, from: fr.from };
        } else if (f.shellExit !== undefined) {
          yield { type: "shell_exit", code: f.shellExit };
        }
        line = undefined;
      }
    }
  } finally {
    conn?.close();
  }
}
