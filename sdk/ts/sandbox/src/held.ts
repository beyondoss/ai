import {
  AuthError,
  ConnectError,
  DiskFullError,
  EndpointError,
  InvalidRequestError,
  JobExistsError,
  JobGoneError,
  KeyConflictError,
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
import { DEFAULT_KILL_GRACE_MS } from "./transport.js";

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
  | { flow: "start"; scope: string; id: string; spec: JobSpec; hold: boolean }
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
      return said.includes("key_conflict")
        ? new KeyConflictError(`job ${id}: ${said}`, id)
        : new JobExistsError(`job ${id} exists: ${said}`, id);
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
    case "transport":
      return new ConnectError(`job ${id}: ${said}`);
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
    stdinOpen: init.flow === "start" && init.spec.stdinOpen === true,
    ptyOpen: init.flow === "start" && init.spec.pty !== undefined,
    kept: init.flow === "start" && !init.hold,
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

  /** Send `open` and report its answer as an event. */
  const send = async (open: Parameters<JobOps["open"]>[0]): Promise<KEvent> => {
    try {
      conn = await ops.open(open);
    } catch (e) {
      if (e instanceof ConnectError) {
        refused = e.detail;
        return { t: "unreached" };
      }
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
                hold: init.hold,
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
          case "started":
            if (
              init.flow === "start" && init.spec.idempotencyKey !== undefined
              && line?.key !== init.spec.idempotencyKey
            ) {
              // The endpoint ignored the key: a start sent again for a lost answer would run twice.
              // Stop what it started (a held job dies with its stream).
              if (!init.hold) {
                await ops.kill(init.scope, [init.id], {
                  remove: true,
                  graceMs: DEFAULT_KILL_GRACE_MS,
                }).catch(() => {});
              }
              throw new NotOfferedError(
                "the endpoint did not echo the start's idempotency key: it ignores keys, so a start sent again for a lost answer would run twice",
              );
            }
            yield {
              type: "started",
              pgid: line?.pgid ?? 0,
              mark: a.mark,
              ...(a.stdin ? { stdin: true } : {}),
              ...(line?.pty === undefined ? {} : { pty: line.pty }),
            };
            break;
          case "attached":
            yield { type: "attached", mark: a.mark, from: a.from };
            break;
          case "shellExit":
            yield { type: "shell_exit", code: a.code };
            break;
          case "kill":
            // A kept start whose echo was refused: nothing else can name the job.
            await ops.kill(init.scope, [init.id], {
              remove: true,
              graceMs: DEFAULT_KILL_GRACE_MS,
            }).catch(() => {});
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
            yield a.killed
              ? { type: "signal", signal: a.signal }
              : { type: "exit", code: a.code };
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
    }
  } finally {
    conn?.close();
  }
}
