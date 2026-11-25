import { fromBase64 } from "../bytes.js";
import { InvalidRequestError, ProtocolError } from "../errors.js";
import { KERNEL_WASM_BASE64 } from "./wasm.js";

/**
 * The job client's kernel (`crates/job-client`, proven in Verus), compiled to WASM and inlined in
 * `wasm.ts`. This file is its codec: bytes in and out, per `crates/job-client-ffi/src/codec.rs`. It
 * decides nothing; `held.ts` does the I/O the kernel's actions name.
 */

/** The typed failure of a handle: the one vocabulary every SDK maps to its own error class. */
export type Fault =
  | "job_gone"
  | "job_exists"
  | "disk_full"
  | "no_pty"
  | "deadline"
  | "not_offered"
  | "protocol"
  | "auth"
  | "transport";
const FAULTS: readonly Fault[] = [
  "job_gone",
  "job_exists",
  "disk_full",
  "no_pty",
  "deadline",
  "not_offered",
  "protocol",
  "auth",
  "transport",
];

/** What the stream's frames said; the payload bytes stay in the shell. `mark` is hex, "" for none. */
export type KFrame =
  | { t: "attached"; mark: string; from: number }
  | { t: "started"; mark: string; stdin: boolean; pty: boolean }
  | { t: "output"; len: number }
  | { t: "gap"; n: number }
  | { t: "heartbeat" }
  | { t: "shellExit"; code: number }
  | { t: "exit"; code: number; signal: number }
  | { t: "bad" };

/** What the shell observed. */
export type KEvent =
  | { t: "go" }
  | { t: "answered"; status: number; retryAfterS: number }
  | { t: "frame"; frame: KFrame }
  | { t: "broke" }
  | { t: "tick" }
  | { t: "exitTaken" }
  | { t: "close" }
  | { t: "deadline" }
  | { t: "unreached" };

/** What the shell does. */
export type KAction =
  | { t: "start"; key: bigint; silenceMs: number }
  | { t: "openAttach"; offset: number; mark: string; silenceMs: number }
  | { t: "closeStream" }
  | { t: "kill" }
  | { t: "started"; mark: string; stdin: boolean; pty: boolean }
  | { t: "attached"; mark: string; from: number }
  | { t: "shellExit"; code: number }
  | { t: "reconnecting"; attempts: number; sinceMs: number }
  | { t: "deliver"; at: number; len: number; dropped: number }
  | {
    t: "deliverExit";
    code: number;
    signal: number;
    dropped: number;
    killed: boolean;
  }
  | { t: "sleep"; untilMs: number }
  | { t: "remove"; mark: string }
  | { t: "fail"; fault: Fault; status: number; offset: number }
  | { t: "finish" }
  | { t: "detached" };

/** The handle's state: opaque bytes the kernel made and the shell only hands back. */
export type KState = Uint8Array & { readonly __kstate: unique symbol };

export type Class =
  | { t: "ok" }
  | { t: "retry" }
  | { t: "terminal"; fault: Fault };

export interface Policy {
  watch?: boolean;
  removeOnExit?: boolean;
  /** A start that needs the job's stdin opened: an echo that says it was not is refused. */
  stdinOpen?: boolean;
  /** A start on a terminal: an echo that says none was opened is refused. */
  ptyOpen?: boolean;
  /** A start of a kept job (`hold: false`): a refused echo stops the job. */
  kept?: boolean;
  /** The heartbeat the request chose, whole seconds; 0 for the default. */
  beatS?: number;
}

interface Exports {
  memory: WebAssembly.Memory;
  jc_abi_version(): number;
  jc_alloc(n: number): number;
  jc_state_for_start: Call;
  jc_state_for_attach: Call;
  jc_state_for_cursor: Call;
  jc_step: Call;
  jc_classify: Call;
  jc_silence_ms: Call;
  jc_cursor_of: Call;
}
type Call = (inp: number, n: number, out: number, cap: number) => number;

/** The ABI `codec.rs` lays out. A change to a layout there bumps `ABI_VERSION`, and this must follow. */
const ABI_VERSION = 5;
/** Room for any one call's input or answer: a state (90 B) and an event (45 B at most), or a cursor. */
const BUF = 1024;
const STATE = 90;

const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);

function safe(v: bigint, what: string): number {
  if (v > MAX_SAFE) throw new ProtocolError(`${what} is past 2^53`);
  return Number(v);
}

/** A writer of the codec's little-endian integers. */
class Wr {
  readonly b: number[] = [];
  u8(v: number): this {
    this.b.push(v);
    return this;
  }
  u16(v: number): this {
    return this.u8(v & 0xff).u8(v >>> 8);
  }
  u32(v: number): this {
    return this.u16(v & 0xffff).u16(v >>> 16);
  }
  i32(v: number): this {
    return this.u32(v >>> 0);
  }
  u64(v: bigint | number): this {
    const x = BigInt.asUintN(64, BigInt(v));
    return this.u32(Number(x & 0xffffffffn)).u32(Number(x >> 32n));
  }
  mark(hex: string): this {
    this.u8(hex.length);
    for (let i = 0; i < 32; i++) {
      this.u8(i < hex.length ? hex.charCodeAt(i) : 0);
    }
    return this;
  }
  bytes(v: Uint8Array): this {
    for (const x of v) this.u8(x);
    return this;
  }
}

/** A reader over a kernel answer. */
class Rd {
  private o = 0;
  private readonly v: DataView;
  readonly a: Uint8Array;
  constructor(a: Uint8Array) {
    this.a = a;
    this.v = new DataView(a.buffer, a.byteOffset, a.byteLength);
  }
  u8(): number {
    return this.v.getUint8(this.o++);
  }
  u16(): number {
    const x = this.v.getUint16(this.o, true);
    this.o += 2;
    return x;
  }
  u32(): number {
    const x = this.v.getUint32(this.o, true);
    this.o += 4;
    return x;
  }
  i32(): number {
    const x = this.v.getInt32(this.o, true);
    this.o += 4;
    return x;
  }
  u64(what: string): number {
    const x = this.v.getBigUint64(this.o, true);
    this.o += 8;
    return safe(x, what);
  }
  big(): bigint {
    const x = this.v.getBigUint64(this.o, true);
    this.o += 8;
    return x;
  }
  mark(): string {
    const len = this.u8();
    let s = "";
    for (let i = 0; i < 32; i++) {
      const c = this.u8();
      if (i < len) s += String.fromCharCode(c);
    }
    return s;
  }
  take(n: number): Uint8Array {
    const x = this.a.slice(this.o, this.o + n);
    this.o += n;
    return x;
  }
}

function frameBytes(w: Wr, f: KFrame): void {
  switch (f.t) {
    case "attached":
      w.u8(0).mark(f.mark).u64(f.from);
      return;
    case "started":
      w.u8(1).mark(f.mark).u8(f.stdin ? 1 : 0).u8(f.pty ? 1 : 0);
      return;
    case "output":
      w.u8(2).u64(f.len);
      return;
    case "gap":
      w.u8(3).u64(f.n);
      return;
    case "heartbeat":
      w.u8(4);
      return;
    case "exit":
      w.u8(5).i32(f.code).i32(f.signal);
      return;
    case "bad":
      w.u8(6);
      return;
    case "shellExit":
      w.u8(7).i32(f.code);
      return;
  }
}

function eventBytes(w: Wr, e: KEvent): void {
  switch (e.t) {
    case "go":
      w.u8(0);
      return;
    case "answered":
      w.u8(1).u16(e.status).u32(e.retryAfterS);
      return;
    case "frame":
      w.u8(2);
      frameBytes(w, e.frame);
      return;
    case "broke":
      w.u8(3);
      return;
    case "tick":
      w.u8(4);
      return;
    case "exitTaken":
      w.u8(5);
      return;
    case "close":
      w.u8(6);
      return;
    case "deadline":
      w.u8(7);
      return;
    case "unreached":
      w.u8(9);
      return;
  }
}

function readAction(r: Rd): KAction {
  const tag = r.u8();
  switch (tag) {
    case 0:
      return { t: "start", key: r.big(), silenceMs: r.u64("silence") };
    case 1:
      return {
        t: "openAttach",
        offset: r.u64("offset"),
        mark: r.mark(),
        silenceMs: r.u64("silence"),
      };
    case 2:
      return { t: "closeStream" };
    case 3:
      return {
        t: "deliver",
        at: r.u64("offset"),
        len: r.u64("length"),
        dropped: r.u64("gap"),
      };
    case 4:
      return {
        t: "deliverExit",
        code: r.i32(),
        signal: r.i32(),
        dropped: r.u64("gap"),
        killed: r.u8() === 1,
      };
    case 5:
      return { t: "sleep", untilMs: r.u64("time") };
    case 6:
      return { t: "remove", mark: r.mark() };
    case 7: {
      const fault = FAULTS[r.u8()];
      if (fault === undefined) throw new ProtocolError("kernel: unknown fault");
      return { t: "fail", fault, status: r.u16(), offset: r.u64("offset") };
    }
    case 8:
      return { t: "finish" };
    case 9:
      return { t: "detached" };
    case 11:
      return {
        t: "started",
        mark: r.mark(),
        stdin: r.u8() === 1,
        pty: r.u8() === 1,
      };
    case 12:
      return { t: "attached", mark: r.mark(), from: r.u64("offset") };
    case 14:
      return { t: "kill" };
    case 13:
      return { t: "shellExit", code: r.i32() };
    case 10:
      return { t: "reconnecting", attempts: r.u32(), sinceMs: r.u64("time") };
    default:
      throw new ProtocolError(`kernel: unknown action ${tag}`);
  }
}

export class Kernel {
  private readonly x: Exports;
  private readonly inp: number;
  private readonly out: number;

  private constructor(x: Exports, inp: number, out: number) {
    this.x = x;
    this.inp = inp;
    this.out = out;
  }

  /** The module compiled once per process, from the bytes inlined in `wasm.ts`. */
  static #loaded: Promise<Kernel> | undefined;

  static load(): Promise<Kernel> {
    Kernel.#loaded ??= Kernel.#instantiate();
    return Kernel.#loaded;
  }

  static async #instantiate(): Promise<Kernel> {
    const bytes = fromBase64(KERNEL_WASM_BASE64);
    if (bytes === null) {
      throw new ProtocolError("kernel: the inlined module is not base64");
    }
    const { instance } = await WebAssembly.instantiate(
      bytes as BufferSource,
      {},
    );
    const x = instance.exports as unknown as Exports;
    if (x.jc_abi_version() !== ABI_VERSION) {
      throw new ProtocolError(
        "kernel: the module's ABI is not the one this codec speaks",
      );
    }
    return new Kernel(x, x.jc_alloc(BUF), x.jc_alloc(BUF));
  }

  /** One call: `input` into the module's memory, the answer out of it. */
  private call(fn: Call, input: Uint8Array): Uint8Array {
    // The block this call's input is copied into holds `BUF` bytes: a longer one would run into the
    // module's own memory.
    if (input.length > BUF) {
      throw new InvalidRequestError(
        `kernel: an input of ${input.length} bytes is past the ${BUF} the module takes`,
      );
    }
    const mem = new Uint8Array(this.x.memory.buffer);
    mem.set(input, this.inp);
    const n = fn(this.inp, input.length, this.out, BUF);
    if (n < 0) throw new ProtocolError(`kernel: refused its input (${n})`);
    return new Uint8Array(this.x.memory.buffer).slice(this.out, this.out + n);
  }

  private state(bytes: Uint8Array): KState {
    return bytes as KState;
  }

  /** A handle that starts a job. `key` is the shell's handle for the idempotency key it holds. */
  stateForStart(key: bigint, p: Policy = {}): KState {
    const w = new Wr().u64(key).u8(p.watch ? 1 : 0).u8(p.removeOnExit ? 1 : 0)
      .u32(p.beatS ?? 0).u8(p.stdinOpen ? 1 : 0).u8(p.ptyOpen ? 1 : 0)
      .u8(p.kept ? 1 : 0);
    return this.state(
      this.call(this.x.jc_state_for_start, Uint8Array.from(w.b)),
    );
  }

  /** A handle that attaches to an incarnation (`mark`, "" for any) at `offset`. */
  stateForAttach(mark: string, offset: number, p: Policy = {}): KState {
    const w = new Wr().mark(mark).u64(offset).u8(p.watch ? 1 : 0).u8(
      p.removeOnExit ? 1 : 0,
    ).u32(p.beatS ?? 0);
    return this.state(
      this.call(this.x.jc_state_for_attach, Uint8Array.from(w.b)),
    );
  }

  /** A handle that resumes the incarnation and offset a cursor names. */
  stateForCursor(cursor: string, p: Policy = {}): KState {
    const w = new Wr().u8(p.watch ? 1 : 0).u8(p.removeOnExit ? 1 : 0).u32(
      p.beatS ?? 0,
    );
    w.bytes(new TextEncoder().encode(cursor));
    return this.state(
      this.call(this.x.jc_state_for_cursor, Uint8Array.from(w.b)),
    );
  }

  /** One step of the handle: its next state and what the shell does. */
  step(
    state: KState,
    ev: KEvent,
    nowMs: number,
    draw: number,
  ): { state: KState; actions: KAction[] } {
    const w = new Wr().bytes(state);
    eventBytes(w, ev);
    w.u64(nowMs).u32(draw);
    const r = new Rd(this.call(this.x.jc_step, Uint8Array.from(w.b)));
    const next = this.state(r.take(STATE));
    const actions: KAction[] = [];
    for (let n = r.u8(); n > 0; n--) actions.push(readAction(r));
    return { state: next, actions };
  }

  /** What an HTTP answer to `op` means. */
  classify(
    op: "start" | "attach" | "remove" | "other",
    status: number,
  ): Class {
    const tag = { start: 0, attach: 1, remove: 2, other: 3 }[op];
    const r = new Rd(
      this.call(
        this.x.jc_classify,
        Uint8Array.from(new Wr().u8(tag).u16(status).b),
      ),
    );
    switch (r.u8()) {
      case 0:
        return { t: "ok" };
      case 1:
        return { t: "retry" };
      default: {
        const fault = FAULTS[r.u8()];
        if (fault === undefined) {
          throw new ProtocolError("kernel: unknown fault");
        }
        return { t: "terminal", fault };
      }
    }
  }

  /** The dead-stream limit for a request that chose heartbeat `beatS`. */
  silenceMs(beatS: number): number {
    return new Rd(
      this.call(this.x.jc_silence_ms, Uint8Array.from(new Wr().u32(beatS).b)),
    ).u64("silence");
  }

  /** The cursor a held state resumes from. */
  cursorOf(state: KState, scope: string, id: string): string {
    const w = new Wr().bytes(state).u8(scope.length).bytes(
      new TextEncoder().encode(scope + id),
    );
    return new TextDecoder().decode(
      this.call(this.x.jc_cursor_of, Uint8Array.from(w.b)),
    );
  }
}
