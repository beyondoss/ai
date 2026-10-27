import { concat, fromBase64, text, toBase64, utf8 } from "./bytes.js";
import {
  classifyFailure,
  ConflictError,
  InvalidQueryError,
  InvalidRequestError,
  NotFoundError,
  SandboxError,
  TimeoutError,
} from "./errors.js";
import {
  GIT_SCRIPT,
  READ_SCRIPT,
  REMOVE_TEMP_SCRIPT,
  STAT_SCRIPT,
  WRITE_CHUNK_SCRIPT,
  WRITE_STDIN_SCRIPT,
} from "./scripts.js";
import {
  classifySearchFailure,
  finalize,
  grepArgs,
  isUnderGitDir,
  parseColonRecords,
  parseRecords,
  rgArgs,
  withDefaults,
} from "./search.js";
import type { SearchEngine, SearchQuery, SearchResult } from "./search.js";
import type { RunResult, Transport } from "./transport.js";

/** Bytes per `read` command: 96 KiB, which is exactly 128 KiB of base64. */
export const READ_CHUNK = 96 * 1024;
/** Raw bytes per chunked-write command: 48 KiB, exactly 64 KiB of base64. */
export const WRITE_CHUNK = 48 * 1024;
/** Per-command timeout of helpers. */
export const COMMAND_TIMEOUT_MS = 120_000;
/** Per-command timeout of capability probes. */
export const PROBE_TIMEOUT_MS = 10_000;

const READ_WINDOW_PROBE =
  "printf abc | dd iflag=skip_bytes,count_bytes skip=1 count=1 status=none 2>/dev/null | base64 -w0 2>/dev/null";

/** What the sandbox has, probed once (FILES-OVER-EXEC section 4). */
export interface Capabilities {
  /** `cat` echoed its stdin: writes send their content on stdin in one command. */
  stdin: boolean;
  /** `find -printf` works: listings are one command. */
  findPrintf: boolean;
  /** `dd iflag=skip_bytes,count_bytes` and `base64 -w0` work: files can be read. */
  readWindow: boolean;
  /** `rg` is present. */
  rg: boolean;
  /** `grep -Z` works. */
  grepNull: boolean;
  searchEngine: SearchEngine;
}

/** Probe results to force, so one sandbox can walk every rung of the fallback ladder. */
export interface CapabilityOverrides {
  stdin?: boolean | undefined;
  findPrintf?: boolean | undefined;
  searchEngine?: SearchEngine | undefined;
}

export interface FilesOptions {
  /** Replace what the probes found (tests, or a sandbox you know better). */
  overrides?: CapabilityOverrides | undefined;
}

export interface Stat {
  kind: "file" | "dir" | "other";
  size: number;
  /** Epoch seconds (the integer part). */
  mtime: number;
  /** A real access check for the calling user, not mode bits. */
  writable: boolean;
}

export interface ReadOptions {
  /** Start at this byte; default 0. */
  offset?: number;
  /** At most this many bytes; default to the end of the file. */
  max?: number;
}

export interface ListOptions {
  includeHidden?: boolean;
  /** Entries kept; default 10000. */
  limit?: number;
}

export interface DirEntry {
  name: string;
  kind: "file" | "dir" | "other";
  size: number;
}

/** Run `SCRIPT` as `sh -c SCRIPT sh ARG...`: the variable parts arrive as `$1`, `$2`, ... */
export function sh(script: string, args: string[]): string[] {
  return ["-c", script, "sh", ...args];
}

function count(name: string, v: number | undefined, dflt: number): number {
  if (v === undefined) return dflt;
  if (!Number.isInteger(v) || v < 0) {
    throw new InvalidRequestError(`\`${name}\` must be a non-negative integer`);
  }
  return v;
}

async function randomHex64(): Promise<string> {
  const bytes = new Uint8Array(8);
  const c = (globalThis as { crypto?: Crypto }).crypto;
  if (c?.getRandomValues) c.getRandomValues(bytes);
  else (await import("node:crypto")).randomFillSync(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Layer 2: files and search over ANY {@link Transport}. Built on `transport.run` and nothing else,
 * following FILES-OVER-EXEC.md. The sandbox is probed once, on the first call.
 */
export class Files {
  readonly transport: Transport;
  private readonly overrides: CapabilityOverrides;
  private caps: Promise<Capabilities> | undefined;
  private readonly gitRoots = new Map<string, boolean>();

  constructor(transport: Transport, options: FilesOptions = {}) {
    this.transport = transport;
    this.overrides = options.overrides ?? {};
  }

  /**
   * What the probes found (after overrides). A probe request that fails at the transport level
   * (unreachable, `auth`, `endpoint`, `timeout`) throws and is not cached: a probe that did not run
   * says nothing about the sandbox.
   */
  capabilities(): Promise<Capabilities> {
    if (!this.caps) {
      const p = this.probe();
      this.caps = p;
      p.catch(() => {
        if (this.caps === p) this.caps = undefined;
      });
    }
    return this.caps;
  }

  private async probe(): Promise<Capabilities> {
    const t = this.transport;
    // A probe that ran and answered "absent" (or could not be started at all) is a capability
    // that is missing; any other failure is returned.
    const settle = async (p: Promise<RunResult>): Promise<RunResult | null> => {
      try {
        return await p;
      } catch (e) {
        if (
          e instanceof SandboxError
          && ["not_found", "not_offered", "too_large"].includes(e.kind)
        ) {
          return null;
        }
        throw e;
      }
    };
    const opts = { timeoutMs: PROBE_TIMEOUT_MS, text: true };
    const [rg, grep, find, stdin, window] = await Promise.all([
      settle(t.run("rg", ["--version"], opts)),
      settle(
        t.run("grep", ["-rnZ", "-e", "__cap_probe__", "--", "/dev/null"], opts),
      ),
      settle(
        t.run("find", ["/dev/null", "-maxdepth", "0", "-printf", "%y"], opts),
      ),
      settle(t.run("cat", [], { ...opts, stdin: utf8("stdin-probe") })),
      settle(t.run("sh", ["-c", READ_WINDOW_PROBE], opts)),
    ]);
    const ranOk = (r: RunResult | null) =>
      r !== null && (r.exitCode === 0 || r.exitCode === 1);
    let hasRg = rg !== null && rg.exitCode === 0;
    let grepNull = ranOk(grep);
    const o = this.overrides;
    switch (o.searchEngine) {
      case "rg":
        hasRg = true;
        grepNull = true;
        break;
      case "grep":
        hasRg = false;
        grepNull = true;
        break;
      case "busybox-grep":
        hasRg = false;
        grepNull = false;
        break;
    }
    return {
      stdin: o.stdin
        ?? (stdin !== null && stdin.exitCode === 0
          && text(stdin.stdout) === "stdin-probe"),
      // busybox `find` has no -printf: it prints usage and exits 1, which "exit 0 or 1" accepts.
      // GNU prints the type of /dev/null, a character device.
      findPrintf: o.findPrintf ?? (ranOk(find) && text(find!.stdout) === "c"),
      readWindow: window !== null && window.exitCode === 0
        && text(window.stdout).trim() === "Yg==",
      rg: hasRg,
      grepNull,
      searchEngine: hasRg ? "rg" : grepNull ? "grep" : "busybox-grep",
    };
  }

  private async exec(
    command: string,
    args: string[],
    stdin?: Uint8Array,
  ): Promise<RunResult> {
    return this.transport.run(command, args, {
      timeoutMs: COMMAND_TIMEOUT_MS,
      text: true,
      ...(stdin !== undefined ? { stdin } : {}),
    });
  }

  // ---- stat ----------------------------------------------------------------------------------

  /** `null` when the path does not exist (a missing parent, a broken link, a path below a file). */
  async stat(path: string): Promise<Stat | null> {
    await this.capabilities();
    const r = await this.exec("sh", sh(STAT_SCRIPT, [path]));
    const out = text(r.stdout);
    if (r.exitCode !== 0 || out.trim() === "") return null;
    return parseStat(out);
  }

  // ---- read ----------------------------------------------------------------------------------

  /**
   * Read `max` bytes (default: to the end) from `offset`, one command per 96 KiB window. A symlink
   * is followed. Refuse `other` kinds yourself with {@link Files.stat}: `dd` on a FIFO with no
   * writer blocks until the command's timeout.
   */
  async read(path: string, options: ReadOptions = {}): Promise<Uint8Array> {
    const offset = count("offset", options.offset, 0);
    const max = count("max", options.max, Infinity);
    const caps = await this.capabilities();
    if (!caps.readWindow) {
      throw new SandboxError(
        "io",
        `read ${path}: this target's dd or base64 does not support iflag=skip_bytes,count_bytes / -w0, so files cannot be read`,
      );
    }
    let out: Uint8Array[] = [];
    let got = 0;
    for (;;) {
      const want = Math.min(max - got, READ_CHUNK);
      const r = await this.exec(
        "sh",
        sh(READ_SCRIPT, [path, String(offset + got), String(want)]),
      );
      if (r.exitCode !== 0) {
        throw classifyFailure(`read ${path}`, text(r.stderr));
      }
      const chunk = fromBase64(text(r.stdout).trim());
      if (!chunk) {
        throw new SandboxError(
          "io",
          `read ${path}: the command's output is not base64`,
        );
      }
      const short = chunk.length < want;
      if (out.length === 0 && (short || want === max)) return chunk;
      out.push(chunk);
      got += chunk.length;
      if (short || got >= max) return concat(out);
    }
  }

  /** {@link Files.read} the whole file as UTF-8 text. */
  async readText(path: string): Promise<string> {
    return text(await this.read(path));
  }

  // ---- write ---------------------------------------------------------------------------------

  /**
   * Replace the file's whole content, atomically: a temp file beside the file the path resolves
   * to, committed with `mv -f`. Keeps the file's mode, writes through symlinks, never leaves the
   * temp behind. The parent directory must exist.
   */
  async write(path: string, data: Uint8Array | string): Promise<void> {
    const bytes = typeof data === "string" ? utf8(data) : data;
    const caps = await this.capabilities();
    const suffix = await randomHex64();
    let failed: SandboxError | undefined;
    try {
      const r = caps.stdin
        ? await this.exec(
          "sh",
          sh(WRITE_STDIN_SCRIPT, [path, suffix, String(bytes.length)]),
          bytes,
        )
        : await this.writeChunks(path, suffix, bytes);
      if (r.exitCode !== 0) {
        failed = classifyFailure(`write ${path}`, text(r.stderr));
      }
    } catch (e) {
      // The compensating action for what the scripts cannot see (a transport error or a timeout
      // between chunks); then the failure is reported as it is.
      await this.exec("sh", sh(REMOVE_TEMP_SCRIPT, [path, suffix])).catch(
        () => {},
      );
      throw e;
    }
    if (failed) {
      await this.exec("sh", sh(REMOVE_TEMP_SCRIPT, [path, suffix])).catch(
        () => {},
      );
      throw failed;
    }
  }

  private async writeChunks(
    path: string,
    suffix: string,
    bytes: Uint8Array,
  ): Promise<RunResult> {
    const size = String(bytes.length);
    let first = true;
    for (let at = 0;; at += WRITE_CHUNK) {
      // An empty file is one empty chunk: the temp must still be created and committed.
      const chunk = bytes.subarray(at, at + WRITE_CHUNK);
      const last = at + WRITE_CHUNK >= bytes.length;
      const r = await this.exec(
        "sh",
        sh(WRITE_CHUNK_SCRIPT, [
          path,
          suffix,
          toBase64(chunk),
          first ? "new" : "append",
          last ? "commit" : "",
          size,
        ]),
      );
      if (r.exitCode !== 0 || last) return r;
      first = false;
    }
  }

  /**
   * Write only if the file's mtime (whole epoch seconds) is `expectedMtime`; `null` means the file
   * must not exist. Throws `conflict` otherwise and writes nothing. Two round trips and not a lock:
   * it catches an edit between a read and a write, and misses one inside the same second.
   */
  async writeIfUnchanged(
    path: string,
    data: Uint8Array | string,
    expectedMtime: number | null,
  ): Promise<void> {
    if (expectedMtime !== null && !Number.isInteger(expectedMtime)) {
      throw new InvalidRequestError(
        "`expectedMtime` must be an integer or null",
      );
    }
    const current = (await this.stat(path))?.mtime ?? null;
    if (current !== expectedMtime) {
      throw new ConflictError("the file changed since it was read");
    }
    await this.write(path, data);
  }

  // ---- mkdir ---------------------------------------------------------------------------------

  /** `mkdir -p`: idempotent. */
  async mkdir(path: string): Promise<void> {
    await this.capabilities();
    const r = await this.exec("mkdir", ["-p", "--", path]);
    if (r.exitCode !== 0) {
      throw classifyFailure(`create ${path}`, text(r.stderr));
    }
  }

  // ---- list ----------------------------------------------------------------------------------

  /** Entries of a directory (order unspecified). Listing a file gives none. */
  async list(path: string, options: ListOptions = {}): Promise<DirEntry[]> {
    const includeHidden = options.includeHidden ?? false;
    const limit = count("limit", options.limit, 10000);
    const caps = await this.capabilities();
    if (caps.findPrintf) {
      const r = await this.exec("find", [
        "-H",
        path,
        "-mindepth",
        "1",
        "-maxdepth",
        "1",
        "-printf",
        "%Y\\t%s\\t%f\\0",
      ]);
      if (r.exitCode !== 0 && r.stdout.length === 0) {
        throw classifyFailure(`ls ${path}`, text(r.stderr));
      }
      return parseDirEntries(text(r.stdout), limit, includeHidden);
    }
    // Busybox: type from a second pass restricted to directories.
    const base = ["-H", path, "-mindepth", "1", "-maxdepth", "1"];
    const listing = await this.exec("find", [...base, "-print0"]);
    if (listing.exitCode !== 0 && listing.stdout.length === 0) {
      throw classifyFailure(`ls ${path}`, text(listing.stderr));
    }
    const dirs = await this.exec("find", [...base, "-type", "d", "-print0"]);
    const dirSet = new Set(
      text(dirs.stdout)
        .split("\0")
        .filter((r) => r !== ""),
    );
    const out: DirEntry[] = [];
    for (const raw of text(listing.stdout).split("\0")) {
      if (raw === "" || out.length >= limit) continue;
      const comps = raw.split("/").filter((c) => c !== "");
      const name = comps[comps.length - 1] ?? "";
      if (
        name === "" || name === ".." || (!includeHidden && name.startsWith("."))
      ) continue;
      out.push({ name, kind: dirSet.has(raw) ? "dir" : "file", size: 0 });
    }
    return out;
  }

  // ---- search --------------------------------------------------------------------------------

  /**
   * Search file contents with `rg`, GNU `grep` or busybox `grep`, whichever the sandbox has
   * ({@link Capabilities.searchEngine}); the engines differ (`.gitignore`, regex dialect), so say
   * which you used.
   */
  async search(query: SearchQuery): Promise<SearchResult> {
    if (typeof query.pattern !== "string" || typeof query.root !== "string") {
      throw new InvalidRequestError("`pattern` and `root` must be strings");
    }
    count("limit", query.limit, 100);
    count("before", query.before, 0);
    count("after", query.after, 0);
    const q = withDefaults(query);
    const caps = await this.capabilities();
    const engine = caps.searchEngine;
    const program = engine === "rg" ? "rg" : "grep";
    const args = engine === "rg"
      ? rgArgs(q, await this.rootInGitRepo(q.root))
      : grepArgs(q, engine === "grep");
    const r = await this.exec(program, args);

    const stdout = text(r.stdout);
    const stderr = text(r.stderr);
    let skipped: string | null = null;
    if (r.exitCode !== 0 && r.exitCode !== 1) {
      if (r.exitCode === 124) {
        throw new TimeoutError(
          `${program} timed out after ${COMMAND_TIMEOUT_MS / 1000}s`,
        );
      }
      const f = classifySearchFailure(stderr, q.root);
      if (f.kind === "bad_regex") {
        const detail = stderr.trim();
        throw new InvalidQueryError(
          `bad regex: ${
            detail === "" ? `${program} rejected the pattern` : detail
          }`,
        );
      }
      if (f.kind === "missing_root") throw new NotFoundError(f.line);
      skipped = f.note;
    }

    let hits = engine === "busybox-grep"
      ? parseColonRecords(stdout, q)
      : parseRecords(stdout, engine === "rg");
    hits = hits.filter((h) => !isUnderGitDir(q.root, h.path));
    const done = finalize(hits, q.limit);
    const first = stderr.trim().split("\n")[0] ?? "";
    return {
      hits: done.hits,
      truncated: done.truncated,
      note: skipped ?? (first === "" ? null : first),
    };
  }

  /** Whether `root` or an ancestor holds a `.git`, asked in the sandbox and cached per root. */
  private async rootInGitRepo(root: string): Promise<boolean> {
    const known = this.gitRoots.get(root);
    if (known !== undefined) return known;
    let inside = false;
    try {
      inside = (await this.exec("sh", sh(GIT_SCRIPT, [root]))).exitCode === 0;
    } catch {
      return false; // a failed probe counts as "not inside"; not cached
    }
    this.gitRoots.set(root, inside);
    return inside;
  }
}

/** Parse the first line of the stat script: `W<TAB>T<TAB>SIZE<TAB>MTIME`. */
export function parseStat(stdout: string): Stat | null {
  const line = stdout.split("\n")[0] ?? "";
  const parts = line.split("\t");
  if (parts.length < 3) return null;
  const size = parts[2]!.trim();
  if (!/^\d+$/.test(size)) return null;
  const secs = (parts[3] ?? "").split(".")[0]!.trim();
  const mtime = /^-?\d+$/.test(secs) ? Number(secs) : 0;
  return {
    writable: parts[0] === "w",
    kind: parts[1] === "f" ? "file" : parts[1] === "d" ? "dir" : "other",
    size: Number(size),
    mtime,
  };
}

/** Parse NUL-terminated `TYPE<TAB>SIZE<TAB>NAME` records of `find -printf '%Y\t%s\t%f\0'`. */
export function parseDirEntries(
  stdout: string,
  limit: number,
  includeHidden: boolean,
): DirEntry[] {
  const out: DirEntry[] = [];
  for (const record of stdout.split("\0")) {
    if (record === "" || out.length >= limit) break;
    const a = record.indexOf("\t");
    if (a < 0) continue;
    const b = record.indexOf("\t", a + 1);
    if (b < 0) continue;
    const type = record.slice(0, a);
    const size = record.slice(a + 1, b);
    const name = record.slice(b + 1);
    if (!includeHidden && name.startsWith(".")) continue;
    if (type === "N" || type === "?") continue;
    const n = /^\d+$/.test(size.trim()) ? Number(size.trim()) : 0;
    out.push({
      name,
      kind: type === "f" ? "file" : type === "d" ? "dir" : "other",
      size: n,
    });
  }
  return out;
}
