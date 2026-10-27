import { utf8 } from "./bytes.js";
import { compileGlob } from "./glob.js";

/** Which program a search uses (FILES-OVER-EXEC section 4). */
export type SearchEngine = "rg" | "grep" | "busybox-grep";

export interface SearchQuery {
  pattern: string;
  root: string;
  literal?: boolean;
  ignoreCase?: boolean;
  /** Only (or, with `globExclude`, never) paths matching this glob. */
  glob?: string;
  globExclude?: boolean;
  /** Matches to return; default 100. Context lines do not count. */
  limit?: number;
  /** Lines of context before and after each match; clamped to 100. */
  before?: number;
  after?: number;
}

export interface SearchHit {
  path: string;
  line: number;
  text: string;
  isMatch: boolean;
}

export interface SearchResult {
  hits: SearchHit[];
  truncated: boolean;
  /** The first line of stderr of a search that succeeded, or what was skipped. */
  note: string | null;
}

export const MAX_LINE = 500;
export const MAX_CONTEXT = 100;
const TRUNCATED_SUFFIX = "… [truncated]";
const WINDOW_PREFIX = "… ";
const WINDOW_MARK = " [truncated; columns ";

type Required_ = Required<Omit<SearchQuery, "glob">> & { glob?: string };

export function withDefaults(q: SearchQuery): Required_ {
  const r: Required_ = {
    pattern: q.pattern,
    root: q.root,
    literal: q.literal ?? false,
    ignoreCase: q.ignoreCase ?? false,
    globExclude: q.globExclude ?? false,
    limit: q.limit ?? 100,
    before: q.before ?? 0,
    after: q.after ?? 0,
  };
  if (q.glob !== undefined) r.glob = q.glob;
  return r;
}

function pushContext(args: string[], q: Required_): void {
  const before = Math.min(q.before, MAX_CONTEXT);
  const after = Math.min(q.after, MAX_CONTEXT);
  if (before > 0) args.push("-B", String(before));
  if (after > 0) args.push("-A", String(after));
}

/** The `rg` argument list (section 5.7). */
export function rgArgs(q: SearchQuery, inGitRepo: boolean): string[] {
  const r = withDefaults(q);
  const args = [
    "--null",
    "--with-filename",
    "--line-number",
    "--column",
    "--no-heading",
    "--color=never",
    "--hidden",
    "--crlf",
  ];
  if (!inGitRepo) args.push("--no-require-git");
  if (r.ignoreCase) args.push("--ignore-case");
  if (r.literal) args.push("--fixed-strings");
  if (r.glob !== undefined) {
    args.push("--glob", r.globExclude ? `!${r.glob}` : r.glob);
  }
  args.push("--glob", "!.git");
  pushContext(args, r);
  // The group separator is the one record with no NUL, so it cannot be told from a path that starts
  // with `--` and a newline: the engine is told not to print it.
  args.push("--no-context-separator");
  args.push("-e", r.pattern, "--", r.root);
  return args;
}

/** The `grep` argument list; `gnu` false is busybox. */
export function grepArgs(q: SearchQuery, gnu: boolean): string[] {
  const r = withDefaults(q);
  const args = ["-r", "-n", "-H"];
  // `-b`: the offset of each line, which a second run's `-o -b` match offsets turn into a column.
  if (gnu) args.push("-Z", "-I", "--color=never", "-b");
  args.push(r.literal ? "-F" : "-E");
  if (r.ignoreCase) args.push("-i");
  if (gnu && r.glob !== undefined) {
    args.push(r.globExclude ? `--exclude=${r.glob}` : `--include=${r.glob}`);
  }
  if (gnu && lastComponent(r.root) !== ".git") args.push("--exclude-dir=.git");
  pushContext(args, r);
  // GNU only (busybox rejects it and prints its `--` line, which the colon parser skips).
  if (gnu) args.push("--no-group-separator");
  args.push(
    "-e",
    r.literal ? r.pattern : crlfTolerantEre(r.pattern),
    "--",
    r.root,
  );
  return args;
}

/** The last path component, as Rust's `Path::file_name` gives it (none for `/`, `..`). */
function lastComponent(p: string): string | undefined {
  const comps = components(p);
  const last = comps[comps.length - 1];
  return last === undefined || last === ".." ? undefined : last;
}

/**
 * Make an ERE's end-of-line anchors tolerate a CRLF line ending: each `$` that anchors the end of
 * a branch (end of pattern, before `|` or `)`) becomes `<CR>?$`. Escaped `$` and `$` inside a
 * bracket expression stay.
 */
export function crlfTolerantEre(pattern: string): string {
  if (!pattern.includes("$")) return pattern;
  const chars = Array.from(pattern);
  let out = "";
  let i = 0;
  while (i < chars.length) {
    const c = chars[i]!;
    if (c === "\\") {
      out += c;
      const next = chars[i + 1];
      if (next !== undefined) {
        out += next;
        i++;
      }
    } else if (c === "[") {
      const start = i;
      i++;
      if (chars[i] === "^") i++;
      if (chars[i] === "]") i++;
      while (i < chars.length && chars[i] !== "]") {
        const open = chars[i + 1];
        if (
          chars[i] === "[" && (open === ":" || open === "=" || open === ".")
        ) {
          i += 2;
          while (
            i + 1 < chars.length && !(chars[i] === open && chars[i + 1] === "]")
          ) i++;
          i++;
        }
        i++;
      }
      const end = Math.min(i, chars.length - 1);
      out += chars.slice(start, end + 1).join("");
    } else if (
      c === "$"
      && (chars[i + 1] === undefined || chars[i + 1] === "|"
        || chars[i + 1] === ")")
    ) {
      out += "\r?$";
    } else {
      out += c;
    }
    i++;
  }
  return out;
}

/** Strip one trailing LF, then one trailing CR, then every remaining CR. */
export function trimEol(s: string): string {
  if (s.endsWith("\n")) s = s.slice(0, -1);
  if (s.endsWith("\r")) s = s.slice(0, -1);
  return s.includes("\r") ? s.replaceAll("\r", "") : s;
}

function isDigit(c: string | undefined): boolean {
  return c !== undefined && c >= "0" && c <= "9";
}

/** A line number from digits, or null if it does not fit a safe integer. */
function lineNumber(digits: string): number | null {
  const n = Number(digits);
  return Number.isSafeInteger(n) ? n : null;
}

/**
 * Parse `PATH<NUL>LINE<SEP>TEXT<LF>` records, as `rg --null` and `grep -Z` emit them with their
 * group separator turned off. Scans for the NUL, not for newlines (a path may contain one), skips
 * no `--` bytes (a path may begin with them), and drops a record that does not fit.
 */
export function parseRecords(
  stdout: string,
  columns: boolean | "gnu" = false,
  longs: LongLine[] = [],
): SearchHit[] {
  const hits: SearchHit[] = [];
  let i = 0;
  let prev = "";
  while (i < stdout.length) {
    // `rg` follows the records of a file it stopped searching at a NUL byte with a line of its own,
    // which has no NUL and so must not be read as the start of a record.
    let first = stdout.indexOf("\n", i);
    if (first < 0) first = stdout.length;
    if (isBinaryWarning(stdout.slice(i, first), prev)) {
      i = first + 1;
      continue;
    }
    const nul = stdout.indexOf("\0", i);
    if (nul < 0) break;
    let nl = stdout.indexOf("\n", nul);
    if (nl < 0) nl = stdout.length;
    prev = stdout.slice(i, nul);
    const hit = parseBody(
      stdout.slice(i, nul),
      stdout.slice(nul + 1, nl),
      columns,
      longs,
      hits.length,
    );
    if (hit) hits.push(hit);
    i = nl + 1;
  }
  return hits;
}

const BINARY_WARNING =
  ": WARNING: stopped searching binary file after match (found \"\\0\" byte around offset ";

/**
 * `PATH: WARNING: stopped searching binary file after match (found "\0" byte around offset N)`,
 * where PATH is the file of the record just before it: `rg` prints it after the matches it found
 * ahead of a NUL byte. Matching the previous record's path, not only the text, keeps a file whose
 * name happens to contain these words a path.
 */
export function isBinaryWarning(line: string, prev: string): boolean {
  if (prev === "" || !line.startsWith(prev + BINARY_WARNING)) return false;
  return /^\d+\)$/.test(line.slice(prev.length + BINARY_WARNING.length));
}

function parseBody(
  path: string,
  rest: string,
  columns: boolean | "gnu",
  longs: LongLine[],
  index: number,
): SearchHit | null {
  let d = 0;
  while (isDigit(rest[d])) d++;
  if (d === 0 || d >= rest.length) return null;
  const line = lineNumber(rest.slice(0, d));
  if (line === null) return null;
  const sep = rest[d];
  if (sep !== ":" && sep !== "-") return null;
  const isMatch = sep === ":";
  let text = rest.slice(d + 1);
  // `rg --column`: a match record is `LINE:COL:TEXT`, a context record `LINE-TEXT`. GNU `grep -b`:
  // `LINE:OFFSET:TEXT` and `LINE-OFFSET-TEXT`, the offset of the line start.
  let col: number | null = null;
  if (columns === "gnu" || (columns && isMatch)) {
    let c = 0;
    while (isDigit(text[c])) c++;
    if (c === 0 || text[c] !== sep) return null;
    const n = lineNumber(text.slice(0, c));
    text = text.slice(c + 1);
    if (columns === "gnu") {
      if (isMatch && n !== null && text.length > MAX_LINE) {
        longs.push({ index, lineStart: n, raw: text });
      }
    } else {
      col = n;
    }
  }
  return {
    path,
    line,
    text: clipAt(trimEol(text), matchOffset(text, col)),
    isMatch,
  };
}

/** A long match line of a GNU `grep` run, whose column the second run finds. */
export interface LongLine {
  index: number;
  lineStart: number;
  raw: string;
}

/**
 * Parse busybox `grep -rn` output (`PATH:LINE:TEXT` or `PATH-LINE-TEXT`): split at the first
 * position where `:` or `-` is followed by digits and the same byte again. The caller's glob is
 * applied here to the full path (busybox has no `--include`).
 */
export function parseColonRecords(stdout: string, q: SearchQuery): SearchHit[] {
  const r = withDefaults(q);
  const re = r.glob !== undefined ? compileGlob(r.glob, r.ignoreCase) : null;
  const hits: SearchHit[] = [];
  const lines = stdout.split("\n");
  if (lines[lines.length - 1] === "") lines.pop();
  for (let line of lines) {
    if (line.endsWith("\r")) line = line.slice(0, -1);
    if (line === "--") continue;
    const rec = splitColonRecord(line);
    if (!rec) continue;
    if (re) {
      const matched = re.test(rec.path);
      if (r.globExclude ? matched : !matched) continue;
    }
    hits.push({
      path: rec.path,
      line: rec.line,
      text: clip(trimEol(rec.text)),
      isMatch: rec.isMatch,
    });
  }
  return hits;
}

export function splitColonRecord(
  line: string,
): { path: string; line: number; isMatch: boolean; text: string } | null {
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c !== ":" && c !== "-") continue;
    let d = i + 1;
    while (isDigit(line[d])) d++;
    const digits = d - (i + 1);
    if (digits === 0 || line[d] !== c) continue;
    const n = lineNumber(line.slice(i + 1, d));
    if (n === null) return null;
    return {
      path: line.slice(0, i),
      line: n,
      isMatch: c === ":",
      text: line.slice(d + 1),
    };
  }
  return null;
}

export type SearchFailure =
  | { kind: "bad_regex" }
  | { kind: "missing_root"; line: string }
  | { kind: "skipped"; note: string };

const OS_ERRORS = [
  "No such file or directory",
  "Permission denied",
  "Is a directory",
  "Not a directory",
  "Too many levels of symbolic links",
  "Input/output error",
  // rg reading a `.git` file (a gitdir pointer) that is not text.
  "stream did not contain valid UTF-8",
];

/**
 * Exit 2 is shared by three situations: the engine rejecting the pattern, a missing root, and
 * per-path read errors. A path may contain a newline, so stderr is read as records that end at an
 * OS error, not as lines; the root's own error is found by its exact string (newlines included).
 */
export function classifySearchFailure(
  stderr: string,
  root: string,
): SearchFailure {
  const endsInOsError = (raw: string): boolean => {
    let l = raw.trimEnd();
    if (l.endsWith(")")) {
      const at = l.lastIndexOf(" (os error ");
      if (at >= 0) l = l.slice(0, at);
    }
    return OS_ERRORS.some((e) => l.endsWith(e));
  };
  const records: string[] = [];
  let start = 0;
  for (const m of stderr.matchAll(/[^\n]*\n|[^\n]+$/g)) {
    const end = m.index! + m[0].length;
    if (endsInOsError(m[0])) {
      records.push(stderr.slice(start, end).trim());
      start = end;
    }
  }
  if (records.length === 0 || stderr.slice(start).trim() !== "") {
    return { kind: "bad_regex" };
  }
  const rootPrefix = `: ${root}: `;
  const ioPrefix = `IO error for operation on ${root}: `;
  const missing = records.find((r) => {
    for (
      let at = r.indexOf(rootPrefix);
      at >= 0;
      at = r.indexOf(rootPrefix, at + 1)
    ) {
      let rest = r.slice(at + rootPrefix.length);
      if (rest.startsWith(ioPrefix)) rest = rest.slice(ioPrefix.length);
      if (
        rest.startsWith("No such file or directory")
        || rest.startsWith("Not a directory")
      ) return true;
    }
    return false;
  });
  if (missing !== undefined) {
    return { kind: "missing_root", line: missing };
  }
  return {
    kind: "skipped",
    note:
      `${records.length} path(s) could not be searched; first: ${records[0]!}`,
  };
}

// ---- clipping --------------------------------------------------------------------------------

/** Clip a line to 500 characters (code points). */
export function clip(line: string): string {
  if (line.length <= MAX_LINE) return line;
  const chars = Array.from(line);
  if (chars.length <= MAX_LINE) return line;
  return chars.slice(0, MAX_LINE).join("") + TRUNCATED_SUFFIX;
}

/**
 * Clip the long match lines of a GNU `grep` run (`longs`) from the output of a second run with
 * `-o -b` and no context: the file offset of a line's first match less the line's offset is the byte
 * column `rg --column` would have given.
 */
export function windowLongLines(
  hits: SearchHit[],
  longs: LongLine[],
  stdout: string,
): void {
  const first = new Map<string, number>();
  let i = 0;
  while (i < stdout.length) {
    const nul = stdout.indexOf("\0", i);
    if (nul < 0) break;
    let nl = stdout.indexOf("\n", nul);
    if (nl < 0) nl = stdout.length;
    const m = /^(\d+):(\d+):/.exec(stdout.slice(nul + 1, nl));
    if (m) {
      const key = `${stdout.slice(i, nul)}\0${m[1]}`;
      if (!first.has(key)) first.set(key, Number(m[2]));
    }
    i = nl + 1;
  }
  for (const long of longs) {
    const hit = hits[long.index]!;
    const off = first.get(`${hit.path}\0${hit.line}`);
    if (off === undefined || off < long.lineStart) continue;
    const cs = matchOffset(long.raw, off - long.lineStart + 1);
    if (cs !== null) hit.text = clipAt(trimEol(long.raw), cs);
  }
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/**
 * The character offset, in the normalized text, of rg's 1-based byte column `col` of the raw line:
 * the characters before it, CRs not counted. The bytes are those of the UTF-8 encoding of the
 * decoded line. Null when there is no column or it is not a character boundary.
 */
export function matchOffset(raw: string, col: number | null): number | null {
  if (col === null || col < 1 || raw.length <= MAX_LINE) return null;
  const bytes = encoder.encode(raw);
  const at = col - 1;
  if (at > bytes.length) return null;
  if (at < bytes.length && (bytes[at]! & 0xc0) === 0x80) return null;
  let n = 0;
  for (const c of decoder.decode(bytes.subarray(0, at))) if (c !== "\r") n++;
  return n;
}

/**
 * Clip a line to 500 characters. With `cs`, the character offset of its first match as the engine
 * reported it, the window starts 250 characters before the match (the head clip when that is the
 * head); the match is never searched for again.
 */
export function clipAt(line: string, cs: number | null): string {
  if (line.length <= MAX_LINE) return line;
  const chars = Array.from(line);
  if (chars.length <= MAX_LINE) return line;
  if (cs === null) return clip(line);
  const total = chars.length;
  let start = Math.max(0, cs - MAX_LINE / 2);
  if (start === 0) return clip(line);
  let end = start + MAX_LINE;
  if (end > total) {
    end = total;
    start = total - MAX_LINE;
  }
  let out = WINDOW_PREFIX + chars.slice(start, end).join("");
  if (end < total) out += " …";
  return `${out}${WINDOW_MARK}${start + 1}-${end} of ${total}]`;
}

// ---- paths -----------------------------------------------------------------------------------

/** The components of a path as Rust's `Path` has them (empty ones dropped). */
export function components(p: string): string[] {
  return p.split("/").filter((c) => c !== "");
}

function compareBytes(a: Uint8Array, b: Uint8Array): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = a[i]! - b[i]!;
    if (d !== 0) return d;
  }
  return a.length - b.length;
}

/** Whether `path` is, or lies inside, a `.git` entry below `root` (root's own components do not count). */
export function isUnderGitDir(root: string, path: string): boolean {
  const r = components(root);
  const p = components(path);
  if (p.length < r.length) return false;
  for (let i = 0; i < r.length; i++) if (p[i] !== r[i]) return false;
  return p.slice(r.length).includes(".git");
}

/**
 * Sort by path then line (paths compare component by component, bytewise), count only matches
 * toward `limit`, cut before the first hit that would be the (limit+1)-th match, and drop the
 * trailing context. The result ends on a match.
 */
export function finalize(
  hits: SearchHit[],
  limit: number,
): { hits: SearchHit[]; truncated: boolean } {
  const keyed = hits.map((h) => ({ h, k: components(h.path).map(utf8) }));
  keyed.sort((a, b) => {
    const n = Math.min(a.k.length, b.k.length);
    for (let i = 0; i < n; i++) {
      const d = compareBytes(a.k[i]!, b.k[i]!);
      if (d !== 0) return d;
    }
    return a.k.length - b.k.length || a.h.line - b.h.line;
  });
  let out = keyed.map((x) => x.h);
  const matches = out.reduce((n, h) => n + (h.isMatch ? 1 : 0), 0);
  const truncated = matches > limit;
  if (truncated) {
    let seen = 0;
    let cut = out.length;
    for (let i = 0; i < out.length; i++) {
      if (out[i]!.isMatch && ++seen > limit) {
        cut = i;
        break;
      }
    }
    out = out.slice(0, cut);
    while (out.length > 0 && !out[out.length - 1]!.isMatch) out.pop();
  }
  return { hits: out, truncated };
}
