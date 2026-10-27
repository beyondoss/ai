import { describe, expect, it } from "vitest";
import { classifyFailure, errorDetail } from "../src/errors.js";
import { parseDirEntries, parseStat } from "../src/files.js";
import { compileGlob } from "../src/glob.js";
import { parseListLine, parseRead } from "../src/jobs.js";
import {
  classifySearchFailure,
  clip,
  crlfTolerantEre,
  finalize,
  grepArgs,
  isUnderGitDir,
  parseColonRecords,
  parseRecords,
  rgArgs,
  splitColonRecord,
} from "../src/search.js";
import type { SearchHit } from "../src/search.js";

const q = (pattern: string, root = ".") => ({ pattern, root });

describe("parseRecords (rg --null / grep -Z)", () => {
  it("parses a match record", () => {
    expect(parseRecords("src/a.rs\0" + "12:let x = 1;\n")).toEqual([
      { path: "src/a.rs", line: 12, text: "let x = 1;", isMatch: true },
    ]);
  });

  it("tells context lines from matches", () => {
    const hits = parseRecords(
      "a.rs\x001-before\na.rs\x002:hit\na.rs\x003-after\n",
    );
    expect(hits.map((h) => h.isMatch)).toEqual([false, true, false]);
  });

  it("treats a path that starts with dashes and a newline as a path, not a separator", () => {
    const hits = parseRecords("--\nb\x002-y\n--\r\nc\x003:z\r\n");
    expect(hits.map((h) => [h.path, h.line])).toEqual([
      ["--\nb", 2],
      ["--\r\nc", 3],
    ]);
  });

  it("does not mistake a file named like a separator for one", () => {
    const hits = parseRecords("--\x001:x\n--\r\x002:y\n--\x003:--\n");
    expect(hits.map((h) => [h.path, h.line])).toEqual([
      ["--", 1],
      ["--\r", 2],
      ["--", 3],
    ]);
    expect(hits[2]!.text).toBe("--");
  });

  it("keeps a record whose path holds a newline whole", () => {
    expect(parseRecords("we\nird.rs\x009:two\n")).toEqual([
      { path: "we\nird.rs", line: 9, text: "two", isMatch: true },
    ]);
  });

  it("ends the path at the NUL, not at a colon", () => {
    const hits = parseRecords("odd:name.rs\x003:body\n");
    expect(hits).toEqual([{
      path: "odd:name.rs",
      line: 3,
      text: "body",
      isMatch: true,
    }]);
  });

  it("strips every carriage return from the text", () => {
    expect(parseRecords("a.rs\x001:crlf line\r\n")[0]!.text).toBe("crlf line");
    expect(parseRecords("a.rs\x001:a\rb\rc\n")[0]!.text).toBe("abc");
  });

  it("drops a record whose line number is not digits", () => {
    expect(parseRecords("a\0x:oops\nb\x002:ok\n").map((h) => h.path)).toEqual([
      "b",
    ]);
  });
});

describe("parseColonRecords (busybox grep -rn)", () => {
  it("splits at the first separator followed by digits and the same separator", () => {
    expect(splitColonRecord("src/a.txt:12:hello: world")).toEqual({
      path: "src/a.txt",
      line: 12,
      isMatch: true,
      text: "hello: world",
    });
    expect(splitColonRecord("src/a.txt-3-context")).toMatchObject({
      line: 3,
      isMatch: false,
    });
    expect(splitColonRecord("no separators here")).toBeNull();
  });

  it("is wrong for a path holding the separator sequence (the documented cost)", () => {
    expect(splitColonRecord("a:1:b.txt:2:x")).toMatchObject({
      path: "a",
      line: 1,
    });
  });

  it("skips `--` and applies the glob to the full path, with globset semantics", () => {
    const out = "/r/a.txt:1:x\n--\n/r/sub/b.rs:2:y\n/r/sub/c.txt-3-z\n";
    expect(
      parseColonRecords(out, { ...q("x"), glob: "*.txt" }).map((h) => h.path),
    ).toEqual([
      "/r/a.txt",
      "/r/sub/c.txt",
    ]);
    expect(
      parseColonRecords(out, { ...q("x"), glob: "*.txt", globExclude: true })
        .map((h) => h.path),
    ).toEqual(["/r/sub/b.rs"]);
  });
});

describe("argument builders", () => {
  it("puts the pattern behind -e and the path behind --", () => {
    const args = rgArgs(q("-v", "-weird-dir"), false);
    expect(args[args.indexOf("-e") + 1]).toBe("-v");
    expect(args[args.indexOf("--") + 1]).toBe("-weird-dir");
  });

  it("passes the exact rg flag order", () => {
    expect(
      rgArgs(
        {
          pattern: "p",
          root: "/r",
          ignoreCase: true,
          literal: true,
          before: 2,
          after: 500,
        },
        false,
      ),
    ).toEqual([
      "--null",
      "--with-filename",
      "--line-number",
      "--column",
      "--no-heading",
      "--color=never",
      "--hidden",
      "--crlf",
      "--no-require-git",
      "--ignore-case",
      "--fixed-strings",
      "--glob",
      "!.git",
      "-B",
      "2",
      "-A",
      "100",
      "--no-context-separator",
      "-e",
      "p",
      "--",
      "/r",
    ]);
    expect(rgArgs(q("p", "/r"), true)).not.toContain("--no-require-git");
  });

  it("turns a negated glob into !GLOB and keeps !.git last so it outranks the caller's", () => {
    const args = rgArgs(
      { ...q("x"), glob: "*.test.rs", globExclude: true },
      false,
    );
    expect(args[args.indexOf("--glob") + 1]).toBe("!*.test.rs");
    const globs = args.flatMap((a, i) => (a === "--glob" ? [args[i + 1]] : []));
    expect(globs).toEqual(["!*.test.rs", "!.git"]);
  });

  it("excludes .git for GNU grep unless the root is .git itself, and never for busybox", () => {
    expect(grepArgs(q("x", "/repo"), true)).toContain("--exclude-dir=.git");
    expect(
      grepArgs(q("x", "/repo/.git"), true).some((a) =>
        a.startsWith("--exclude-dir")
      ),
    ).toBe(
      false,
    );
    expect(
      grepArgs(q("x", "/repo"), false).some((a) =>
        a.startsWith("--exclude-dir")
      ),
    ).toBe(false);
  });

  it("maps globs to --include/--exclude for GNU grep only", () => {
    expect(grepArgs({ ...q("x"), glob: "*.rs" }, true)).toContain(
      "--include=*.rs",
    );
    expect(grepArgs({ ...q("x"), glob: "*.rs", globExclude: true }, true))
      .toContain(
        "--exclude=*.rs",
      );
    expect(
      grepArgs({ ...q("x"), glob: "*.rs" }, false).some((a) =>
        a.startsWith("--inc")
      ),
    ).toBe(
      false,
    );
  });

  it("turns the group separator off for rg and GNU grep, and only there", () => {
    expect(grepArgs(q("x"), true)).toContain("--no-group-separator");
    expect(grepArgs(q("x"), false)).not.toContain("--no-group-separator");
  });

  it("uses -F for literal and never pairs it with -E; busybox gets no GNU-only flags", () => {
    const lit = grepArgs({ ...q("a.b(c)"), literal: true }, true);
    expect(lit).toContain("-F");
    expect(lit).not.toContain("-E");
    const bb = grepArgs(q("x"), false);
    for (const gnu of ["-Z", "-I", "--color=never"]) {
      expect(bb).not.toContain(gnu);
    }
    expect(bb.slice(0, 4)).toEqual(["-r", "-n", "-H", "-E"]);
  });
});

describe("crlfTolerantEre", () => {
  it("rewrites end anchors, and only those", () => {
    expect(crlfTolerantEre("end$")).toBe("end\r?$");
    expect(crlfTolerantEre("(one$|two$)")).toBe("(one\r?$|two\r?$)");
    expect(crlfTolerantEre("a$|b")).toBe("a\r?$|b");
    expect(crlfTolerantEre("cost\\$")).toBe("cost\\$");
    expect(crlfTolerantEre("[$]")).toBe("[$]");
    expect(crlfTolerantEre("[]$]x$")).toBe("[]$]x\r?$");
    expect(crlfTolerantEre("[[:space:]$]$")).toBe("[[:space:]$]\r?$");
    expect(crlfTolerantEre("plain")).toBe("plain");
  });

  it("does not rewrite a literal search", () => {
    expect(grepArgs({ ...q("end$"), literal: true }, true)).toContain("end$");
    expect(grepArgs(q("end$"), true)).toContain("end\r?$");
  });
});

describe("a newline in a path", () => {
  it("does not hide the OS error", () => {
    const root = "/work/no\nwhere";
    for (
      const missing of [
        "rg: /work/no\nwhere: No such file or directory (os error 2)\n",
        "rg: /work/no\nwhere: IO error for operation on /work/no\nwhere: No such file or directory (os error 2)\n",
        "grep: /work/no\nwhere: No such file or directory\n",
      ]
    ) {
      expect(classifySearchFailure(missing, root).kind).toBe("missing_root");
    }
    const f = classifySearchFailure(
      "rg: /p/a\nb: Permission denied (os error 13)\nrg: /p/c: Permission denied (os error 13)\n",
      "/p",
    );
    expect((f as { note: string }).note.startsWith("2 path(s)")).toBe(true);
    expect(
      classifyFailure("cat x", "cat: a\nb: No such file or directory\n").kind,
    )
      .toBe("not_found");
    expect(errorDetail("one\ntwo\n")).toBe("one");
  });
});

describe("a root below a file", () => {
  it("is a root error", () => {
    const f = classifySearchFailure(
      "rg: /w/file/sub: IO error for operation on /w/file/sub: Not a directory (os error 20)\n",
      "/w/file/sub",
    );
    expect(f.kind).toBe("missing_root");
    expect(
      classifySearchFailure("grep: /w/file/sub: Not a directory", "/w/file")
        .kind,
    ).toBe("skipped");
  });
});

describe("classifySearchFailure (exit 2)", () => {
  const root = "/work/proj";
  it("reads an engine complaint, or silence, as a rejected pattern", () => {
    for (
      const bad of [
        "regex parse error:\n    (\n    ^\nerror: unclosed group",
        "grep: Unmatched ( or \\(",
        "grep: bad regex '(': Unmatched ( or \\(",
        "",
      ]
    ) {
      expect(classifySearchFailure(bad, root).kind).toBe("bad_regex");
    }
  });

  it("reads a missing root", () => {
    for (
      const missing of [
        "rg: /work/proj: No such file or directory (os error 2)",
        "rg: /work/proj: IO error for operation on /work/proj: No such file or directory (os error 2)",
        "grep: /work/proj: No such file or directory",
      ]
    ) {
      expect(classifySearchFailure(missing, root).kind).toBe("missing_root");
    }
  });

  it("keeps the hits when only paths below the root failed", () => {
    const f = classifySearchFailure(
      "rg: /work/proj/a: Permission denied (os error 13)\nrg: /work/proj/b: Permission denied (os error 13)\n",
      root,
    );
    expect(f).toMatchObject({ kind: "skipped" });
    expect((f as { note: string }).note.startsWith("2 path(s)")).toBe(true);
    expect(
      classifySearchFailure(
        "grep: /work/proj/x: No such file or directory",
        root,
      ).kind,
    ).toBe(
      "skipped",
    );
  });
});

describe("classifyFailure (a command that ran and failed)", () => {
  it("classifies by the first line of stderr, case-insensitively, first rule wins", () => {
    const kind = (s: string) => classifyFailure("op", s).kind;
    expect(kind("cat: x: No such file or directory")).toBe("not_found");
    expect(kind("mv: Directory nonexistent")).toBe("not_found");
    expect(kind("cp: nonexistent directory")).toBe("not_found");
    expect(kind("IS A DIRECTORY")).toBe("is_directory");
    expect(kind("mkdir: Not a directory")).toBe("not_a_directory");
    expect(kind("sh: Permission denied")).toBe("permission_denied");
    expect(kind("something else")).toBe("io");
    expect(kind("")).toBe("io");
    expect(classifyFailure("write /a", "").detail).toBe(
      "write /a: command failed",
    );
    expect(classifyFailure("write /a", "\n  boom\nsecond").detail).toBe(
      "write /a: boom",
    );
  });

  it("does not classify by the path in the prefix", () => {
    expect(classifyFailure("read /no such file or directory/x", "weird").kind)
      .toBe("io");
  });
});

describe("parseStat", () => {
  it("parses the line, keeping the integer part of the mtime", () => {
    expect(parseStat("w\tf\t12\t1700000000.9876\n")).toEqual({
      writable: true,
      kind: "file",
      size: 12,
      mtime: 1700000000,
    });
    expect(parseStat("-\td\t4096\t5\n")).toMatchObject({
      writable: false,
      kind: "dir",
    });
    expect(parseStat("w\to\t0\t0\n")).toMatchObject({ kind: "other" });
  });

  it("treats a line that does not parse as absent", () => {
    expect(parseStat("w\tf\tnot-a-number\t5\n")).toBeNull();
    expect(parseStat("garbage\n")).toBeNull();
  });

  it("tolerates a stat that failed (0, 0) and a missing mtime", () => {
    expect(parseStat("w\tf\t0\t0\n")).toMatchObject({ size: 0, mtime: 0 });
    expect(parseStat("w\tf\t3\t")).toMatchObject({ size: 3, mtime: 0 });
  });
});

describe("parseDirEntries (find -printf)", () => {
  const rec = (...r: string[]) => r.join("\0") + "\0";
  it("splits at the first two tabs only, and maps the types", () => {
    const out = rec("f\t5\ta\tb", "d\t4096\tsub", "p\t0\tfifo");
    expect(parseDirEntries(out, 100, true)).toEqual([
      { name: "a\tb", kind: "file", size: 5 },
      { name: "sub", kind: "dir", size: 4096 },
      { name: "fifo", kind: "other", size: 0 },
    ]);
  });

  it("skips broken links (N) and unstattable entries (?)", () => {
    expect(
      parseDirEntries(rec("N\t0\tbroken", "?\t0\tx", "f\t1\tok"), 100, true),
    ).toHaveLength(1);
  });

  it("drops hidden names before counting toward the limit", () => {
    const out = rec("f\t1\t.h1", "f\t1\t.h2", "f\t1\ta", "f\t1\tb", "f\t1\tc");
    expect(parseDirEntries(out, 2, false).map((e) => e.name)).toEqual([
      "a",
      "b",
    ]);
    expect(parseDirEntries(out, 2, true).map((e) => e.name)).toEqual([
      ".h1",
      ".h2",
    ]);
  });

  it("makes an unparseable size 0 and skips a record with fewer than two tabs", () => {
    expect(parseDirEntries(rec("f\tNaN\tz", "f\tonly"), 10, true)).toEqual([
      { name: "z", kind: "file", size: 0 },
    ]);
  });

  it("keeps a name with a newline whole", () => {
    expect(parseDirEntries(rec("f\t1\ta\nb"), 10, true)[0]!.name).toBe("a\nb");
  });
});

describe("finalize (sort, limit, truncation)", () => {
  const h = (path: string, line: number, isMatch = true): SearchHit => ({
    path,
    line,
    text: "",
    isMatch,
  });

  it("sorts component by component, bytewise: a/b.txt before a.txt", () => {
    const out =
      finalize([h("/r/a.txt", 1), h("/r/a/b.txt", 1), h("/r/a/b.txt", 0)], 10)
        .hits;
    expect(out.map((x) => `${x.path}:${x.line}`)).toEqual([
      "/r/a/b.txt:0",
      "/r/a/b.txt:1",
      "/r/a.txt:1",
    ]);
  });

  it("compares in UTF-8 byte order, not UTF-16 order", () => {
    // U+FF5E (3 bytes in UTF-8, one UTF-16 unit) sorts before U+1F600 (4 bytes, a surrogate pair).
    const out = finalize([h("/\u{1F600}", 1), h("/～", 1)], 10).hits;
    expect(out.map((x) => x.path)).toEqual(["/～", "/\u{1F600}"]);
  });

  it("counts only matches toward the limit and ends on a match", () => {
    const hits = [
      h("a", 1, false),
      h("a", 2),
      h("a", 3, false),
      h("a", 4),
      h("a", 5, false),
      h("a", 6),
    ];
    const r = finalize(hits, 2);
    expect(r.truncated).toBe(true);
    expect(r.hits.map((x) => x.line)).toEqual([1, 2, 3, 4]);
    expect(finalize(hits, 3).truncated).toBe(false);
    expect(finalize(hits, 3).hits).toHaveLength(6);
  });
});

describe("isUnderGitDir", () => {
  it("counts only components below the root", () => {
    expect(isUnderGitDir("/r", "/r/.git/HEAD")).toBe(true);
    expect(isUnderGitDir("/r", "/r/a/.git")).toBe(true);
    expect(isUnderGitDir("/r", "/r/a/b")).toBe(false);
    expect(isUnderGitDir("/r/.git", "/r/.git/HEAD")).toBe(false);
    expect(isUnderGitDir("/r/.git", "/r/.git/refs/.git/x")).toBe(true);
    expect(isUnderGitDir("/r", "/elsewhere/.git/x")).toBe(false);
  });
});

describe("clipping", () => {
  it("leaves a line of at most 500 characters alone and cuts a context line at 500", () => {
    expect(clip("x".repeat(500))).toBe("x".repeat(500));
    expect(clip("x".repeat(501))).toBe("x".repeat(500) + "… [truncated]");
  });

  it("counts characters, not UTF-16 units", () => {
    const line = "\u{1F600}".repeat(400); // 800 UTF-16 units, 400 characters
    expect(clip(line)).toBe(line);
    expect(clip("\u{1F600}".repeat(501))).toBe(
      "\u{1F600}".repeat(500) + "… [truncated]",
    );
  });

  it("windows a long match line around the column rg reported", () => {
    const line = "a".repeat(1000) + "NEEDLE" + "b".repeat(1000);
    const [hit] = parseRecords(`f\x001:1001:${line}\n`, true);
    // total = 2006; start = 1000 - 250 = 750; end = 1250
    expect(hit!.text.startsWith("… ")).toBe(true);
    expect(hit!.text).toContain("NEEDLE");
    expect(hit!.text.endsWith(" … [truncated; columns 751-1250 of 2006]")).toBe(
      true,
    );
  });

  it("clips the head when the match starts in the first 250 characters", () => {
    const line = "NEEDLE" + "b".repeat(1000);
    const [hit] = parseRecords(`f\x001:1:${line}\n`, true);
    expect(hit!.text).toBe(line.slice(0, 500) + "… [truncated]");
  });

  it("anchors the window to the end of a line when the match is near it", () => {
    const line = "a".repeat(1000) + "NEEDLE";
    const [hit] = parseRecords(`f\x001:1001:${line}\n`, true);
    expect(hit!.text.endsWith("NEEDLE [truncated; columns 507-1006 of 1006]"))
      .toBe(true);
  });

  it("counts the column in bytes, the offset in characters, and ignores CRs", () => {
    // 600 two-byte characters, a CR, then the match at byte column 1202.
    const line = "é".repeat(600) + "\r" + "x".repeat(600);
    const [hit] = parseRecords(`f\x001:1202:${line}\n`, true);
    expect(hit!.text.endsWith("[truncated; columns 351-850 of 1200]")).toBe(
      true,
    );
  });

  it("falls back to the head with no column or one inside a character", () => {
    const line = "é".repeat(1000);
    expect(parseRecords(`f\x001:2:${line}\n`, true)[0]!.text).toBe(clip(line));
    expect(parseRecords(`f\x001:${line}\n`)[0]!.text).toBe(clip(line));
  });
});

describe("compileGlob (globset semantics)", () => {
  const m = (glob: string, path: string, ic = false) =>
    compileGlob(glob, ic).test(path);
  it("lets * and ? cross /", () => {
    expect(m("*.txt", "/r/sub/a.txt")).toBe(true);
    expect(m("*.txt", "/r/sub/a.txt.bak")).toBe(false);
    expect(m("/r/?/a", "/r/x/a")).toBe(true);
    expect(m("/r/?a", "/r//a")).toBe(true);
  });
  it("handles **, classes and alternation", () => {
    expect(m("**/a.txt", "a.txt")).toBe(true);
    expect(m("**/a.txt", "/x/y/a.txt")).toBe(true);
    expect(m("/r/**", "/r/x/y")).toBe(true);
    expect(m("/r/**/a", "/r/a")).toBe(true);
    expect(m("/r/**/a", "/r/x/y/a")).toBe(true);
    expect(m("*.{rs,toml}", "x.toml")).toBe(true);
    expect(m("*.{rs,toml}", "x.md")).toBe(false);
    expect(m("[a-c]x", "bx")).toBe(true);
    expect(m("[!a-c]x", "bx")).toBe(false);
    expect(m("[^a-c]x", "dx")).toBe(true);
    expect(m("a\\*b", "a*b")).toBe(true);
    expect(m("a\\*b", "axb")).toBe(false);
  });
  it("is case-insensitive on request and rejects malformed globs", () => {
    expect(m("*.TXT", "a.txt", true)).toBe(true);
    expect(m("*.TXT", "a.txt")).toBe(false);
    for (const bad of ["[a", "{a,b", "a}", "a**b"]) {
      expect(() => compileGlob(bad, false), bad).toThrow(/bad glob/);
    }
  });
});

describe("jobs parsers", () => {
  it("parses the baseline's read answer", () => {
    const r = parseRead("running - 0 5\naGVsbG8=\n");
    expect(r).toMatchObject({
      state: "running",
      code: null,
      offset: 0,
      end: 5,
    });
    expect(new TextDecoder().decode(r.data)).toBe("hello");
    expect(parseRead("ended 3 10 20\n\n")).toMatchObject({
      state: "ended",
      code: 3,
      offset: 10,
      end: 20,
    });
    expect(parseRead("gone\n")).toMatchObject({ state: "gone", end: 0 });
    expect(() => parseRead("bogus")).toThrow(/unreadable/);
    expect(() => parseRead("running - 0 5\n***\n")).toThrow(/unreadable/);
  });

  it("parses a list line, label with spaces whole", () => {
    expect(
      parseListLine(
        "0123ab ended 0 7 background 120 1700000000 1700000009 sleep 1 && echo hi",
      ),
    ).toEqual({
      id: "0123ab",
      state: "ended",
      code: 0,
      kept: true,
      num: 7,
      kind: "background",
      seen: 120,
      started: 1700000000,
      ended: 1700000009,
      label: "sleep 1 && echo hi",
    });
    expect(parseListLine("abc running - - - 0 5 0 ")).toMatchObject({
      kept: false,
      num: null,
      kind: null,
      label: "",
    });
    expect(parseListLine("ABC running - - - 0 5 0 x")).toBeNull();
    expect(parseListLine("")).toBeNull();
  });
});
