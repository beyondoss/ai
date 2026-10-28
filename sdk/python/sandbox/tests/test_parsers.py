"""Parsers and command builders, with the cases of the Rust reference's unit tests."""

from __future__ import annotations

import pytest

from beyond_sandbox import _fs
from beyond_sandbox._fs import (
    Query,
    classify_search_failure,
    clip,
    compile_glob,
    crlf_tolerant_ere,
    finalize,
    first_line,
    is_under_git_dir,
    parse_colon_records,
    parse_dir_entries,
    parse_records,
    parse_stat,
    posix_grep_args,
    rg_args,
    split_colon_record,
)
from beyond_sandbox.errors import classify_stderr
from beyond_sandbox.models import ExecResult, Hit


def q(pattern: str = "x", root: str = ".", **kw: object) -> Query:
    return Query(pattern, root, **kw)  # type: ignore[arg-type]


# ---- parse_records ---------------------------------------------------------------------------------


def test_parses_a_match_record() -> None:
    hits = parse_records("src/a.rs\x0012:let x = 1;\n")
    assert hits == [Hit("src/a.rs", 12, "let x = 1;", True)]


def test_distinguishes_context_lines_from_matches() -> None:
    hits = parse_records("a.rs\x001-before\na.rs\x002:hit\na.rs\x003-after\n")
    assert [h.is_match for h in hits] == [False, True, False]


def test_skips_the_group_separator() -> None:
    hits = parse_records("a.rs\x001:one\n--\nb.rs\x009:two\n")
    assert [(h.path, h.line) for h in hits] == [("a.rs", 1), ("b.rs", 9)]


def test_skips_the_crlf_group_separator_rg_writes_under_crlf() -> None:
    hits = parse_records("a.rs\x001:one\r\n--\r\nb.rs\x009:two\r\n")
    assert [(h.path, h.line) for h in hits] == [("a.rs", 1), ("b.rs", 9)]


def test_a_file_named_like_a_separator_is_a_file() -> None:
    hits = parse_records("--\x001:x\n--\r\x002:y\n--\n--\x003:--\n")
    assert [(h.path, h.line) for h in hits] == [("--", 1), ("--\r", 2), ("--", 3)]
    assert hits[2].text == "--"


def test_a_trailing_separator_without_a_newline_ends_the_records() -> None:
    assert len(parse_records("a\x001:x\n--")) == 1
    assert len(parse_records("a\x001:x\n--\r")) == 1


def test_a_newline_inside_a_filename_does_not_shred_the_record() -> None:
    hits = parse_records("we\nird.rs\x009:two\n")
    assert hits == [Hit("we\nird.rs", 9, "two", True)]


def test_a_colon_in_the_path_is_not_the_separator() -> None:
    assert parse_records("odd:name.rs\x003:body\n") == [Hit("odd:name.rs", 3, "body", True)]


def test_carriage_returns_are_removed_from_text() -> None:
    assert parse_records("a.rs\x001:crlf line\r\n")[0].text == "crlf line"
    assert parse_records("a.rs\x001:a\rb\r\n")[0].text == "ab"


def test_a_record_that_does_not_fit_is_dropped() -> None:
    assert parse_records("a.rs\x00x:nope\nb.rs\x002:yes\n") == [Hit("b.rs", 2, "yes", True)]


# ---- busybox records -------------------------------------------------------------------------------


def test_colon_records_split_at_the_first_sep_digits_sep() -> None:
    assert split_colon_record("src/a.rs:12:let x = 1;") == ("src/a.rs", 12, True, "let x = 1;")
    assert split_colon_record("src/a.rs-12-ctx") == ("src/a.rs", 12, False, "ctx")
    assert split_colon_record("nothing here") is None


def test_colon_records_are_wrong_for_paths_with_sep_digits_sep() -> None:
    # Documented cost of the busybox rung.
    assert split_colon_record("a:1:b.txt:2:text") == ("a", 1, True, "b.txt:2:text")


def test_colon_records_skip_separators_and_apply_the_glob() -> None:
    out = "/r/a.rs:1:one\n--\n/r/b.txt:2:two\r\n"
    hits = parse_colon_records(out, q(glob="*.rs"))
    assert [h.path for h in hits] == ["/r/a.rs"]
    hits = parse_colon_records(out, q(glob="*.rs", glob_exclude=True))
    assert [h.path for h in hits] == ["/r/b.txt"]
    assert hits[0].text == "two"


# ---- stat ------------------------------------------------------------------------------------------


def test_stat_parses_the_one_line() -> None:
    st = parse_stat("w\tf\t12\t1700000000\n")
    assert st is not None
    assert (st.kind, st.size, st.mtime, st.writable) == ("file", 12, 1700000000, True)


def test_stat_keeps_a_fractional_mtime() -> None:
    st = parse_stat("-\td\t4096\t1700000000.987654321\n")
    assert st is not None and (st.kind, st.mtime, st.writable) == ("dir", 1700000000.9876542, False)


def test_stat_of_other_and_garbage() -> None:
    st = parse_stat("w\to\t0\t5\n")
    assert st is not None and st.kind == "other"
    assert parse_stat("w\tf\tbig\t5\n") is None
    assert parse_stat("w\tf\n") is None
    assert parse_stat("") is None


def test_stat_mtime_overflow_is_not_a_crash() -> None:
    st = parse_stat("w\tf\t1\t99999999999999999999999\n")
    assert st is not None and st.mtime == float(99999999999999999999999)


# ---- listing ---------------------------------------------------------------------------------------


def test_listing_records() -> None:
    out = "f\t5\ta.txt\x00d\t4096\tsub\x00N\t0\tbroken\x00?\t0\tx\x00p\t0\tpipe\x00"
    names = [(e.name, e.kind, e.size) for e in parse_dir_entries(out, 100, True)]
    assert names == [("a.txt", "file", 5), ("sub", "dir", 4096), ("pipe", "other", 0)]


def test_listing_drops_hidden_names_before_counting() -> None:
    out = "f\t1\t.h1\x00f\t1\t.h2\x00f\t1\ta\x00f\t1\tb\x00f\t1\tc\x00"
    assert [e.name for e in parse_dir_entries(out, 2, False)] == ["a", "b"]
    assert [e.name for e in parse_dir_entries(out, 2, True)] == [".h1", ".h2"]


def test_a_name_may_contain_tabs_and_newlines() -> None:
    entries = parse_dir_entries("f\t3\ta\tb\nc\x00", 10, True)
    assert entries[0].name == "a\tb\nc"


def test_an_unparseable_size_is_zero() -> None:
    assert parse_dir_entries("f\tbig\ta\x00", 10, True)[0].size == 0


# ---- search ----------------------------------------------------------------------------------------


def test_a_newline_in_a_path_does_not_hide_the_os_error() -> None:
    root = "/work/no\nwhere"
    for missing in [
        "rg: /work/no\nwhere: No such file or directory (os error 2)\n",
        "rg: /work/no\nwhere: IO error for operation on /work/no\nwhere: No such file or directory (os error 2)\n",
        "grep: /work/no\nwhere: No such file or directory\n",
    ]:
        assert classify_search_failure(missing, root).kind == "missing_root", missing
    skipped = classify_search_failure(
        "rg: /p/a\nb: Permission denied (os error 13)\nrg: /p/c: Permission denied (os error 13)\n",
        "/p",
    )
    assert skipped.kind == "skipped" and skipped.note.startswith("2 path(s)")
    assert (
        first_line("cat: a\nb: No such file or directory\n")
        == "cat: a\nb: No such file or directory"
    )
    assert first_line("one\ntwo\n") == "one"
    assert first_line("  \n") == "command failed"


def test_stat_keeps_the_fraction_of_the_mtime() -> None:
    assert _fs.parse_stat("w\tf\t3\t1793127211.536029123\n").mtime == 1793127211.536029123  # type: ignore[union-attr]
    assert _fs.parse_stat("w\tf\t3\t981173106.000000000\n").mtime == 981173106  # type: ignore[union-attr]
    assert _fs.parse_stat("w\tf\t3\t981173106\n").mtime == 981173106  # type: ignore[union-attr]


def test_a_root_below_a_file_is_a_root_error() -> None:
    f = classify_search_failure(
        "rg: /w/file/sub: IO error for operation on /w/file/sub: Not a directory (os error 20)\n",
        "/w/file/sub",
    )
    assert f.kind == "missing_root" and "Not a directory" in f.note
    assert (
        classify_search_failure("grep: /w/file/sub: Not a directory", "/w/file").kind == "skipped"
    )


def test_search_exit_two_is_told_apart_by_stderr() -> None:
    root = "/work/proj"
    for bad in [
        "regex parse error:\n    (\n    ^\nerror: unclosed group",
        "grep: Unmatched ( or \\(",
        "grep: bad regex '(': Unmatched ( or \\(",
        "",
    ]:
        assert classify_search_failure(bad, root).kind == "bad_regex", bad
    for missing in [
        "rg: /work/proj: No such file or directory (os error 2)",
        "rg: /work/proj: IO error for operation on /work/proj: No such file or directory (os error 2)",
        "grep: /work/proj: No such file or directory",
    ]:
        assert classify_search_failure(missing, root).kind == "missing_root", missing
    skipped = classify_search_failure(
        "rg: /work/proj/a: Permission denied (os error 13)\n"
        "rg: /work/proj/b: Permission denied (os error 13)\n",
        root,
    )
    assert skipped.kind == "skipped" and skipped.note.startswith("2 path(s)")
    assert (
        classify_search_failure("grep: /work/proj/x: No such file or directory", root).kind
        == "skipped"
    )


def test_end_anchors_tolerate_a_crlf_line_ending() -> None:
    assert crlf_tolerant_ere("end$") == "end\r?$"
    assert crlf_tolerant_ere("(one$|two$)") == "(one\r?$|two\r?$)"
    assert crlf_tolerant_ere("a$|b") == "a\r?$|b"
    # literals stay literal: escaped, or inside a bracket expression
    assert crlf_tolerant_ere("cost\\$") == "cost\\$"
    assert crlf_tolerant_ere("[$]") == "[$]"
    assert crlf_tolerant_ere("[]$]x$") == "[]$]x\r?$"
    assert crlf_tolerant_ere("[[:space:]$]$") == "[[:space:]$]\r?$"
    assert crlf_tolerant_ere("plain") == "plain"


def test_rg_args_put_the_pattern_behind_dash_e_and_the_path_behind_dash_dash() -> None:
    args = rg_args(q("-v", "-weird-dir"), False)
    assert args[args.index("-e") + 1] == "-v"
    assert args[args.index("--") + 1] == "-weird-dir"
    assert args[:8] == ["--null", "--with-filename", "--line-number", "--column", "--no-heading",
                        "--color=never", "--hidden", "--crlf"]  # fmt: skip


def test_rg_only_drops_require_git_outside_a_repo() -> None:
    assert "--no-require-git" in rg_args(q(), False)
    assert "--no-require-git" not in rg_args(q(), True)


def test_rg_gets_only_a_restricting_glob_and_git_outranks_it() -> None:
    # `--glob '!G'` would also prune a directory whose name matches; the hits are filtered instead
    args = rg_args(q(glob="*.test.rs", glob_exclude=True), False)
    globs = [args[i + 1] for i, a in enumerate(args) if a == "--glob"]
    assert globs == ["!.git"]
    args = rg_args(q(glob="*", glob_exclude=False), False)
    assert [args[i + 1] for i, a in enumerate(args) if a == "--glob"] == ["*", "!.git"]


def test_context_is_clamped_to_one_hundred_and_omitted_at_zero() -> None:
    args = rg_args(q(before=500, after=3), False)
    assert args[args.index("-B") + 1] == "100" and args[args.index("-A") + 1] == "3"
    assert "-B" not in rg_args(q(), False)


def test_gnu_grep_excludes_git_unless_the_root_is_git_itself() -> None:
    assert "--exclude-dir=.git" in posix_grep_args(q(root="/repo"), True)
    assert not any(
        a.startswith("--exclude-dir") for a in posix_grep_args(q(root="/repo/.git"), True)
    )
    assert not any(a.startswith("--exclude-dir") for a in posix_grep_args(q(root="/repo"), False))


def test_posix_grep_glob_flags_and_busybox_has_none() -> None:
    assert "--exclude=*.t" in posix_grep_args(q(glob="*.t", glob_exclude=True), True)
    assert "--include=*.rs" in posix_grep_args(q(glob="*.rs"), True)
    busybox = posix_grep_args(q(glob="*.rs"), False)
    assert busybox[:4] == ["-r", "-n", "-H", "-E"]
    assert not any(a.startswith("--include") or a in ("-Z", "-I") for a in busybox)


def test_literal_mode_uses_fixed_strings_and_never_pairs_f_with_e() -> None:
    args = posix_grep_args(q("a.b(c)", literal=True), True)
    assert "-F" in args and "-E" not in args
    assert "end$" in posix_grep_args(q("end$", literal=True), True)
    assert "end\r?$" in posix_grep_args(q("end$"), True)


# ---- glob (globset semantics: `*` crosses `/`) -----------------------------------------------------


@pytest.mark.parametrize(
    ("glob", "path", "ci", "want"),
    [
        ("*.rs", "/r/src/a.rs", False, True),
        ("*.rs", "/r/src/a.RS", False, False),
        ("*.rs", "/r/src/a.RS", True, True),
        ("a?c", "abc", False, True),
        ("a?c", "a/c", False, True),
        ("[ab]x", "bx", False, True),
        ("[!ab]x", "bx", False, False),
        ("{foo,bar}.txt", "bar.txt", False, True),
        ("{foo,bar}.txt", "baz.txt", False, False),
        ("**/c.txt", "a/b/c.txt", False, True),
        ("**/c.txt", "c.txt", False, True),
        ("a/**", "a/b/c", False, True),
        ("a/**/c", "a/c", False, True),
        ("a/**/c", "a/b/b/c", False, True),
        ("\\*.rs", "*.rs", False, True),
        ("\\*.rs", "x.rs", False, False),
        ("a.b", "axb", False, False),
    ],
)
def test_globs(glob: str, path: str, ci: bool, want: bool) -> None:
    rx = compile_glob(glob, ci)
    assert rx is not None
    assert (rx.fullmatch(path) is not None) is want


def test_a_glob_that_does_not_parse_filters_nothing() -> None:
    assert compile_glob("[abc", False) is None
    assert compile_glob("{a,{b}}", False) is None


# ---- post-processing -------------------------------------------------------------------------------


def test_is_under_git_dir_counts_only_components_below_the_root() -> None:
    assert is_under_git_dir("/repo", "/repo/.git/config")
    assert is_under_git_dir("/repo", "/repo/sub/.git")
    assert not is_under_git_dir("/repo/.git", "/repo/.git/config")
    assert not is_under_git_dir("/repo", "/repo/src/a.rs")
    assert not is_under_git_dir("/repo", "/other/.git/x")
    assert not is_under_git_dir("/repo", "/repo/.gitignore")


def hit(path: str, line: int, is_match: bool = True) -> Hit:
    return Hit(path, line, "t", is_match)


def test_finalize_sorts_by_path_components_then_line() -> None:
    hits, truncated = finalize([hit("a.txt", 1), hit("a/b.txt", 2), hit("a/b.txt", 1)], 100)
    assert [(h.path, h.line) for h in hits] == [("a/b.txt", 1), ("a/b.txt", 2), ("a.txt", 1)]
    assert not truncated


def test_finalize_counts_only_matches_toward_the_limit() -> None:
    hits = [hit("a", 1, False), hit("a", 2), hit("a", 3, False), hit("a", 4), hit("a", 5, False)]
    kept, truncated = finalize(hits, 1)
    assert truncated
    assert [(h.line, h.is_match) for h in kept] == [(1, False), (2, True)]


def test_finalize_drops_dangling_context() -> None:
    hits = [hit("a", 1), hit("a", 2, False), hit("a", 3)]
    kept, truncated = finalize(hits, 1)
    assert truncated and [h.line for h in kept] == [1]


def test_clip_counts_characters_not_bytes() -> None:
    assert clip("é" * 500) == "é" * 500
    assert clip("é" * 501) == "é" * 500 + "… [truncated]"


def _rec(col: int, line: str) -> str:
    return f"f\x001:{col}:{line}\n"


def test_a_match_deep_in_a_long_line_is_windowed_around_rgs_column() -> None:
    line = "a" * 2000 + "NEEDLE" + "b" * 2000
    [hit] = parse_records(_rec(2001, line), True)
    assert hit.text.startswith("… ") and "NEEDLE" in hit.text
    assert hit.text.endswith(" … [truncated; columns 1751-2250 of 4006]")


def test_a_match_near_the_end_shows_the_last_window() -> None:
    line = "a" * 2000 + "NEEDLE"
    [hit] = parse_records(_rec(2001, line), True)
    assert hit.text == "… " + line[-500:] + " [truncated; columns 1507-2006 of 2006]"


def test_a_match_in_the_first_250_characters_keeps_the_head_clip() -> None:
    line = "NEEDLE" + "b" * 1000
    [hit] = parse_records(_rec(1, line), True)
    assert hit.text == line[:500] + "… [truncated]"


def test_the_column_is_bytes_the_offset_characters_and_crs_do_not_count() -> None:
    line = "é" * 600 + "\r" + "x" * 600
    [hit] = parse_records(_rec(1202, line), True)
    assert hit.text.endswith("[truncated; columns 351-850 of 1200]")


def test_no_column_or_one_inside_a_character_clips_the_head() -> None:
    line = "é" * 1000
    assert parse_records(_rec(2, line), True)[0].text == clip(line)
    assert parse_records(f"f\x001:{line}\n")[0].text == clip(line)


# ---- error classification --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stderr", "kind"),
    [
        ("cat: x: No such file or directory", "not_found"),
        ("mv: cannot move: Directory nonexistent", "not_found"),
        ("Is a directory (os error 21)", "is_directory"),
        ("mkdir: cannot create directory 'a/b': Not a directory", "not_a_directory"),
        ("sh: 1: cannot create /x: Permission denied", "permission_denied"),
        ("short write: 3 of 9 bytes arrived on stdin", "io"),
        ("command failed", "io"),
    ],
)
def test_stderr_classification(stderr: str, kind: str) -> None:
    assert classify_stderr(stderr).kind == kind


def test_the_classifier_reads_the_stderr_line_not_the_path() -> None:
    err = _fs.fail(
        "read /tmp/is a directory/x", ExecResult(1, b"", b"Permission denied (os error 13)\n")
    )
    assert err.kind == "permission_denied"
    assert str(err.detail) == "read /tmp/is a directory/x: Permission denied (os error 13)"


def test_rgs_binary_file_warning_is_not_the_start_of_a_record() -> None:
    w = ': WARNING: stopped searching binary file after match (found "\\0" byte around offset 3)'
    out = f"a\x001:1:x\nb\x001:1:y\nb{w}\nc\x001:1:z\n"
    assert [h.path for h in parse_records(out, True)] == ["a", "b", "c"]
    # Not the previous record's file: a path that merely contains the words.
    out = f"a\x001:1:x\nb{w}\nc\x001:1:z\n"
    assert [h.path for h in parse_records(out, True)] == ["a", f"b{w}\nc"]
