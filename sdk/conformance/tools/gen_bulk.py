#!/usr/bin/env python3
"""Regenerates sdk/conformance/scenarios/bulk.json.

The scenarios carry tars and digests that are tedious to keep by hand (hostile archives, a 50 MB
file's SHA-256); this is where they come from. Run from anywhere:

    python3 sdk/conformance/tools/gen_bulk.py            # rewrite bulk.json
    python3 sdk/conformance/tools/gen_bulk.py --check    # exit 1 if bulk.json is out of date
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "scenarios" / "bulk.json"
B64 = lambda b: base64.b64encode(b).decode()  # noqa: E731


# ---- trees ----------------------------------------------------------------------------------


def f(path, data=b"", mode="0644", mtime=None, size=None, pattern=None):
    e = {"path": path, "type": "file", "mode": mode}
    if pattern is not None:
        e["pattern_b64"] = B64(pattern)
        e["size"] = size
    else:
        e["data_b64"] = B64(data)
    if mtime is not None:
        e["mtime"] = mtime
    return e


def d(path, mode="0755", mtime=None):
    e = {"path": path, "type": "dir", "mode": mode}
    if mtime is not None:
        e["mtime"] = mtime
    return e


def ln(path, target):
    return {"path": path, "type": "symlink", "target": target}


def content(e):
    if "pattern_b64" in e:
        pat = base64.b64decode(e["pattern_b64"])
        n = e["size"]
        return (pat * (n // len(pat) + 1))[:n]
    return base64.b64decode(e["data_b64"])


def expect_entry(e, mtimes=True):
    if e["type"] == "file":
        c = content(e)
        r = {"path": e["path"], "type": "file", "mode": e["mode"], "size": len(c),
             "sha256": hashlib.sha256(c).hexdigest()}
        if mtimes and "mtime" in e:
            r["mtime"] = e["mtime"]
        return r
    if e["type"] == "dir":
        return {"path": e["path"], "type": "dir", "mode": e["mode"]}
    return {"path": e["path"], "type": "symlink", "target": e["target"]}


def tree(entries, **kw):
    """The listing a download of this tree gives: the explicit entries plus the directories
    implied by their paths (which the builder creates with the default mode)."""
    seen = {e["path"] for e in entries}
    out = [expect_entry(e, **kw) for e in entries]
    for e in entries:
        parts = e["path"].split("/")[:-1]
        for i in range(1, len(parts) + 1):
            p = "/".join(parts[:i])
            if p not in seen:
                seen.add(p)
                out.append({"path": p, "type": "dir", "mode": "0755"})
    return out


# ---- tars -----------------------------------------------------------------------------------


def raw_tar(members, end=True, corrupt=False):
    """Hand-built ustar: members are (name, typeflag, data, linkname, mode)."""
    out = b""
    for name, kind, data, link, mode in members:
        h = bytearray(512)
        n = name.encode()
        h[0:len(n)] = n
        h[100:108] = b"%07o\0" % mode
        h[108:116] = b"0000000\0"
        h[116:124] = b"0000000\0"
        h[124:136] = b"%011o\0" % len(data)
        h[136:148] = b"%011o\0" % 1500000000
        h[148:156] = b"        "
        h[156] = ord(kind)
        lk = link.encode()
        h[157:157 + len(lk)] = lk
        h[257:263] = b"ustar\0"
        h[263:265] = b"00"
        h[148:154] = b"%06o" % sum(h)
        h[154] = 0
        h[155] = 0x20
        if corrupt:
            h[0] ^= 1
        out += bytes(h) + data + b"\0" * (-len(data) % 512)
    if end:
        out += b"\0" * 1024
    return out


def py_tar(fmt, members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as t:
        for name, kind, data, link, mode in members:
            ti = tarfile.TarInfo(name)
            ti.mode = mode
            ti.mtime = 1500000000
            if kind == "5":
                ti.type = tarfile.DIRTYPE
                t.addfile(ti)
            elif kind == "2":
                ti.type = tarfile.SYMTYPE
                ti.linkname = link
                t.addfile(ti)
            else:
                ti.size = len(data)
                t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def file_m(name, data=b"x", mode=0o644):
    return (name, "0", data, "", mode)


# ---- scenarios ------------------------------------------------------------------------------

SC = []


def sc(id_, description, steps, setup=None, requires=None, **extra):
    s = {"id": "bulk." + id_, "description": description}
    if requires:
        s["requires"] = requires
    s.update(extra)
    if setup:
        s["setup"] = setup
    s["steps"] = steps
    SC.append(s)


def up(sid, path, entries, expect=None, **kw):
    s = {"id": sid, "op": "upload", "path": path, "entries": entries}
    s.update(kw)
    s["expect"] = expect if expect is not None else {"ok": {}}
    return s


def down(sid, path, entries=None, expect=None, **kw):
    s = {"id": sid, "op": "download", "path": path}
    s.update(kw)
    if expect is None:
        expect = {"ok": {"entries": {"$unordered": entries}}}
    s["expect"] = expect
    return s


def extract(sid, tar, entries=None, expect=None, **kw):
    s = {"id": sid, "op": "extract", "tar_b64": B64(tar)}
    s.update(kw)
    if expect is None:
        expect = {"ok": {"entries": {"$unordered": entries}}}
    s["expect"] = expect
    return s


def unsafe(why):
    return {"error": {"kind": "protocol", "detail": {"$re": "unsafe archive member.*" + why}}}


R = "${ROOT}"

# -- round trips

odd = [
    f("a b.txt", b"spaces"),
    f("new\nline", b"newline"),
    f("-rf", b"dash"),
    f("-", b"just a dash"),
    f("quote'd\"name", b"quotes"),
    f("semi;colon&amp|pipe", b"shell"),
    f("$(touch pwned)", b"subst"),
    f("`touch pwned2`", b"backtick"),
    f("glob*?[x]", b"glob"),
    f(".hidden", b"hidden"),
    f("tab\tname", b"tab"),
    f("ünï/cödé.txt", "unicode".encode()),
    f("emoji-\U0001F600", b"emoji"),
    f("n" * 200, b"long name"),
    f("p" * 60 + "/" + "q" * 60 + "/" + "r" * 60 + "/file", b"long path"),
    f("e/" + "x" * 120 + "/" + "y" * 120, b"longer than 255"),
    d("empty dir"),
    f("dir with space/inner file", b"in"),
]
sc("round-trip.odd-names",
   "Names with spaces, newlines, quotes, shell syntax, a leading dash, unicode and lengths beyond "
   "100 and 255 bytes survive upload and download, and no name is interpreted by a shell.",
   [up("up", R + "/dst", odd),
    down("down", R + "/dst", tree(odd)),
    {"id": "no-side-effects", "op": "stat", "path": R + "/dst/pwned", "expect": {"ok": {"exists": False}}},
    {"id": "no-side-effects-2", "op": "stat", "path": R + "/pwned", "expect": {"ok": {"exists": False}}}])

shapes = [
    d("empty"),
    d("private", "0700"),
    f("private/secret", b"s", "0600"),
    f("run.sh", b"#!/bin/sh\necho hi\n", "0755", mtime=1500000000),
    f("ro.txt", b"read only", "0444", mtime=1400000000),
    f("plain-exec", b"x", "0755"),
    d("locked", "0555"),
    f("locked/inside", b"inside", "0444"),
    ln("rel-file", "run.sh"),
    ln("rel-dir", "private"),
    ln("dangling", "no/such/thing"),
    ln("abs", "/etc/hostname"),
    ln("up", "../.."),
    ln("odd link\nname", "-target with spaces"),
    ln("long-link", "t" * 130),
]
sc("round-trip.modes-links-and-empty-dirs",
   "Empty directories, permission bits (including a read-only directory that still holds files), "
   "mtimes, and symbolic links of every shape (relative, to a directory, dangling, absolute, upward, "
   "very long) come back exactly; links are stored by their target text and never followed.",
   [up("up", R + "/dst", shapes),
    down("down", R + "/dst", tree(shapes)),
    {"id": "remote-mode", "op": "exec", "command": "sh",
     "args": ["-c", 'stat -c %a "$1/run.sh" "$1/private" "$1/locked" "$1/ro.txt"', "sh", R + "/dst"],
     "expect": {"ok": {"exit_code": 0, "stdout_utf8": "755\n700\n555\n444\n"}}},
    {"id": "remote-link", "op": "exec", "command": "readlink", "args": [R + "/dst/rel-file"],
     "expect": {"ok": {"stdout_utf8": "run.sh\n"}}},
    # A read-only directory cannot be removed by the harness's cleanup.
    {"id": "unlock", "op": "exec", "command": "chmod", "args": ["-R", "u+rwx", R + "/dst"],
     "expect": {"ok": {"exit_code": 0}}}])

PAT = hashlib.sha256(b"bulk").digest() + b"\x00\xff\n" + bytes(range(7))
BIG = 50 * 1024 * 1024 + 17
big = [f("big.bin", pattern=PAT, size=BIG, mode="0640", mtime=1600000000), f("small.txt", b"after the big one")]
sc("round-trip.large-file",
   "A 50 MiB file (and a small one) goes up in several chunks and comes back byte for byte: the "
   "digest is the file's, and no stage file is left beside the destination.",
   [up("up", R + "/dst", big, expect={"ok": {"files": 2, "bytes": BIG + 17}}),
    down("down", R + "/dst", tree(big), expect={"ok": {"files": 2, "bytes": BIG + 17, "entries": {"$unordered": tree(big)}}}),
    {"id": "no-stage", "op": "list", "path": R, "include_hidden": True,
     "expect": {"ok": {"entries": {"$unordered": [{"name": "dst"}]}}}}])

small = [f(f"f{i}.bin", pattern=PAT, size=3000 + i * 997) for i in range(12)] + [d("sub"), f("sub/z", b"z" * 5000)]
sc("round-trip.staged-in-small-chunks",
   "With a small chunk size the upload goes through the stage (several appends, then one "
   "extraction) and ends the same as a one-request upload; the stage is gone.",
   [up("up", R + "/dst", small, chunk_bytes=4096),
    down("down", R + "/dst", tree(small)),
    {"id": "no-stage", "op": "list", "path": R, "include_hidden": True,
     "expect": {"ok": {"entries": {"$unordered": [{"name": "dst"}]}}}}])

sc("round-trip.exact-chunk-multiple",
   "An archive that is exactly one chunk long is sent staged with an empty tail that is not sent, "
   "and extracts.",
   # header 512 + 3584 of data = 4096 per file; two entries + end 1024 -> pick one file: 512 + 2560 pad
   # Simply cover several nearby sizes; the ends are what matters.
   [up(f"up{n}", f"{R}/dst{n}", [f("a", b"a" * n)], chunk_bytes=1024)
    for n in (1, 511, 512, 513, 1024)]
   + [down(f"down{n}", f"{R}/dst{n}", [{"path": "a", "type": "file", "size": n}]) for n in (1, 511, 512, 513, 1024)])

sc("upload.sources-list",
   "A list of sources uploads each by its base name (a file, a directory with its tree) and only "
   "those.",
   [up("up", R + "/dst",
       [f("a.txt", b"a"), f("skipped.txt", b"no"), d("d"), f("d/inner", b"in"), f("other/x", b"x")],
       sources=["a.txt", "d"]),
    down("down", R + "/dst", [
        {"path": "a.txt", "type": "file", "size": 1},
        {"path": "d", "type": "dir"},
        {"path": "d/inner", "type": "file", "size": 2},
    ])])

sc("upload.merges-into-existing",
   "Uploading into a directory that exists keeps what is there, replaces files of the same name, "
   "and does not change the directory's own mode.",
   [up("up", R + "/dst", [f("new.txt", b"new"), f("clash.txt", b"from upload")]),
    down("down", R + "/dst", [
        {"path": "keep.txt", "type": "file", "size": 4},
        {"path": "new.txt", "type": "file", "size": 3},
        {"path": "clash.txt", "type": "file", "size": 11},
    ]),
    {"id": "mode", "op": "exec", "command": "stat", "args": ["-c", "%a", R + "/dst"],
     "expect": {"ok": {"stdout_utf8": "700\n"}}}],
   setup=[{"id": "mk", "op": "mkdir", "path": R + "/dst"},
          {"id": "w1", "op": "write", "path": R + "/dst/keep.txt", "data_b64": B64(b"keep")},
          {"id": "w2", "op": "write", "path": R + "/dst/clash.txt", "data_b64": B64(b"old")},
          {"id": "chmod", "op": "exec", "command": "chmod", "args": ["700", R + "/dst"]}])

sc("upload.creates-missing-parents",
   "The destination may be several levels below an existing directory.",
   [up("up", R + "/a/b/c/dst", [f("x", b"x")]),
    down("down", R + "/a/b/c/dst", [{"path": "x", "type": "file", "size": 1}])])

sc("upload.empty-tree",
   "An empty directory uploads as an empty destination.",
   [up("up", R + "/dst", [], expect={"ok": {"files": 0, "dirs": 0, "links": 0, "bytes": 0}}),
    {"id": "is-dir", "op": "stat", "path": R + "/dst", "expect": {"ok": {"exists": True, "kind": "dir"}}},
    down("down", R + "/dst", [])])

sc("upload.trailing-slash",
   "A trailing slash on the destination names the same directory.",
   [up("up", R + "/dst/", [f("x", b"x")]),
    down("down", R + "/dst", [{"path": "x", "type": "file", "size": 1}])])

# -- refused requests

sc("upload.destination-is-a-file",
   "A destination that is a file is not_a_directory, and the file is not touched.",
   [up("up", R + "/f", [f("x", b"x")], expect={"error": {"kind": "not_a_directory"}}),
    {"id": "intact", "op": "read", "path": R + "/f", "expect": {"ok": {"data_utf8": "I am a file"}}}],
   setup=[{"id": "w", "op": "write", "path": R + "/f", "data_b64": B64(b"I am a file")}])

sc("upload.destination-parent-is-a-file",
   "A destination below a file is not_a_directory and creates nothing.",
   [up("up", R + "/f/dst", [f("x", b"x")], expect={"error": {"kind": "not_a_directory"}})],
   setup=[{"id": "w", "op": "write", "path": R + "/f", "data_b64": B64(b"file")}])

sc("upload.refuses-bad-requests",
   "A relative destination, the root, and a source that cannot be archived are invalid_request "
   "before anything is sent.",
   [up("relative", "relative/dir", [f("x", b"x")], expect={"error": {"kind": "invalid_request"}}),
    up("root", "/", [f("x", b"x")], expect={"error": {"kind": "invalid_request"}}),
    up("fifo", R + "/dst", [f("x", b"x"), {"path": "pipe", "type": "fifo"}],
       expect={"error": {"kind": "invalid_request", "detail": {"$re": "pipe"}}}),
    {"id": "nothing-made", "op": "stat", "path": R + "/dst", "expect": {"ok": {"exists": False}}}])

sc("download.missing-path",
   "Downloading a path that does not exist is not_found and writes nothing.",
   [down("down", R + "/nope", expect={"error": {"kind": "not_found"}}),
    down("relative", "nope", expect={"error": {"kind": "invalid_request"}})])

sc("download.single-file",
   "Downloading one file puts it in the local directory under its base name; a file whose name "
   "starts with a dash works.",
   [down("file", R + "/note.txt", [{"path": "note.txt", "type": "file", "size": 5, "sha256": hashlib.sha256(b"hello").hexdigest()}]),
    down("dash", R + "/-rf", [{"path": "-rf", "type": "file", "size": 1}])],
   setup=[{"id": "w1", "op": "write", "path": R + "/note.txt", "data_b64": B64(b"hello")},
          {"id": "w2", "op": "write", "path": R + "/-rf", "data_b64": B64(b"x")}])

sc("download.empty-dir",
   "An empty directory downloads as nothing.",
   [down("down", R + "/empty", [])],
   setup=[{"id": "mk", "op": "mkdir", "path": R + "/empty"}])

sc("download.links-are-not-followed",
   "Symbolic links in the sandbox come down as links (to a file, to a directory, dangling, absolute, "
   "pointing out), never as the content they point at, and hard links come down as files.",
   [down("down", R + "/src", [
       {"path": "a", "type": "file", "size": 1},
       {"path": "hard", "type": "file", "size": 1, "sha256": hashlib.sha256(b"a").hexdigest()},
       {"path": "abs", "type": "symlink", "target": "/etc/passwd"},
       {"path": "up", "type": "symlink", "target": ".."},
       {"path": "dangling", "type": "symlink", "target": "nowhere"},
       {"path": "dir", "type": "dir"},
       {"path": "dirlink", "type": "symlink", "target": "dir"},
   ])],
   setup=[{"id": "mk", "op": "mkdir", "path": R + "/src/dir"},
          {"id": "w", "op": "write", "path": R + "/src/a", "data_b64": B64(b"a")},
          {"id": "ln", "op": "exec", "command": "sh", "args": ["-c",
           'cd "$1" && ln a hard && ln -s /etc/passwd abs && ln -s .. up && ln -s nowhere dangling && ln -s dir dirlink',
           "sh", R + "/src"], "expect": {"ok": {"exit_code": 0}}}])

sc("download.unreadable-file",
   "A file the sandbox user may not read fails the download with permission_denied; the error is "
   "reported, not a partial success.",
   [down("down", R + "/src", expect={"error": {"kind": "permission_denied"}})],
   setup=[{"id": "mk", "op": "mkdir", "path": R + "/src"},
          {"id": "w", "op": "write", "path": R + "/src/secret", "data_b64": B64(b"s")},
          {"id": "chmod", "op": "exec", "command": "chmod", "args": ["000", R + "/src/secret"]}],
   requires={"not_root": True})

sc("download.into-existing-local-directory",
   "Extraction merges into a local directory that exists: other files stay, same-named files are "
   "replaced.",
   [down("down", R + "/src", [
       {"path": "keep.txt", "type": "file", "size": 4},
       {"path": "f.txt", "type": "file", "size": 6, "sha256": hashlib.sha256(b"remote").hexdigest()},
   ], prefill=[f("keep.txt", b"keep"), f("f.txt", b"local and longer")])],
   setup=[{"id": "mk", "op": "mkdir", "path": R + "/src"},
          {"id": "w", "op": "write", "path": R + "/src/f.txt", "data_b64": B64(b"remote")}])

sc("download.replaces-a-local-link",
   "A local symbolic link in the way of a downloaded file is replaced by the file; what it pointed "
   "at is not written (the driver fails the op if it was).",
   [down("down", R + "/src", [
       {"path": "f.txt", "type": "file", "size": 6},
   ], prefill=[ln("f.txt", "${OUTSIDE}/target")])],
   setup=[{"id": "mk", "op": "mkdir", "path": R + "/src"},
          {"id": "w", "op": "write", "path": R + "/src/f.txt", "data_b64": B64(b"remote")}])

# -- hostile archives (the local rules)

evil_ok = [("good.txt", "0", b"fine", "", 0o644)]
sc("extract.refuses-parent-components",
   "A member named with `..` is refused (protocol, `unsafe archive member`) and nothing is written "
   "outside the target, whatever its position in the name.",
   [extract("dotdot", raw_tar([file_m("../escape.txt", b"pwned")]), expect=unsafe(r"\.\.")),
    extract("deep", raw_tar([file_m("a/b/../../../escape.txt", b"pwned")]), expect=unsafe(r"\.\.")),
    extract("dir", raw_tar([("../escape-dir/", "5", b"", "", 0o755)]), expect=unsafe(r"\.\.")),
    extract("trailing", raw_tar([file_m("a/..", b"x")]), expect=unsafe(r"\.\.")),
    extract("after-good", raw_tar(evil_ok + [file_m("../escape.txt", b"pwned")]), expect=unsafe(r"\.\."))])

sc("extract.refuses-absolute-names",
   "An absolute member name is refused.",
   [extract("file", raw_tar([file_m("/tmp/bulk-abs-escape.txt", b"pwned")]), expect=unsafe("absolute")),
    extract("dir", raw_tar([("/tmp/bulk-abs-dir/", "5", b"", "", 0o755)]), expect=unsafe("absolute")),
    {"id": "not-written", "op": "stat", "path": "/tmp/bulk-abs-escape.txt", "expect": {"ok": {"exists": False}}}])

sc("extract.refuses-writing-through-a-link",
   "Nothing is written through a symbolic link: not one the archive made earlier, not one that was "
   "already in the target; a link in the way of a directory member is replaced by the directory.",
   [extract("by-the-archive",
            raw_tar([("up", "2", b"", "..", 0o777), file_m("up/escape.txt", b"pwned")]),
            expect=unsafe("not a directory")),
    extract("by-the-archive-dir",
            raw_tar([("up", "2", b"", "..", 0o777), ("up/sub/", "5", b"", "", 0o755)]),
            expect=unsafe("not a directory")),
    extract("already-there",
            raw_tar([file_m("link/f.txt", b"pwned")]),
            expect=unsafe("not a directory"), prefill=[ln("link", "${OUTSIDE}")]),
    extract("link-over-dir-member",
            raw_tar([("link/", "5", b"", "", 0o755), file_m("link/f.txt", b"safe")]),
            [{"path": "link", "type": "dir", "mode": "0755"},
             {"path": "link/f.txt", "type": "file", "size": 4}],
            prefill=[ln("link", "${OUTSIDE}")]),
    extract("file-over-link",
            raw_tar([file_m("link", b"now a file")]),
            [{"path": "link", "type": "file", "size": 10}],
            prefill=[ln("link", "${OUTSIDE}/target")])])

sc("extract.links-in-the-archive",
   "A symlink member is created as it says, whatever it points at, because it is never followed; a "
   "hard link is made only to a regular file the archive wrote, and never to anything else.",
   [extract("symlink-out",
            raw_tar([("abs", "2", b"", "/etc/passwd", 0o777), ("up", "2", b"", "../../..", 0o777)]),
            [{"path": "abs", "type": "symlink", "target": "/etc/passwd"},
             {"path": "up", "type": "symlink", "target": "../../.."}]),
    extract("hard-ok",
            raw_tar([file_m("a", b"data"), ("b", "1", b"", "a", 0o644), ("./c", "1", b"", "./a", 0o644)]),
            [{"path": "a", "type": "file", "size": 4}, {"path": "b", "type": "file", "size": 4},
             {"path": "c", "type": "file", "size": 4}]),
    extract("hard-to-outside",
            raw_tar([("h", "1", b"", "../../../etc/passwd", 0o644)]), expect=unsafe("link target")),
    extract("hard-to-absolute",
            raw_tar([("h", "1", b"", "/etc/passwd", 0o644)]), expect=unsafe("absolute")),
    extract("hard-to-unseen",
            raw_tar([("h", "1", b"", "never-written", 0o644)]), expect=unsafe("already wrote")),
    extract("hard-to-symlink",
            raw_tar([("s", "2", b"", "/etc/passwd", 0o777), ("h", "1", b"", "s", 0o644)]),
            expect=unsafe("already wrote"))])

sc("extract.refuses-what-it-cannot-make",
   "A device or FIFO member is io (unsupported member type); a header with a bad checksum, and an "
   "archive cut off before its end, are protocol errors.",
   [extract("device", raw_tar([("dev", "3", b"", "", 0o644)]),
            expect={"error": {"kind": "io", "detail": {"$re": "unsupported member type"}}}),
    extract("fifo", raw_tar([("pipe", "6", b"", "", 0o644)]),
            expect={"error": {"kind": "io", "detail": {"$re": "unsupported member type"}}}),
    extract("checksum", raw_tar([file_m("a")], corrupt=True), expect={"error": {"kind": "protocol"}}),
    extract("no-end", raw_tar([file_m("a")], end=False), expect={"error": {"kind": "protocol"}}),
    extract("cut-in-data", raw_tar([file_m("a", b"x" * 2000)], end=False)[:700],
            expect={"error": {"kind": "protocol"}}),
    extract("not-a-tar", b"this is not a tar archive at all" * 40, expect={"error": {"kind": "protocol"}})])

sc("extract.normalizes-names",
   "Empty and `.` components are dropped, the root member sets nothing, set-id bits are not carried, "
   "a directory's mode is applied after its content, and a file is created owner-writable first.",
   [extract("names",
            raw_tar([("./", "5", b"", "", 0o700), ("a/./b//c.txt", "0", b"x", "", 0o4755),
                     ("d/", "5", b"", "", 0o555), ("d/f", "0", b"f", "", 0o444)]),
            [{"path": "a", "type": "dir", "mode": "0755"}, {"path": "a/b", "type": "dir", "mode": "0755"},
             {"path": "a/b/c.txt", "type": "file", "mode": "0755", "size": 1},
             {"path": "d", "type": "dir", "mode": "0555", "mtime": 1500000000},
             {"path": "d/f", "type": "file", "mode": "0444", "size": 1, "mtime": 1500000000}])])

sc("extract.refuses-replacing-a-directory-with-a-file",
   "A file member over a local directory is is_directory and the directory stays.",
   [extract("clash", raw_tar([file_m("d", b"x")]),
            expect={"error": {"kind": "is_directory"}},
            prefill=[d("d"), f("d/keep", b"keep")])])

# Tars as other writers make them: the reader takes ustar, GNU long names and pax.
longname = "L" * 130 + "/" + "n" * 120
members = [("plain.txt", "0", b"plain", "", 0o644), (longname, "0", b"long", "", 0o600),
           ("dir/", "5", b"", "", 0o750), ("dir/ln", "2", b"", "t" * 150, 0o777),
           ("üñï.txt", "0", b"uni", "", 0o644)]
exp_members = [
    {"path": "plain.txt", "type": "file", "size": 5, "mode": "0644"},
    {"path": "L" * 130, "type": "dir"},
    {"path": longname, "type": "file", "size": 4, "mode": "0600"},
    {"path": "dir", "type": "dir", "mode": "0750"},
    {"path": "dir/ln", "type": "symlink", "target": "t" * 150},
    {"path": "üñï.txt", "type": "file", "size": 3},
]
sc("extract.reads-other-writers",
   "Archives written as GNU tar (long names with `L`/`K`), as pax (`path`, `linkpath`), and as ustar "
   "with a `prefix` field extract the same.",
   [extract("gnu", py_tar(tarfile.GNU_FORMAT, members), exp_members),
    extract("pax", py_tar(tarfile.PAX_FORMAT, members), exp_members),
    extract("ustar-prefix", py_tar(tarfile.USTAR_FORMAT, [
        ("plain.txt", "0", b"plain", "", 0o644),
        ("p" * 90 + "/" + "q" * 60 + "/file.txt", "0", b"deep", "", 0o644)]),
        [{"path": "plain.txt", "type": "file", "size": 5},
         {"path": "p" * 90, "type": "dir"}, {"path": "p" * 90 + "/" + "q" * 60, "type": "dir"},
         {"path": "p" * 90 + "/" + "q" * 60 + "/file.txt", "type": "file", "size": 4}])])

# -- the environments

sc("environments.read-only-destination",
   "Uploading to a read-only filesystem is an error that names it, creates nothing, and leaves "
   "what was there.",
   [up("new", "${RO}/dst", [f("x", b"x")], expect={"error": {"kind": "io", "detail": {"$re": "(?i)read-only file system"}}}),
    up("existing", "${RO}", [f("x", b"x")],
       expect={"error": {"kind": "io", "detail": {"$re": "(?i)read-only file system"}}}),
    {"id": "gone", "op": "stat", "path": "${RO}/dst", "expect": {"ok": {"exists": False}}},
    {"id": "no-x", "op": "stat", "path": "${RO}/x", "expect": {"ok": {"exists": False}}},
    {"id": "old-intact", "op": "read", "path": "${RO}/existing", "expect": {"ok": {"data_utf8": "old\n"}}},
    up("staged", "${RO}/dst", small, chunk_bytes=4096,
       expect={"error": {"kind": "io", "detail": {"$re": "(?i)read-only file system"}}}),
    {"id": "no-stage", "op": "list", "path": "${RO}", "include_hidden": True,
     "expect": {"ok": {"entries": {"$unordered": [{"name": "existing"}]}}}}],
   requires={"vars": ["RO"]})

sc("environments.locked-destination",
   "Uploading to a directory the sandbox user may not write is permission_denied, and the old file "
   "is intact.",
   [up("new", "${LOCKED}/dst", [f("x", b"x")], expect={"error": {"kind": "permission_denied"}}),
    up("existing", "${LOCKED}", [f("x", b"x")], expect={"error": {"kind": "permission_denied"}}),
    {"id": "old-intact", "op": "read", "path": "${LOCKED}/existing", "expect": {"ok": {"data_utf8": "old\n"}}},
    {"id": "no-x", "op": "stat", "path": "${LOCKED}/x", "expect": {"ok": {"exists": False}}}],
   requires={"vars": ["LOCKED"], "not_root": True})

BIG2 = 2 * 1024 * 1024
disk = [f("big.bin", pattern=PAT, size=BIG2)]
sc("environments.no-space",
   "A destination filesystem that fills up fails the upload with io (No space left on device); a "
   "destination this call created is removed, a stage is removed, and the filesystem works again.",
   [up("one-request", "${TINY}/dst", disk,
       expect={"error": {"kind": "io", "detail": {"$re": "(?i)no space left|wrote only"}}}),
    {"id": "dst-gone", "op": "stat", "path": "${TINY}/dst", "expect": {"ok": {"exists": False}}},
    up("staged", "${TINY}/dst2", disk, chunk_bytes=65536,
       expect={"error": {"kind": "io", "detail": {"$re": "(?i)no space left|wrote only"}}}),
    {"id": "dst2-gone", "op": "stat", "path": "${TINY}/dst2", "expect": {"ok": {"exists": False}}},
    {"id": "no-stage", "op": "list", "path": "${TINY}", "include_hidden": True,
     "expect": {"ok": {"entries": {"$not": {"$includes": [{"name": {"$re": "upload"}}]}}}}},
    up("existing-dir", "${TINY}/old", disk, chunk_bytes=65536,
       expect={"error": {"kind": "io", "detail": {"$re": "(?i)no space left|wrote only"}}}),
    {"id": "no-stage-2", "op": "list", "path": "${TINY}", "include_hidden": True,
     "expect": {"ok": {"entries": {"$not": {"$includes": [{"name": {"$re": "upload"}}]}}}}},
    {"id": "clean", "op": "exec", "command": "rm", "args": ["-rf", "${TINY}/old"], "expect": {"ok": {"exit_code": 0}}},
    up("works-again", "${TINY}/ok", [f("small", b"small")]),
    {"id": "cleanup", "op": "exec", "command": "rm", "args": ["-rf", "${TINY}/ok"], "expect": {"ok": {"exit_code": 0}}}],
   requires={"vars": ["TINY"]},
   setup=[{"id": "mk", "op": "mkdir", "path": "${TINY}/old"}])

# -- two clients on one sandbox

relay_tree = [f("a.txt", b"written by A"), d("sub"), f("sub/b.bin", pattern=PAT, size=300000, mode="0600"),
              ln("link", "a.txt"), d("empty"), f("näme with space", b"n")]
sc("relay.written-by-one-read-by-another",
   "A tree one client uploads is the tree another client downloads: each client reads the tar the "
   "other one wrote, and the sandbox's `tar` accepts it.",
   [up("up", R + "/dst", relay_tree, chunk_bytes=65536),
    dict(down("down", R + "/dst", tree(relay_tree)), via="B")])


def build():
    return {
        "suite": "bulk",
        "version": 1,
        "description": "Layer 2: bulk copy as tar over exec (FILES-OVER-EXEC.md 5.8): trees up and down, "
        "hostile archives, and the environments that refuse. Generated by tools/gen_bulk.py.",
        "scenarios": SC,
    }


def main():
    text = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        return 0 if OUT.exists() and OUT.read_text() == text else 1
    OUT.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
