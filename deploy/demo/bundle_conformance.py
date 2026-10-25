#!/usr/bin/env python3
"""Runs the bundle conformance kit (crates/agent, `bundle_conformance`) against the demo's bundle path.

    python3 deploy/demo/bundle_conformance.py       # needs cargo, tar, sha256sum; no AWS

The demo builds a bundle in `uc_bundle` (usecases.sh): a copy of the agent folder, archived by
`find | sort | tar --format=ustar --no-recursion`, named by `sha256sum`, uploaded to S3 and handed to the
replica as a presigned link. There is no deploy step that reads the source, and S3 cannot be used here, so
this runs the part that runs locally: the kit's operator hook (the table is in
`conformance::bundle`) is this file's `hook` subcommand.

    deploy <tenant> <source.tar>   extract the source into a folder (what a demo author has), run the
                                   uc_bundle pipeline on it, store it as <sha>.tar; exit 1 if tar
                                   cannot extract the source
    url <tenant> <name> <ttl>      http://127.0.0.1:<port>/<name>.tar from a server that serves any
                                   stored name (the load proof's bundle host serves by hash the same way)

The run passes when the kit's failures are exactly KNOWN_GAPS: what the demo's bundle path does not do. A
gap that closes fails the run too, so the list cannot go stale. The S3 link itself is https and signed
(`aws s3 presign`) but lasts 6 h (`--expires-in 21600`), not the grant's life plus 120 s.
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

KNOWN_GAPS = {
    # Nothing reads the source at build: tar's own refusals are the only ones, so a symlink, a hardlink,
    # an absolute path (tar strips the slash) and a 64 MiB+ file all become a bundle.
    "a bundle with an unsafe entry fails at deploy",
    # The local stand-in serves http; the real link (S3 presign) is https.
    "a bundle URL is https",
    # The local stand-in serves any stored name; S3's link is signed.
    "a bundle URL is signed",
    # The local stand-in never expires; S3's lasts 6 h, not the grant's life plus 120 s.
    "a bundle URL lives the grant's life plus the fetch window",
}


def hook(store, port, op, args):
    if op == "deploy":
        _tenant, source = args
        with tempfile.TemporaryDirectory(prefix="bundle-src-") as stage:
            if subprocess.run(["tar", "-xf", source, "-C", stage], capture_output=True).returncode != 0:
                sys.exit(1)
            out = os.path.join(stage + ".tar")
            # The uc_bundle pipeline, verbatim.
            r = subprocess.run(
                f"cd {stage} && find . -mindepth 1 | sed 's|^\\./||' | LC_ALL=C sort"
                f" | tar --format=ustar --owner=0 --group=0 --numeric-owner --mtime=@0 --no-recursion -cf {out} -T -",
                shell=True, capture_output=True)
            if r.returncode != 0:
                sys.exit(1)
            sha = subprocess.run(["sha256sum", out], capture_output=True, text=True).stdout.split()[0]
            shutil.move(out, os.path.join(store, sha + ".tar"))
            print(sha)
    elif op == "url":
        print(f"http://127.0.0.1:{port}/{args[1]}.tar")
    else:
        sys.exit(f"unknown hook op {op!r}")


def run():
    with tempfile.TemporaryDirectory(prefix="bundle-store-") as store:
        class Quiet(SimpleHTTPRequestHandler):
            def __init__(self, *a, **k):
                super().__init__(*a, directory=store, **k)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        env = dict(
            os.environ,
            BUNDLE_CONFORMANCE_STORE=store,
            BUNDLE_CONFORMANCE_PORT=str(srv.server_port),
            AGENT_CONFORMANCE_BUNDLE_HOOK=f"{sys.executable} -I {os.path.abspath(__file__)} hook",
            AGENT_CONFORMANCE_BUNDLE_FETCH_WINDOW_SECS="10",
        )
        out = subprocess.run(
            ["cargo", "test", "-p", "beyond-ai-agent", "--test", "bundle_conformance",
             "the_configured_store_conforms", "--", "--nocapture"],
            cwd=ROOT, env=env, capture_output=True, text=True).stdout
        srv.shutdown()
    print(out)
    if not re.search(r"^\s+(PASS|FAIL)\s", out, re.M):
        sys.exit("the kit did not run")
    failed = set(re.findall(r"^\s+FAIL\s+(.*?): ", out, re.M))
    new, closed = failed - KNOWN_GAPS, KNOWN_GAPS - failed
    for name in sorted(new):
        print(f"NEW FAILURE: {name}")
    for name in sorted(closed):
        print(f"GAP CLOSED (drop it from KNOWN_GAPS): {name}")
    if new or closed:
        sys.exit(1)
    print(f"demo bundle path: {len(KNOWN_GAPS)} known gaps, no other failure")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hook":
        hook(os.environ["BUNDLE_CONFORMANCE_STORE"], os.environ["BUNDLE_CONFORMANCE_PORT"], sys.argv[2], sys.argv[3:])
    else:
        run()
