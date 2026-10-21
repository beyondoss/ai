#!/usr/bin/env python3
"""Checks for front.py and e2b.sh against stubs: no E2B, no network beyond localhost.

    python3 deploy/demo/test_front.py
"""
import http.client
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix="front-test-")
VMDIR = os.path.join(TMP, "vm")
os.makedirs(VMDIR)

# A stand-in for e2b.sh: vm_for "creates" a VM whose url nothing listens on, and e2b_kill logs the id.
STUB_E2B = os.path.join(TMP, "e2b.sh")
with open(STUB_E2B, "w") as f:
    f.write(f'''
vm_for() {{
  echo x >> "{TMP}/creates"
  n=$(wc -l < "{TMP}/creates")
  echo '{{"id":"vm'$n'","envd_token":"t","bearer":"b","traffic_token":"","url":"https://127.0.0.1:1/exec"}}' > "$VMDIR/$1.json"
  [ -z "${{STUB_SLOW:-}}" ] || sleep "$STUB_SLOW"
}}
vm_field() {{ jq -r ".$2" "$VMDIR/$1.json"; }}
e2b_kill() {{ echo "$1" >> "{TMP}/kills"; }}
''')

os.environ.update(FRONT_VMDIR=VMDIR, FRONT_E2B_SH=STUB_E2B, FRONT_CA_FILE=os.path.join(TMP, "ca.pem"))
sys.path.insert(0, HERE)
import front  # noqa: E402


def lines(name):
    try:
        with open(os.path.join(TMP, name)) as f:
            return f.read().split("\n")[:-1]
    except FileNotFoundError:
        return []


def serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), front.Front)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def post(srv, session):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=20)
    c.request("POST", f"/exec/{session}", body=b"{}")
    r = c.getresponse()
    r.read()
    c.close()
    return r.status


class FrontTest(unittest.TestCase):
    def setUp(self):
        front.sessions.clear()
        for n in ("creates", "kills", "deleted"):
            try:
                os.remove(os.path.join(TMP, n))
            except FileNotFoundError:
                pass
        for n in os.listdir(VMDIR):
            os.remove(os.path.join(VMDIR, n))
        os.environ.pop("STUB_SLOW", None)

    def test_a_vm_that_cannot_be_reached_is_dropped_and_the_next_request_makes_a_fresh_one(self):
        srv = serve()
        self.assertEqual(post(srv, "s1"), 502)
        self.assertNotIn("s1", front.sessions)
        self.assertFalse(os.path.exists(f"{VMDIR}/s1.json"))
        self.assertEqual(lines("kills"), ["vm1"])
        self.assertEqual(post(srv, "s1"), 502)
        self.assertEqual(len(lines("creates")), 2, "the second request must not reuse the dead entry")
        srv.shutdown()

    def test_an_entry_past_the_vm_timeout_is_pruned(self):
        srv = serve()
        front.sessions["old"] = {"lock": threading.Lock(), "vm": {"id": "vmold"}, "born": time.time() - front.VM_TTL - 1}
        with open(f"{VMDIR}/old.json", "w") as f:
            f.write("{}")
        post(srv, "other")
        self.assertNotIn("old", front.sessions)
        self.assertFalse(os.path.exists(f"{VMDIR}/old.json"))
        srv.shutdown()


class E2bShTest(unittest.TestCase):
    def test_a_vm_created_then_failed_is_killed(self):
        # A fake curl: creating answers a VM with no traffic token (a private VM needs one); DELETE is logged.
        bindir = os.path.join(TMP, "bin")
        os.makedirs(bindir, exist_ok=True)
        curl = os.path.join(bindir, "curl")
        with open(curl, "w") as f:
            f.write(f'''#!/usr/bin/env bash
cat > /dev/null
case "$*" in
  *DELETE*) echo "$*" >> "{TMP}/deleted"; printf 204 ;;
  *sandboxes*) echo '{{"sandboxID":"vmX","envdAccessToken":"tok"}}' ;;
esac
''')
        os.chmod(curl, os.stat(curl).st_mode | stat.S_IXUSR)
        env = dict(os.environ, PATH=bindir + ":" + os.environ["PATH"], E2B_API_KEY="k", VMDIR=VMDIR,
                   SANDBOX_PORT="8081", TAG_KEY="k", NAME="n", VM_PRIVATE="1", STUB="/dev/null", CA_FILE="/dev/null")
        p = subprocess.run(["bash", "-c", f'. "{HERE}/e2b.sh"; vm_for sess'], env=env, capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertTrue(any("sandboxes/vmX" in l for l in lines("deleted")), p.stderr)
        self.assertFalse(os.path.exists(f"{VMDIR}/sess.json.part"))


if __name__ == "__main__":
    unittest.main()
