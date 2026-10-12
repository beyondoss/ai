"""The run's private filesystem view: where the agent works and where every test suite is graded.

    python3 sandbox.py '<ops json>' <cwd> -- <cmd> [args…]

A user + mount + PID namespace (no root, no setuid helper) in which:

- the run's workspace sits at the fixed path `WS` (`/tmp/longctx/work`), whatever the run dir is
  called on the host, so the agent and the grader see the same paths in every arm. boltons'
  `test_iter_find_files` strips characters, not a prefix (`p.lstrip(BOLTONS_PATH)`), and failed in
  every run whose host path held an `f` ("default");
- the pristine trees, the built tasks, every other run, the repo and the whole of `/home` are empty
  read-only tmpfs mounts, with only what the run needs (the venv, its base interpreter minus its
  site-packages, the agent binary, `rg`) bound back read-only at their own paths;
- `/tmp`, `/var/tmp` and `/dev/shm` are private, so concurrent runs can't read or clobber each
  other's scratch files (agents put helper scripts in `/tmp`);
- a fresh `/proc` hides every process outside the namespace (`/proc/<pid>/root` of the harness
  would otherwise reach the host's view), `no_new_privs` is set and host root is unmapped, so
  setuid binaries (sudo) do nothing; the docker socket and the per-user runtime dir (systemd --user,
  ssh-agent) are masked.

The network is shared: the agent must reach its model provider. Fetching a pristine copy over the
network is what the prompt forbids and `run._SUSPECT` flags.

`ops` is an ordered list: `["tmpfs", path, writable]` mounts an empty tmpfs, `["bind", src, dst,
writable]` binds a host path. Read-only tmpfs mounts are remounted read-only after every bind under
them exists. The command runs as the caller's uid with no capabilities, so it can't undo a mount.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import stat
import sys

WS = "/tmp/longctx/work"  # the workspace, as the agent and the grader see it
ROOT = "/tmp/longctx"
SETUP_FAILED = 125  # exit status when the namespace could not be built

HIDE = ("/home", "/root", "/mnt", "/media", "/srv", "/opt", "/run/user")  # empty, read-only
PRIVATE = ("/tmp", "/var/tmp", "/dev/shm")  # empty, writable, this run's own
NULLED = ("/run/docker.sock",)  # a docker socket mounts anything


def view(hide: list[str], ro: list[tuple[str, str]], after: list[str], rw: list[tuple[str, str]]) -> list:
    """The ordered ops: hide `HIDE` + `hide`, privatize `PRIVATE`, bind `ro` read-only, hide `after`
    (paths inside an `ro` bind), then bind `rw`."""
    ops: list = []
    hidden: list[str] = []
    for p in (*HIDE, *hide):
        p = os.path.realpath(p)
        if os.path.isdir(p) and not any(p == h or p.startswith(h + "/") for h in hidden):
            hidden.append(p)
            ops.append(["tmpfs", p, False])
    ops += [["tmpfs", p, True] for p in PRIVATE if os.path.isdir(p)]
    ops += [["bind", "/dev/null", p, False] for p in NULLED if os.path.exists(p)]
    ops += [["bind", s, d, False] for s, d in ro]
    ops += [["tmpfs", p, False] for p in after]
    ops += [["bind", s, d, True] for s, d in rw]
    return ops


def argv(ops: list, cwd: str, cmd: list[str]) -> list[str]:
    """The command line that runs `cmd` in `cwd` inside the view `ops`."""
    return [sys.executable, "-I", os.path.abspath(__file__), json.dumps(ops), cwd, "--", *cmd]

_NEWNS, _NEWUSER, _NEWPID = 0x00020000, 0x10000000, 0x20000000
_RDONLY, _NOSUID, _NODEV, _NOEXEC = 1, 2, 4, 8
_REMOUNT, _BIND, _REC, _PRIVATE = 32, 4096, 16384, 1 << 18
_NOATIME, _NODIRATIME, _RELATIME = 1024, 2048, 1 << 21
_PR_SET_PDEATHSIG, _PR_SET_NO_NEW_PRIVS = 1, 38
_libc = ctypes.CDLL(None, use_errno=True)


def _mount(src: str | None, dst: str, fstype: str | None, flags: int, data: str | None = None) -> None:
    enc = lambda s: s.encode() if s is not None else None
    if _libc.mount(enc(src), enc(dst), enc(fstype), ctypes.c_ulong(flags), enc(data)) != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"mount {src} -> {dst} ({fstype}): {os.strerror(e)}")


def _locked(path: str) -> int:
    """The mount flags a remount of `path` must keep (a user namespace may not clear them)."""
    f = os.statvfs(path).f_flag
    out = 0
    for st, ms in ((os.ST_NOSUID, _NOSUID), (os.ST_NODEV, _NODEV), (os.ST_NOEXEC, _NOEXEC),
                   (os.ST_NOATIME, _NOATIME), (os.ST_NODIRATIME, _NODIRATIME), (os.ST_RELATIME, _RELATIME)):
        if f & st:
            out |= ms
    return out


def _target(dst: str, is_dir: bool) -> None:
    if is_dir:
        os.makedirs(dst, exist_ok=True)
    elif not os.path.lexists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w").close()


def _setup(ops: list) -> None:
    _mount(None, "/", None, _REC | _PRIVATE)
    # Open every bind source before any mask hides it.
    fds = {i: os.open(op[1], os.O_PATH | os.O_CLOEXEC) for i, op in enumerate(ops) if op[0] == "bind"}
    isdir = {i: stat.S_ISDIR(os.fstat(fd).st_mode) for i, fd in fds.items()}
    readonly = []
    for i, op in enumerate(ops):
        if op[0] == "tmpfs":
            _, dst, rw = op
            _target(dst, True)
            _mount("tmpfs", dst, "tmpfs", _NOSUID | _NODEV, "mode=1777" if rw else "mode=0755")
            if not rw:
                readonly.append(dst)
        else:
            _, src, dst, rw = op
            _target(dst, isdir[i])
            _mount(f"/proc/self/fd/{fds[i]}", dst, None, _BIND | _REC)
            if not rw:
                _mount(None, dst, None, _BIND | _REMOUNT | _RDONLY | _locked(dst))
    for dst in readonly:
        _mount(None, dst, None, _REMOUNT | _RDONLY | _NOSUID | _NODEV)
    for fd in fds.values():
        os.close(fd)
    _mount("proc", "/proc", "proc", _NOSUID | _NODEV | _NOEXEC)


def main(argv: list[str]) -> int:
    ops, cwd = json.loads(argv[0]), argv[1]
    assert argv[2] == "--", argv
    cmd = argv[3:]
    uid, gid = os.getuid(), os.getgid()
    os.unshare(_NEWUSER | _NEWNS | _NEWPID)
    for name, text in (("setgroups", "deny"), ("uid_map", f"{uid} {uid} 1"), ("gid_map", f"{gid} {gid} 1")):
        with open(f"/proc/self/{name}", "w") as f:
            f.write(text)
    pid1 = os.fork()
    if pid1:
        # Outside: wait for the namespace's init and hand back its status.
        _, st = os.waitpid(pid1, 0)
        return os.waitstatus_to_exitcode(st) if os.WIFEXITED(st) else 128 + os.WTERMSIG(st)
    # PID 1 of the namespace: when it exits the kernel kills everything left inside, so a killed or
    # finished run leaves no test process behind. It dies with the outer process too.
    try:
        _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        _setup(ops)
        os.chdir(cwd)
        if _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
    except Exception as e:  # noqa: BLE001 — report and fail; the caller treats it as a broken run
        print(f"sandbox: {e}", file=sys.stderr, flush=True)
        os._exit(SETUP_FAILED)
    child = os.fork()
    if child == 0:
        try:
            os.execvp(cmd[0], cmd)
        except OSError as e:
            print(f"sandbox: exec {cmd[0]}: {e}", file=sys.stderr, flush=True)
            os._exit(127)
    while True:  # reap orphans until the command itself exits
        pid, st = os.wait()
        if pid == child:
            os._exit(os.waitstatus_to_exitcode(st) if os.WIFEXITED(st) else 128 + os.WTERMSIG(st))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
