"""The shell decides nothing. A status table, an offset sum, a retry counter or a time constant in the
handle would be a second copy of the kernel, outside the proof; this test fails when the handle or its
driver grows one. Twin of ``crates/job-client-ffi/tests/no_decision.rs`` (the FFI crate) and of the
shell-structure test the client-kernel design asks of every SDK.

It reads the source with ``ast``. A branch is a comparison, a loop's condition, a counter or arithmetic on
something the kernel owns; what is allowed is plumbing: which exception is which fact, what to hand to the
consumer, which effect to do.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "beyond_sandbox"

#: The shell's two files: the handle (``_held``) and the drivers that perform its effects (``_drive``).
SHELL = ("_held.py", "_drive.py")

#: Identifier parts of what the kernel owns: a status, a count of tries, an offset, a time.
OWNED = {
    "status", "code", "attempt", "attempts", "retry", "retries", "miss", "misses", "budget", "offset",
    "elapsed", "until", "since", "deadline", "silence", "backoff", "wait", "delay", "jitter", "timeout",
}  # fmt: skip


def owned(node: ast.AST) -> str | None:
    """The name of a thing the kernel owns that ``node`` mentions, if it does."""
    for n in ast.walk(node):
        name = n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else None
        if name and OWNED & set(name.lower().split("_")):
            return name
    return None


#: Exec's ``Pipe`` folds a command's frames into a result (stderr kept to a limit, 128 + a signal): not the
#: job handle, and not a decision of the protocol it speaks to a job endpoint.
NOT_THE_HANDLE = {"_PipeAcc"}


def walk(tree: ast.AST) -> list[ast.AST]:
    """Every node of ``tree`` except the classes that are not the job handle."""
    out: list[ast.AST] = []
    todo = [tree]
    while todo:
        node = todo.pop()
        if isinstance(node, ast.ClassDef) and node.name in NOT_THE_HANDLE:
            continue
        out.append(node)
        todo.extend(ast.iter_child_nodes(node))
    return out


def violations(source: str, handle: bool) -> list[str]:
    """What in ``source`` is a decision. ``handle``: the handle itself, which also has no number at all."""
    out: list[str] = []
    for node in walk(ast.parse(source)):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Compare) and (name := owned(node)):
            out.append(f"{line}: a comparison on `{name}`")
        elif isinstance(node, ast.Compare) and any(
            isinstance(c, ast.Constant)
            and isinstance(c.value, (int, float))
            and not isinstance(c.value, bool)
            for c in [node.left, *node.comparators]
        ):
            out.append(f"{line}: a comparison with a number")
        elif isinstance(node, ast.BinOp) and (name := owned(node)):
            out.append(f"{line}: arithmetic on `{name}`")
        elif isinstance(node, ast.AugAssign) and (
            owned(node) or isinstance(node.op, (ast.Add, ast.Sub, ast.Mult))
        ):
            out.append(f"{line}: a counter or a sum")
        elif isinstance(node, (ast.While, ast.If)) and node is not None and owned(node.test):
            out.append(f"{line}: a branch on `{owned(node.test)}`")
        elif isinstance(node, ast.Call) and _call_name(node) in {
            "range",
            "randint",
            "uniform",
            "pow",
        }:
            out.append(f"{line}: `{_call_name(node)}` (a retry count, a jitter or a backoff)")
        elif isinstance(node, ast.Call) and _call_name(node) == "sleep" and _has_number(node):
            out.append(f"{line}: a sleep for a fixed time")
        elif isinstance(node, ast.Constant) and isinstance(node.value, float):
            out.append(f"{line}: a float literal ({node.value})")
        elif (
            handle
            and isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
            and node.value not in (0, 1)
        ):
            out.append(f"{line}: a number ({node.value})")
    return out


def _call_name(call: ast.Call) -> str:
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def _has_number(call: ast.Call) -> bool:
    return any(isinstance(a, ast.Constant) and isinstance(a.value, (int, float)) for a in call.args)


@pytest.mark.parametrize("name", SHELL)
def test_the_shell_makes_no_decision(name: str) -> None:
    found = violations((SRC / name).read_text(), handle=name == "_held.py")
    assert found == [], f"{name} decides what the kernel must:\n" + "\n".join(found)


# The scan itself can fail: each of these is what a shell must never grow, and each is caught.
@pytest.mark.parametrize(
    "snippet",
    [
        "if opened.status == 429:\n    pass",
        "if status in (502, 503, 504):\n    pass",
        "if a.status >= 500:\n    pass",
        "offset = offset + len(data)",
        "self.offset += n",
        "attempts += 1",
        "if attempts > budget:\n    pass",
        "while retries:\n    pass",
        "for _ in range(5):\n    pass",
        "time.sleep(0.25)",
        "await asyncio.sleep(2)",
        "delay = min(delay * 2, 30)",
        "if elapsed_ms > 30_000:\n    pass",
        "x = 429",
        "x = 0.5",
        "if code != 0:\n    pass",
        "backoff = 2 ** attempt",
    ],
)
def test_the_scan_catches_a_decision(snippet: str) -> None:
    assert violations(snippet, handle=True), snippet


@pytest.mark.parametrize(
    "snippet",
    [
        "if a.dropped:\n    yield Emit(JobGap(a.dropped))",
        "if event is None:\n    event, line = yield from _read()",
        "except (TransportError, TimeoutError):\n    return K.Event.BROKE, ''",
        "time.sleep(_kernel.wait_s(effect.until_ms))",
        "answer = next(lines, None)",
        "if conn is not None:\n    conn.close()",
    ],
)
def test_the_scan_leaves_plumbing_alone(snippet: str) -> None:
    src = re.sub(r"^except", "try:\n    pass\nexcept", snippet, flags=re.M)
    assert violations(src, handle=True) == [], snippet
