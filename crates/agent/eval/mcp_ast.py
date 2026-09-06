#!/usr/bin/env python3
"""Stdio MCP server: Python AST queries (tree-sitter-shaped, stdlib only).

Newline-delimited JSON-RPC, matching the agent's fixture MCP client.
Tools: list_functions, list_classes, find_calls, get_function, module_imports.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

SERVER_NAME = "ast"
PROTOCOL = "2025-11-25"


def _py_files(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".py" else []
    if not root.is_dir():
        return []
    if recursive:
        return sorted(p for p in root.rglob("*.py") if p.is_file())
    return sorted(p for p in root.glob("*.py") if p.is_file())


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _qual(stack: list[str], name: str) -> str:
    return ".".join([*stack, name])


class Index(ast.NodeVisitor):
    def __init__(self, file: str) -> None:
        self.file = file
        self.stack: list[str] = []
        self.functions: list[dict[str, Any]] = []
        self.classes: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.imports: list[dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._fn(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._fn(node)

    def _fn(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        args = [a.arg for a in node.args.args]
        self.functions.append(
            {
                "file": self.file,
                "name": _qual(self.stack, node.name),
                "line": node.lineno,
                "async": isinstance(node, ast.AsyncFunctionDef),
                "args": args,
            }
        )
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [ast.unparse(b) for b in node.bases]
        self.classes.append(
            {
                "file": self.file,
                "name": _qual(self.stack, node.name),
                "line": node.lineno,
                "bases": bases,
            }
        )
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        callee = ast.unparse(node.func)
        self.calls.append(
            {
                "file": self.file,
                "function": _qual(self.stack, "")[:-1] or "<module>",
                "line": node.lineno,
                "callee": callee,
            }
        )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                {
                    "file": self.file,
                    "line": node.lineno,
                    "module": alias.name,
                    "name": alias.asname or alias.name,
                    "from": False,
                }
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            self.imports.append(
                {
                    "file": self.file,
                    "line": node.lineno,
                    "module": f"{mod}.{alias.name}".strip("."),
                    "name": alias.asname or alias.name,
                    "from": True,
                }
            )


def _index_path(path: str, recursive: bool = True) -> Index:
    root = Path(path)
    base = root if root.is_dir() else root.parent
    acc = Index("<merged>")
    acc.file = str(root)
    for py in _py_files(root, recursive):
        src = py.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=str(py))
        idx = Index(_rel(py, base) if root.is_dir() else py.name)
        idx.visit(tree)
        acc.functions.extend(idx.functions)
        acc.classes.extend(idx.classes)
        acc.calls.extend(idx.calls)
        acc.imports.extend(idx.imports)
    return acc


def _arg(args: dict[str, Any], name: str, default: Any = None) -> Any:
    if name not in args or args[name] is None:
        return default
    return args[name]


TOOL_DEFS = [
    {
        "name": "list_functions",
        "description": (
            "List function and async-function definitions in a Python file or directory "
            "using the stdlib ast module (real definitions, not string/comment matches). "
            "Returns name, line, args, and enclosing class when nested. Path may be a "
            "file or a directory; recursive defaults true for directories."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory to index."},
                "recursive": {"type": "boolean", "description": "Walk subdirectories (default true)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_classes",
        "description": (
            "List class definitions in a Python file or directory via ast.ClassDef. "
            "Includes bases as unparsed source. Does not match class names in comments."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "find_calls",
        "description": (
            "Find Call nodes whose callee source equals `callee` (e.g. eval, exec, "
            "os.system, subprocess.Popen). This is AST-precise: comments and string "
            "literals that mention the name do not match. Returns file, enclosing "
            "function, line, and callee."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "callee": {
                    "type": "string",
                    "description": "Exact ast.unparse(func) match, e.g. eval or subprocess.Popen.",
                },
                "recursive": {"type": "boolean"},
            },
            "required": ["path", "callee"],
        },
    },
    {
        "name": "get_function",
        "description": (
            "Return one function record by dotted name (Class.method or plain def) "
            "in a file or tree. Empty list if missing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "name": {"type": "string"},
                "recursive": {"type": "boolean"},
            },
            "required": ["path", "name"],
        },
    },
    {
        "name": "module_imports",
        "description": (
            "List Import and ImportFrom nodes. Use to see whether os/subprocess/eval "
            "aliases exist before searching for calls."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean"},
            },
            "required": ["path"],
        },
    },
]

CAPABILITIES = {"tools": {}, "resources": {}, "prompts": {}}


def _ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _err(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def call_tool(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name") or ""
    args = params.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    try:
        recursive = bool(_arg(args, "recursive", True))
        path = str(_arg(args, "path", "/app"))
        idx = _index_path(path, recursive)
        if name == "list_functions":
            return _ok(json.dumps(idx.functions, indent=2))
        if name == "list_classes":
            return _ok(json.dumps(idx.classes, indent=2))
        if name == "find_calls":
            callee = str(_arg(args, "callee", ""))
            hits = [c for c in idx.calls if c["callee"] == callee]
            return _ok(json.dumps(hits, indent=2))
        if name == "get_function":
            want = str(_arg(args, "name", ""))
            hits = [f for f in idx.functions if f["name"] == want]
            return _ok(json.dumps(hits, indent=2))
        if name == "module_imports":
            return _ok(json.dumps(idx.imports, indent=2))
        return _err(f"unknown tool {name}")
    except Exception as exc:  # noqa: BLE001 — MCP tool errors are user-facing
        return _err(f"{type(exc).__name__}: {exc}")


def handle(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "server/discover":
        return {
            "resultType": "complete",
            "supportedVersions": ["2026-07-28", "2025-11-25"],
            "capabilities": CAPABILITIES,
            "ttlMs": 0,
            "cacheScope": "private",
            "_meta": {
                "io.modelcontextprotocol/serverInfo": {
                    "name": SERVER_NAME,
                    "version": "0.1.0",
                }
            },
        }
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL,
            "capabilities": CAPABILITIES,
            "serverInfo": {"name": SERVER_NAME, "version": "0.1.0"},
        }
    if method == "tools/list":
        return {"tools": TOOL_DEFS}
    if method == "tools/call":
        return call_tool(params)
    if method == "resources/list":
        return {"resources": []}
    if method == "prompts/list":
        return {"prompts": []}
    if method == "ping":
        return {}
    raise KeyError(method)


def main() -> None:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = request.get("method")
        if not method:
            continue
        req_id = request.get("id")
        if req_id is None:
            continue
        params = request.get("params") or {}
        try:
            result = handle(method, params if isinstance(params, dict) else {})
            envelope = {"jsonrpc": "2.0", "id": req_id, "result": result}
        except KeyError:
            envelope = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        sys.stdout.write(json.dumps(envelope) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
