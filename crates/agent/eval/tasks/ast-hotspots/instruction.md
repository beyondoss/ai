The Python package under `/app/svc` has a few dangerous runtime calls mixed with comments, docstrings, and string literals that *mention* the same names.

Write `/app/report.json`: a JSON array of every **AST Call** to exactly one of:

- `eval`
- `exec`
- `os.system`
- `subprocess.Popen`

Do **not** include comments, docstrings, or string literals. Each item must be:

```json
{"file": "<path relative to /app/svc>", "function": "<enclosing def, or <module>>", "line": <int>, "callee": "<ast.unparse of the call func>"}
```

Sort by `file`, then `line`. An `ast` MCP server is configured for structural queries — prefer it over grepping.
