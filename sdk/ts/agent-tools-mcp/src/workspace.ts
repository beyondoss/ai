import path from "node:path";

/**
 * Sandbox root for `ctx.workspaceRoot` and `ctx.getSandbox()`.
 *
 * `--dir` is only the folder of tool modules. The root is `--workspace` when
 * the caller passes it, otherwise the process cwd — the seeded bundle, which
 * the guest `cd`s into before `serve`.
 */
export function resolveWorkspace(
  explicit: string | undefined,
  cwd = process.cwd(),
): string {
  return path.resolve(explicit ?? cwd);
}
