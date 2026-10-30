/**
 * Eve-compatible tool `ctx` (eve.dev/docs/tools § The ctx parameter).
 */
import {
  createLocalSandboxHandle,
  type SandboxHandle,
} from "./sandbox-handle.js";

export type ToolSession = {
  id: string;
  turn?: number;
  auth?: { current?: unknown };
  parent?: { id?: string };
};

export type ToolChannel = {
  kind: string;
  continuationToken?: string;
  metadata?: Record<string, unknown>;
};

export type ToolContext = {
  /** Grant-injected / process env (Beyond). */
  env: NodeJS.ProcessEnv;
  /** Absolute workspace root (Beyond bundle seed / cwd). */
  workspaceRoot: string;
  /** Eve: session metadata. */
  session: ToolSession;
  /** Eve: id of the current tool call. */
  callId: string;
  /** Eve: runtime tool name the model called. */
  toolName: string;
  /** Eve: model messages for the step that requested this call. */
  messages: unknown[];
  /** Eve: aborts when the active turn is cancelled. */
  abortSignal: AbortSignal;
  /** Eve: live sandbox handle (`run`, files, …). */
  getSandbox: () => Promise<SandboxHandle>;
  /** Eve: channel serving the session (optional when not on a channel). */
  channel?: ToolChannel;
  /**
   * Eve: start/continue a session on a channel.
   * Beyond MCP guest: not wired — throws (host concern).
   */
  to: (channel: string, target: unknown) => never;
};

export type CreateToolContextInput = {
  env?: NodeJS.ProcessEnv;
  workspaceRoot: string;
  toolName: string;
  callId?: string;
  session?: Partial<ToolSession>;
  messages?: unknown[];
  abortSignal?: AbortSignal;
  channel?: ToolChannel;
};

export function createToolContext(input: CreateToolContextInput): ToolContext {
  const workspaceRoot = input.workspaceRoot;
  const abortSignal = input.abortSignal ?? new AbortController().signal;
  const session: ToolSession = {
    id:
      input.session?.id ??
      input.env?.BEYOND_SESSION_ID ??
      process.env.BEYOND_SESSION_ID ??
      "local",
    turn: input.session?.turn,
    auth: input.session?.auth,
    parent: input.session?.parent,
  };

  return {
    env: input.env ?? process.env,
    workspaceRoot,
    session,
    callId: input.callId ?? crypto.randomUUID(),
    toolName: input.toolName,
    messages: input.messages ?? [],
    abortSignal,
    channel: input.channel,
    getSandbox: async () => createLocalSandboxHandle(workspaceRoot, abortSignal),
    to: () => {
      throw new Error(
        "ctx.to() is an Eve channel host API; not available in the Beyond tools-mcp guest",
      );
    },
  };
}
