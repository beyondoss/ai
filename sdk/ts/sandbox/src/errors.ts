/** The error vocabulary of FILES-OVER-EXEC.md section 3.1 (plus `unsupported`, which only a driver reports). */
export type ErrorKind =
  | "not_found"
  | "is_directory"
  | "not_a_directory"
  | "permission_denied"
  | "io"
  | "invalid_query"
  | "conflict"
  | "too_large"
  | "auth"
  | "timeout"
  | "transport"
  | "protocol"
  | "endpoint"
  | "not_offered"
  | "invalid_request";

/** Every failure this package reports. `kind` is the spec's error kind; `detail` is free text. */
export class SandboxError extends Error {
  readonly kind: ErrorKind;
  readonly detail: string;

  constructor(kind: ErrorKind, detail: string, options?: { cause?: unknown }) {
    super(`${kind}: ${detail}`, options);
    this.name = new.target.name;
    this.kind = kind;
    this.detail = detail;
  }
}

/** The endpoint could not be reached, or a stream broke. */
export class TransportError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("transport", detail, options);
  }
}

/** The request deadline passed (not a command's own exit 124). */
export class TimeoutError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("timeout", detail, options);
  }
}

/** HTTP 401 or 403. */
export class AuthError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("auth", detail, options);
  }
}

/** The endpoint's answer exceeded the response cap. The command ran. */
export class TooLargeError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("too_large", detail, options);
  }
}

/** The endpoint answered with something that is not the protocol. */
export class ProtocolError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("protocol", detail, options);
  }
}

/** The endpoint refused the request with another status. */
export class EndpointError extends SandboxError {
  readonly status: number;

  constructor(detail: string, status: number, options?: { cause?: unknown }) {
    super("endpoint", detail, options);
    this.status = status;
  }
}

/** The endpoint lacks the capability (it does not stream; it has no jobs). */
export class NotOfferedError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("not_offered", detail, options);
  }
}

/** The caller's arguments cannot form a request; nothing was sent. */
export class InvalidRequestError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("invalid_request", detail, options);
  }
}

/** A path (or its parent) does not exist. */
export class NotFoundError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("not_found", detail, options);
  }
}

/** The path is a directory where a file was required. */
export class IsDirectoryError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("is_directory", detail, options);
  }
}

/** The path is not a directory where one was required. */
export class NotADirectoryError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("not_a_directory", detail, options);
  }
}

/** The sandbox refused access. */
export class PermissionDeniedError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("permission_denied", detail, options);
  }
}

/** A command ran and failed for a reason that is not a more specific kind. */
export class IoError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("io", detail, options);
  }
}

/** `writeIfUnchanged`: the file's mtime is not the expected one. */
export class ConflictError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("conflict", detail, options);
  }
}

/** The search pattern or glob was rejected. */
export class InvalidQueryError extends SandboxError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super("invalid_query", detail, options);
  }
}

// ---- job conditions: finer types of the kinds above, so a `catch (e instanceof NotFoundError)` keeps working ----

/**
 * The job does not exist, or was removed: `attach` answered 404, or `read` reported state `gone`.
 * Also a stdin `write` or a `keep`/`seen` to a job that is not there.
 */
export class JobGoneError extends NotFoundError {
  readonly id: string;

  constructor(detail: string, id: string, options?: { cause?: unknown }) {
    super(detail, options);
    this.id = id;
  }
}

/** `start` of an id that already exists in the scope (409). The first job is left alone. */
export class JobExistsError extends ConflictError {
  readonly id: string;

  constructor(detail: string, id: string, options?: { cause?: unknown }) {
    super(detail, options);
    this.id = id;
  }
}

/** `start` refused because the sandbox's disk is full (507): there is no room for the job's output. */
export class DiskFullError extends IoError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super(detail, options);
  }
}

/**
 * A stdin `write` the job cannot take (409): its stdin was never opened (`never_opened`), or is
 * closed, the job ended or its reader left (`closed`). Retrying cannot succeed. `written` is the
 * stream's length applied so far, when the endpoint says.
 */
export class StdinClosedError extends ConflictError {
  readonly reason: "closed" | "never_opened";
  readonly written: number | undefined;

  constructor(
    detail: string,
    reason: "closed" | "never_opened",
    written?: number,
    options?: { cause?: unknown },
  ) {
    super(detail, options);
    this.reason = reason;
    this.written = written;
  }
}

/**
 * A positional stdin `write` that waited for the bytes before it and they never came (409 `gap`).
 * Writing the missing bytes first, then this one again, succeeds. `written` is the length applied.
 */
export class StdinGapError extends ConflictError {
  readonly written: number | undefined;

  constructor(detail: string, written?: number, options?: { cause?: unknown }) {
    super(detail, options);
    this.written = written;
  }
}

/**
 * A stdin `write` while 16 others already wait on the job (429). Retryable: the same write at the
 * same position later applies nothing twice.
 */
export class StdinBusyError extends ConflictError {
  constructor(detail: string, options?: { cause?: unknown }) {
    super(detail, options);
  }
}

/**
 * The exception class of `kind`, for code that has only the name (and optional HTTP status /
 * cause).
 */
export function errorFor(
  kind: ErrorKind,
  detail: string,
  options?: { cause?: unknown; status?: number },
): SandboxError {
  const { cause, status } = options ?? {};
  const o = cause !== undefined ? { cause } : undefined;
  switch (kind) {
    case "not_found":
      return new NotFoundError(detail, o);
    case "is_directory":
      return new IsDirectoryError(detail, o);
    case "not_a_directory":
      return new NotADirectoryError(detail, o);
    case "permission_denied":
      return new PermissionDeniedError(detail, o);
    case "io":
      return new IoError(detail, o);
    case "invalid_query":
      return new InvalidQueryError(detail, o);
    case "conflict":
      return new ConflictError(detail, o);
    case "too_large":
      return new TooLargeError(detail, o);
    case "auth":
      return new AuthError(detail, o);
    case "timeout":
      return new TimeoutError(detail, o);
    case "transport":
      return new TransportError(detail, o);
    case "protocol":
      return new ProtocolError(detail, o);
    case "endpoint":
      return new EndpointError(detail, status ?? 0, o);
    case "not_offered":
      return new NotOfferedError(detail, o);
    case "invalid_request":
      return new InvalidRequestError(detail, o);
  }
}

const OS_ERROR_NAMES = [
  "no such file or directory",
  "directory nonexistent",
  "nonexistent directory",
  "is a directory",
  "not a directory",
  "permission denied",
];

/**
 * The detail of a failed command: the first line of trimmed stderr; when that line names no OS
 * error, the lines through the first one that does (a path containing a newline splits the message).
 */
export function errorDetail(stderr: string): string {
  const lines = stderr.trim().split("\n");
  const at = lines.findIndex((l) => {
    const low = l.toLowerCase();
    return OS_ERROR_NAMES.some((n) => low.includes(n));
  });
  return (at < 0 ? lines[0] ?? "" : lines.slice(0, at + 1).join("\n"))
    .trimEnd();
}

/**
 * A command ran and failed (FILES-OVER-EXEC section 3.3): the first line of its stderr, classified
 * case-insensitively by the first rule that matches.
 */
export function classifyFailure(prefix: string, stderr: string): SandboxError {
  const line = errorDetail(stderr);
  const detail = `${prefix}: ${line === "" ? "command failed" : line}`;
  const m = line.toLowerCase();
  if (
    m.includes("no such file or directory")
    || m.includes("directory nonexistent")
    || m.includes("nonexistent directory")
  ) {
    return new NotFoundError(detail);
  }
  if (m.includes("is a directory")) {
    return new IsDirectoryError(detail);
  }
  if (m.includes("not a directory")) {
    return new NotADirectoryError(detail);
  }
  if (m.includes("permission denied")) {
    return new PermissionDeniedError(detail);
  }
  return new IoError(detail);
}
