"""Errors. Every failure is a :class:`SandboxError` carrying the spec's error ``kind``."""

from __future__ import annotations


class SandboxError(Exception):
    """Base class. ``kind`` is a name from the error vocabulary of FILES-OVER-EXEC.md section 3."""

    kind: str = "io"

    def __init__(self, detail: str = "", *, status: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        #: The HTTP status, for errors that came from a non-2xx answer.
        self.status = status

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}" if self.detail else self.kind


# A command ran and failed (section 3.3).
class NotFoundError(SandboxError):
    kind = "not_found"


class IsDirectoryError(SandboxError):
    kind = "is_directory"


class NotADirectoryError(SandboxError):
    kind = "not_a_directory"


class PermissionDeniedError(SandboxError):
    kind = "permission_denied"


class IoError(SandboxError):
    kind = "io"


# The request or the caller.
class InvalidQueryError(SandboxError):
    kind = "invalid_query"


class ConflictError(SandboxError):
    kind = "conflict"


class InvalidRequestError(SandboxError):
    kind = "invalid_request"


# The transport (section 3.2).
class TooLargeError(SandboxError):
    kind = "too_large"


class AuthError(SandboxError):
    kind = "auth"


class TimeoutError(SandboxError):
    kind = "timeout"


class TransportError(SandboxError):
    kind = "transport"


class ConnectError(TransportError):
    """The connection was never made: refused, no route, or the name did not resolve. Nothing was sent, so
    nothing happened. (A ``TransportError``: a caller that catches that keeps working.)"""


class ProtocolError(SandboxError):
    kind = "protocol"


class EndpointError(SandboxError):
    kind = "endpoint"


class NotOfferedError(SandboxError):
    """The endpoint or transport lacks the capability (no stream, no jobs, …)."""

    kind = "not_offered"


# Job conditions: finer types of the kinds above, so ``except NotFoundError`` keeps working.
class JobGoneError(NotFoundError):
    """The job does not exist or was removed: ``attach`` answered 404, or ``read`` said ``gone``.
    Also a stdin ``write``, ``keep`` or ``seen`` to a job that is not there."""

    def __init__(self, detail: str = "", *, id: str = "", status: int | None = None) -> None:  # noqa: A002
        super().__init__(detail, status=status)
        self.id = id


class JobExistsError(ConflictError):
    """``start`` of an id that already exists in the scope (409). The first job is left alone."""

    def __init__(self, detail: str = "", *, id: str = "", status: int | None = None) -> None:  # noqa: A002
        super().__init__(detail, status=status)
        self.id = id


class DiskFullError(IoError):
    """``start`` refused because the sandbox's disk is full (507): no room for the job's output."""


class StdinClosedError(ConflictError):
    """A stdin ``write`` the job cannot take (409): ``reason`` is ``"never_opened"`` (started without
    ``stdin_open``) or ``"closed"`` (EOF sent, the job ended or its reader left) or ``"hung_up"`` (a terminal's keeper is gone). Retrying cannot
    succeed. ``written`` is the stream's length applied so far, when the endpoint says."""

    def __init__(
        self,
        detail: str = "",
        *,
        reason: str = "closed",
        written: int | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(detail, status=status)
        self.reason = reason
        self.written = written


class StdinGapError(ConflictError):
    """A positional stdin ``write`` waited for the bytes before it and they never came (409 ``gap``).
    Writing the missing bytes first, then this one again, succeeds."""

    def __init__(
        self, detail: str = "", *, written: int | None = None, status: int | None = None
    ) -> None:
        super().__init__(detail, status=status)
        self.written = written


class StdinBusyError(ConflictError):
    """A stdin ``write`` while 16 others already wait on the job (429). Retryable: the same write at
    the same position later applies nothing twice."""


_BY_KIND: dict[str, type[SandboxError]] = {
    cls.kind: cls
    for cls in (
        NotFoundError,
        IsDirectoryError,
        NotADirectoryError,
        PermissionDeniedError,
        IoError,
        InvalidQueryError,
        ConflictError,
        InvalidRequestError,
        TooLargeError,
        AuthError,
        TimeoutError,
        TransportError,
        ProtocolError,
        EndpointError,
        NotOfferedError,
    )
}


def error_for(kind: str, detail: str) -> SandboxError:
    """The exception class of ``kind``, for code that has only the name."""
    return _BY_KIND.get(kind, IoError)(detail)


def classify_stderr(detail: str) -> SandboxError:
    """FILES-OVER-EXEC.md 3.3: the kind of a command that ran and failed, from its stderr."""
    low = detail.lower()
    if (
        "no such file or directory" in low
        or "directory nonexistent" in low
        or "nonexistent directory" in low
    ):
        return NotFoundError(detail)
    if "is a directory" in low:
        return IsDirectoryError(detail)
    if "not a directory" in low:
        return NotADirectoryError(detail)
    if "permission denied" in low:
        return PermissionDeniedError(detail)
    return IoError(detail)
