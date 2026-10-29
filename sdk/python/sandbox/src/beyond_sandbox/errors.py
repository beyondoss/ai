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


class ProtocolError(SandboxError):
    kind = "protocol"


class EndpointError(SandboxError):
    kind = "endpoint"


class NotOfferedError(SandboxError):
    """The endpoint or transport lacks the capability (no stream, no jobs, …)."""

    kind = "not_offered"


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
