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
class NotFound(SandboxError):
    kind = "not_found"


class IsDirectory(SandboxError):
    kind = "is_directory"


class NotADirectory(SandboxError):
    kind = "not_a_directory"


class PermissionDenied(SandboxError):
    kind = "permission_denied"


class IoError(SandboxError):
    kind = "io"


# The request or the caller.
class InvalidQuery(SandboxError):
    kind = "invalid_query"


class Conflict(SandboxError):
    kind = "conflict"


class InvalidRequest(SandboxError):
    kind = "invalid_request"


# The transport (section 3.2).
class TooLarge(SandboxError):
    kind = "too_large"


class AuthError(SandboxError):
    kind = "auth"


class Timeout(SandboxError):
    kind = "timeout"


class TransportError(SandboxError):
    kind = "transport"


class ProtocolError(SandboxError):
    kind = "protocol"


class EndpointError(SandboxError):
    kind = "endpoint"


class NotOffered(SandboxError):
    kind = "not_offered"


_BY_KIND: dict[str, type[SandboxError]] = {
    cls.kind: cls
    for cls in (
        NotFound,
        IsDirectory,
        NotADirectory,
        PermissionDenied,
        IoError,
        InvalidQuery,
        Conflict,
        InvalidRequest,
        TooLarge,
        AuthError,
        Timeout,
        TransportError,
        ProtocolError,
        EndpointError,
        NotOffered,
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
        return NotFound(detail)
    if "is a directory" in low:
        return IsDirectory(detail)
    if "not a directory" in low:
        return NotADirectory(detail)
    if "permission denied" in low:
        return PermissionDenied(detail)
    return IoError(detail)
