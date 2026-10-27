"""beyond-sandbox: run commands in a sandbox, and read, write and search its files, over exec.

Three layers, each usable alone:

1. **Transport**: raw exec (``run``, ``stream``, jobs). :class:`HttpTransport` /
   :class:`AsyncHttpTransport` speak the exec protocol; anything with ``run`` and ``stream`` works.
2. **Helpers**: :class:`Files` and :class:`Jobs` (and ``Async*``), built on a transport's public calls.
3. **Surface**: :class:`Sandbox` / :class:`AsyncSandbox`, which gather both and keep raw exec in reach.
"""

from .atransport import AsyncFrameStream, AsyncHttpTransport, AsyncJobTransport, AsyncTransport
from .errors import (
    AuthError,
    Conflict,
    EndpointError,
    InvalidQuery,
    InvalidRequest,
    IoError,
    IsDirectory,
    NotADirectory,
    NotFound,
    PermissionDenied,
    ProtocolError,
    SandboxError,
    Timeout,
    TooLarge,
    TransportError,
)
from .files import AsyncFiles, CapsOverride, Files
from .jobs import AsyncJobs, Jobs
from .models import (
    Attached,
    Capabilities,
    Entry,
    ExecResult,
    Exit,
    Frame,
    Hit,
    JobEnd,
    JobGap,
    JobInfo,
    JobOutput,
    JobRead,
    JobShellExit,
    JobStarted,
    SearchResult,
    ShellExit,
    Signal,
    Stat,
    Stderr,
    Stdout,
    Survivor,
    TimedOut,
)
from .sandbox import AsyncFrames, AsyncSandbox, Frames, Sandbox
from .transport import FrameStream, HttpTransport, JobTransport, Transport

__version__ = "0.1.0.dev0"

__all__ = [
    "AsyncFiles",
    "AsyncFrameStream",
    "AsyncFrames",
    "AsyncHttpTransport",
    "AsyncJobTransport",
    "AsyncJobs",
    "AsyncSandbox",
    "AsyncTransport",
    "Attached",
    "AuthError",
    "Capabilities",
    "CapsOverride",
    "Conflict",
    "EndpointError",
    "Entry",
    "ExecResult",
    "Exit",
    "Files",
    "Frame",
    "FrameStream",
    "Frames",
    "Hit",
    "HttpTransport",
    "InvalidQuery",
    "InvalidRequest",
    "IoError",
    "IsDirectory",
    "JobEnd",
    "JobGap",
    "JobInfo",
    "JobOutput",
    "JobRead",
    "JobShellExit",
    "JobStarted",
    "JobTransport",
    "Jobs",
    "NotADirectory",
    "NotFound",
    "PermissionDenied",
    "ProtocolError",
    "Sandbox",
    "SandboxError",
    "SearchResult",
    "ShellExit",
    "Signal",
    "Stat",
    "Stderr",
    "Stdout",
    "Survivor",
    "TimedOut",
    "Timeout",
    "TooLarge",
    "Transport",
    "TransportError",
    "__version__",
]
