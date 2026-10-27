//! Layers 2 and 3: [`Jobs`], and [`Sandbox`], the object that owns all three layers.

use std::sync::Arc;
use std::time::Duration;

use crate::exec::{ChunkSink, CommandRunner, ExecOutput, ExecResult};
use crate::exec_endpoint::HttpExecRunner;
use crate::exec_endpoint::jobs::{SandboxJobs, ShellJobs};
use crate::fs::shell::ShellFs;
use crate::token::TokenSource;

/// How long [`Command`] lets a command run unless told otherwise.
pub const DEFAULT_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

/// Jobs over any [`Exec`](crate::Exec): commands that outlive the call that started them, are listed,
/// read, followed and killed by id. An HTTP endpoint that offers them natively (v1.3) gets its own
/// jobs; anything else gets the baseline built from plain exec calls. Derefs to [`SandboxJobs`],
/// whose methods are the operations.
#[derive(Clone)]
pub struct Jobs(Arc<dyn SandboxJobs>);

impl Jobs {
    /// Jobs on the target `exec` reaches.
    pub fn over(exec: Arc<dyn CommandRunner>) -> Self {
        Self(
            exec.clone()
                .sandbox_jobs()
                .unwrap_or_else(|| Arc::new(ShellJobs::new(exec))),
        )
    }
}

impl std::ops::Deref for Jobs {
    type Target = dyn SandboxJobs;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// A connected sandbox: the raw [`Exec`](crate::Exec), plus [`Files`](crate::Files) and [`Jobs`]
/// built on it.
#[derive(Clone)]
pub struct Sandbox {
    exec: Arc<dyn CommandRunner>,
    files: Arc<ShellFs>,
    jobs: Jobs,
}

impl Sandbox {
    /// Connect to the exec endpoint at `url`, authenticating with `token`. Probes the sandbox once
    /// for the tools the file helpers can use; fails if the endpoint cannot be reached.
    pub async fn connect(
        url: impl Into<String>,
        token: impl Into<TokenSource>,
    ) -> std::io::Result<Self> {
        let exec = HttpExecRunner::new(url)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?
            .bearer(token);
        Self::over(Arc::new(exec)).await
    }

    /// A sandbox over any [`Exec`](crate::Exec): `ssh`, `docker exec`, a vendor SDK, a test double.
    pub async fn over(exec: Arc<dyn CommandRunner>) -> std::io::Result<Self> {
        let files = Arc::new(ShellFs::connect(exec.clone()).await?);
        let jobs = Jobs::over(exec.clone());
        Ok(Self { exec, files, jobs })
    }

    /// Run `program` with `args`.
    pub fn exec<I, S>(&self, program: impl Into<String>, args: I) -> Command<'_>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Command {
            exec: &*self.exec,
            program: program.into(),
            args: args.into_iter().map(Into::into).collect(),
            cwd: None,
            timeout: DEFAULT_COMMAND_TIMEOUT,
            stdin: None,
        }
    }

    /// Read, write, edit, list, search and glob files.
    pub fn files(&self) -> &ShellFs {
        &self.files
    }

    /// Long-running commands.
    pub fn jobs(&self) -> &Jobs {
        &self.jobs
    }

    /// The exec this sandbox runs on, for anything the helpers do not cover.
    pub fn raw(&self) -> &Arc<dyn CommandRunner> {
        &self.exec
    }
}

/// One command, as [`Sandbox::exec`] builds it. Run it with [`run`](Self::run), or
/// [`stream`](Self::stream) its output as it is produced.
pub struct Command<'a> {
    exec: &'a dyn CommandRunner,
    program: String,
    args: Vec<String>,
    cwd: Option<String>,
    timeout: Duration,
    stdin: Option<Vec<u8>>,
}

impl Command<'_> {
    /// Run in this directory.
    pub fn cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Kill the command after this long ([`DEFAULT_COMMAND_TIMEOUT`] otherwise).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Feed these bytes to the command's standard input, then close it. Not every exec can; one
    /// that cannot answers [`std::io::ErrorKind::Unsupported`].
    pub fn stdin(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.stdin = Some(bytes.into());
        self
    }

    /// Run to completion and return the captured output as **bytes**, with
    /// [`exact`](ExecOutput::exact) saying whether they are all the command wrote: they are over an
    /// endpoint that honours `output_base64` or streams, and not over a plain v1.1 endpoint, which
    /// can only answer text.
    pub async fn run(self) -> std::io::Result<ExecOutput> {
        self.exec
            .run_exact(
                &self.program,
                &self.args,
                self.cwd.as_deref(),
                self.timeout,
                self.stdin.as_deref(),
            )
            .await
    }

    /// Run to completion, handing `on_chunk` each chunk of output as it arrives. Over an exec that
    /// streams (HTTP v1.2, the template transports) that is live; over one that does not, `on_chunk`
    /// is never called and the output is in the result. A command given [`stdin`](Self::stdin) is
    /// not streamed.
    pub async fn stream(self, on_chunk: ChunkSink<'_>) -> std::io::Result<ExecResult> {
        if self.stdin.is_some() {
            let out = self.run().await?;
            return Ok(ExecResult {
                code: Some(out.code),
                stdout: out.stdout_text().into_owned(),
                stderr: out.stderr_text().into_owned(),
                ..Default::default()
            });
        }
        self.exec
            .run_streaming(
                &self.program,
                &self.args,
                self.cwd.as_deref(),
                self.timeout,
                on_chunk,
            )
            .await
    }
}
