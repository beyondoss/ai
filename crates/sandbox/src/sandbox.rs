//! Layers 2 and 3: [`Jobs`], and [`Sandbox`], the object that owns all three layers.

use std::sync::Arc;
use std::time::Duration;

use crate::exec::{ChunkSink, CommandRunner, ExecOutput, ExecResult};
use crate::exec_endpoint::HttpExecRunner;
use crate::exec_endpoint::jobs::{JobId, JobScope, JobSpec, SandboxJobs};
use crate::fs::shell::ShellFs;
use crate::token::TokenSource;

/// How long [`Command`] lets a command run unless told otherwise.
pub const DEFAULT_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

/// Jobs on the target an [`Exec`](crate::Exec) reaches: commands that outlive the call that started
/// them, are listed, read, followed and killed by id. The exec endpoint owns them (the target's
/// `execd`). Derefs to [`SandboxJobs`], whose methods are the operations.
#[derive(Clone)]
pub struct Jobs(Arc<dyn SandboxJobs>);

impl Jobs {
    /// Jobs on the target `exec` reaches, or `None` when it has none of its own (an exec that runs
    /// commands on this host).
    pub fn over(exec: Arc<dyn CommandRunner>) -> Option<Self> {
        exec.sandbox_jobs().map(Self)
    }

    /// Start a job the sandbox keeps from the start (no hold). Returns its process group once it
    /// runs. Same shape as the TypeScript/Python `jobs.start`: program + args, with optional
    /// [`cwd`](StartJob::cwd) / [`label`](StartJob::label) before [`.run()`](StartJob::run).
    ///
    /// ```no_run
    /// # async fn demo(jobs: &beyond_sandbox::Jobs) -> std::io::Result<()> {
    /// use beyond_sandbox::exec_endpoint::jobs::{JobId, JobScope};
    /// let scope = JobScope::named("demo").map_err(std::io::Error::from)?;
    /// let id = JobId::fresh();
    /// jobs.start(&scope, &id, "npm", ["run", "dev"])
    ///     .cwd("/work")
    ///     .label("dev server")
    ///     .run()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn start<'a, I, S>(
        &'a self,
        scope: &'a JobScope,
        id: &'a JobId,
        program: impl Into<String>,
        args: I,
    ) -> StartJob<'a>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        StartJob {
            jobs: self,
            scope,
            id,
            spec: JobSpec {
                program: program.into(),
                args: args.into_iter().map(Into::into).collect(),
                cwd: None,
                label: String::new(),
            },
        }
    }
}

impl std::ops::Deref for Jobs {
    type Target = dyn SandboxJobs;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// A kept job as [`Jobs::start`] builds it. Configure with [`cwd`](Self::cwd) /
/// [`label`](Self::label), then [`run`](Self::run).
pub struct StartJob<'a> {
    jobs: &'a Jobs,
    scope: &'a JobScope,
    id: &'a JobId,
    spec: JobSpec,
}

impl StartJob<'_> {
    /// Run in this directory.
    pub fn cwd(mut self, cwd: impl Into<String>) -> Self {
        self.spec.cwd = Some(cwd.into());
        self
    }

    /// One-line label; newlines are replaced before it is written.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.spec.label = label.into();
        self
    }

    /// Start the kept job; returns its process group once it runs.
    pub async fn run(self) -> std::io::Result<u32> {
        self.jobs.start_kept(self.scope, self.id, &self.spec).await
    }
}

/// A connected sandbox: the raw [`Exec`](crate::Exec), plus [`Files`](crate::Files) and [`Jobs`]
/// built on it.
#[derive(Clone)]
pub struct Sandbox {
    exec: Arc<dyn CommandRunner>,
    files: Arc<ShellFs>,
    jobs: Option<Jobs>,
}

impl Sandbox {
    /// Connect to the exec endpoint at `url`, authenticating with `token`, and probe once.
    /// A bad URL or token fails here. For an [`Exec`](crate::Exec) you already hold with no probe,
    /// use [`Self::over`] or [`Self::with_capabilities`].
    pub async fn connect(
        url: impl Into<String>,
        token: impl Into<TokenSource>,
    ) -> std::io::Result<Self> {
        let exec = HttpExecRunner::new(url)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?
            .bearer(token);
        let sb = Self::over(Arc::new(exec)).await?;
        sb.files.capabilities().await?;
        Ok(sb)
    }

    /// A sandbox over any [`Exec`](crate::Exec): a vendor SDK, a test double. Does not probe;
    /// the file helpers look at the target once when the first one needs to know.
    pub async fn over(exec: Arc<dyn CommandRunner>) -> std::io::Result<Self> {
        let files = Arc::new(ShellFs::connect(exec.clone()).await?);
        let jobs = Jobs::over(exec.clone());
        Ok(Self { exec, files, jobs })
    }

    /// A sandbox over `exec` whose tools are already known, so nothing is probed: to pin the
    /// fallback rung the file helpers use (a test of a sandbox without `rg`, `grep -Z` or `find
    /// -printf`).
    pub fn with_capabilities(
        exec: Arc<dyn CommandRunner>,
        caps: crate::fs::shell::Capabilities,
    ) -> Self {
        let files = Arc::new(ShellFs::with_capabilities(exec.clone(), caps));
        let jobs = Jobs::over(exec.clone());
        Self { exec, files, jobs }
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

    /// Long-running commands, when the target owns jobs: [`std::io::ErrorKind::Unsupported`] for an
    /// exec that has none.
    pub fn jobs(&self) -> std::io::Result<&Jobs> {
        self.jobs.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "this exec does not own jobs",
            )
        })
    }

    /// Copy local files and directories into the sandbox directory `remote_dir` as a tar stream
    /// ([`crate::bulk`]).
    pub async fn upload(
        &self,
        sources: crate::bulk::Sources,
        remote_dir: &str,
    ) -> Result<crate::bulk::Transfer, crate::bulk::BulkError> {
        crate::bulk::upload(&*self.exec, sources, remote_dir).await
    }

    /// Copy a sandbox directory's contents (or one file) into the local directory `local_dir`
    /// ([`crate::bulk`]).
    pub async fn download(
        &self,
        remote_path: &str,
        local_dir: &std::path::Path,
    ) -> Result<crate::bulk::Transfer, crate::bulk::BulkError> {
        crate::bulk::download(&*self.exec, remote_path, local_dir).await
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

    /// Run to completion and return the exit code and the output as **bytes**: what the command wrote,
    /// unaltered.
    pub async fn run(self) -> std::io::Result<ExecOutput> {
        Ok(self.run_result().await?.into())
    }

    async fn run_result(self) -> std::io::Result<ExecResult> {
        match self.stdin.as_deref() {
            Some(stdin) => {
                self.exec
                    .run_with_stdin(
                        &self.program,
                        &self.args,
                        self.cwd.as_deref(),
                        self.timeout,
                        stdin,
                    )
                    .await
            }
            None => {
                self.exec
                    .run(&self.program, &self.args, self.cwd.as_deref(), self.timeout)
                    .await
            }
        }
    }

    /// Run to completion, handing `on_chunk` each chunk of output as it arrives.
    ///
    /// A command given [`stdin`](Self::stdin) cannot be streamed: the one-shot path is the only
    /// way to feed stdin today. Calling this with stdin set is
    /// [`std::io::ErrorKind::Unsupported`] — use [`run`](Self::run) instead, so silence on the
    /// sink is never mistaken for a live empty stream.
    pub async fn stream(self, on_chunk: ChunkSink<'_>) -> std::io::Result<ExecResult> {
        if self.stdin.is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "streaming a command with stdin is not offered; call run() instead",
            ));
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
