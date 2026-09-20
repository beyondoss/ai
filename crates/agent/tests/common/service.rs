//! Scaffolding for the `serve --service` suites: a running replica plus everything a tenant needs
//! to reach it.
//!
//! The pieces are deliberately *real*. The sandbox is a directory on this host reached only through
//! an HTTP exec endpoint, so "the tool ran in the sandbox" and "the tool ran on the replica" are
//! distinguishable by which directory a marker file lands in. The grants are minted by
//! [`super::grant`], which is written from the spec rather than from `src/grant.rs`. The shards are
//! separate temp directories, so tenant scoping and cross-shard routing are observable on disk.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use beyond_ai_agent::memory::file::FileBackend;
use beyond_ai_agent::session_store::{Layout, RepoOptions, SessionRepo, TenantCodec};

use super::exec_mock::ExecMock;
use super::grant::{Claims, Minter, Secrets};
use super::{ChildGuard, SpawnGuarded, free_port, serve_service_cmd, wait_for_port};

/// The header value every exec request from a service session must carry — the grant's sealed
/// `exec_headers`. A test asserts both that it *reaches the sandbox* and that it never appears in a
/// protocol response or on disk.
pub const EXEC_HEADER: (&str, &str) = ("X-Sandbox-Token", "exec-secret-do-not-log");

/// The gateway credential sealed into every grant these tests mint.
pub const GATEWAY_KEY: &str = "bai_v1.grant-key-not-the-hosts";

pub struct Service {
    pub port: u16,
    pub minter: Minter,
    pub exec: ExecMock,
    /// The tenant's workspace inside the sandbox.
    pub workspace: PathBuf,
    /// The sandbox's `$HOME` — what the startup probe reports.
    pub sandbox_home: PathBuf,
    /// `<shard name>` → directory, in `--shard` flag order.
    pub shards: Vec<(String, PathBuf)>,
    pub child: ChildGuard,
    /// The model server this replica was pointed at, so a [`Peer`] can share it.
    pub model_base: String,
    /// Held so every directory above outlives the replica.
    pub dir: tempfile::TempDir,
}

/// A **second** replica on the same mounts, exec endpoint and grant keys — what makes storage
/// fencing observable: two processes, one shard directory, exactly the deployment the epoch fence
/// and the session lock exist for.
pub struct Peer {
    pub port: u16,
    pub child: ChildGuard,
}

/// How the replica is started, for the cases that need something other than the default.
#[derive(Default)]
pub struct Options {
    /// Extra arguments after the standard `--service` set.
    pub extra_args: Vec<String>,
    /// Override the replica's own `HOME` (default: a path that doesn't exist), so a test can plant
    /// `~/.claude/SYSTEM.md` and skills on the replica and prove they never reach a tenant.
    pub host_home: Option<PathBuf>,
    /// Extra environment for the replica — an ambient `ANTHROPIC_API_KEY`, say.
    pub env: Vec<(String, String)>,
}

impl Service {
    pub async fn start(model_base: &str, shard_names: &[&str]) -> Self {
        Self::start_with(model_base, shard_names, Options::default()).await
    }

    pub async fn start_with(model_base: &str, shard_names: &[&str], opts: Options) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("sandbox/workspace");
        let sandbox_home = dir.path().join("sandbox/home");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&sandbox_home).unwrap();
        let exec = ExecMock::start_with_home(&workspace, Some(&sandbox_home), true).await;

        let shards: Vec<(String, PathBuf)> = shard_names
            .iter()
            .map(|name| {
                let path = dir.path().join(format!("shard-{name}"));
                std::fs::create_dir_all(&path).unwrap();
                ((*name).to_string(), path)
            })
            .collect();
        let minter = Minter::new(dir.path());
        let port = free_port();
        let flag_shards: Vec<(&str, &Path)> = shards
            .iter()
            .map(|(n, p)| (n.as_str(), p.as_path()))
            .collect();
        let mut cmd = serve_service_cmd(
            super::BIN,
            model_base,
            port,
            &minter.grant_key_flag(),
            minter.seal_key(),
            &flag_shards,
        );
        cmd.args(&opts.extra_args);
        if let Some(home) = &opts.host_home {
            cmd.env("HOME", home);
        }
        for (k, v) in &opts.env {
            cmd.env(k, v);
        }
        let child = cmd.spawn_guarded();
        wait_for_port(port);
        Self {
            port,
            minter,
            exec,
            workspace,
            sandbox_home,
            shards,
            child,
            model_base: model_base.to_string(),
            dir,
        }
    }

    /// Start a second replica against the same shards and the same keyring — the two-process case.
    /// Grants minted by `self` are valid on it, and both write the same mounted directories.
    pub fn start_peer(&self, extra_args: &[&str]) -> Peer {
        let flag_shards: Vec<(&str, &Path)> = self
            .shards
            .iter()
            .map(|(n, p)| (n.as_str(), p.as_path()))
            .collect();
        let port = free_port();
        let mut cmd = serve_service_cmd(
            super::BIN,
            &self.model_base,
            port,
            &self.minter.grant_key_flag(),
            self.minter.seal_key(),
            &flag_shards,
        );
        cmd.args(extra_args);
        let child = cmd.spawn_guarded();
        wait_for_port(port);
        Peer { port, child }
    }

    /// Plant a file inside the tenant's workspace, creating parents. The replica can only reach it
    /// through the exec endpoint, so anything a test finds in the prompt afterwards came from there.
    pub fn write_in_workspace(&self, rel: &str, contents: &str) -> PathBuf {
        write_at(&self.workspace.join(rel), contents)
    }

    /// Plant a file under the sandbox's own `$HOME` — the tenant's home, not the replica's.
    pub fn write_in_sandbox_home(&self, rel: &str, contents: &str) -> PathBuf {
        write_at(&self.sandbox_home.join(rel), contents)
    }

    /// A `SKILL.md` in the tenant's workspace: `<workspace>/<root>/<name>/SKILL.md`.
    pub fn write_sandbox_skill(&self, root: &str, name: &str, description: &str, body: &str) {
        self.write_in_workspace(
            &format!("{root}/{name}/SKILL.md"),
            &format!("---\nname: {name}\ndescription: {description}\n---\n{body}\n"),
        );
    }

    pub fn shard(&self, name: &str) -> &Path {
        self.shards
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p.as_path())
            .unwrap_or_else(|| panic!("no shard {name}"))
    }

    /// Where a tenant's sessions land on `shard`.
    pub fn sessions_dir(&self, shard: &str, tenant: &str) -> PathBuf {
        self.shard(shard).join(tenant).join("sessions")
    }

    /// Where a tenant's durable memory lands on `shard`.
    pub fn memory_dir(&self, shard: &str, tenant: &str) -> PathBuf {
        self.shard(shard).join(tenant).join("memory")
    }

    /// The tenant's sealing keys, derived from the `dek` every grant here carries — what the next
    /// owner of a shard holds, and the only way to read anything this replica wrote.
    pub fn codec(&self, tenant: &str) -> Arc<TenantCodec> {
        Arc::new(TenantCodec::new(tenant, &self.secrets().dek))
    }

    /// A tenant's session repo on `shard`, opened the way a replica opens it: epoch segments, sealed.
    pub fn tenant_repo(&self, shard: &str, tenant: &str) -> SessionRepo {
        SessionRepo::open_with(
            self.sessions_dir(shard, tenant),
            RepoOptions {
                layout: Layout::Segmented {
                    codec: Some(self.codec(tenant)),
                },
                id_prefix: Some(shard.to_string()),
            },
        )
        .unwrap()
    }

    /// A tenant's durable memory store on `shard`, sealed with its own key.
    pub fn tenant_memory(&self, shard: &str, tenant: &str) -> FileBackend {
        FileBackend::at(self.memory_dir(shard, tenant)).sealed(self.codec(tenant))
    }

    pub fn claims(&self, tenant: &str, session_id: &str, home_shard: &str) -> Claims {
        Claims {
            tenant: tenant.into(),
            session_id: session_id.into(),
            home_shard: home_shard.into(),
            workspace_root: self.workspace.to_string_lossy().into_owned(),
            exec_url: self.exec.url.clone(),
            mcp: Vec::new(),
            exp: far_future(),
        }
    }

    pub fn secrets(&self) -> Secrets {
        Secrets {
            exec_headers: vec![(EXEC_HEADER.0.into(), EXEC_HEADER.1.into())],
            mcp_headers: BTreeMap::new(),
            gateway_key: GATEWAY_KEY.into(),
            dek: [9u8; 32],
        }
    }

    /// A grant for `tenant`'s `session_id`, homed on the first `--shard`.
    pub fn token(&self, tenant: &str, session_id: &str) -> String {
        let home = self.shards[0].0.clone();
        self.minter
            .mint(&self.claims(tenant, session_id, &home), &self.secrets())
    }

    /// A grant with the claims tweaked — an expired `exp`, a relative `workspace_root`, a shard this
    /// replica doesn't mount.
    pub fn token_with(
        &self,
        tenant: &str,
        session_id: &str,
        edit: impl FnOnce(&mut Claims),
    ) -> String {
        let home = self.shards[0].0.clone();
        let mut claims = self.claims(tenant, session_id, &home);
        edit(&mut claims);
        self.minter.mint(&claims, &self.secrets())
    }

    /// The `x-beyond-grant` header for [`Self::token`].
    pub fn header<'a>(&self, token: &'a str) -> [(&'static str, &'a str); 1] {
        [("x-beyond-grant", token)]
    }
}

/// Every regular file under `dir`, recursively — what a test that asserts "nothing readable sits on
/// this mount" walks.
pub fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(next) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&next) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    out
}

fn write_at(path: &Path, contents: &str) -> PathBuf {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, contents).unwrap();
    path.to_path_buf()
}

/// A replica `HOME` stocked with the full `~/.claude` surface a tenant must never see: a `SYSTEM.md`,
/// an `APPEND_SYSTEM.md`, a `CLAUDE.md`, a skill, a prompt template and an agent definition, each
/// carrying its own marker string. Returns the directory (hold it: dropping it deletes the tree) and
/// the markers, so a test can assert every one of them is absent from the prompt.
pub fn host_claude_home() -> (tempfile::TempDir, Vec<&'static str>) {
    const MARKERS: &[&str] = &[
        "HOST-SYSTEM-MARKER",
        "HOST-APPEND-MARKER",
        "HOST-CONTEXT-MARKER",
        "host-only-skill",
        "host-only-prompt",
        "host-only-agent",
    ];
    let home = tempfile::tempdir().unwrap();
    let claude = home.path().join(".claude");
    write_at(&claude.join("SYSTEM.md"), "You are HOST-SYSTEM-MARKER.");
    write_at(&claude.join("APPEND_SYSTEM.md"), "HOST-APPEND-MARKER.");
    write_at(&claude.join("CLAUDE.md"), "HOST-CONTEXT-MARKER");
    write_at(
        &claude.join("skills/host-only-skill/SKILL.md"),
        "---\nname: host-only-skill\ndescription: the replica's own skill\n---\nbody\n",
    );
    write_at(
        &claude.join("prompts/host-only-prompt.md"),
        "The replica's own prompt template.",
    );
    write_at(
        &claude.join("agents/host-only-agent.md"),
        "---\nname: host-only-agent\ndescription: the replica's own agent\n---\nbody\n",
    );
    (home, MARKERS.to_vec())
}

/// Far enough out that these tests never expire, near enough to stay a plausible unix second.
pub fn far_future() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        + 3600
}
