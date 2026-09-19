//! Service mode: one `serve` replica, many tenants, nothing of the replica's own reaching any of
//! them.
//!
//! `serve --listen … --service` turns the daemon from "a headless agent for whoever is on the other
//! end of this socket" into a **fail-closed multi-tenant service**. Every connection presents a
//! [`bsg_v1` session grant](crate::grant) in the `x-beyond-grant` header; the grant says which tenant
//! and session it is, where that session's sandbox and workspace are, and — sealed — the credentials
//! to reach them. This module is the seam between that grant and the rest of `serve`:
//!
//! - [`authorize`] turns a token plus a `?session_id=` into a [`ServiceSession`], or into exactly one
//!   [`Refusal`] (and so exactly one HTTP status). Nothing downstream re-derives tenancy.
//! - [`ServiceSession`] then answers every "where does this session's *X* live?" question `serve`
//!   used to answer from the replica host: persistence ([`ServiceSession::session_dir`]), durable
//!   memory ([`ServiceSession::memory_backend`]), the exec target every tool runs in
//!   ([`ServiceSession::connect_exec`]), and the gateway credential
//!   ([`ServiceSession::gateway_key`]).
//!
//! **The no-host rule.** In service mode the replica's own `$HOME`, cwd, and stored settings are not
//! a fallback for anything: no `~/.claude` skills/prompts/agents/`SYSTEM.md`, no `models.json` or
//! `auth.json`, no per-cwd session directory, no `RealRunner`/`LocalFs` behind an unset exec
//! endpoint. Where a host default would have applied, service mode **refuses** instead — see
//! [`refused_command`] and [`ExecCell::strict`](crate::exec_endpoint::ExecCell::strict). Everything a
//! tenant can see comes from its grant or its sandbox.
//!
//! Sandbox *discovery* (skills, context files, agents, prompt templates) and per-session MCP are not
//! wired yet: [`Resources`] is empty and [`ServiceSession::mcp`] returns nothing, so the features
//! that would otherwise silently read the replica host are simply off. Later PRs fill those two seams
//! in; this module is where they land, so the rest of `serve` does not move again.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::grant::{Grant, GrantVerifier};

/// Cap on the `x-beyond-grant` header's value. The whole header block is capped at 16 KiB
/// (`serve_ws::MAX_HEADER_BYTES`), so a token larger than this could never arrive beside a `Host`
/// and a `Sec-WebSocket-Key` anyway — rejecting it here means the crypto never runs on it.
pub const MAX_GRANT_BYTES: usize = 12 * 1024;

/// The header the edge relays a session grant in.
pub const GRANT_HEADER: &str = "x-beyond-grant";

/// The probe every service session runs before it touches a tool: the sandbox's own `$HOME`. It is
/// the smallest command that proves three things at once — the exec endpoint answers, its
/// credentials work, and there is a shell on the other side — and its output is exactly what
/// [`ShellFs::with_home`](crate::tools::fs::shell::ShellFs::with_home) needs to expand a `~`.
const HOME_PROBE: &str = r#"printf %s "$HOME""#;

/// How long the startup probe gets. Generous for one round trip to a sandbox that has to be awake
/// anyway, short enough that a dead endpoint fails the session rather than hanging the client.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

// ---------------------------------------------------------------------------------------------
// Shards
// ---------------------------------------------------------------------------------------------

/// The tenant-data mounts this replica serves: `--shard <name>=</abs/path>`, repeatable.
///
/// A shard is a storage domain, not a tenant: many tenants live on one shard, each under its own
/// `<root>/<tenant>/` subtree. A session id minted for a shard carries it as a `<shard>.` prefix, so
/// an id alone says which mount to open — which is what lets one replica serve sessions from several
/// mounts without a lookup, and what makes a session for a mount this replica doesn't have a
/// **421** (ask another replica) rather than a 404.
#[derive(Debug, Default)]
pub struct Shards(BTreeMap<String, PathBuf>);

impl Shards {
    /// Parse `--shard` values. Each is `<name>=</absolute/path>`; the name is a path segment (so it
    /// can be a directory component and an id prefix) with no `.` (so `<shard>.<opaque>` splits
    /// unambiguously), and the path must be absolute — a relative one would resolve against the
    /// replica's cwd, which service mode has no business reading.
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut map = BTreeMap::new();
        for arg in args {
            let (name, path) = arg
                .split_once('=')
                .ok_or_else(|| format!("--shard {arg:?}: expected <name>=</absolute/path>"))?;
            let (name, path) = (name.trim(), Path::new(path.trim()));
            if !is_shard_name(name) {
                return Err(format!(
                    "--shard {arg:?}: the name must be letters, digits, '-' or '_' (no '.', which \
                     separates the shard from the session id)"
                ));
            }
            if !path.is_absolute() {
                return Err(format!("--shard {arg:?}: the path must be absolute"));
            }
            if map.insert(name.to_owned(), path.to_path_buf()).is_some() {
                return Err(format!("--shard: {name:?} is given more than once"));
            }
        }
        Ok(Self(map))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&Path> {
        self.0.get(name).map(PathBuf::as_path)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Path)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_path()))
    }

    /// Every mounted shard's directory for `tenant` — the set a tenant-scoped listing walks.
    pub fn tenant_dirs(&self, tenant: &str, leaf: &str) -> Vec<PathBuf> {
        self.0
            .values()
            .map(|root| root.join(tenant).join(leaf))
            .collect()
    }
}

/// The shard a session id names: the part before its first `.`, or `home` for an id with no prefix
/// (which is every id minted before the repo learns to prefix derived ids).
pub fn shard_of<'a>(session_id: &'a str, home: &'a str) -> &'a str {
    session_id.split_once('.').map_or(home, |(shard, _)| shard)
}

/// A shard name: a directory component *and* an id prefix, so no `.` and nothing path-special.
fn is_shard_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// A tenant id or connector name: one safe path segment. Same shape a session id must have
/// ([`is_valid_session_id`](crate::session_store::is_valid_session_id)), for the same reason — it
/// becomes part of a filesystem path.
fn is_safe_segment(s: &str) -> bool {
    s.len() <= 128 && crate::session_store::is_valid_session_id(s)
}

// ---------------------------------------------------------------------------------------------
// Authorization
// ---------------------------------------------------------------------------------------------

/// Why a connection was refused. One variant per HTTP status, so the connection path maps rather
/// than decides — an operator reading a 421 knows it is a routing problem and not an auth one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// 401 — no grant, or one that does not verify (unknown kid, bad signature, expired, unsealable).
    Unauthorized(&'static str),
    /// 400 — the request or the grant's own claims are unusable (no `?session_id=`, a tenant that
    /// isn't a path segment, a relative `workspace_root`).
    BadRequest(&'static str),
    /// 403 — the grant verifies, but is not for this session.
    Forbidden(&'static str),
    /// 421 Misdirected Request — the grant is fine; this replica just doesn't mount its shard. The
    /// edge should try another replica rather than the client retrying here.
    Misdirected(&'static str),
}

/// Unix seconds, failing **closed**: a clock the process cannot read at all yields `u64::MAX`, which
/// expires every grant, rather than `0`, which would accept every expired one.
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(u64::MAX)
}

/// Verify `token` and bind it to `session_id`, producing this connection's [`ServiceSession`].
///
/// The order is the refusal table: a token that doesn't verify is a 401 before anything in it is
/// read; a grant that verifies but names another session is a 403; claims that aren't the right
/// *shape* are a 400; and only a well-formed grant for a shard this replica doesn't mount is a 421.
pub fn authorize(
    verifier: &GrantVerifier,
    shards: &Arc<Shards>,
    token: Option<&str>,
    session_id: Option<&str>,
    now: u64,
) -> Result<ServiceSession, Refusal> {
    let token = token.ok_or(Refusal::Unauthorized("a session grant is required"))?;
    if token.len() > MAX_GRANT_BYTES {
        return Err(Refusal::Unauthorized("the session grant is too large"));
    }
    let grant = verifier
        .verify(token, now)
        .map_err(|_| Refusal::Unauthorized("the session grant was rejected"))?;

    let session_id = session_id.ok_or(Refusal::BadRequest(
        "a session grant needs an explicit ?session_id=",
    ))?;
    if grant.session_id != session_id {
        return Err(Refusal::Forbidden(
            "the session grant is for a different session",
        ));
    }

    if !is_safe_segment(&grant.tenant) {
        return Err(Refusal::BadRequest("the session grant's tenant is invalid"));
    }
    if !is_shard_name(&grant.home_shard) {
        return Err(Refusal::BadRequest(
            "the session grant's home shard is invalid",
        ));
    }
    if !Path::new(&grant.workspace_root).is_absolute() {
        return Err(Refusal::BadRequest(
            "the session grant's workspace root must be absolute",
        ));
    }
    if !grant.mcp.iter().all(|c| is_safe_segment(&c.name)) {
        return Err(Refusal::BadRequest(
            "a session grant connector name is invalid",
        ));
    }

    let shard = shard_of(session_id, &grant.home_shard);
    if !is_shard_name(shard) {
        return Err(Refusal::BadRequest("the session id's shard is invalid"));
    }
    let session_root = shards.get(shard).ok_or(Refusal::Misdirected(
        "this replica does not mount that shard",
    ))?;
    let home_root = shards.get(&grant.home_shard).ok_or(Refusal::Misdirected(
        "this replica does not mount the grant's home shard",
    ))?;

    let session_dir = session_root.join(&grant.tenant).join("sessions");
    let memory_dir = home_root.join(&grant.tenant).join("memory");
    Ok(ServiceSession {
        grant: Arc::new(grant),
        shards: Arc::clone(shards),
        session_dir,
        memory_dir,
        sandbox: OnceLock::new(),
    })
}

// ---------------------------------------------------------------------------------------------
// The session
// ---------------------------------------------------------------------------------------------

/// What the sandbox answered at session start. Recorded once, read by every tool build after it.
#[derive(Debug, Clone)]
pub struct Sandbox {
    /// The sandbox's `$HOME`, so a model-supplied `~/notes.md` expands against the tenant's own home
    /// rather than being left alone (or, far worse, resolved against the replica's).
    pub home: String,
    /// `bash` if the sandbox has one, else `sh`. Recorded here rather than probed per tool call.
    pub shell: &'static str,
}

/// One connection's verified grant, and everything derived from it that `serve` would otherwise have
/// taken from the replica host.
///
/// Shared by `Arc`: the connection that *spawns* a session hands its copy to that session's task,
/// where it fixes the session's secrets for the task's life (see the module doc). A later attach
/// still needs its own valid grant for the same `(tenant, session_id)` — that is what the
/// supervisor's per-slot tenant check enforces — but it does not re-point a live session's storage
/// or credentials.
pub struct ServiceSession {
    pub grant: Arc<Grant>,
    pub shards: Arc<Shards>,
    /// `<shard of the session id>/<tenant>/sessions` — where this session's transcript lives.
    pub session_dir: PathBuf,
    /// `<home shard>/<tenant>/memory` — the tenant's durable memory, shared by all of its sessions.
    pub memory_dir: PathBuf,
    /// Filled by [`Self::connect_exec`] at session start; read by the tool builds after it.
    sandbox: OnceLock<Sandbox>,
}

impl ServiceSession {
    pub fn tenant(&self) -> &str {
        &self.grant.tenant
    }

    pub fn session_id(&self) -> &str {
        &self.grant.session_id
    }

    /// The tenant's workspace inside the sandbox — this session's `cwd` everywhere `serve` reports
    /// or records one.
    pub fn workspace_root(&self) -> &str {
        &self.grant.workspace_root
    }

    /// The grant's gateway credential. Service mode never resolves a credential any other way: no
    /// `--key`, no `models.json`, no OAuth store, no ambient provider env.
    pub fn gateway_key(&self) -> &str {
        self.grant.secrets.gateway_key.expose()
    }

    /// The sandbox, once probed. `None` before [`Self::connect_exec`] has run — no tool can run in
    /// that window, because the session fails to start if the probe fails.
    pub fn sandbox(&self) -> Option<&Sandbox> {
        self.sandbox.get()
    }

    /// This tenant's session directories across **every** mounted shard — what a tenant-scoped
    /// listing walks, so a session minted on another shard is still listable here.
    pub fn tenant_session_dirs(&self) -> Vec<PathBuf> {
        self.shards.tenant_dirs(self.tenant(), "sessions")
    }

    /// The directory holding `id`, by `id`'s own shard prefix. `None` when that shard isn't mounted
    /// — the caller reports it rather than silently operating on the wrong mount.
    pub fn session_dir_for(&self, id: &str) -> Option<PathBuf> {
        let shard = shard_of(id, &self.grant.home_shard);
        Some(self.shards.get(shard)?.join(self.tenant()).join("sessions"))
    }

    /// Durable memory, rooted at the tenant's own directory rather than resolved from the replica's
    /// cwd or `--memory` DSN.
    pub fn memory_backend(&self) -> Arc<dyn crate::memory::MemoryBackend> {
        Arc::new(crate::memory::file::FileBackend::at(
            self.memory_dir.clone(),
        ))
    }

    /// Build this session's exec target from the grant, **strictly**: the endpoint is probed before
    /// any tool can reach it, and a failure is returned to the caller, which ends the session.
    ///
    /// The probe is `sh -c 'printf %s "$HOME"'`. Its output becomes the backend's home (so `~`
    /// expands inside the sandbox, not on the replica), and a second `command -v bash` records which
    /// shell the remote `bash` tool should ask for. Both are cached on this `ServiceSession`.
    pub async fn connect_exec(
        &self,
        max_response_bytes: usize,
    ) -> Result<crate::exec_endpoint::ExecTarget, String> {
        use crate::exec_endpoint::{ExecTarget, HttpExecRunner};
        use crate::tools::exec::CommandRunner;

        let mut runner = HttpExecRunner::new(self.grant.exec_url.as_str())?
            .with_max_response_bytes(max_response_bytes);
        for header in &self.grant.secrets.exec_headers {
            runner = runner.with_header(header.name.as_str(), header.value.expose());
        }
        let runner: Arc<dyn CommandRunner> = Arc::new(runner);

        let home = runner
            .run(
                "sh",
                &["-c".to_string(), HOME_PROBE.to_string()],
                None,
                PROBE_TIMEOUT,
            )
            .await
            .map_err(|e| format!("the sandbox did not answer: {e}"))?;
        if home.code != Some(0) || home.stdout.trim().is_empty() {
            return Err(format!(
                "the sandbox did not report a home directory (exit {:?})",
                home.code
            ));
        }
        let home = home.stdout.trim().to_string();
        // The remote `bash` tool prefers a real bash and falls back to `sh`, which every sandbox has
        // by construction (the probe above just ran one).
        let shell = runner
            .run(
                "command",
                &["-v".to_string(), "bash".to_string()],
                None,
                PROBE_TIMEOUT,
            )
            .await
            .is_ok_and(|r| r.code == Some(0) && !r.stdout.trim().is_empty());
        let _ = self.sandbox.set(Sandbox {
            home: home.clone(),
            shell: if shell { "bash" } else { "sh" },
        });
        Ok(ExecTarget::over_with_home(runner, Some(home)).await)
    }

    /// What a tenant's own sandbox contributes to the prompt. Empty until sandbox discovery lands:
    /// the replica host's skills, agents, prompt templates, and context files must never reach a
    /// tenant, so "not discovered yet" fails closed as "none".
    pub fn resources(&self) -> Resources {
        Resources::default()
    }

    /// The tenant's MCP connectors. Empty until per-session MCP lands — the grant carries the
    /// connector list and its headers, but nothing dials them yet.
    pub fn mcp(
        &self,
    ) -> (
        Vec<Arc<dyn agent_core::Tool>>,
        crate::tools::mcp::McpCatalog,
    ) {
        (Vec::new(), crate::tools::mcp::McpCatalog::default())
    }

    /// Strip replica-host paths out of `text` — an error from the storage layer names the file it
    /// failed on, and a tenant has no business learning this replica's mount layout. Shard roots
    /// become `<store>`; the replica's own `$HOME`/cwd become `<host>`.
    pub fn redact(&self, text: &str) -> String {
        let mut out = text.to_owned();
        for (_, root) in self.shards.iter() {
            let root = root.to_string_lossy();
            if !root.is_empty() && out.contains(root.as_ref()) {
                out = out.replace(root.as_ref(), "<store>");
            }
        }
        for host in [
            std::env::var_os("HOME").map(PathBuf::from),
            std::env::current_dir().ok(),
        ]
        .into_iter()
        .flatten()
        {
            let host = host.to_string_lossy().into_owned();
            if host.len() > 1 && out.contains(&host) {
                out = out.replace(&host, "<host>");
            }
        }
        out
    }
}

/// Everything sandbox discovery contributes to a session, as one value so the wiring is written
/// once. Empty today; a later PR fills it from the sandbox through the exec backend.
#[derive(Default)]
pub struct Resources {
    pub skills: Vec<crate::skills::Skill>,
    pub prompt_templates: Vec<crate::prompts::PromptTemplate>,
    pub agents: Vec<crate::agents::AgentDef>,
    pub context_files: bool,
}

// ---------------------------------------------------------------------------------------------
// Command gating
// ---------------------------------------------------------------------------------------------

/// Why a control command is refused in service mode, or `None` if it is allowed.
///
/// Two groups. The first are host operations that have no tenant meaning at all and would, if left
/// on, reach the replica: re-pointing the exec endpoint (the grant decides that), the whole
/// interactive login surface (an operator's own credential store), switching to a session this
/// connection has no grant for, and `reload` (which re-walks the replica's filesystem — re-enabled
/// once discovery reads the sandbox instead).
///
/// The second group — `fork`, `clone`, `new_session` — is refused only until derived ids carry their
/// shard prefix. Minting an unprefixed id on a multi-shard replica would create a session nothing
/// could route back to, which is worse than refusing the command.
pub fn refused_command(command: &str) -> Option<&'static str> {
    Some(match command {
        "set_exec_endpoint" => {
            "refused in service mode: the session grant fixes this session's exec endpoint"
        }
        "login" | "submit_code" | "abort_login" | "logout" | "auth_status" => {
            "refused in service mode: credentials come from the session grant"
        }
        "switch_session" => {
            "refused in service mode: connect at ?session_id=<id> with a grant for that session"
        }
        "reload" => "refused in service mode: resource discovery is not available yet",
        "fork" | "clone" | "new_session" => {
            "refused in service mode: derived sessions are not available yet"
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shards(pairs: &[(&str, &str)]) -> Arc<Shards> {
        Arc::new(
            Shards::parse(
                &pairs
                    .iter()
                    .map(|(n, p)| format!("{n}={p}"))
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn shard_flags_demand_a_name_and_an_absolute_path() {
        assert!(Shards::parse(&["a=/mnt/a".into()]).is_ok());
        assert!(Shards::parse(&["a".into()]).is_err(), "no '='");
        assert!(Shards::parse(&["a=rel".into()]).is_err(), "relative");
        assert!(
            Shards::parse(&["a.b=/mnt/a".into()]).is_err(),
            "dot in name"
        );
        assert!(Shards::parse(&["=/mnt/a".into()]).is_err(), "empty name");
        assert!(
            Shards::parse(&["a=/mnt/a".into(), "a=/mnt/b".into()]).is_err(),
            "duplicate"
        );
    }

    #[test]
    fn an_id_names_its_own_shard_and_an_unprefixed_one_falls_to_home() {
        assert_eq!(shard_of("s1.abc", "home"), "s1");
        assert_eq!(shard_of("abc", "home"), "home");
        assert_eq!(shard_of("s1.a.b", "home"), "s1");
    }

    #[test]
    fn tenant_dirs_cover_every_mounted_shard() {
        let s = shards(&[("a", "/mnt/a"), ("b", "/mnt/b")]);
        let dirs = s.tenant_dirs("t1", "sessions");
        assert_eq!(
            dirs,
            vec![
                PathBuf::from("/mnt/a/t1/sessions"),
                PathBuf::from("/mnt/b/t1/sessions"),
            ]
        );
    }

    #[test]
    fn every_host_command_that_could_reach_the_replica_is_refused() {
        for command in [
            "set_exec_endpoint",
            "login",
            "submit_code",
            "abort_login",
            "logout",
            "auth_status",
            "switch_session",
            "reload",
            "fork",
            "clone",
            "new_session",
        ] {
            assert!(refused_command(command).is_some(), "{command}");
        }
        for command in [
            "prompt",
            "get_state",
            "list_sessions",
            "export_html",
            "bash",
        ] {
            assert!(refused_command(command).is_none(), "{command}");
        }
    }

    #[test]
    fn redaction_removes_the_mount_layout_from_an_error() {
        let s = shards(&[("a", "/mnt/efs/a")]);
        let session = ServiceSession {
            grant: Arc::new(Grant {
                tenant: "t1".into(),
                session_id: "a.x".into(),
                home_shard: "a".into(),
                workspace_root: "/w".into(),
                exec_url: "http://x/".into(),
                mcp: Vec::new(),
                exp: 0,
                secrets: serde_json::from_str(
                    r#"{"exec_headers":[],"mcp_headers":{},"gateway_key":"k","dek":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="}"#,
                )
                .unwrap(),
            }),
            shards: s,
            session_dir: PathBuf::from("/mnt/efs/a/t1/sessions"),
            memory_dir: PathBuf::from("/mnt/efs/a/t1/memory"),
            sandbox: OnceLock::new(),
        };
        let redacted =
            session.redact("failed to open /mnt/efs/a/t1/sessions/1_x.jsonl: No such file");
        assert!(!redacted.contains("/mnt/efs"), "{redacted}");
        assert!(redacted.contains("<store>/t1/sessions"), "{redacted}");
    }
}
