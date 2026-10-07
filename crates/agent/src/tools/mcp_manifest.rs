//! A cache of what each MCP server advertises, so the agent need not start one to know.
//!
//! ## Why
//!
//! Reaping an idle server gives its memory back, but it still *starts* one at boot and holds it
//! until the idle window expires — on a guest that never browses, that is a whole language runtime
//! spawned, initialized, and killed for nothing. Measured on the vps primitive, `@playwright/mcp`
//! costs 82.7 MB of anonymous memory sitting idle before a browser exists.
//!
//! The only reason to start it at boot is discovery: the model must be told what it can call before
//! it can call anything, and `tools/list` needs a live server. But that answer is *stable* — it is a
//! property of the server binary and its arguments, not of this boot. So it is worth caching, and
//! then a boot that never calls a browser tool never starts a browser server.
//!
//! ## Staleness
//!
//! The cache key is the server's full invocation (command, args, resolved env keys) plus the schema
//! version below. Change the pinned `@playwright/mcp` version in the config and the key changes with
//! it, so a stale manifest cannot outlive the server it described.
//!
//! What the key deliberately does *not* cover is a server whose tool list changes without its
//! invocation changing — a server that advertises different tools on different days. That is why a
//! mismatch is repaired rather than trusted blindly: the first real `tools/call` connects, and if the
//! tool it is calling is gone the call fails loudly with the server's own error, which is exactly
//! what would have happened without a cache. The cache can make the agent advertise a tool that no
//! longer exists; it cannot make a call silently do the wrong thing.

use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

use crate::settings::{McpServerConfig, McpTransport};

/// Bumped whenever the cached shape changes. Part of the key, so an older manifest is simply a miss
/// rather than something that has to be migrated.
const MANIFEST_VERSION: u32 = 3;

/// One server's advertised tools / resources / prompts, as they were when last discovered.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerManifest {
    /// Identifies the exact invocation this was discovered from — see [`invocation_key`].
    pub key: String,
    pub tools: Vec<CachedTool>,
    /// `resources/list` entries, exposed as `mcp__<server>__resource__<name>` tools.
    #[serde(default)]
    pub resources: Vec<CachedResource>,
    /// `prompts/list` entries, exposed as `mcp__<server>__prompt__<name>` tools.
    #[serde(default)]
    pub prompts: Vec<CachedPrompt>,
    /// SEP-2640 `skills/list` entries — `None` when the server does not declare the extension.
    /// Their digests are the server's at discovery time; a stale one fails verification on load and
    /// is refreshed through `skills/get` (see `mcp_skills`), so the cache cannot serve wrong content.
    #[serde(default)]
    pub skills: Option<Vec<crate::tools::mcp_skills::SkillEntry>>,
    /// Why entries were left out of that listing, so a boot from the cache still tells the user.
    #[serde(default)]
    pub skill_diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CachedTool {
    /// The bare name as the server knows it, unprefixed.
    pub remote_name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    /// The tool's MCP Apps `_meta.ui` (view resource, visibility). Recorded only in the apps
    /// manifest ([`ManifestDir::for_apps`]); absent from every plain one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ui: Option<crate::tools::mcp_apps::ToolUi>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CachedResource {
    /// Bare resource name (not the URI).
    pub name: String,
    pub uri: String,
    pub description: String,
    /// The size the server advertised, if it did — lets the host refuse an oversized MCP App view
    /// without reading it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CachedPrompt {
    pub name: String,
    pub description: String,
    /// JSON Schema derived from the prompt's declared arguments (same shape live discovery builds).
    pub input_schema: serde_json::Value,
}

/// A stable identity for "this server, invoked this way".
///
/// Env is included by **key only, never value**: a server's credentials commonly arrive through its
/// `env` (a `GITHUB_TOKEN`, say), and this string is written to disk. Which variables are set changes
/// what a server exposes; their values do not need to be recorded to know that.
pub fn invocation_key(config: &McpServerConfig) -> String {
    let mut parts: Vec<String> = vec![format!("v{MANIFEST_VERSION}"), config.name.clone()];
    match &config.transport {
        McpTransport::Stdio { command, args, env } => {
            parts.push("stdio".into());
            parts.push(command.clone());
            parts.extend(args.iter().cloned());
            parts.extend(env.keys().map(|k| format!("env:{k}")));
        }
        McpTransport::Http { url, .. } => {
            parts.push("http".into());
            parts.push(url.clone());
        }
    }
    parts.join("\u{1f}")
}

/// Where the cache lives.
///
/// Passed in rather than read from `HOME` at the point of use. Two reasons, and the second is the
/// one that bit: an ambient `HOME` makes this untestable (setting an env var in-process is `unsafe`
/// in edition 2024, which this crate forbids), and — worse — a test run would write into the
/// developer's own `~/.claude`, so one test's discovery silently changed the next test's behavior.
/// That actually happened: a reaping test started passing for the wrong reason because a previous
/// run had left a manifest behind.
#[derive(Debug, Clone)]
pub struct ManifestDir(PathBuf, &'static str);

impl ManifestDir {
    /// The real location, under the agent's own state directory. It describes the *machine's*
    /// configured servers, so it belongs beside the other agent state rather than in a session.
    pub fn from_home() -> Option<Self> {
        std::env::var_os("HOME").map(|h| Self(PathBuf::from(h).join(".claude"), FILE))
    }

    /// An explicit directory — what tests use, and what an embedder can point wherever it likes.
    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self(dir.into(), FILE)
    }

    /// The same directory's manifest for connections that advertise MCP Apps. A file of its own,
    /// not a key in the shared one: a server may list different tools once the extension is
    /// advertised, and the two flavors must never overwrite each other's answer.
    pub fn for_apps(&self) -> Self {
        Self(self.0.clone(), APPS_FILE)
    }

    fn file(&self) -> PathBuf {
        self.0.join(self.1)
    }
}

const FILE: &str = "mcp-manifest.json";
const APPS_FILE: &str = "mcp-manifest-apps.json";

type Store = BTreeMap<String, ServerManifest>;

fn read_store(dir: &ManifestDir) -> Store {
    let Ok(bytes) = std::fs::read(dir.file()) else {
        return Store::new();
    };
    // A corrupt or half-written manifest is a cache miss, never an error: the worst case is that the
    // agent connects at boot exactly as it did before this existed.
    serde_json::from_slice(&bytes).unwrap_or_default()
}

/// The cached manifest for `config`, if one was recorded for this exact invocation.
pub fn load(dir: &ManifestDir, config: &McpServerConfig) -> Option<ServerManifest> {
    let key = invocation_key(config);
    read_store(dir)
        .remove(&config.name)
        .filter(|m| m.key == key)
        .filter(|m| {
            !m.tools.is_empty()
                || !m.resources.is_empty()
                || !m.prompts.is_empty()
                || m.skills.is_some()
        })
}

/// Read the store, apply `change`, and write it back if `change` says it changed anything — under
/// the cross-process [`FileLock`], since several connections (and agents) update one file: a skills
/// refresh on one server racing a connect on another must not drop either's entry. Best-effort like
/// everything here: a lock that cannot be had skips the write, which costs a server spawn at the next
/// boot, never a wrong answer.
fn update_store(dir: &ManifestDir, change: impl FnOnce(&mut Store) -> bool) {
    let Ok(_lock) = FileLock::acquire(&dir.file()) else {
        return;
    };
    let mut all = read_store(dir);
    if change(&mut all) {
        write_store(dir, &all);
    }
}

/// Drop `config`'s cached manifest, if any — for a server whose answer must not be cached here (a
/// skills listing marked `cacheScope: "private"`). Best-effort, like [`store`].
pub fn forget(dir: &ManifestDir, config: &McpServerConfig) {
    update_store(dir, |all| all.remove(&config.name).is_some());
}

/// Replace only the skills listing in `config`'s cached manifest — for a listing re-fetched after
/// connect (`ttlMs` ran out, or the server said it changed), which would otherwise reach the cache
/// only at the next live connect, leaving a restart to advertise the stale one. A `private` listing
/// (`cacheScope: "private"`) forgets the server instead, as at connect. Without a manifest already
/// recorded for this exact invocation there is nothing to amend: the next live connect writes a
/// whole one. Best-effort, like [`store`].
pub fn store_skills(
    dir: &ManifestDir,
    config: &McpServerConfig,
    skills: Vec<crate::tools::mcp_skills::SkillEntry>,
    skill_diagnostics: Vec<String>,
    private: bool,
) {
    if private {
        forget(dir, config);
        return;
    }
    let key = invocation_key(config);
    update_store(dir, |all| {
        let Some(manifest) = all.get_mut(&config.name).filter(|m| m.key == key) else {
            return false;
        };
        manifest.skills = Some(skills);
        manifest.skill_diagnostics = skill_diagnostics;
        true
    });
}

/// Record what `config`'s server advertises. Best-effort: a cache that cannot be written costs a
/// server spawn on the next boot, which is the behavior without it.
pub fn store(
    dir: &ManifestDir,
    config: &McpServerConfig,
    tools: Vec<CachedTool>,
    resources: Vec<CachedResource>,
    prompts: Vec<CachedPrompt>,
    skills: Option<Vec<crate::tools::mcp_skills::SkillEntry>>,
    skill_diagnostics: Vec<String>,
) {
    let manifest = ServerManifest {
        key: invocation_key(config),
        tools,
        resources,
        prompts,
        skills,
        skill_diagnostics,
    };
    update_store(dir, |all| {
        all.insert(config.name.clone(), manifest);
        true
    });
}

fn write_store(dir: &ManifestDir, all: &Store) {
    let path = dir.file();
    let Ok(bytes) = serde_json::to_vec_pretty(all) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Write-then-rename: a crash mid-write must not leave a truncated manifest that reads as a
    // *different* tool set. A miss is fine; a plausible-looking wrong answer is not. The temporary
    // name is this writer's own, so a writer that lost the lock to a stale-lock break still cannot
    // rename another's half-written file into place.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = path.with_extension(format!("json.{}.{seq}.tmp", std::process::id()));
    if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

const LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(5);
const STALE_LOCK_AGE: Duration = Duration::from_secs(10);

/// A cross-process advisory lock via atomic lockfile creation — `mcp_auth_store.rs::FileLock`'s
/// pattern (itself `trust_store.rs`'s and `settings.rs`'s), duplicated rather than shared like each of
/// those. Held for one read-modify-write of the manifest; a lockfile older than [`STALE_LOCK_AGE`] is
/// a crashed holder's and is broken.
struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(store_path: &Path) -> std::io::Result<Self> {
        let mut os = store_path.as_os_str().to_owned();
        os.push(".lock");
        let lock_path = PathBuf::from(os);
        if let Some(parent) = lock_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let deadline = Instant::now() + LOCK_TIMEOUT;
        loop {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(_) => return Ok(Self { path: lock_path }),
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    let stale = std::fs::metadata(&lock_path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| SystemTime::now().duration_since(t).ok())
                        .is_some_and(|age| age > STALE_LOCK_AGE);
                    if stale {
                        let _ = std::fs::remove_file(&lock_path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        return Err(std::io::Error::new(
                            ErrorKind::TimedOut,
                            format!("timed out waiting for {}", lock_path.display()),
                        ));
                    }
                    std::thread::sleep(LOCK_RETRY_INTERVAL);
                }
                Err(e) => return Err(e),
            }
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stdio(name: &str, command: &str, args: &[&str]) -> McpServerConfig {
        McpServerConfig {
            name: name.into(),
            events: Vec::new(),
            transport: McpTransport::Stdio {
                command: command.into(),
                args: args.iter().map(|s| (*s).to_string()).collect(),
                env: Default::default(),
            },
        }
    }

    /// Connects and skills refreshes on many servers at once — the manifest is one file they all
    /// read, change and write back — lose nobody's entry and nobody's refresh.
    #[test]
    fn concurrent_stores_and_skills_refreshes_lose_no_update() {
        const SERVERS: usize = 24;
        let dir = tempfile::tempdir().unwrap();
        let manifest = ManifestDir::at(dir.path());
        let configs: Vec<McpServerConfig> = (0..SERVERS)
            .map(|i| stdio(&format!("s{i}"), "npx", &[&format!("server-{i}")]))
            .collect();
        let race = |work: &(dyn Fn(usize) + Sync)| {
            let barrier = std::sync::Barrier::new(SERVERS);
            std::thread::scope(|scope| {
                for i in 0..SERVERS {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        work(i);
                    });
                }
            });
        };
        race(&|i| {
            store(
                &manifest,
                &configs[i],
                vec![],
                vec![],
                vec![],
                Some(vec![]),
                vec![],
            );
        });
        for (i, c) in configs.iter().enumerate() {
            assert!(load(&manifest, c).is_some(), "s{i}'s connect was lost");
        }
        race(&|i| {
            store_skills(
                &manifest,
                &configs[i],
                vec![],
                vec![format!("refreshed-{i}")],
                false,
            );
        });
        for (i, c) in configs.iter().enumerate() {
            let m = load(&manifest, c).unwrap_or_else(|| panic!("s{i}'s entry was lost"));
            assert_eq!(
                m.skill_diagnostics,
                [format!("refreshed-{i}")],
                "s{i}'s refresh was lost"
            );
        }
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n != FILE)
            .collect();
        assert!(
            leftovers.is_empty(),
            "no lock or temporary file is left: {leftovers:?}"
        );
    }

    #[test]
    fn the_key_changes_when_the_pinned_version_does() {
        // The realistic staleness case: bumping `@playwright/mcp@0.0.78` to `@0.0.79` in settings.json
        // must not keep serving the old server's tool list.
        let a = stdio("playwright", "node", &["cli.js", "--headless"]);
        let b = stdio("playwright", "node", &["cli.js", "--headed"]);
        assert_ne!(invocation_key(&a), invocation_key(&b));
    }

    #[test]
    fn the_key_is_stable_for_an_unchanged_invocation() {
        let a = stdio("playwright", "node", &["cli.js"]);
        assert_eq!(
            invocation_key(&a),
            invocation_key(&stdio("playwright", "node", &["cli.js"]))
        );
    }

    #[test]
    fn env_contributes_its_names_but_never_its_values() {
        // Credentials commonly arrive via `env`, and this key is written to disk.
        let mut with_env = stdio("s", "cmd", &[]);
        if let McpTransport::Stdio { env, .. } = &mut with_env.transport {
            env.insert("GITHUB_TOKEN".into(), "ghp_super_secret_value".into());
        }
        let key = invocation_key(&with_env);
        assert!(key.contains("env:GITHUB_TOKEN"), "{key}");
        assert!(
            !key.contains("ghp_super_secret_value"),
            "a secret value must never reach the manifest key: {key}"
        );
        assert_ne!(invocation_key(&stdio("s", "cmd", &[])), key);
    }

    #[test]
    fn a_transport_change_changes_the_key() {
        let s = stdio("s", "cmd", &[]);
        let h = McpServerConfig {
            name: "s".into(),
            events: Vec::new(),
            transport: McpTransport::Http {
                url: "https://example.com/mcp".into(),
                headers: Default::default(),
            },
        };
        assert_ne!(invocation_key(&s), invocation_key(&h));
    }
}
