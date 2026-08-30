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
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::settings::{McpServerConfig, McpTransport};

/// Bumped whenever the cached shape changes. Part of the key, so an older manifest is simply a miss
/// rather than something that has to be migrated.
const MANIFEST_VERSION: u32 = 1;

/// One server's advertised tools, as they were when last discovered.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerManifest {
    /// Identifies the exact invocation this was discovered from — see [`invocation_key`].
    pub key: String,
    pub tools: Vec<CachedTool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CachedTool {
    /// The bare name as the server knows it, unprefixed.
    pub remote_name: String,
    pub description: String,
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
pub struct ManifestDir(PathBuf);

impl ManifestDir {
    /// The real location, under the agent's own state directory. It describes the *machine's*
    /// configured servers, so it belongs beside the other agent state rather than in a session.
    pub fn from_home() -> Option<Self> {
        std::env::var_os("HOME").map(|h| Self(PathBuf::from(h).join(".claude")))
    }

    /// An explicit directory — what tests use, and what an embedder can point wherever it likes.
    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self(dir.into())
    }

    fn file(&self) -> PathBuf {
        self.0.join("mcp-manifest.json")
    }
}

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
        .filter(|m| !m.tools.is_empty())
}

/// Record what `config`'s server advertises. Best-effort: a cache that cannot be written costs a
/// server spawn on the next boot, which is the behavior without it.
pub fn store(dir: &ManifestDir, config: &McpServerConfig, tools: Vec<CachedTool>) {
    let path = dir.file();
    let mut all = read_store(dir);
    all.insert(
        config.name.clone(),
        ServerManifest {
            key: invocation_key(config),
            tools,
        },
    );
    let Ok(bytes) = serde_json::to_vec_pretty(&all) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Write-then-rename: a crash mid-write must not leave a truncated manifest that reads as a
    // *different* tool set. A miss is fine; a plausible-looking wrong answer is not.
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, &bytes).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stdio(name: &str, command: &str, args: &[&str]) -> McpServerConfig {
        McpServerConfig {
            name: name.into(),
            transport: McpTransport::Stdio {
                command: command.into(),
                args: args.iter().map(|s| (*s).to_string()).collect(),
                env: Default::default(),
            },
        }
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
            transport: McpTransport::Http {
                url: "https://example.com/mcp".into(),
                headers: Default::default(),
            },
        };
        assert_ne!(invocation_key(&s), invocation_key(&h));
    }
}
