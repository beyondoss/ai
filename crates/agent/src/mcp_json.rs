//! `.mcp.json`: the de facto project MCP config file (Claude Code, Cursor).
//!
//! ```json
//! {"mcpServers": {
//!   "files": {"command": "npx", "args": ["-y", "server-filesystem", "."], "env": {"K": "v"}},
//!   "docs":  {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${TOKEN}"}}
//! }}
//! ```
//!
//! An entry with a `command` is a stdio server, one with a `url` an HTTP server; a `"type"` of
//! `stdio`, `http`, `streamable-http` or `sse` is accepted and must agree with that shape (an `sse`
//! entry is dialed over streamable HTTP, the only HTTP transport this client speaks).
//!
//! Two consumers read the same parse and differ in what they trust:
//!
//! - **Local** ([`local_servers`]): the entries become [`McpServerConfig`]s and join the settings'
//!   `mcp_servers` (`settings::effective_settings_for_cwd`, behind the project-trust check), so a
//!   header or `env` value goes through the settings' own `resolve_config_value` (`$VAR`, `${VAR}`,
//!   `!command`) exactly as if it had been written in `settings.json`.
//! - **Hosted** ([`hosted_connectors`]): the file is the tenant's, read from its sandbox (or its
//!   bundle) on a replica. A `url` server is dialed from the replica, and only the `url` survives:
//!   a header would be a secret lookup, so headers come from the session grant alone. A `command`
//!   server is not run on the replica either: it becomes a connector with
//!   [`McpConnector::command`] set, and [`crate::tools::mcp_sandbox_stdio`] runs it as a job in the
//!   session's own sandbox, where the tenant's files and nothing of the replica's are. Its `env` is
//!   carried as the literal strings the file holds, never resolved on the replica.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::grant::{McpConnector, SandboxCommand};
use crate::settings::{McpServerConfig, McpTransport};

/// The file name, at the workspace root.
pub const FILE: &str = ".mcp.json";

/// Larger than any real config; a bigger file is ignored rather than read whole.
pub const MAX_BYTES: u64 = 256 * 1024;

/// The agent's hard ceiling on connectors a hosted session takes from the file, whatever the grant
/// says and when it says nothing. Each is an outbound handshake from the shared replica, dialed
/// together at session start, and every tool it lists lands in the prompt. The real number is the
/// control plane's (`limits.max_mcp_json_servers` in the grant); this only keeps a missing or huge
/// one from meaning thousands of dials. Connectors from the session grant are not counted.
pub const MAX_HOSTED_CONNECTORS: usize = 64;

enum Entry {
    Stdio {
        command: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
    },
    Http {
        url: String,
        headers: BTreeMap<String, String>,
    },
}

fn string_map(v: Option<&Value>) -> BTreeMap<String, String> {
    v.and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_owned())))
                .collect()
        })
        .unwrap_or_default()
}

fn entry(v: &Value) -> Result<Entry, String> {
    let command = v.get("command").and_then(Value::as_str);
    let url = v.get("url").and_then(Value::as_str);
    let kind = match v.get("type").and_then(Value::as_str) {
        Some("stdio") => "stdio",
        Some("http" | "streamable-http" | "streamableHttp" | "sse") => "http",
        Some(other) => return Err(format!("unsupported type `{}`", other.escape_default())),
        None if command.is_some() => "stdio",
        None => "http",
    };
    match (kind, command, url) {
        ("stdio", Some(command), _) => Ok(Entry::Stdio {
            command: command.to_owned(),
            args: v
                .get("args")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|s| s.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default(),
            env: string_map(v.get("env")),
        }),
        ("http", _, Some(url)) => Ok(Entry::Http {
            url: url.to_owned(),
            headers: string_map(v.get("headers")),
        }),
        _ => Err("needs a `command` or a `url`".to_owned()),
    }
}

fn parse(text: &str) -> (Vec<(String, Entry)>, Vec<String>) {
    let root: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => return (Vec::new(), vec![format!("{FILE}: not valid JSON ({e})")]),
    };
    let mut entries = Vec::new();
    let mut warnings = Vec::new();
    for (name, v) in root
        .get("mcpServers")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        match entry(v) {
            Ok(e) => entries.push((name.clone(), e)),
            Err(why) => warnings.push(format!("{FILE}: server `{name}` skipped: {why}")),
        }
    }
    (entries, warnings)
}

/// The file's servers as settings entries, plus one warning per entry that could not be used.
pub fn local_servers(text: &str) -> (Vec<McpServerConfig>, Vec<String>) {
    let (entries, warnings) = parse(text);
    let servers = entries
        .into_iter()
        .map(|(name, entry)| McpServerConfig {
            name,
            transport: match entry {
                Entry::Stdio { command, args, env } => McpTransport::Stdio { command, args, env },
                Entry::Http { url, headers } => McpTransport::Http { url, headers },
            },
            events: Vec::new(),
        })
        .collect();
    (servers, warnings)
}

/// The file's usable servers as grant-shaped connectors, plus a warning for each entry skipped or
/// stripped. See the module doc for what a hosted session takes from the file. Names are held to
/// the grant's own rule, so a name the grant would refuse never reaches the connect path.
pub fn hosted_connectors(text: &str, grant_limit: Option<u32>) -> (Vec<McpConnector>, Vec<String>) {
    let (entries, mut warnings) = parse(text);
    let (cap, limit) = match grant_limit {
        Some(n) if (n as usize) < MAX_HOSTED_CONNECTORS => {
            (n as usize, "the grant's max_mcp_json_servers")
        }
        _ => (MAX_HOSTED_CONNECTORS, "the agent's ceiling"),
    };
    let mut connectors = Vec::new();
    let mut over_cap = 0usize;
    for (name, entry) in entries {
        if !crate::service::is_connector_name(&name) {
            warnings.push(format!("{FILE}: server `{name}` skipped: invalid name"));
            continue;
        }
        match entry {
            Entry::Stdio { .. } if connectors.len() >= cap => over_cap += 1,
            Entry::Stdio { command, args, env } => {
                if command.is_empty() || command.contains('\0') {
                    warnings.push(format!(
                        "{FILE}: server `{name}` skipped: invalid `command`"
                    ));
                } else if args.iter().any(|a| a.contains('\0'))
                    || env
                        .iter()
                        .any(|(k, v)| k.is_empty() || k.contains(['=', '\0']) || v.contains('\0'))
                {
                    warnings.push(format!(
                        "{FILE}: server `{name}` skipped: a NUL, or an empty or `=` env name"
                    ));
                } else {
                    connectors.push(McpConnector {
                        name,
                        url: String::new(),
                        sandbox: true,
                        managed: false,
                        command: Some(SandboxCommand { command, args, env }),
                    });
                }
            }
            Entry::Http { .. } if connectors.len() >= cap => over_cap += 1,
            Entry::Http { url, headers } => {
                if !headers.is_empty() {
                    warnings.push(format!(
                        "{FILE}: server `{name}`: `headers` in the file are ignored; \
                         a hosted connector's headers come from the session grant"
                    ));
                }
                connectors.push(McpConnector {
                    name,
                    url,
                    sandbox: false,
                    managed: false,
                    command: None,
                });
            }
        }
    }
    if over_cap > 0 {
        warnings.push(format!(
            "{FILE}: {over_cap} server(s) skipped: {limit} lets a hosted session take \
             at most {cap} from the file"
        ));
    }
    (connectors, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE_TEXT: &str = r#"{"mcpServers": {
        "files": {"command": "npx", "args": ["-y", "srv"], "env": {"K": "${V}"}},
        "docs": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${TOKEN}"}},
        "typed": {"type": "http", "url": "https://example.com/t"},
        "legacy": {"type": "sse", "url": "https://example.com/sse"},
        "explicit": {"type": "stdio", "command": "srv"}
    }}"#;

    #[test]
    fn local_servers_reads_stdio_and_http_entries_and_accepts_type() {
        let (servers, warnings) = local_servers(FILE_TEXT);
        assert!(warnings.is_empty(), "{warnings:?}");
        let by_name = |n: &str| {
            servers
                .iter()
                .find(|s| s.name == n)
                .unwrap()
                .transport
                .clone()
        };
        assert_eq!(
            by_name("files"),
            McpTransport::Stdio {
                command: "npx".into(),
                args: vec!["-y".into(), "srv".into()],
                env: [("K".to_owned(), "${V}".to_owned())].into(),
            }
        );
        assert_eq!(
            by_name("docs"),
            McpTransport::Http {
                url: "https://example.com/mcp".into(),
                headers: [("Authorization".to_owned(), "Bearer ${TOKEN}".to_owned())].into(),
            }
        );
        assert!(matches!(by_name("typed"), McpTransport::Http { .. }));
        assert!(matches!(by_name("legacy"), McpTransport::Http { .. }));
        assert!(matches!(by_name("explicit"), McpTransport::Stdio { .. }));
        assert_eq!(servers.len(), 5);
    }

    #[test]
    fn local_servers_warns_on_a_bad_entry_and_keeps_the_rest() {
        let (servers, warnings) = local_servers(
            r#"{"mcpServers": {"ok": {"url": "https://a/mcp"}, "neither": {}, "odd": {"type": "ws", "url": "wss://a"}}}"#,
        );
        assert_eq!(servers.len(), 1);
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(warnings.iter().all(|w| w.starts_with(".mcp.json")));
    }

    #[test]
    fn a_malformed_file_is_one_warning_and_no_servers() {
        let (servers, warnings) = local_servers("{ not json");
        assert!(servers.is_empty());
        assert_eq!(warnings.len(), 1, "{warnings:?}");
    }

    #[test]
    fn hosted_connectors_take_a_command_as_a_sandbox_job_and_drop_headers() {
        let (connectors, warnings) = hosted_connectors(FILE_TEXT, None);
        let mut names: Vec<_> = connectors.iter().map(|c| c.name.as_str()).collect();
        names.sort();
        assert_eq!(names, ["docs", "explicit", "files", "legacy", "typed"]);
        let docs = connectors.iter().find(|c| c.name == "docs").unwrap();
        assert_eq!(docs.url, "https://example.com/mcp");
        assert!(docs.command.is_none() && !docs.sandbox);
        let files = connectors.iter().find(|c| c.name == "files").unwrap();
        assert!(files.sandbox && files.url.is_empty());
        let cmd = files.command.as_ref().unwrap();
        assert_eq!(cmd.command, "npx");
        assert_eq!(cmd.args, ["-y", "srv"]);
        // Literal: the replica resolves nothing from the file.
        assert_eq!(cmd.env["K"], "${V}");
        assert!(
            warnings.iter().all(|w| !w.contains("skipped: a `command`")),
            "{warnings:?}"
        );
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("`docs`") && w.contains("headers")),
            "{warnings:?}"
        );
        assert!(
            warnings.iter().all(|w| !w.contains("TOKEN")),
            "a warning must not echo a value from the file: {warnings:?}"
        );
    }

    #[test]
    fn hosted_connectors_refuse_names_the_grant_would_refuse() {
        let (connectors, warnings) = hosted_connectors(
            r#"{"mcpServers": {"a__b": {"url": "https://a/mcp"}, "has space": {"url": "https://a/mcp"}, "fine": {"url": "https://a/mcp"}}}"#,
            None,
        );
        assert_eq!(connectors.len(), 1);
        assert_eq!(connectors[0].name, "fine");
        assert_eq!(warnings.len(), 2, "{warnings:?}");
    }

    fn file_of(n: usize) -> String {
        let servers: Vec<String> = (0..n)
            .map(|i| format!(r#""s{i}": {{"url": "https://a/{i}"}}"#))
            .collect();
        format!(r#"{{"mcpServers": {{{}}}}}"#, servers.join(","))
    }

    #[test]
    fn the_grants_limit_below_the_ceiling_caps_the_file_and_the_warning_names_it() {
        let (connectors, warnings) = hosted_connectors(&file_of(10), Some(3));
        assert_eq!(connectors.len(), 3);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("7 server(s) skipped"), "{warnings:?}");
        assert!(warnings[0].contains("max_mcp_json_servers"), "{warnings:?}");
        assert!(warnings[0].contains("at most 3"), "{warnings:?}");
    }

    #[test]
    fn no_limit_means_the_ceiling_never_unlimited() {
        let (connectors, warnings) = hosted_connectors(&file_of(MAX_HOSTED_CONNECTORS + 5), None);
        assert_eq!(connectors.len(), MAX_HOSTED_CONNECTORS);
        assert_eq!(MAX_HOSTED_CONNECTORS, 64);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("5 server(s) skipped"), "{warnings:?}");
        assert!(warnings[0].contains("ceiling"), "{warnings:?}");
    }

    #[test]
    fn a_limit_above_the_ceiling_still_stops_at_the_ceiling() {
        let (connectors, warnings) =
            hosted_connectors(&file_of(MAX_HOSTED_CONNECTORS + 5), Some(1000));
        assert_eq!(connectors.len(), MAX_HOSTED_CONNECTORS);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("ceiling"), "{warnings:?}");
    }

    #[test]
    fn a_file_within_the_limit_warns_of_nothing() {
        let (connectors, warnings) = hosted_connectors(&file_of(3), Some(3));
        assert_eq!(connectors.len(), 3);
        assert!(warnings.is_empty(), "{warnings:?}");
    }
}
