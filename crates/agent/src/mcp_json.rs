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
//! - **Hosted** ([`hosted_connectors`]): the file is the tenant's, read from its sandbox on a
//!   replica. Only the `url` survives. A header, an `env` value or a `command` would be a secret
//!   lookup or a process on the replica, so none is taken from the file: headers come from the
//!   session grant alone, and `command` entries are skipped.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::grant::McpConnector;
use crate::settings::{McpServerConfig, McpTransport};

/// The file name, at the workspace root.
pub const FILE: &str = ".mcp.json";

/// Larger than any real config; a bigger file is ignored rather than read whole.
pub const MAX_BYTES: u64 = 256 * 1024;

/// Most connectors a hosted session takes from the file. Each is an outbound handshake from the
/// shared replica, dialed together at session start, and every tool it lists lands in the prompt;
/// a real project config names a handful, so 16 leaves room without letting a tenant's file fan a
/// session out to thousands of dials. Connectors from the session grant are not counted: those are
/// the control plane's call.
pub const MAX_HOSTED_CONNECTORS: usize = 16;

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
pub fn hosted_connectors(text: &str) -> (Vec<McpConnector>, Vec<String>) {
    let (entries, mut warnings) = parse(text);
    let mut connectors = Vec::new();
    let mut over_cap = 0usize;
    for (name, entry) in entries {
        if !crate::service::is_connector_name(&name) {
            warnings.push(format!("{FILE}: server `{name}` skipped: invalid name"));
            continue;
        }
        match entry {
            Entry::Stdio { .. } => warnings.push(format!(
                "{FILE}: server `{name}` skipped: a `command` server is not run in a hosted session"
            )),
            Entry::Http { .. } if connectors.len() >= MAX_HOSTED_CONNECTORS => over_cap += 1,
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
                });
            }
        }
    }
    if over_cap > 0 {
        warnings.push(format!(
            "{FILE}: {over_cap} server(s) skipped: a hosted session takes at most \
             {MAX_HOSTED_CONNECTORS} from the file"
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
    fn hosted_connectors_keep_only_the_url_and_say_what_they_dropped() {
        let (connectors, warnings) = hosted_connectors(FILE_TEXT);
        let mut names: Vec<_> = connectors.iter().map(|c| c.name.as_str()).collect();
        names.sort();
        assert_eq!(names, ["docs", "legacy", "typed"]);
        let docs = connectors.iter().find(|c| c.name == "docs").unwrap();
        assert_eq!(docs.url, "https://example.com/mcp");
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("`files`") && w.contains("command")),
            "{warnings:?}"
        );
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("`explicit`") && w.contains("command")),
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
        );
        assert_eq!(connectors.len(), 1);
        assert_eq!(connectors[0].name, "fine");
        assert_eq!(warnings.len(), 2, "{warnings:?}");
    }

    #[test]
    fn hosted_connectors_stop_at_the_cap_and_say_so() {
        let n = MAX_HOSTED_CONNECTORS + 5;
        let servers: Vec<String> = (0..n)
            .map(|i| format!(r#""s{i}": {{"url": "https://a/{i}"}}"#))
            .collect();
        let (connectors, warnings) =
            hosted_connectors(&format!(r#"{{"mcpServers": {{{}}}}}"#, servers.join(",")));
        assert_eq!(connectors.len(), MAX_HOSTED_CONNECTORS);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("5 server(s) skipped"), "{warnings:?}");
    }
}
