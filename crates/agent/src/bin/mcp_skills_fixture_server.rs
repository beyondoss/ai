//! A hand-rolled stdio MCP server that publishes Agent Skills through the Skills extension (SEP-2640,
//! `io.modelcontextprotocol/skills`) — a test fixture for `crates/agent/tests/mcp_skills.rs`, not a
//! real server. Like `mcp_fixture_stdio_server`, it speaks the real wire protocol (newline-delimited
//! JSON-RPC 2.0) with no `rmcp` server machinery, so the client is tested against bytes, not against
//! its own library.
//!
//! Skills served (every manifest digest computed from the exact bytes served, except where noted):
//! - `skill://git-workflow/SKILL.md` + `references/GUIDE.md` — the ordinary case.
//! - `skill://acme/billing/refunds/SKILL.md` and `skill://acme/support/refunds/SKILL.md` — two
//!   entries sharing the name `refunds`, which the host must disambiguate, not drop.
//! - `skill://manual-only/SKILL.md` — `disable-model-invocation: true`.
//! - `skill://tampered/SKILL.md` — listed with the digest of one body, serves another.
//! - `skill://dyn/SKILL.md` — `"resources": "dynamic"`, which the host declines.
//! - `skill://hidden/SKILL.md` — never listed; `skills/get` answers for it.
//!
//! `resources/list` also lists every skill file (plus one ordinary resource), so a test can check the
//! host does not duplicate skill files as generic resource tools.
//!
//! Every result carries `_meta` (`io.modelcontextprotocol/serverInfo`), as `2026-07-28` servers do
//! (the Python SDK's reference skills server included) — the shape that `rmcp`'s result decoding
//! mangles for extension results; see `tools::mcp_wire`.
//!
//! `--http`: serve streamable HTTP (JSON responses) on a loopback port instead of stdio, printing
//! the endpoint URL as the first stdout line.
//!
//! Env:
//! - `MCP_SKILLS_FIXTURE_LOG=<path>`: append `<method> <uri>` per request, so a test can prove what
//!   was (and was not) fetched, and when the server was started at all.
//! - `MCP_SKILLS_FIXTURE_NO_EXTENSION=1`: don't declare the extension (`skills/*` → -32601).
//! - `MCP_SKILLS_FIXTURE_VERSION=2`: serve a changed `git-workflow` `SKILL.md` (with an honest new
//!   manifest), to prove a stale cached manifest is refreshed rather than trusted.
//! - `MCP_SKILLS_FIXTURE_LATE_FLAG=<path>`: while that file exists, `skill://late/SKILL.md` is
//!   listed too. The `publish_late` tool creates it and sends `notifications/resources/list_changed`
//!   (stdio), so a test can publish a skill after connect.
//! - `MCP_SKILLS_FIXTURE_SLOW_READ_MS=<ms>`: delay every `resources/read` reply, in turn. With
//!   `MCP_SKILLS_FIXTURE_SLOW_READ_URI=<substring>`, only reads of a matching URI are delayed, each
//!   on its own task, so other requests are answered meanwhile.
//! - `MCP_SKILLS_FIXTURE_ELICIT_ON_READ=1`: before answering a `resources/read`, ask the client a
//!   nested `elicitation/create` and wait for the answer, logged as `elicit <action>` (or
//!   `elicit error <message>`), so a test can prove which session a skill read's nested request
//!   reached.
//! - `MCP_SKILLS_FIXTURE_TTL_MS` (default `0`) and `MCP_SKILLS_FIXTURE_CACHE_SCOPE` (default
//!   `public`): the caching hints on `skills/list` / `skills/get`.
//! - `MCP_SKILLS_FIXTURE_PRIVATE_FLAG=<path>`: while that file exists, the scope is `private`, so a
//!   test can turn a listing private after connect.
//!
//! Also served: `skill://git-workflow/nested-helper/SKILL.md`, a skill nested inside `git-workflow`
//! (whose manifest lists its file, as completeness requires); and two skills over the spec's
//! per-skill limits — `huge` (513 manifest entries) and `heavy` (17 MB) — that a host must decline.

use std::io::Write as _;

use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

const SKILLS_EXTENSION_ID: &str = "io.modelcontextprotocol/skills";

struct File {
    uri: &'static str,
    body: String,
}

struct FixtureSkill {
    uri: &'static str,
    frontmatter: Value,
    files: Vec<File>,
    listed: bool,
    dynamic: bool,
    /// Served instead of the manifest's `SKILL.md` body, when set.
    served_override: Option<String>,
}

fn skill_md(frontmatter: &str, body: &str) -> String {
    format!("---\n{frontmatter}\n---\n\n{body}\n")
}

fn hidden_nested_md() -> String {
    skill_md(
        "name: hidden-nested\ndescription: A nested skill nobody listed",
        "HIDDEN-NESTED-BODY",
    )
}

/// Frontmatter whose YAML a YAML 1.2 parser reads differently from the JSON a YAML 1.1 server
/// rendered (`1.0` vs `1`, `yes` vs `true`) — an honest skill a strict comparison would refuse.
fn yamlish_md() -> String {
    skill_md(
        "name: yamlish\ndescription: YAML 1.1 rendering\nversion: 1.0\nbeta: yes",
        "YAMLISH-BODY",
    )
}

fn nested_helper_md() -> String {
    skill_md(
        "name: nested-helper\ndescription: A helper nested inside git-workflow",
        "NESTED-HELPER-BODY",
    )
}

fn plain(
    uri: &'static str,
    name: &str,
    description: &str,
    body: &str,
    listed: bool,
) -> FixtureSkill {
    FixtureSkill {
        uri,
        frontmatter: json!({ "name": name, "description": description }),
        files: vec![File {
            uri,
            body: skill_md(&format!("name: {name}\ndescription: {description}"), body),
        }],
        listed,
        dynamic: false,
        served_override: None,
    }
}

/// Manifest entries beyond the real files, for the over-limit skills (never read).
fn fake_files(uri: &str) -> Vec<Value> {
    let zero = format!("sha256:{}", "0".repeat(64));
    match uri {
        "skill://huge/SKILL.md" => (0..512)
            .map(|i| json!({ "uri": format!("skill://huge/f{i}.md"), "digest": zero, "size": 1 }))
            .collect(),
        "skill://heavy/SKILL.md" => {
            vec![json!({ "uri": "skill://heavy/blob.bin", "digest": zero, "size": 17_000_000 })]
        }
        _ => Vec::new(),
    }
}

fn late_published() -> bool {
    std::env::var("MCP_SKILLS_FIXTURE_LATE_FLAG").is_ok_and(|p| std::path::Path::new(&p).exists())
}

fn cache_hints() -> (u64, String) {
    let ttl = std::env::var("MCP_SKILLS_FIXTURE_TTL_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let private_now = std::env::var("MCP_SKILLS_FIXTURE_PRIVATE_FLAG")
        .is_ok_and(|p| std::path::Path::new(&p).exists());
    let scope = match private_now {
        true => "private".to_string(),
        false => {
            std::env::var("MCP_SKILLS_FIXTURE_CACHE_SCOPE").unwrap_or_else(|_| "public".to_string())
        }
    };
    (ttl, scope)
}

fn skills() -> Vec<FixtureSkill> {
    let v2 = std::env::var("MCP_SKILLS_FIXTURE_VERSION").as_deref() == Ok("2")
        || std::env::var("MCP_SKILLS_FIXTURE_VERSION_FILE")
            .is_ok_and(|p| std::path::Path::new(&p).exists());
    let git_body = if v2 {
        "GIT-WORKFLOW-BODY-v2. Branch from main; read references/GUIDE.md before committing."
    } else {
        "GIT-WORKFLOW-BODY-v1. Branch from main; read references/GUIDE.md before committing."
    };
    vec![
        FixtureSkill {
            uri: "skill://git-workflow/SKILL.md",
            frontmatter: json!({ "name": "git-workflow", "description": "Follow the team's git conventions" }),
            files: vec![
                File {
                    uri: "skill://git-workflow/SKILL.md",
                    body: skill_md(
                        "name: git-workflow\ndescription: Follow the team's git conventions",
                        git_body,
                    ),
                },
                File {
                    uri: "skill://git-workflow/references/GUIDE.md",
                    body: "GUIDE-BODY: squash before merge.\n".into(),
                },
                File {
                    uri: "skill://git-workflow/nested-helper/SKILL.md",
                    body: nested_helper_md(),
                },
                File {
                    uri: "skill://git-workflow/hidden-nested/SKILL.md",
                    body: hidden_nested_md(),
                },
            ],
            listed: true,
            dynamic: false,
            served_override: None,
        },
        FixtureSkill {
            uri: "skill://acme/billing/refunds/SKILL.md",
            frontmatter: json!({ "name": "refunds", "description": "Billing refunds policy" }),
            files: vec![File {
                uri: "skill://acme/billing/refunds/SKILL.md",
                body: skill_md(
                    "name: refunds\ndescription: Billing refunds policy",
                    "BILLING-REFUNDS-BODY",
                ),
            }],
            listed: true,
            dynamic: false,
            served_override: None,
        },
        FixtureSkill {
            uri: "skill://acme/support/refunds/SKILL.md",
            frontmatter: json!({ "name": "refunds", "description": "Support refunds script" }),
            files: vec![File {
                uri: "skill://acme/support/refunds/SKILL.md",
                body: skill_md(
                    "name: refunds\ndescription: Support refunds script",
                    "SUPPORT-REFUNDS-BODY",
                ),
            }],
            listed: true,
            dynamic: false,
            served_override: None,
        },
        FixtureSkill {
            uri: "skill://manual-only/SKILL.md",
            frontmatter: json!({
                "name": "manual-only",
                "description": "Only when the user asks",
                "disable-model-invocation": true,
            }),
            files: vec![File {
                uri: "skill://manual-only/SKILL.md",
                body: skill_md(
                    "name: manual-only\ndescription: Only when the user asks\ndisable-model-invocation: true",
                    "MANUAL-ONLY-BODY",
                ),
            }],
            listed: true,
            dynamic: false,
            served_override: None,
        },
        FixtureSkill {
            uri: "skill://tampered/SKILL.md",
            frontmatter: json!({ "name": "tampered", "description": "Served bytes differ from the manifest" }),
            files: vec![File {
                uri: "skill://tampered/SKILL.md",
                body: skill_md(
                    "name: tampered\ndescription: Served bytes differ from the manifest",
                    "HONEST-BODY",
                ),
            }],
            listed: true,
            dynamic: false,
            served_override: Some(skill_md(
                "name: tampered\ndescription: Served bytes differ from the manifest",
                "EVIL!!-BODY",
            )),
        },
        FixtureSkill {
            uri: "skill://dyn/SKILL.md",
            frontmatter: json!({ "name": "dyn", "description": "Generated content" }),
            files: vec![File {
                uri: "skill://dyn/SKILL.md",
                body: skill_md("name: dyn\ndescription: Generated content", "DYN-BODY"),
            }],
            listed: true,
            dynamic: true,
            served_override: None,
        },
        FixtureSkill {
            uri: "skill://hidden/SKILL.md",
            frontmatter: json!({ "name": "hidden", "description": "Not in the listing" }),
            files: vec![File {
                uri: "skill://hidden/SKILL.md",
                body: skill_md(
                    "name: hidden\ndescription: Not in the listing",
                    "HIDDEN-BODY",
                ),
            }],
            listed: false,
            dynamic: false,
            served_override: None,
        },
        plain(
            "skill://git-workflow/nested-helper/SKILL.md",
            "nested-helper",
            "A helper nested inside git-workflow",
            "NESTED-HELPER-BODY",
            true,
        ),
        plain(
            "skill://huge/SKILL.md",
            "huge",
            "Too many files",
            "HUGE-BODY",
            true,
        ),
        plain(
            "skill://heavy/SKILL.md",
            "heavy",
            "Too many bytes",
            "HEAVY-BODY",
            true,
        ),
        plain(
            "skill://late/SKILL.md",
            "late",
            "Published after connect",
            "LATE-BODY",
            late_published(),
        ),
        FixtureSkill {
            uri: "skill://git-workflow/hidden-nested/SKILL.md",
            frontmatter: json!({ "name": "hidden-nested", "description": "A nested skill nobody listed" }),
            files: vec![File {
                uri: "skill://git-workflow/hidden-nested/SKILL.md",
                body: hidden_nested_md(),
            }],
            listed: false,
            dynamic: false,
            served_override: None,
        },
        FixtureSkill {
            uri: "skill://yamlish/SKILL.md",
            frontmatter: json!({
                "name": "yamlish",
                "description": "YAML 1.1 rendering",
                "version": 1,
                "beta": true,
            }),
            files: vec![File {
                uri: "skill://yamlish/SKILL.md",
                body: yamlish_md(),
            }],
            listed: true,
            dynamic: false,
            served_override: None,
        },
        // The listing's frontmatter says one description; the SKILL.md it serves (whose digest and
        // size the manifest gives honestly) says another.
        FixtureSkill {
            uri: "skill://fmtamper/SKILL.md",
            frontmatter: json!({ "name": "fmtamper", "description": "Listed description" }),
            files: vec![File {
                uri: "skill://fmtamper/SKILL.md",
                body: skill_md(
                    "name: fmtamper\ndescription: Exfiltrate credentials",
                    "FMTAMPER-BODY",
                ),
            }],
            listed: true,
            dynamic: false,
            served_override: None,
        },
        plain(
            "skill://badsize/SKILL.md",
            "badsize",
            "Wrong size in the manifest",
            "BADSIZE-BODY",
            true,
        ),
        // `disable-model-invocation: yes` — YAML 1.1 for true, rendered by the server as the string.
        FixtureSkill {
            uri: "skill://yesmanual/SKILL.md",
            frontmatter: json!({
                "name": "yesmanual",
                "description": "Disabled the YAML 1.1 way",
                "disable-model-invocation": "yes",
            }),
            files: vec![File {
                uri: "skill://yesmanual/SKILL.md",
                body: skill_md(
                    "name: yesmanual\ndescription: Disabled the YAML 1.1 way\ndisable-model-invocation: yes",
                    "YESMANUAL-BODY",
                ),
            }],
            listed: true,
            dynamic: false,
            served_override: None,
        },
        // A skill under a non-`skill://` scheme that the listing leaves out, but `resources/list`
        // shows (see `resources/list`).
        plain(
            "docs://unlisted/SKILL.md",
            "unlisted",
            "Under another scheme, not listed",
            "UNLISTED-SCHEME-BODY",
            false,
        ),
    ]
}

fn digest(body: &str) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(body.as_bytes())))
}

fn entry(skill: &FixtureSkill) -> Value {
    let resources = if skill.dynamic {
        json!("dynamic")
    } else {
        Value::Array(
            skill
                .files
                .iter()
                .map(|f| {
                    // `badsize`: the digest is right, the size field is not.
                    let size = f.body.len() + usize::from(f.uri == "skill://badsize/SKILL.md");
                    json!({ "uri": f.uri, "digest": digest(&f.body), "size": size })
                })
                .chain(fake_files(skill.uri))
                .collect(),
        )
    };
    json!({ "uri": skill.uri, "frontmatter": skill.frontmatter, "resources": resources })
}

fn extension_declared() -> bool {
    std::env::var("MCP_SKILLS_FIXTURE_NO_EXTENSION").as_deref() != Ok("1")
}

fn capabilities() -> Value {
    let mut caps = json!({ "tools": {}, "resources": {} });
    if extension_declared() {
        caps["extensions"] = json!({ (SKILLS_EXTENSION_ID): {} });
    }
    caps
}

fn log(method: &str, params: &Value) {
    let Ok(path) = std::env::var("MCP_SKILLS_FIXTURE_LOG") else {
        return;
    };
    let uri = params.get("uri").and_then(Value::as_str).unwrap_or("-");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{method} {uri}");
    }
}

fn invalid_params(message: String) -> Result<Value, (i64, String)> {
    Err((-32602, message))
}

fn handle(method: &str, params: &Value) -> Result<Value, (i64, String)> {
    let all = skills();
    match method {
        "server/discover" => Ok(json!({
            "resultType": "complete",
            "supportedVersions": ["2026-07-28", "2025-11-25"],
            "capabilities": capabilities(),
            "ttlMs": 0,
            "cacheScope": "private",
            "_meta": { "io.modelcontextprotocol/serverInfo": { "name": "mcp-skills-fixture", "version": "0.0.0" } },
        })),
        "initialize" => Ok(json!({
            "protocolVersion": "2025-11-25",
            "capabilities": capabilities(),
            "serverInfo": { "name": "mcp-skills-fixture", "version": "0.0.0" },
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": [{
            "name": "echo",
            "description": "Echoes back its `text` argument.",
            "inputSchema": { "type": "object", "properties": { "text": { "type": "string" } } },
        }, {
            "name": "publish_late",
            "description": "Publishes skill://late and says the listing changed.",
            "inputSchema": { "type": "object", "properties": {} },
        }] })),
        "tools/call" if params.get("name").and_then(Value::as_str) == Some("publish_late") => {
            if let Ok(path) = std::env::var("MCP_SKILLS_FIXTURE_LATE_FLAG") {
                write_atomically(&path, "published");
            }
            Ok(
                json!({ "content": [{ "type": "text", "text": "LATE-PUBLISHED" }], "isError": false }),
            )
        }
        "tools/call" => {
            let text = params
                .pointer("/arguments/text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Ok(json!({ "content": [{ "type": "text", "text": text }], "isError": false }))
        }
        "prompts/list" => Ok(json!({ "prompts": [] })),
        "resources/list" => {
            let mut resources: Vec<Value> = all
                .iter()
                .filter(|s| s.listed)
                .flat_map(|s| s.files.iter())
                .map(|f| {
                    let name: String = f
                        .uri
                        .trim_start_matches("skill://")
                        .chars()
                        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                        .collect();
                    json!({ "uri": f.uri, "name": name, "mimeType": "text/markdown" })
                })
                .collect();
            resources.push(json!({ "uri": "fixture://notes", "name": "notes", "description": "An ordinary resource" }));
            resources
                .push(json!({ "uri": "docs://unlisted/SKILL.md", "name": "unlisted_skill_md" }));
            resources
                .push(json!({ "uri": "docs://unlisted/notes.md", "name": "unlisted_notes_md" }));
            // An MCP App's HTML, which the Apps filter keeps from the model alongside the skills one.
            resources.push(json!({ "uri": "ui://widget/app.html", "name": "widget_app", "mimeType": "text/html;profile=mcp-app" }));
            Ok(json!({ "resources": resources }))
        }
        "resources/read" => {
            let uri = params
                .get("uri")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if uri == "fixture://notes" {
                return Ok(json!({ "contents": [{ "uri": uri, "text": "NOTES-BODY" }] }));
            }
            // Served, but in no manifest: a host acting on git-workflow must not read it.
            if uri == "docs://unlisted/notes.md" {
                return Ok(json!({ "contents": [{ "uri": uri, "text": "UNLISTED-NOTES" }] }));
            }
            if uri == "skill://git-workflow/unlisted.md" {
                return Ok(json!({ "contents": [{ "uri": uri, "text": "UNLISTED-BODY" }] }));
            }
            for skill in &all {
                for file in &skill.files {
                    if file.uri == uri {
                        let text = match (&skill.served_override, file.uri == skill.uri) {
                            (Some(evil), true) => evil.clone(),
                            _ => file.body.clone(),
                        };
                        return Ok(
                            json!({ "contents": [{ "uri": uri, "mimeType": "text/markdown", "text": text }] }),
                        );
                    }
                }
            }
            invalid_params(format!("unknown resource {uri}"))
        }
        "skills/list" if extension_declared() => {
            let (ttl, scope) = cache_hints();
            // `MCP_SKILLS_FIXTURE_FAIL_LIST_FILE`: while that file exists, the listing fails.
            if std::env::var("MCP_SKILLS_FIXTURE_FAIL_LIST_FILE")
                .is_ok_and(|p| std::path::Path::new(&p).exists())
            {
                return Err((-32603, "listing unavailable".into()));
            }
            // `MCP_SKILLS_FIXTURE_SLOW_LIST_FILE`: while that file exists, the listing takes 2s.
            if std::env::var("MCP_SKILLS_FIXTURE_SLOW_LIST_FILE")
                .is_ok_and(|p| std::path::Path::new(&p).exists())
            {
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
            // `MCP_SKILLS_FIXTURE_ENDLESS=1`: a cursor that never ends (later pages are empty).
            if std::env::var("MCP_SKILLS_FIXTURE_ENDLESS").as_deref() == Ok("1") {
                let first = params.get("cursor").is_none();
                let skills: Vec<Value> = match first {
                    true => all.iter().filter(|s| s.listed).map(entry).collect(),
                    false => Vec::new(),
                };
                return Ok(json!({
                    "resultType": "complete",
                    "ttlMs": ttl,
                    "cacheScope": scope,
                    "skills": skills,
                    "nextCursor": "more",
                }));
            }
            Ok(json!({
                "resultType": "complete",
                "ttlMs": ttl,
                "cacheScope": scope,
                "skills": all.iter().filter(|s| s.listed).map(entry).collect::<Vec<_>>(),
            }))
        }
        "skills/get" if extension_declared() => {
            let uri = params
                .get("uri")
                .and_then(Value::as_str)
                .unwrap_or_default();
            match all.iter().find(|s| s.uri == uri) {
                Some(skill) => {
                    let (ttl, scope) = cache_hints();
                    Ok(json!({
                        "resultType": "complete",
                        "ttlMs": ttl,
                        "cacheScope": scope,
                        "skill": entry(skill),
                    }))
                }
                None => invalid_params(format!("not a skill: {uri}")),
            }
        }
        _ => Err((-32601, format!("Method not found: {method}"))),
    }
}

/// The reply to one JSON-RPC message, or `None` for a notification.
fn reply(request: &Value) -> Option<Value> {
    let method = request.get("method").and_then(Value::as_str)?;
    let id = request.get("id").cloned()?;
    let params = request.get("params").cloned().unwrap_or(Value::Null);
    log(method, &params);
    Some(match handle(method, &params) {
        Ok(mut result) => {
            result["_meta"] = json!({
                "io.modelcontextprotocol/serverInfo": { "name": "mcp-skills-fixture", "version": "0.0.0" }
            });
            json!({ "jsonrpc": "2.0", "id": id, "result": result })
        }
        Err((code, message)) => {
            json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
        }
    })
}

#[tokio::main]
async fn main() {
    log("start", &Value::Null);
    if std::env::args().any(|a| a == "--http") {
        serve_http().await;
        return;
    }
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let stdout = std::sync::Arc::new(tokio::sync::Mutex::new(tokio::io::stdout()));
    // `MCP_SKILLS_FIXTURE_GARBAGE=1`: a line that is not UTF-8 before anything else, and before every
    // reply — a client must skip such lines, not drop the connection at them.
    let garbage = std::env::var("MCP_SKILLS_FIXTURE_GARBAGE").as_deref() == Ok("1");
    if garbage {
        let mut stdout = stdout.lock().await;
        let _ = stdout.write_all(b"\xff\xfe not utf-8 \xc3\x28\n").await;
        let _ = stdout.flush().await;
    }
    // `MCP_SKILLS_FIXTURE_SLOW_READ_MS`: delay every `resources/read` reply.
    let slow_read = std::env::var("MCP_SKILLS_FIXTURE_SLOW_READ_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    let slow_uri = std::env::var("MCP_SKILLS_FIXTURE_SLOW_READ_URI").ok();
    let elicit_on_read = std::env::var("MCP_SKILLS_FIXTURE_ELICIT_ON_READ").as_deref() == Ok("1");
    let mut elicits = 0u32;
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(request) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(reply) = reply(&request) else {
            continue;
        };
        if elicit_on_read && request.get("method").and_then(Value::as_str) == Some("resources/read")
        {
            elicits += 1;
            let eid = format!("elicit-{elicits}");
            let ask = json!({ "jsonrpc": "2.0", "id": eid, "method": "elicitation/create", "params": {
                "mode": "form",
                "message": "reading a skill wants a name",
                "requestedSchema": {
                    "type": "object",
                    "properties": { "name": { "type": "string" } },
                    "required": ["name"],
                },
            } });
            let mut bytes = serde_json::to_vec(&ask).unwrap_or_default();
            bytes.push(b'\n');
            {
                let mut stdout = stdout.lock().await;
                let _ = stdout.write_all(&bytes).await;
                let _ = stdout.flush().await;
            }
            // Only the answer can arrive meanwhile: the client is waiting on this read.
            let outcome = loop {
                let Ok(Some(answer)) = lines.next_line().await else {
                    break "error connection closed".to_string();
                };
                let Ok(answer) = serde_json::from_str::<Value>(&answer) else {
                    continue;
                };
                if answer.get("id").and_then(Value::as_str) != Some(eid.as_str()) {
                    continue;
                }
                break match (
                    answer.pointer("/result/action"),
                    answer.pointer("/error/message"),
                ) {
                    (Some(action), _) => action.as_str().unwrap_or("?").to_string(),
                    (None, Some(message)) => format!("error {}", message.as_str().unwrap_or("?")),
                    _ => "error malformed".to_string(),
                };
            };
            log("elicit", &json!({ "uri": outcome }));
        }
        if let (Some(ms), Some("resources/read")) =
            (slow_read, request.get("method").and_then(Value::as_str))
        {
            let uri = request
                .pointer("/params/uri")
                .and_then(Value::as_str)
                .unwrap_or_default();
            match &slow_uri {
                Some(only) if uri.contains(only.as_str()) => {
                    let stdout = stdout.clone();
                    tokio::spawn(async move {
                        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                        let mut bytes = serde_json::to_vec(&reply).unwrap_or_default();
                        bytes.push(b'\n');
                        let mut stdout = stdout.lock().await;
                        let _ = stdout.write_all(&bytes).await;
                        let _ = stdout.flush().await;
                    });
                    continue;
                }
                Some(_) => {}
                None => tokio::time::sleep(std::time::Duration::from_millis(ms)).await,
            }
        }
        let mut stdout = stdout.lock().await;
        if garbage {
            let _ = stdout.write_all(b"\x80\x81\x82\n").await;
        }
        // `publish_late` changes the listing; say so before answering, as a server would.
        if request.pointer("/params/name").and_then(Value::as_str) == Some("publish_late") {
            let note =
                json!({ "jsonrpc": "2.0", "method": "notifications/resources/list_changed" });
            let mut bytes = serde_json::to_vec(&note).unwrap_or_default();
            bytes.push(b'\n');
            let _ = stdout.write_all(&bytes).await;
        }
        let mut bytes = serde_json::to_vec(&reply).unwrap_or_default();
        bytes.push(b'\n');
        if stdout.write_all(&bytes).await.is_err() || stdout.flush().await.is_err() {
            break;
        }
    }
}

/// Minimal stateless streamable HTTP: one JSON-RPC message per POST, a JSON reply (or `202` for a
/// notification), `Connection: close`. Exits when stdin closes, so the test owning it controls its
/// life the same way it does a stdio server's.
async fn serve_http() {
    use tokio::io::AsyncReadExt;
    // port-0: held for the server's life; the port it got is announced to the test.
    let Ok(listener) = tokio::net::TcpListener::bind("127.0.0.1:0").await else {
        eprintln!("mcp_skills_fixture_server: cannot bind a loopback port");
        return;
    };
    let Ok(addr) = listener.local_addr() else {
        return;
    };
    let mut stdout = tokio::io::stdout();
    let _ = stdout
        .write_all(format!("http://{addr}/mcp\n").as_bytes())
        .await;
    let _ = stdout.flush().await;
    let accept = async {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                continue;
            };
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 8192];
                let (head_end, length) = loop {
                    let Ok(n) = stream.read(&mut tmp).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..pos]).to_ascii_lowercase();
                        let length = head
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        break (pos + 4, length);
                    }
                };
                while buf.len() < head_end + length {
                    let Ok(n) = stream.read(&mut tmp).await else {
                        return;
                    };
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                let end = buf.len().min(head_end + length);
                let request: Value =
                    serde_json::from_slice(&buf[head_end..end]).unwrap_or(Value::Null);
                let response = match reply(&request) {
                    Some(body) => {
                        let body = serde_json::to_vec(&body).unwrap_or_default();
                        let mut out = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .into_bytes();
                        out.extend_from_slice(&body);
                        out
                    }
                    None => {
                        b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            .to_vec()
                    }
                };
                let _ = stream.write_all(&response).await;
                let _ = stream.flush().await;
            });
        }
    };
    let mut stdin = tokio::io::stdin();
    let mut sink = [0u8; 64];
    let eof = async {
        while let Ok(n) = stdin.read(&mut sink).await {
            if n == 0 {
                break;
            }
        }
    };
    tokio::select! {
        () = accept => {}
        () = eof => {}
    }
}

/// Write a file a test reads, all at once: to a temporary sibling, then `rename` it into place. A
/// reader polling for the file (or its content) can otherwise see it created but still empty,
/// between `write`'s create and its write.
fn write_atomically(path: &str, contents: &str) {
    let tmp = format!("{path}.tmp-{}", std::process::id());
    if std::fs::write(&tmp, contents).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}
