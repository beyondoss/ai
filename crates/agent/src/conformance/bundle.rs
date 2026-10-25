//! The conformance kit for the control plane's bundles and the bundle store: `BEYOND.md`, "What
//! Beyond builds, per part", rows "Control plane: bundles" and "Bundle store", and decision 8 (bundle
//! URLs). Beyond runs it against its own deploy path to prove it meets this repo's contract. Run it
//! with
//!
//! ```sh
//! AGENT_CONFORMANCE_BUNDLE_HOOK=/path/to/operator-hook \
//!   cargo nextest run -p beyond-ai-agent -E 'test(/^bundle_conformance::/)' --no-capture
//! ```
//!
//! - `AGENT_CONFORMANCE_BUNDLE_HOOK`: the operator hook, below.
//! - `AGENT_CONFORMANCE_BUNDLE_FETCH_WINDOW_SECS`: the replica's fetch window (default 120). The URL
//!   check waits out the grant's life plus this window, so a full run takes about that long; set
//!   `AGENT_CONFORMANCE_BUNDLE_WAIT=0` to skip the wait (the expiry check is then skipped, with a
//!   reason).
//!
//! # The operator hook
//!
//! Called as `<hook> <op> <args…>`. Beyond's deploy takes a source (an agent folder) and builds the
//! bundle; this kit hands it the source as a tar file, since a tar is the one shape that can carry
//! every unsafe entry. Each op is justified by the checks that cannot run without it.
//!
//! | op | answer | needed by |
//! |---|---|---|
//! | `deploy <tenant> <source.tar>` | exit 0 and the stored bundle's name (its SHA-256 hex) on stdout when the deploy succeeded; exit 1 when it refused the source; any other exit is a broken hook | unsafe entries, hash, saved tools |
//! | `url <tenant> <name> <ttl_secs>` | the bundle URL a grant of `ttl_secs` would carry | scheme, signature, lifetime, hash |

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use super::{CommandHook, Report, fail, pass, skip};
use crate::exec_endpoint::conformance::Check;
use crate::tools::mcp_manifest::CachedTool;

pub const UNSAFE_REFUSED: &str = "a bundle with an unsafe entry fails at deploy";
pub const HASH_IS_NAME: &str = "a stored bundle hashes to its name";
pub const URL_HTTPS: &str = "a bundle URL is https";
pub const URL_SIGNED: &str = "a bundle URL is signed";
pub const URL_SHORT_LIVED: &str = "a bundle URL lives the grant's life plus the fetch window";
pub const SAVED_TOOLS: &str = "a bundle keeps .mcp-tools.json at its root";

/// Every check, in run order.
pub const CHECKS: [&str; 6] = [
    UNSAFE_REFUSED,
    HASH_IS_NAME,
    URL_HTTPS,
    URL_SIGNED,
    URL_SHORT_LIVED,
    SAVED_TOOLS,
];

/// The operator hook's ops (the module doc has the contract). [`CommandHook`] runs them as a
/// command; a test of the kit implements them in-process.
#[allow(async_fn_in_trait)]
pub trait Hooks {
    /// `Ok(Some(name))`: stored. `Ok(None)`: the deploy refused the source.
    async fn deploy(
        &self,
        tenant: &str,
        source: &std::path::Path,
    ) -> Result<Option<String>, String>;
    async fn url(&self, tenant: &str, name: &str, ttl_secs: u64) -> Result<String, String>;
}

impl Hooks for CommandHook {
    async fn deploy(
        &self,
        tenant: &str,
        source: &std::path::Path,
    ) -> Result<Option<String>, String> {
        let source = source.to_string_lossy();
        match self.run_status("deploy", &[tenant, &source]).await? {
            (0, out) => Ok(Some(out.trim().to_string())),
            (1, _) => Ok(None),
            (code, _) => Err(format!(
                "hook `deploy` exited {code}: neither stored (0) nor refused (1)"
            )),
        }
    }

    async fn url(&self, tenant: &str, name: &str, ttl_secs: u64) -> Result<String, String> {
        let ttl = ttl_secs.to_string();
        Ok(self
            .run("url", &[tenant, name, &ttl])
            .await?
            .trim()
            .to_string())
    }
}

/// What the kit holds the store to.
#[derive(Debug, Clone)]
pub struct BundleTarget {
    /// The replica's fetch window after a grant ends (decision 8: 120 s).
    pub fetch_window: Duration,
    /// How far from the window's edge the expiry probes sit.
    pub slack: Duration,
    /// Wait out the expiry. Off, the expiry check is skipped.
    pub wait: bool,
    /// Accept a plain `http://127.0.0.1` URL: for the reference implementations in this repo, which
    /// cannot terminate TLS. The scheme check is then skipped, with a reason.
    pub loopback_http: bool,
}

impl Default for BundleTarget {
    fn default() -> Self {
        Self {
            fetch_window: Duration::from_secs(120),
            slack: Duration::from_secs(5),
            wait: true,
            loopback_http: false,
        }
    }
}

// ---- tar ------------------------------------------------------------------------------------

/// One tar entry. `kind` is the raw type flag byte: `b'0'` file, `b'5'` directory, `b'2'` symlink,
/// `b'1'` hardlink, `b'3'`/`b'4'` devices, `b'x'` a pax header, `b'L'` a GNU long name.
struct Entry<'a> {
    path: &'a str,
    kind: u8,
    body: &'a [u8],
    link: &'a str,
}

fn file<'a>(path: &'a str, body: &'a str) -> Entry<'a> {
    Entry {
        path,
        kind: b'0',
        body: body.as_bytes(),
        link: "",
    }
}

fn header(path: &str, kind: u8, size: usize, link: &str) -> [u8; 512] {
    let mut h = [0u8; 512];
    h[..path.len().min(100)].copy_from_slice(&path.as_bytes()[..path.len().min(100)]);
    h[100..107].copy_from_slice(b"0000644");
    h[108..115].copy_from_slice(b"0000000");
    h[116..123].copy_from_slice(b"0000000");
    h[124..135].copy_from_slice(format!("{size:011o}").as_bytes());
    h[136..147].copy_from_slice(b"00000000000");
    h[148..156].copy_from_slice(b"        ");
    h[156] = kind;
    h[157..157 + link.len()].copy_from_slice(link.as_bytes());
    h[257..263].copy_from_slice(b"ustar\0");
    h[263..265].copy_from_slice(b"00");
    let sum: u32 = h.iter().map(|b| u32::from(*b)).sum();
    h[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
    h
}

fn tar(entries: &[Entry<'_>]) -> Vec<u8> {
    let mut out = Vec::new();
    for e in entries {
        out.extend_from_slice(&header(e.path, e.kind, e.body.len(), e.link));
        out.extend_from_slice(e.body);
        out.resize(out.len().div_ceil(512) * 512, 0);
    }
    out.resize(out.len() + 1024, 0);
    out
}

/// The source with one hostile entry added to an otherwise good folder.
fn with(entry: Entry<'_>) -> Vec<u8> {
    tar(&[file("AGENTS.md", "be brief\n"), entry])
}

/// A source whose one file claims `bytes` of content: a sparse file, so the cap costs no disk.
fn oversize(path: &std::path::Path, bytes: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut f = std::fs::File::create(path)?;
    f.write_all(&header("big.bin", b'0', bytes as usize, ""))?;
    let padded = bytes.div_ceil(512) * 512;
    f.seek(SeekFrom::Start(512 + padded))?;
    f.write_all(&[0u8; 1024])
}

/// A scratch directory removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(nonce: &str) -> std::io::Result<Self> {
        let dir = std::env::temp_dir().join(format!("bundle-conformance-{nonce}"));
        std::fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---- the checks -----------------------------------------------------------------------------

struct Kit<'a, H> {
    target: &'a BundleTarget,
    hooks: &'a H,
    client: reqwest::Client,
    scratch: &'a std::path::Path,
    tenant: String,
}

/// Run every check against the control plane and store.
pub async fn verify<H: Hooks>(target: &BundleTarget, hooks: &H) -> Report {
    agent_core::ensure_provider();
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    );
    let scratch = match Scratch::new(&nonce) {
        Ok(s) => s,
        Err(e) => {
            return Report {
                kit: "bundles",
                checks: vec![fail(UNSAFE_REFUSED, format!("no scratch directory: {e}"))],
            };
        }
    };
    let kit = Kit {
        target,
        hooks,
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default(),
        scratch: &scratch.0,
        tenant: format!("t-{nonce}"),
    };
    let checks = vec![
        kit.unsafe_refused().await,
        kit.hash_is_name().await,
        kit.url_https().await,
        kit.url_signed().await,
        kit.url_short_lived().await,
        kit.saved_tools().await,
    ];
    Report {
        kit: "bundles",
        checks,
    }
}

/// A good source, deployed.
const GOOD: [(&str, &str); 2] = [
    ("AGENTS.md", "be brief\n"),
    (".agents/skills/s/SKILL.md", "a skill\n"),
];

impl<H: Hooks> Kit<'_, H> {
    fn write(&self, name: &str, bytes: &[u8]) -> Result<PathBuf, String> {
        let p = self.scratch.join(name);
        std::fs::write(&p, bytes).map_err(|e| e.to_string())?;
        Ok(p)
    }

    async fn deploy(&self, name: &str, bytes: &[u8]) -> Result<Option<String>, String> {
        let p = self.write(name, bytes)?;
        self.hooks.deploy(&self.tenant, &p).await
    }

    async fn deploy_good(&self) -> Result<String, String> {
        let bytes = tar(&GOOD.iter().map(|(p, b)| file(p, b)).collect::<Vec<_>>());
        self.deploy("good.tar", &bytes)
            .await?
            .ok_or_else(|| "setup: the deploy refused a safe source".to_string())
    }

    /// `BEYOND.md`: "A bundle with an unsafe entry fails the deploy, not the session open": an
    /// absolute path, `..`, a symlink, a hardlink, a device, a pax or GNU header, or more than the
    /// size cap ([`crate::bundle::MAX_BYTES`]) is refused at deploy. A safe source must be stored,
    /// or refusing everything would pass.
    async fn unsafe_refused(&self) -> Check {
        if let Err(e) = self.deploy_good().await {
            return fail(UNSAFE_REFUSED, e);
        }
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("an absolute path", with(file("/etc/evil", "x"))),
            ("a `..` path", with(file("../evil", "x"))),
            (
                "a path that climbs out midway",
                with(file("a/../../evil", "x")),
            ),
            (
                "a symlink",
                with(Entry {
                    path: "link",
                    kind: b'2',
                    body: b"",
                    link: "/etc/passwd",
                }),
            ),
            (
                "a hardlink",
                with(Entry {
                    path: "link",
                    kind: b'1',
                    body: b"",
                    link: "AGENTS.md",
                }),
            ),
            (
                "a character device",
                with(Entry {
                    path: "dev",
                    kind: b'3',
                    body: b"",
                    link: "",
                }),
            ),
            (
                "a block device",
                with(Entry {
                    path: "dev",
                    kind: b'4',
                    body: b"",
                    link: "",
                }),
            ),
            (
                "a pax header",
                with(Entry {
                    path: "pax",
                    kind: b'x',
                    body: b"",
                    link: "",
                }),
            ),
            (
                "a GNU long-name header",
                with(Entry {
                    path: "gnu",
                    kind: b'L',
                    body: b"",
                    link: "",
                }),
            ),
        ];
        let mut accepted = Vec::new();
        for (i, (what, bytes)) in cases.iter().enumerate() {
            match self.deploy(&format!("unsafe-{i}.tar"), bytes).await {
                Ok(None) => {}
                Ok(Some(_)) => accepted.push(*what),
                Err(e) => return fail(UNSAFE_REFUSED, e),
            }
        }
        let big = self.scratch.join("oversize.tar");
        let cap = crate::bundle::MAX_BYTES as u64;
        match oversize(&big, cap + 1) {
            Ok(()) => match self.hooks.deploy(&self.tenant, &big).await {
                Ok(None) => {}
                Ok(Some(_)) => accepted.push("more than the size cap"),
                Err(e) => return fail(UNSAFE_REFUSED, e),
            },
            Err(e) => {
                return fail(
                    UNSAFE_REFUSED,
                    format!("could not build the oversize source: {e}"),
                );
            }
        }
        if accepted.is_empty() {
            pass(
                UNSAFE_REFUSED,
                format!("{} unsafe sources refused at deploy", cases.len() + 1),
            )
        } else {
            fail(
                UNSAFE_REFUSED,
                format!("the deploy stored a source with: {}", accepted.join(", ")),
            )
        }
    }

    async fn fetch(&self, url: &str) -> Result<(u16, Vec<u8>), String> {
        let r = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("fetching the bundle: {e}"))?;
        let status = r.status().as_u16();
        Ok((status, r.bytes().await.map_err(|e| e.to_string())?.to_vec()))
    }

    /// `BEYOND.md`: "A stored bundle hashes to its name." Fetched through its URL, the bytes hash to
    /// the name the deploy returned, and are a plain tar the agent's own unpacker accepts, holding
    /// the files that were deployed.
    async fn hash_is_name(&self) -> Check {
        let run = async {
            let name = self.deploy_good().await?;
            let url = self.hooks.url(&self.tenant, &name, 600).await?;
            let (status, bytes) = self.fetch(&url).await?;
            Ok::<_, String>((name, status, bytes))
        };
        let (name, status, bytes) = match run.await {
            Ok(v) => v,
            Err(e) => return fail(HASH_IS_NAME, e),
        };
        if status != 200 {
            return fail(HASH_IS_NAME, format!("the bundle URL answered {status}"));
        }
        let got: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if got != name {
            return fail(
                HASH_IS_NAME,
                format!("stored under {name}, but its bytes hash to {got}"),
            );
        }
        let dest = self.scratch.join("unpacked");
        if let Err(e) = std::fs::create_dir_all(&dest)
            .map_err(|e| e.to_string())
            .and_then(|()| crate::bundle::unpack(&bytes, &dest))
        {
            return fail(
                HASH_IS_NAME,
                format!("the agent cannot unpack the stored bundle: {e}"),
            );
        }
        match GOOD
            .iter()
            .find(|(p, b)| std::fs::read_to_string(dest.join(p)).ok().as_deref() != Some(*b))
        {
            Some((p, _)) => fail(
                HASH_IS_NAME,
                format!("{p} is not in the stored bundle as deployed"),
            ),
            None => pass(
                HASH_IS_NAME,
                format!(
                    "{} bytes, sha256 {}…",
                    bytes.len(),
                    &name[..12.min(name.len())]
                ),
            ),
        }
    }

    async fn url_for(&self, ttl: u64) -> Result<String, String> {
        let name = self.deploy_good().await?;
        self.hooks.url(&self.tenant, &name, ttl).await
    }

    /// Decision 8: "Every bundle URL in a grant is https."
    async fn url_https(&self) -> Check {
        match self.url_for(600).await {
            Err(e) => fail(URL_HTTPS, e),
            Ok(u) if u.starts_with("https://") => pass(URL_HTTPS, "https"),
            Ok(u) if self.target.loopback_http && u.starts_with("http://127.0.0.1") => skip(
                URL_HTTPS,
                "a loopback reference implementation cannot terminate TLS",
            ),
            Ok(u) => fail(
                URL_HTTPS,
                format!("the URL is {}", u.split('?').next().unwrap_or(&u)),
            ),
        }
    }

    /// Decision 8: "a short-lived signed link". The link works whole, and not without its signature
    /// or with a changed one.
    async fn url_signed(&self) -> Check {
        let url = match self.url_for(600).await {
            Ok(u) => u,
            Err(e) => return fail(URL_SIGNED, e),
        };
        let Some((base, query)) = url.split_once('?').filter(|(_, q)| !q.is_empty()) else {
            return fail(URL_SIGNED, "the URL carries no signature (it has no query)");
        };
        let mut tampered = query.to_string();
        let last = tampered.pop().unwrap_or('0');
        tampered.push(if last == '0' { '1' } else { '0' });
        let probes = [
            ("whole", url.clone(), true),
            ("without its signature", base.to_string(), false),
            (
                "with a changed signature",
                format!("{base}?{tampered}"),
                false,
            ),
        ];
        for (what, u, ok) in probes {
            match self.fetch(&u).await {
                Ok((200, _)) if !ok => {
                    return fail(URL_SIGNED, format!("the link was served {what}"));
                }
                Ok((status, _)) if ok && status != 200 => {
                    return fail(URL_SIGNED, format!("the link {what} answered {status}"));
                }
                Ok(_) => {}
                Err(e) => return fail(URL_SIGNED, e),
            }
        }
        pass(
            URL_SIGNED,
            "served whole; refused unsigned and with a changed signature",
        )
    }

    /// Decision 8: "valid for the grant's life plus the replica's 120 s fetch window". A link for a
    /// 2 s grant still works just inside that window and not just outside it.
    async fn url_short_lived(&self) -> Check {
        if !self.target.wait {
            return skip(
                URL_SHORT_LIVED,
                "AGENT_CONFORMANCE_BUNDLE_WAIT=0: not waiting out the expiry",
            );
        }
        let ttl = Duration::from_secs(2);
        let started = Instant::now();
        let url = match self.url_for(ttl.as_secs()).await {
            Ok(u) => u,
            Err(e) => return fail(URL_SHORT_LIVED, e),
        };
        let (inside, outside) = (
            ttl + self.target.fetch_window - self.target.slack,
            ttl + self.target.fetch_window + self.target.slack,
        );
        for (when, expect_ok) in [(inside, true), (outside, false)] {
            tokio::time::sleep_until(tokio::time::Instant::from_std(started + when)).await;
            match self.fetch(&url).await {
                Ok((200, _)) if !expect_ok => {
                    return fail(
                        URL_SHORT_LIVED,
                        format!(
                            "still served {:?} after a {ttl:?} grant (window {:?})",
                            started.elapsed(),
                            self.target.fetch_window
                        ),
                    );
                }
                Ok((status, _)) if expect_ok && status != 200 => {
                    return fail(
                        URL_SHORT_LIVED,
                        format!(
                            "refused ({status}) at {:?}, inside the grant's life plus the window",
                            started.elapsed()
                        ),
                    );
                }
                Ok(_) => {}
                Err(e) => return fail(URL_SHORT_LIVED, e),
            }
        }
        pass(
            URL_SHORT_LIVED,
            format!("served until {inside:?}, refused by {outside:?}"),
        )
    }

    /// `BEYOND.md`: "Save `.mcp-tools.json` at its root for in-sandbox MCP servers". A source that
    /// carries the file is stored with it at the root, intact, in the shape the replica reads
    /// (`{"<connector>": [tool…]}`), so the session opens from it without dialing the server. (That
    /// opening makes no exec and no MCP request is the agent's own, proven in
    /// `tests/serve_service_bundle_mcp_tools.rs`.)
    async fn saved_tools(&self) -> Check {
        let tools = r#"{"box":[{"remote_name":"echo","description":"echo","input_schema":{"type":"object"}}]}"#;
        let bytes = tar(&[
            file("AGENTS.md", "be brief\n"),
            file(
                ".mcp.json",
                r#"{"mcpServers":{"box":{"url":"http://sandbox.invalid/mcp"}}}"#,
            ),
            file(".mcp-tools.json", tools),
        ]);
        let run = async {
            let name = self
                .deploy("tools.tar", &bytes)
                .await?
                .ok_or("the deploy refused a source with .mcp-tools.json")?;
            let url = self.hooks.url(&self.tenant, &name, 600).await?;
            let (status, body) = self.fetch(&url).await?;
            if status != 200 {
                return Err(format!("the bundle URL answered {status}"));
            }
            let dest = self.scratch.join("tools-unpacked");
            std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;
            crate::bundle::unpack(&body, &dest)?;
            std::fs::read_to_string(dest.join(".mcp-tools.json"))
                .map_err(|_| ".mcp-tools.json is not at the bundle's root".to_string())
        };
        match run.await {
            Err(e) => fail(SAVED_TOOLS, e),
            Ok(saved) => match serde_json::from_str::<BTreeMap<String, Vec<CachedTool>>>(&saved) {
                Ok(m)
                    if m.get("box")
                        .is_some_and(|t| t.len() == 1 && t[0].remote_name == "echo") =>
                {
                    pass(SAVED_TOOLS, "at the root, in the shape the replica reads")
                }
                Ok(_) => fail(SAVED_TOOLS, ".mcp-tools.json was changed by the deploy"),
                Err(e) => fail(
                    SAVED_TOOLS,
                    format!(".mcp-tools.json no longer parses as the replica reads it: {e}"),
                ),
            },
        }
    }
}
