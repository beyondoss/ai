//! The reference check for `sdk/conformance`: every scenario file is replayed through the agent's own
//! `ShellFs` + `HttpExecRunner` (the code the spec was derived from) against a loopback `execd`, in
//! several backend profiles. If this passes, the scenarios say what the agent does today; a client in
//! another language that passes the same files does what the agent does.
//!
//! The evaluator is `crates/sandbox-conformance` (the one the `sandbox-matrix` binary also uses); this
//! file supplies the driver in-process and the backend profiles. The driver it feeds is in
//! `sdk_conformance_driver.rs`.
//!
//! A scenario marked `planned` is skipped, visibly, and never counted as a pass.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use base64::Engine as _;
use beyond_test_support::exec_mock::{ExecMock, MockOptions, Protocol};
use sandbox_conformance::{Backend, Connector, Link, Status, Tally, run_scenario};
use serde_json::{Value, json};

use crate::sdk_conformance_driver::{B64, CapsOverride, Config, Driver};

const BEARER: &str = "conformance-secret";

fn scenario_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../sdk/conformance/scenarios")
}

/// A backend configuration: what the endpoint is, and which probe results the driver is told to
/// believe instead (`CONFORMANCE_CAPS`), so one machine can walk every rung of the fallback ladder.
struct Profile {
    name: &'static str,
    options: MockOptions,
    bearer: bool,
    caps: CapsOverride,
    /// The suites this profile runs. The two profiles served by the mock's own (deliberately
    /// imperfect) endpoint exist to prove the client's probes and fallbacks, so they run the
    /// suites that go through those and leave the endpoint-behaviour suites to execd.
    suites: &'static [&'static str],
}

fn profiles() -> Vec<Profile> {
    let base = || Profile {
        name: "",
        options: MockOptions::default(),
        bearer: false,
        caps: CapsOverride::default(),
        suites: &[],
    };
    vec![
        Profile {
            name: "execd",
            ..base()
        },
        Profile {
            name: "execd-bearer",
            bearer: true,
            ..base()
        },
        Profile {
            // A v1 endpoint: ignores `stdin_base64`, so the probe finds no stdin by itself.
            name: "v1-no-stdin",
            options: MockOptions {
                honors_stdin: false,
                ..MockOptions::default()
            },
            suites: &["files", "errors", "helpers"],
            ..base()
        },
        Profile {
            // An endpoint that never learned streaming: `stream` is an unknown field to it.
            name: "v11-endpoint",
            options: MockOptions {
                protocol: Protocol::V11,
                ..MockOptions::default()
            },
            suites: &["files", "errors", "helpers"],
            ..base()
        },
        Profile {
            name: "execd-no-stdin",
            caps: CapsOverride {
                stdin: Some(false),
                ..CapsOverride::default()
            },
            ..base()
        },
        Profile {
            name: "execd-grep",
            caps: CapsOverride {
                search_engine: Some("grep"),
                ..CapsOverride::default()
            },
            ..base()
        },
        Profile {
            name: "execd-busybox-grep",
            caps: CapsOverride {
                search_engine: Some("busybox-grep"),
                find_printf: Some(false),
                ..CapsOverride::default()
            },
            ..base()
        },
        Profile {
            // Native jobs off: the endpoint offers plain exec only, so jobs are the baseline.
            name: "execd-baseline-jobs",
            options: MockOptions {
                jobs: false,
                ..MockOptions::default()
            },
            ..base()
        },
    ]
}

/// The in-process driver, as the evaluator's link.
struct Reference(Driver);

#[async_trait]
impl Link for Reference {
    async fn call(&self, req: &Value) -> Value {
        self.0.handle(req).await
    }
}

struct Reconnect(Config);

#[async_trait]
impl Connector for Reconnect {
    async fn connect(&self, bearer: Option<Option<String>>, url: Option<String>) -> Box<dyn Link> {
        let mut cfg = self.0.clone();
        if let Some(b) = bearer {
            cfg.bearer = b;
        }
        if let Some(u) = url {
            cfg.url = u;
        }
        Box::new(Reference(Driver::connect(&cfg).await))
    }
}

/// Every scenario file, in name order.
fn load() -> Vec<(String, Value)> {
    sandbox_conformance::load_scenarios(&scenario_dir()).expect("sdk/conformance/scenarios exists")
}

#[test]
fn scenario_files_are_well_formed() {
    let files = load();
    assert!(!files.is_empty());
    let mut ids = std::collections::BTreeSet::new();
    for (name, f) in &files {
        let suite = f["suite"]
            .as_str()
            .unwrap_or_else(|| panic!("{name}: no suite"));
        assert_eq!(
            format!("{suite}.json"),
            *name,
            "a file is named for its suite"
        );
        for sc in f["scenarios"].as_array().unwrap() {
            let id = sc["id"].as_str().unwrap();
            assert!(
                id.starts_with(&format!("{suite}.")),
                "{id} is not in {suite}"
            );
            assert!(ids.insert(id.to_string()), "duplicate scenario id {id}");
            assert!(sc["description"].is_string(), "{id}: no description");
            let mut step_ids = std::collections::BTreeSet::new();
            for s in sc["steps"].as_array().unwrap() {
                let sid = s["id"]
                    .as_str()
                    .unwrap_or_else(|| panic!("{id}: step without id"));
                assert!(step_ids.insert(sid), "{id}: duplicate step id {sid}");
                assert!(s["op"].is_string(), "{id}/{sid}: no op");
            }
            assert!(
                !sc["steps"].as_array().unwrap().is_empty(),
                "{id}: no steps"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reference_agent_passes_every_scenario() {
    let files = load();
    let mut grand = Tally::default();
    let mut grand_failures: Vec<String> = Vec::new();
    for profile in profiles() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().canonicalize().unwrap();
        let mut options = profile.options.clone();
        if profile.bearer {
            options.require_bearer = Some(BEARER.to_string());
        }
        let mock = ExecMock::start_with(&work, options).await;
        let cfg = Config {
            url: mock.url.clone(),
            bearer: profile.bearer.then(|| BEARER.to_string()),
            caps: profile.caps,
            jobs_root: work.join(".jobs"),
        };
        let mut be = Backend::open(
            profile.name,
            Box::new(Reconnect(cfg)),
            profile.bearer,
            &work.to_string_lossy(),
        )
        .await;
        be.work = work.to_string_lossy().into_owned();
        let mut tally = Tally::default();
        for (_, file) in &files {
            let suite = file["suite"].as_str().unwrap();
            if !profile.suites.is_empty() && !profile.suites.contains(&suite) {
                continue;
            }
            for sc in file["scenarios"].as_array().unwrap() {
                let checks = run_scenario(sc, &be).await;
                for c in &checks {
                    match c.status {
                        Status::Fail => grand_failures
                            .push(format!("[{}] {}: {}", profile.name, c.id, c.detail)),
                        Status::KnownDefect => {
                            println!("[{}] known defect: {}: {}", profile.name, c.id, c.detail)
                        }
                        Status::Skipped | Status::Unsupported if !c.detail.is_empty() => {
                            println!("[{}] {:?}: {}: {}", profile.name, c.status, c.detail, c.id)
                        }
                        _ => {}
                    }
                }
                tally.add(&checks);
            }
        }
        println!(
            "== {}: {} checks passed, {} failed, {} known defects, {} unsupported, {} skipped",
            profile.name,
            tally.passed,
            tally.failed,
            tally.known_defect,
            tally.unsupported,
            tally.skipped,
        );
        grand.known_defect += tally.known_defect;
        grand.unsupported += tally.unsupported;
        grand.passed += tally.passed;
        grand.failed += tally.failed;
        grand.skipped += tally.skipped;
    }
    println!(
        "== total: {} checks passed, {} failed, {} known defects, {} unsupported, {} skipped",
        grand.passed, grand.failed, grand.known_defect, grand.unsupported, grand.skipped
    );
    assert!(
        grand_failures.is_empty(),
        "{} check(s) failed:\n{}",
        grand_failures.len(),
        grand_failures.join("\n")
    );
    assert!(grand.passed > 0);
    assert_eq!(grand.unsupported, 0, "the reference driver has every op");
}

/// The wire shapes FILES-OVER-EXEC.md states as exact: what a write sends with stdin, and without.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn write_wire_shapes_match_the_spec() {
    for stdin in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().canonicalize().unwrap();
        let mock = ExecMock::start(&work, stdin).await;
        let cfg = Config {
            url: mock.url.clone(),
            jobs_root: work.join(".jobs"),
            ..Config::default()
        };
        let d = Driver::connect(&cfg).await;
        let n = 100_000usize;
        let before = mock.bodies().len();
        let r = d
            .handle(
                &json!({"id": 1, "op": "write", "path": work.join("f").to_string_lossy(),
                "data_b64": B64.encode(vec![b'x'; n])}),
            )
            .await;
        assert!(r.get("ok").is_some(), "{r}");
        let sent = &mock.bodies()[before..];
        for b in sent {
            assert_eq!(b["command"], "sh");
            assert_eq!(b["args"][0], "-c");
            assert_eq!(b["args"][2], "sh");
        }
        if stdin {
            assert_eq!(sent.len(), 1, "one command with the content on stdin");
            assert_eq!(sent[0]["args"].as_array().unwrap().len(), 6);
            assert_eq!(sent[0]["args"][5], n.to_string());
            assert_eq!(
                sent[0]["stdin_base64"].as_str().unwrap().len(),
                n.div_ceil(3) * 4
            );
        } else {
            assert_eq!(sent.len(), n.div_ceil(48 * 1024), "one command per 48 KiB");
            for (i, b) in sent.iter().enumerate() {
                let a = b["args"].as_array().unwrap();
                assert_eq!(
                    a.len(),
                    9,
                    "-c, script, sh, target, suffix, chunk, mode, commit, size"
                );
                assert!(a[5].as_str().unwrap().len() <= 65536);
                assert_eq!(a[6], if i == 0 { "new" } else { "append" });
                assert_eq!(a[7], if i + 1 == sent.len() { "commit" } else { "" });
                assert_eq!(a[8], n.to_string());
                assert!(b.get("stdin_base64").is_none());
            }
        }
    }
}
