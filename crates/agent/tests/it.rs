//! The agent's integration tests, compiled and linked as ONE test binary.
//!
//! Each `tests/<domain>.rs` is a module here, not its own binary: 240 separate binaries each linked
//! the whole dependency graph (~3.9 GB of test artifacts, most of CI's build time), one binary links
//! it once (~100 MB). The one-file-per-domain layout is unchanged; only how they are compiled moved.
//!
//! - Run one file's tests: `cargo nextest run -p beyond-agent -E 'test(/^serve_approval::/)'`.
//!   Under plain `cargo test` every file now shares one process, so use nextest (a process per
//!   test, which is also what CI does).
//! - A new `tests/<domain>.rs` must be declared below. `every_test_file_is_declared` fails if it
//!   isn't, since an undeclared file would silently never run. Helpers shared by several files live
//!   in `common/` and `mcp_tasks_env/` and are reached as `use crate::common;`.
//! - `code_mode_eval.rs` is not here: it needs `--features code-mode` and keeps its own target.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[macro_use]
mod common;
mod mcp_tasks_env;

mod agent_frontmatter_run;
mod agent_tools_e2b;
mod agent_tools_matrix;
mod agent_tools_mcp;
mod auth_cli;
mod bash_background_remote;
mod bash_background_run;
mod bash_background_serve;
mod bash_jobs_panic_exit;
mod bash_jobs_reopen;
mod bash_jobs_reopen_after_kill;
mod bash_jobs_run;
mod bash_jobs_serve;
mod bash_jobs_serve_abort;
mod bash_jobs_serve_carry;
mod bash_jobs_serve_daemon;
mod bash_jobs_serve_host;
mod bash_jobs_serve_new_session;
mod bash_jobs_serve_sandbox;
mod bash_jobs_stop_remote;
mod bash_remote_exit;
mod bash_remote_leftovers;
mod bash_remote_stream;
mod bash_remote_world;
mod bash_sandbox_jobs;
mod broker_conformance;
mod bundle_conformance;
mod cli_errors;
mod cli_tool_selection;
mod code_mode_density;
mod direct_routing_wire;
mod egress_proxy_env;
mod exec_conformance;
mod exec_endpoint;
mod exec_endpoint_live;
mod exec_endpoint_response_cap;
mod exec_endpoint_writes;
mod exec_front_conformance;
mod exec_multi_tenant;
mod exec_notify_vectors;
mod fs_backend_cost;
mod fs_backend_fuzz;
mod fs_backend_parity;
mod fs_backend_remote_world;
mod gateway_e2e;
mod home_isolation;
mod job_notify;
mod mcp_apps;
mod mcp_apps_durable;
mod mcp_apps_events;
mod mcp_apps_resilience;
mod mcp_apps_security;
mod mcp_apps_session;
mod mcp_cancel_propagates;
mod mcp_client;
mod mcp_elicitation;
mod mcp_events_busy;
mod mcp_events_durability;
mod mcp_events_lifecycle;
mod mcp_events_nested;
mod mcp_events_poll;
mod mcp_events_push;
mod mcp_events_receiver;
mod mcp_events_redial;
mod mcp_events_routing;
mod mcp_events_v1a;
mod mcp_events_webhook;
mod mcp_handshake_timeout;
mod mcp_hotplug;
mod mcp_hotplug_subagent;
mod mcp_late_discover;
mod mcp_list_changed;
mod mcp_message_cap;
mod mcp_names;
mod mcp_oauth;
mod mcp_oauth_refresh;
mod mcp_oauth_routing;
mod mcp_progress;
mod mcp_reap;
mod mcp_redial;
mod mcp_sandbox_connector;
mod mcp_sandbox_stdio;
mod mcp_skills;
mod mcp_skills_approval;
mod mcp_skills_integrity;
mod mcp_skills_listing;
mod mcp_skills_resume;
mod mcp_skills_sessions;
mod mcp_skills_steer;
mod mcp_stdio_orphans;
mod mcp_stdio_transport;
mod mcp_surfaces;
mod mcp_tasks;
mod mcp_tasks_conformance;
mod mcp_tasks_daemon;
mod mcp_tasks_durable;
mod mcp_tasks_resume;
mod mcp_tasks_service;
mod mcp_tool_search;
mod mcp_unauthorized;
mod notify_e2e_callbacks;
mod notify_e2e_lost;
mod notify_e2e_orderings;
mod notify_e2e_release;
mod notify_e2e_takeover;
mod notify_model;
mod notify_token;
mod oauth_e2e;
mod ports_lint;
mod release_seams;
mod run_cli_flags;
mod run_compaction_persist;
mod run_compaction_reminder;
mod run_continue_after_failure;
mod run_core_flow;
mod run_memory;
mod run_prompt_prefix;
mod run_rate_limit;
mod run_resume_after_kill;
mod run_resume_model;
mod run_session_management;
mod run_session_memory;
mod run_signal_handling;
mod run_skills_prompts;
mod run_stdin_open_pipe;
mod run_stdout_retry;
mod run_stdout_robustness;
mod run_structured_output;
mod run_subagent;
mod run_subagent_concurrent_merge;
mod run_subagent_loop;
mod run_subagent_memory;
mod run_subagent_parallel;
mod run_subagent_structured_output;
mod run_subagent_worktree;
mod run_subagent_worktree_keep;
mod run_subagent_worktree_merge;
mod run_subagent_worktree_note;
mod run_todo;
mod run_wrap_up_and_export;
mod sandbox_jobs;
mod serve_accept_error;
mod serve_agent_frontmatter;
mod serve_approval;
mod serve_auth;
mod serve_block_images;
mod serve_browser_guard;
mod serve_busy_host_commands;
mod serve_busy_self_abort;
mod serve_compact_cancellable;
mod serve_compaction_persist;
mod serve_compaction_retry;
mod serve_cut_off_tool_call;
mod serve_drain;
mod serve_exec_endpoint;
mod serve_gateway_headers;
mod serve_grant;
mod serve_harness_deadlines;
mod serve_health;
mod serve_http;
mod serve_job_ended;
mod serve_lifecycle;
mod serve_list_paging;
mod serve_list_sessions;
mod serve_masking_two_passes;
mod serve_memory;
mod serve_models_thinking;
mod serve_oauth_model_switch;
mod serve_prompt_flow;
mod serve_reaper;
mod serve_reattach_catchup;
mod serve_replay_replies;
mod serve_resume_after_kill;
mod serve_service_agent_frontmatter;
mod serve_service_bundle;
mod serve_service_bundle_lazy;
mod serve_service_bundle_mcp_tools;
mod serve_service_discovery;
mod serve_service_grant;
mod serve_service_isolation;
mod serve_service_lazy_sandbox;
mod serve_service_limits;
mod serve_service_lock;
mod serve_service_managed_mcp;
mod serve_service_mcp;
mod serve_service_mcp_json;
mod serve_service_mode;
mod serve_service_policy;
mod serve_service_probe_once;
mod serve_service_seal_env;
mod serve_service_session_end;
mod serve_service_sessions;
mod serve_service_storage;
mod serve_service_subagent;
mod serve_service_subprotocol;
mod serve_service_tls;
mod serve_session_addressing;
mod serve_session_density;
mod serve_session_isolation;
mod serve_session_lifecycle;
mod serve_session_memory;
mod serve_session_title;
mod serve_session_tree;
mod serve_shared_client;
mod serve_slow_client;
mod serve_sse_events;
mod serve_startup_stall;
mod serve_state_reporting;
mod serve_stdin_bad_line;
mod serve_structured_output;
mod serve_subagent;
mod serve_subagent_model;
mod serve_systemd;
mod serve_todo;
mod serve_tools_bash;
mod serve_trust_prompts;
mod serve_uds;
mod serve_websocket;
mod session_compaction_continuity;
mod session_create_recovery;
mod session_file_integrity;
mod session_listing_index;
mod session_new_session_reset;
mod session_retry_persist;
mod session_segments;
mod session_segments_fencing;
mod session_segments_paths;
mod session_segments_sealing;
mod session_writer_lock;
mod settings_project_override;
mod settings_store;
mod shellfs_props;
mod shellfs_props_env;
mod shellfs_props_faults;
mod shellfs_props_known;
mod shellfs_props_readonly;
mod shellfs_props_tiny;
mod smoke;
mod state_dir;
mod steer_parity;
mod subagent_policy;
mod subagent_usage;
mod tool_output_masking;
mod tool_reactor_stall;
mod trust_cli;
mod web_parse_isolation;
mod web_tool;

/// Every `tests/*.rs` except this file and the feature-gated `code_mode_eval` is a `mod` above.
#[test]
fn every_test_file_is_declared() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let this = include_str!("it.rs");
    let mut missing = Vec::new();
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let name = path.file_stem().unwrap().to_str().unwrap();
        if name != "it" && name != "code_mode_eval" && !this.contains(&format!("\nmod {name};\n")) {
            missing.push(name.to_string());
        }
    }
    assert!(missing.is_empty(), "declare in tests/it.rs: {missing:?}");
}
