//! `serve` e2e: wire-layer `--block-images` enforcement (`ModelRequest::block_images`) — the two gaps
//! an ingestion-only gate (the `read` tool / CLI `@file` attachments, already covered by
//! `serve_tools_bash.rs`'s `serve_block_images_flag_forces_a_read_tool_image_to_a_text_placeholder`)
//! can't close on its own: an RPC client pushing an image directly into a running session, and an
//! image already persisted in session history before the flag was toggled on.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};

use common::{SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text};
use serde_json::json;

/// A tiny valid PNG (standard base64), reused verbatim as the RPC `images[].data` field — doesn't need
/// decoding to bytes here since it's shipped straight through as the wire-level base64 payload, not
/// read off disk by a tool.
const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAANklEQVR42u3OQQ0AAAgAoetfWls4H2wEoKlXEhISEhISEhISEhISEhISEhISEhISEhISEhK6s98T93mKDkyKAAAAAElFTkSuQmCC";

#[test]
fn serve_rpc_prompt_pushed_image_is_downgraded_at_the_wire_when_block_images_is_set() {
    // Task 1 (pi-parity fix, serve pass 19): the `prompt` RPC handler pushes a client-supplied
    // `images` attachment straight into the session with zero reference to `Agent.block_images`
    // anywhere in that path — before this pass's fix, `--block-images` only gated the ingestion-time
    // `read`-tool/CLI-attachment path, so an RPC client could bypass it completely. Wiring
    // `ModelRequest::block_images` from `Agent.block_images` at every turn build
    // (`agent_core::agent::Agent`'s per-turn `ModelRequest::new` call site) closes this uniformly, with
    // no special-casing needed in `serve.rs` itself.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file)
        .args(["--block-images"])
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({
            "type": "prompt",
            "message": "what is this?",
            "images": [{ "media_type": "image/png", "data": RED_PNG_B64 }],
        })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");
    drop(stdin);
    child.wait().unwrap();

    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("image omitted: model does not support images"),
        "an RPC-pushed image must be downgraded to the non-vision placeholder at the wire layer when \
         --block-images is set: {}",
        bodies[0]
    );
    assert!(
        !bodies[0]
            .to_ascii_lowercase()
            .contains(&RED_PNG_B64.to_ascii_lowercase()),
        "the raw image bytes must not reach the wire once downgraded: {}",
        bodies[0]
    );
}

#[test]
fn serve_persisted_image_is_downgraded_on_resend_once_block_images_is_toggled_on() {
    // Task 1 (pi-parity fix, serve pass 19): enforcement was previously ingestion-time only, so an
    // image already persisted in session history *before* `--block-images` was toggled on still got
    // resent verbatim on a resumed session. Reproduced here as two successive `serve` processes
    // sharing the same session file: the first ingests a real image with no `--block-images`; the
    // second reopens that same file *with* `--block-images` and sends an unrelated follow-up prompt —
    // the full conversation history (including the first turn's now-already-persisted image) is resent
    // on every turn, so the fix must catch it there too, not just on newly-ingested images.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("first"), turn_text("second")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // First process: no `--block-images` — the image is ingested and persisted as-is.
    {
        let mut child = serve_cmd(bin, &base, &session_file).spawn_guarded();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(
            stdin,
            "{}",
            json!({
                "type": "prompt",
                "message": "what is this?",
                "images": [{ "media_type": "image/png", "data": RED_PNG_B64 }],
            })
        )
        .unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "prompt");
        assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");
        drop(stdin);
        child.wait().unwrap();
    }
    {
        let bodies = bodies.lock().unwrap();
        assert!(
            bodies[0]
                .to_ascii_lowercase()
                .contains(&RED_PNG_B64.to_ascii_lowercase()),
            "sanity check: the image must actually reach the wire before --block-images is ever set: {}",
            bodies[0]
        );
    }

    // Second process: same session file, resumed with `--block-images` this time — nothing new is
    // ingested this run, but the *already-persisted* image from the first turn is still part of
    // `session.messages` and gets resent on this turn's own request.
    {
        let mut child = serve_cmd(bin, &base, &session_file)
            .args(["--block-images"])
            .spawn_guarded();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(
            stdin,
            "{}",
            json!({ "type": "prompt", "message": "and now?" })
        )
        .unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "prompt");
        assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");
        drop(stdin);
        child.wait().unwrap();
    }

    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[1].contains("image omitted: model does not support images"),
        "the already-persisted image from the first turn must be downgraded once the resumed \
         session's --block-images is set, not just newly-ingested images: {}",
        bodies[1]
    );
    assert!(
        !bodies[1]
            .to_ascii_lowercase()
            .contains(&RED_PNG_B64.to_ascii_lowercase()),
        "the raw persisted image bytes must not be resent once --block-images is set: {}",
        bodies[1]
    );
}

#[test]
fn serve_set_block_images_toggles_and_rejects_a_non_boolean() {
    // Pass 20 (pi-parity fix): `block_images` had a `--block-images` startup flag but no live RPC
    // toggle at all — `build_agent` read `cfg.block_images` straight off the static startup config on
    // every rebuild, so an already-running `serve` process had no way to change it short of a restart.
    // Mirrors `serve_set_auto_compaction_toggles_and_rejects_a_non_boolean`
    // (`serve_compaction_retry.rs`) exactly.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_block_images", "enabled": true })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_block_images");
    assert_eq!(frames.last().unwrap()["success"], true);
    assert_eq!(frames.last().unwrap()["data"]["block_images"], true);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_block_images", "enabled": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_block_images");
    assert_eq!(frames.last().unwrap()["data"]["block_images"], false);

    // Missing/non-boolean `enabled` is rejected, not silently coerced.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_block_images", "enabled": "yes" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_block_images");
    assert_eq!(frames.last().unwrap()["success"], false);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_block_images_true_downgrades_a_persisted_image_mid_session_without_a_restart() {
    // The live-toggle counterpart to `serve_persisted_image_is_downgraded_on_resend_once_block_images_is_toggled_on`
    // just above: that test proves the flag reaches a *second process* reopening the same session file;
    // this proves the very same effect within *one* running process, via `set_block_images`, with the
    // agent rebuilt in place and no restart at all — the concrete gap this pass's fix closes.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("first"), turn_text("second")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // No `--block-images` — the image is ingested and sent to the wire as-is.
    let mut child = serve_cmd(bin, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    writeln!(
        stdin,
        "{}",
        json!({
            "type": "prompt",
            "message": "what is this?",
            "images": [{ "media_type": "image/png", "data": RED_PNG_B64 }],
        })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");

    {
        let bodies = bodies.lock().unwrap();
        assert!(
            bodies[0]
                .to_ascii_lowercase()
                .contains(&RED_PNG_B64.to_ascii_lowercase()),
            "sanity check: the image must actually reach the wire before block_images is ever set: {}",
            bodies[0]
        );
    }

    // Same process, no restart: toggle the live setting, then send an unrelated follow-up prompt —
    // the first turn's already-persisted image is still part of `session.messages` and gets resent on
    // this turn's own request.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_block_images", "enabled": true })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_block_images");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "and now?" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");
    drop(stdin);
    child.wait().unwrap();

    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[1].contains("image omitted: model does not support images"),
        "set_block_images(true) must downgrade the already-persisted image on the very next request, \
         in the same process, with no restart: {}",
        bodies[1]
    );
    assert!(
        !bodies[1]
            .to_ascii_lowercase()
            .contains(&RED_PNG_B64.to_ascii_lowercase()),
        "the raw persisted image bytes must not be resent once set_block_images(true) takes effect: {}",
        bodies[1]
    );
}
