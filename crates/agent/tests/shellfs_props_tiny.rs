//! `ShellFs` on a filesystem that fills up: a 96 KiB tmpfs in a private user+mount namespace (no
//! root needed). A write that does not fit must fail with the file untouched and no temp left behind,
//! and one that fits must succeed. Commands reach the namespace through `nsenter`; the oracle reads it
//! through `/proc/PID/root`. Skipped, with the reason printed, where unprivileged namespaces are off.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::shellfs_props::run_machine;
use crate::shellfs_props_env::{self as env, Config, PLAIN};
use hegel::TestCase;

fn tiny(tc: TestCase, stdin: bool) {
    if let Some(why) = env::tiny_unavailable() {
        eprintln!("skipping: no tiny filesystem here ({why})");
        return;
    }
    run_machine(
        tc,
        Config {
            tiny_kib: Some(96),
            stdin,
            ..PLAIN
        },
    );
}

#[hegel::test(env::settings(50, 0x5e11_f531))]
fn a_filesystem_that_fills_up_stdin_writes(tc: TestCase) {
    tiny(tc, true);
}

#[hegel::test(env::settings(50, 0x5e11_f532))]
fn a_filesystem_that_fills_up_chunked_writes(tc: TestCase) {
    tiny(tc, false);
}
