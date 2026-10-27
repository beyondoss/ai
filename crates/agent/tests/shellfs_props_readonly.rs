//! `ShellFs` where the tree starts with a read-only directory and a read-only file, on top of the
//! chmod steps every run has. Writes there must be refused with `permission_denied` and change
//! nothing, reads must still work, and no temp may be left in a directory that refused one.
//!
//! The permission checks rely on the caller not being root; as root every access succeeds and the
//! oracle (which asks the same question of the real tree) predicts that.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::shellfs_props::run_machine;
use crate::shellfs_props_env::{self as env, Config, PLAIN};
use hegel::TestCase;

#[hegel::test(env::settings(40, 0x5e11_f521))]
fn read_only_directory_and_file(tc: TestCase) {
    if env::truth_is_root() {
        eprintln!("skipping: running as root, where permission bits do not bind");
        return;
    }
    run_machine(
        tc,
        Config {
            readonly_start: true,
            ..PLAIN
        },
    );
}
