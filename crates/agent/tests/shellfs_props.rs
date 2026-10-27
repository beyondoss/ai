//! `ShellFs` against ground truth, on a clean transport. See `shellfs_props_env` for the model, the
//! oracle and the generators.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::shellfs_props_env::{self as env, Config, PLAIN};
use hegel::TestCase;

fn run(tc: TestCase, cfg: Config) {
    let m = env::machine(&tc, cfg);
    hegel::stateful::machine(m).steps(cfg.steps).run(tc);
}

#[hegel::test(env::settings(40, 0x5e11_f5_01))]
fn plain_stdin_writes(tc: TestCase) {
    run(tc, PLAIN);
}

#[hegel::test(env::settings(30, 0x5e11_f5_02))]
fn chunked_writes_without_stdin(tc: TestCase) {
    run(tc, Config { stdin: false, ..PLAIN });
}

#[hegel::test(env::settings(30, 0x5e11_f5_03))]
fn small_response_cap(tc: TestCase) {
    run(tc, Config { cap: Some(40_000), ..PLAIN });
}
