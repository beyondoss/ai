# Gateway verification

This directory holds every claim the gateway makes, and every suspected or confirmed defect against
those claims. Nobody writes down whether a claim holds. The `verify` tool (`crates/verify`) computes
it from the tests and their results.

| File           | What it holds                                                                  |
| -------------- | ------------------------------------------------------------------------------ |
| `claims.toml`  | Each claim's id, priority, layer, clients, `touches` paths and linked defects. |
| `defects.toml` | Each defect's severity, evidence, linked claims, state, test, fix and note.    |

## Tagging a test

```rust
/// claim: SEC-1, SEC-5
/// defect: D01
#[tokio::test]
#[ignore = "D01 reproduced: managed /{provider} forwards any path"]
async fn provider_route_rejects_files_api() {
    // asserts the CORRECT behavior
}
```

A test may claim several claims. The tag lines go directly above the attributes and the `fn`.

## Defect lifecycle

| State        | What the gate requires                                                          |
| ------------ | ------------------------------------------------------------------------------- |
| `suspected`  | Nothing yet. The finding comes from reading code.                               |
| `reproduced` | `test` asserts the correct behavior, is `#[ignore]`d, and **fails** when run.   |
| `fixed`      | `test` passes, is no longer ignored, and `fix` names the commit.                |
| `refuted`    | `test` passes and `note` records why the behavior was already correct.          |
| `accepted`   | `note` records the decision, and `test` asserts the behavior as now documented. |

When a reproduction starts passing, the gate stops and asks for the defect to be marked `fixed` or
`refuted`. This keeps the ledger from saying "broken" after a fix, or "fixed" while the code is
still broken. An `#[ignore]`d test that reproduces no open defect is also an error. That is the only
reason a tagged test may be ignored.

## Claim status

| Status     | Meaning                                                                       |
| ---------- | ----------------------------------------------------------------------------- |
| `PROVEN`   | Every tagged test passed and no linked defect is open.                        |
| `PARTIAL`  | Hermetic tests pass, but live cells are pending, or some tests didn't run.    |
| `RED`      | A tagged test fails, or a linked defect is still `suspected` or `reproduced`. |
| `UNTESTED` | No test carries the id.                                                       |

## Commands

```sh
mise run verify:status            # run tagged tests (ignored ones too), then print status
mise run verify:status -- --json  # the same as JSON (feeds the status page)
mise run verify:gate              # registry/tag/result consistency; non-zero on any problem
```

Live cells, which run real SDKs and agent harnesses against real providers, will land as tagged
tests in their own layer. `STALE` status will land with the run ledger.
