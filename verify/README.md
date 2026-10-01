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

A defect that only a real client can settle (`refuted` or `fixed` by a live cell) names the cell as
`test = "live:<cell name>"`; the gate checks that cell's outcome in the last live run.

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

## Billing reconciliation (BIL-5)

`crates/verify/tests/reconcile_live.rs` checks the ledger against the providers' own books. Each
trial (`BIL-5::raw::openai::reconcile`, `BIL-5::raw::anthropic::reconcile`) boots a gateway with
that provider's pool key, sends a fixed batch through it (non-stream and stream, a translated
pairing, a ~3k-token system prompt reused so cache writes and reads occur, and Responses on
OpenAI), and sums the batch's `ai.usage` rows, normalized by `usage_wire`. It then reads the
provider's organization usage report for the same minutes, filtered to the pool key's id and the
batch's model (`gpt-4.1-nano`, `claude-sonnet-4-5`, which the other live suites don't use). It
passes only when uncached input, cache reads, cache writes and output agree exactly (and the request
count, where OpenAI reports one). So it proves that every token the provider charged our key for is
in a billing row, and that no row bills a token the provider didn't charge.

A trial is listed only with `VERIFY_LIVE=1` and both of that provider's keys set: `OPENAI_ADMIN_KEY`
and `OPENAI_API_KEY`, or `ANTHROPIC_ADMIN_KEY` and `ANTHROPIC_API_KEY`. A missing key means the
trial isn't listed, never that it fails. The admin keys are used only on read-only endpoints (key
listings and usage reports).

Usage reports lag traffic by minutes. A trial polls every 30 seconds until the provider's totals
match and hold, so it can take up to ~15 minutes.

## Live cells

`crates/verify/tests/live.rs` runs real SDKs and agent harnesses through a gateway built from this
tree to real providers, one cell per `(claims, client, route, probe)`. Each probe
(`verify/clients/py/probe.py`, `verify/clients/node/probe.mjs`) asserts its claim's oracle and
reports every HTTP call it made. The cell then holds each call to exactly one `ai.usage` row with
the tokens the client saw. A call whose row is not the ordinary one says so with `expect`: no row (a
free token count, a BYO key), a refusal billed nothing, a cut-short estimate bounded by the same
request completed, the provider it must land on, or a row field's minimum (cache reads, 1-hour
writes, server tool calls). Image and PDF fixtures live in `verify/clients/fixtures/`.

```sh
VERIFY_LIVE=1 cargo nextest run -p beyond-ai-verify --test live -E 'test(/::stream_abort$/)'
VERIFY_ROWS_OUT=$PWD/target/rows.jsonl VERIFY_LIVE=1 cargo nextest run ...  # also keep every row
```

`STALE` status will land with the run ledger.
