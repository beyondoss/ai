# beyond-job-client

The job client's kernel. This first slice is the job cursor: one job incarnation (scope, id, mark) at one
output offset, as a string a caller keeps and hands back to `resume`. A cursor names exactly one incarnation
at exactly one offset, so a resume never reads another incarnation's bytes (a reused id has another mark) and
never re-reads or skips a byte (the offset is 16 fixed hex digits, so a cursor cut short cannot decode to a
smaller one). The step function (reconnect, backoff, exit handling) lands in B1 and is laid out to join this
crate: the same `verus!` module, the same `mut_*` features.

No I/O, no clock, no randomness, no dependency but `vstd`. Every decision is written inside `verus!`, so the
function production runs is the one [Verus](https://github.com/verus-lang/verus) checks, with no bound on
anything. A normal build erases the specs and proofs; `mise run verify:kernels`
(`.github/scripts/verify-kernels.sh`, the release pinned in `.github/scripts/verus.env`, run by CI's `verus`
job) checks them.

```rust
let c = Cursor::from_parts("session-1", "a1b2c3", "00ff00ff00ff00ff1a", 4096)?;
let s: String = job_client::encode(&c);   // j1.session-1.a1b2c3.00ff00ff00ff00ff1a.0000000000001000.0a3f
let back = job_client::decode(&s)?;       // Ok(c); Err(CursorError) for anything else
```

## Format

The leading `j1` is the cursor format's own version, not a wire protocol version: a change to the format
bumps it and old cursors are refused as `BadVersion`.

```text
j1.<scope>.<id>.<mark>.<offset>.<check>
```

| Field  | Rule                                                                                | Taken from                                                                                                                |
| ------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `j1`   | the version                                                                         | this crate                                                                                                                |
| scope  | 1 to 64 bytes of `[0-9a-z-]`                                                        | `JobScope::named` (`crates/sandbox/src/exec_endpoint/jobs.rs`), the TypeScript scope check (`sdk/ts/sandbox/src/jobs.ts`) |
| id     | 1 to 32 bytes of `[0-9a-z]`                                                         | `JobId::parse` (same file), the TypeScript id check (same)                                                                |
| mark   | 1 to 32 bytes of `[0-9a-f]`                                                         | execd's `new_job_mark` (`crates/execd/src/proc.rs`): 16 hex, then the sequence in hex                                     |
| offset | 16 bytes of `[0-9a-f]`: the `u64`                                                   | fixed width                                                                                                               |
| check  | 4 bytes of `[0-9a-f]`: the sum of every byte before the `.` that precedes the check | an additive check                                                                                                         |

`.` is outside every alphabet, so a string has one parse and a cursor has one string. The check is for a
cursor cut or garbled in a database column or a log line, not for security (the endpoint enforces scope and
auth; a mark is unguessable). It is additive, not a CRC: a one-byte change moves the sum by a non-zero amount
below 256, which Verus proves with no bound. It does not detect two bytes swapped, or two changes that cancel
in the sum; the alphabets and the structure may still refuse those, and nothing is claimed.

## Proven

In `src/lib.rs`. `accepts(s)` says `s` is the encoding of some cursor whose fields obey the rules. Each
property has one `mut_*` feature that removes the guard it rests on and turns the proof red
(`mise run verify:kernels -- --features <mut>`).

| Property                                                                                                                                                                                     | Where                                                        | Mutant                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| C1 `decode(encode(c)) == Ok(c)` for every cursor: nothing valid is refused (the completeness partner of C3). `decode` is `Ok` exactly for the strings `encode` writes, and there is one each | `decode_bytes`'s ensures, `encode_fields`, `lemma_enc_facts` | `mut_rejects_max_offset`                       |
| C2 the offset is exact: all 64 bits survive, `u64::MAX` included                                                                                                                             | `lemma_offset_hex_round_trip`, `lemma_hex_roundtrip`         | `mut_offset_truncated`                         |
| C3a changing any one byte of an accepted string leaves a string that is refused                                                                                                              | `lemma_single_byte_change_refused`, `lemma_sum_one_diff`     | `mut_check_skipped`, `mut_check_skips_a_byte`  |
| C3b a strict prefix of an accepted string is refused (a cursor cut short)                                                                                                                    | `lemma_prefix_refused`                                       | `mut_trailing_accepted` (the extension mirror) |
| C3c an accepted string with bytes after it is refused                                                                                                                                        | `lemma_extension_refused`, `decode_bytes`'s `Ok ==> accepts` | `mut_trailing_accepted`                        |
| C4 a decoded or constructed cursor has a scope of 1 to 64 `[0-9a-z-]`, an id of 1 to 32 `[0-9a-z]` and a mark of 1 to 32 `[0-9a-f]`: the type invariant, whoever made the value              | `Cursor::inv`, `Cursor::scope`, `Cursor::id`, `Cursor::mark` | `mut_scope_unchecked`, `mut_id_unchecked`      |

C1 is also the liveness partner of the refusals (C3): a valid cursor is always accepted, so a retry with the
cursor a handle held never fails on the cursor. `decode` and `encode` are total and take no step.

No `assume`, `admit` or `external_body`.

## What stays outside `verus!`

- `encode` and `decode` over `String` and `&str`: one-line wrappers that turn the proven byte functions'
  ASCII output into a `String` (no failure path) and hand `str::as_bytes` in.
- `Cursor`'s `Clone` (a clone of valid fields is valid), `Debug`, `PartialEq`, `Eq`, and the `*_str` accessors
  that reshape a field.
- The type invariant holds where a cursor is made (`Cursor::new`, `decode_bytes`): Verus checks it at both.
