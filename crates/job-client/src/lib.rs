//! The job cursor: one job incarnation (scope, id, mark) at one output offset, as a string a caller can keep
//! and hand back to `resume`. The cursor is what lets a client reconnect without reading another
//! incarnation's bytes: the mark names the incarnation, the offset names the next byte not yet delivered.
//! Everything here is pure (no I/O, no clock, no randomness), so a model checker and production call what
//! Verus proves: every decision is written inside `verus!`.
//!
//! # Format
//!
//! ```text
//! j1.<scope>.<id>.<mark>.<offset>.<check>
//! ```
//!
//! | field | rule | where the rule comes from |
//! |---|---|---|
//! | `j1` | the cursor format's own version | this crate |
//! | scope | 1 to 64 bytes of `[0-9a-z-]` | `JobScope::named` (`crates/sandbox/src/exec_endpoint/jobs.rs`), the TypeScript `Jobs` scope check (`sdk/ts/sandbox/src/jobs.ts`) |
//! | id | 1 to 32 bytes of `[0-9a-z]` | `JobId::parse` (same file), the TypeScript id check (same) |
//! | mark | 1 to 32 bytes of `[0-9a-f]` | execd's `new_job_mark` (`crates/execd/src/proc.rs`): 16 hex of a hash, then the sequence in hex |
//! | offset | 16 bytes of `[0-9a-f]`, the u64 | fixed width, so a truncated cursor can never decode to a smaller offset |
//! | check | 4 bytes of `[0-9a-f]`: the sum of every byte before the `.` that precedes it | an additive check, not a MAC |
//!
//! The separator `.` is outside every field's alphabet, so a string has exactly one parse and exactly one
//! encoding. The check is about being well-formed (a cursor cut off in a database column or a log line),
//! not about security: the endpoint enforces scope and auth, and a mark is unguessable.
//!
//! What the check guarantees, and no more: of two accepted strings of equal length, none differs from the
//! other in exactly one byte; no accepted string is a strict prefix or a strict extension of another.
//! A transposition of two bytes, or two changed bytes whose sum is unchanged, is not detected by the
//! check (the alphabets and the structure may still refuse it).

// The code inside `verus!` takes the shapes Verus can prove (an explicit `if` where `matches!` reads better,
// a mutation's `cfg!` branch beside its twin); clippy's rewrites are not those.
#![allow(
    clippy::match_like_matches_macro,
    clippy::if_same_then_else,
    clippy::collapsible_if,
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::len_zero,
    non_shorthand_field_patterns
)]

use vstd::prelude::*;

pub mod step;

verus! {

// ---------------------------------------------------------------------------------------------
// The alphabets. A byte is a `u8`: every field is ASCII, and the encoded string is ASCII.
// ---------------------------------------------------------------------------------------------

/// `[0-9]`.
pub open spec fn is_digit(b: u8) -> bool {
    48 <= b && b <= 57
}

/// `[0-9a-f]`.
pub open spec fn is_hex(b: u8) -> bool {
    is_digit(b) || (97 <= b && b <= 102)
}

/// A scope's bytes: `[0-9a-z-]`.
pub open spec fn scope_byte(b: u8) -> bool {
    is_digit(b) || (97 <= b && b <= 122) || b == 45
}

/// An id's bytes: `[0-9a-z]`.
pub open spec fn id_byte(b: u8) -> bool {
    is_digit(b) || (97 <= b && b <= 122)
}

/// A mark's bytes: `[0-9a-f]`.
pub open spec fn mark_byte(b: u8) -> bool {
    is_hex(b)
}

/// The rules for the three variable fields.
pub open spec fn fields_ok(scope: Seq<u8>, id: Seq<u8>, mark: Seq<u8>) -> bool {
    &&& 1 <= scope.len() <= 64
    &&& 1 <= id.len() <= 32
    &&& 1 <= mark.len() <= 32
    &&& forall|i: int| 0 <= i < scope.len() ==> scope_byte(#[trigger] scope[i])
    &&& forall|i: int| 0 <= i < id.len() ==> id_byte(#[trigger] id[i])
    &&& forall|i: int| 0 <= i < mark.len() ==> mark_byte(#[trigger] mark[i])
}

// ---------------------------------------------------------------------------------------------
// Hex and sums, as specs.
// ---------------------------------------------------------------------------------------------

/// The lowercase hex digit of `n < 16`.
pub open spec fn hexc(n: nat) -> u8 {
    if n < 10 { (48 + n) as u8 } else { (87 + n) as u8 }
}

/// The value of a hex digit.
pub open spec fn nib(b: u8) -> nat {
    if b <= 57 { (b - 48) as nat } else { (b - 87) as nat }
}

pub open spec fn pow16(n: nat) -> nat
    decreases n,
{
    if n == 0 { 1 } else { 16 * pow16((n - 1) as nat) }
}

/// `v` in `n` hex digits, most significant first (the low `n` digits if `v` is larger).
pub open spec fn hex_n(v: nat, n: nat) -> Seq<u8>
    decreases n,
{
    if n == 0 { Seq::empty() } else { hex_n(v / 16, (n - 1) as nat).push(hexc(v % 16)) }
}

/// The number a run of hex digits spells.
pub open spec fn hex_val(s: Seq<u8>) -> nat
    decreases s.len(),
{
    if s.len() == 0 { 0 } else { hex_val(s.drop_last()) * 16 + nib(s.last()) }
}

/// The sum of the bytes.
pub open spec fn sum(s: Seq<u8>) -> nat
    decreases s.len(),
{
    if s.len() == 0 { 0 } else { sum(s.drop_last()) + s.last() as nat }
}

/// How many `.` are in `s`.
pub open spec fn dots(s: Seq<u8>) -> nat
    decreases s.len(),
{
    if s.len() == 0 { 0 } else { dots(s.drop_last()) + if s.last() == 46 { 1nat } else { 0nat } }
}

// ---------------------------------------------------------------------------------------------
// The encoding, as a spec: the one string a cursor has.
// ---------------------------------------------------------------------------------------------

pub open spec fn dot() -> Seq<u8> {
    seq![46u8]
}

/// Everything the check covers: `j1.<scope>.<id>.<mark>.<offset>`.
pub open spec fn body(scope: Seq<u8>, id: Seq<u8>, mark: Seq<u8>, offset: u64) -> Seq<u8> {
    seq![106u8, 49u8, 46u8] + scope + dot() + id + dot() + mark + dot() + hex_n(offset as nat, 16)
}

/// `.<check>`, where the check is `x` in four hex digits.
pub open spec fn tail(x: nat) -> Seq<u8> {
    dot() + hex_n(x, 4)
}

/// The encoding of a cursor's fields.
pub open spec fn spec_enc(scope: Seq<u8>, id: Seq<u8>, mark: Seq<u8>, offset: u64) -> Seq<u8> {
    body(scope, id, mark, offset) + tail(sum(body(scope, id, mark, offset)))
}

/// `s` is the encoding of some valid cursor.
pub open spec fn accepts(s: Seq<u8>) -> bool {
    exists|scope: Seq<u8>, id: Seq<u8>, mark: Seq<u8>, offset: u64|
        fields_ok(scope, id, mark) && s == spec_enc(scope, id, mark, offset)
}

// ---------------------------------------------------------------------------------------------
// Lemmas about the specs.
// ---------------------------------------------------------------------------------------------

proof fn lemma_hex_n_len(v: nat, n: nat)
    ensures
        hex_n(v, n).len() == n,
    decreases n,
{
    if n > 0 {
        lemma_hex_n_len(v / 16, (n - 1) as nat);
    }
}

proof fn lemma_hex_n_bytes(v: nat, n: nat)
    ensures
        forall|i: int| 0 <= i < n ==> is_hex(#[trigger] hex_n(v, n)[i]),
    decreases n,
{
    if n > 0 {
        lemma_hex_n_bytes(v / 16, (n - 1) as nat);
        lemma_hex_n_len(v / 16, (n - 1) as nat);
        assert(v % 16 < 16);
        assert forall|i: int| 0 <= i < n implies is_hex(#[trigger] hex_n(v, n)[i]) by {
            if i < n - 1 {
                assert(hex_n(v, n)[i] == hex_n(v / 16, (n - 1) as nat)[i]);
            }
        }
    }
}

proof fn lemma_pow16_pos(n: nat)
    ensures
        pow16(n) >= 1,
    decreases n,
{
    if n > 0 {
        lemma_pow16_pos((n - 1) as nat);
    }
}

proof fn lemma_pow16_mono(a: nat, b: nat)
    requires
        a <= b,
    ensures
        pow16(a) <= pow16(b),
    decreases b,
{
    if a < b {
        lemma_pow16_mono(a, (b - 1) as nat);
    }
}

proof fn lemma_pow16_values()
    ensures
        pow16(4) == 65536,
        pow16(16) == 18446744073709551616,
{
    reveal_with_fuel(pow16, 17);
}

/// `v` written in `n` digits and read back is `v` modulo 16^n: with `n = 16` and a `u64`, exactly `v`.
proof fn lemma_hex_roundtrip(v: nat, n: nat)
    ensures
        hex_val(hex_n(v, n)) == v % pow16(n),
    decreases n,
{
    if n == 0 {
        assert(v % 1 == 0);
    } else {
        lemma_hex_roundtrip(v / 16, (n - 1) as nat);
        lemma_pow16_pos((n - 1) as nat);
        let h = hex_n(v, n);
        assert(h.drop_last() =~= hex_n(v / 16, (n - 1) as nat));
        assert(h.last() == hexc(v % 16));
        assert(nib(hexc(v % 16)) == v % 16);
        vstd::arithmetic::div_mod::lemma_mod_breakdown(v as int, 16, pow16((n - 1) as nat) as int);
    }
}

/// A run of hex digits is below 16^len.
proof fn lemma_hex_val_bound(s: Seq<u8>)
    requires
        forall|i: int| 0 <= i < s.len() ==> is_hex(#[trigger] s[i]),
    ensures
        hex_val(s) < pow16(s.len()),
    decreases s.len(),
{
    if s.len() > 0 {
        let p = s.drop_last();
        assert forall|i: int| 0 <= i < p.len() implies is_hex(#[trigger] p[i]) by {
            assert(p[i] == s[i]);
        }
        lemma_hex_val_bound(p);
        assert(nib(s.last()) < 16);
    }
}

/// The offset a cursor carries survives its 16 digits whole, `u64::MAX` included.
pub proof fn lemma_offset_hex_round_trip(v: u64)
    ensures
        hex_val(hex_n(v as nat, 16)) == v,
        hex_n(v as nat, 16).len() == 16,
{
    lemma_hex_roundtrip(v as nat, 16);
    lemma_pow16_values();
    lemma_hex_n_len(v as nat, 16);
    assert(v as nat % 18446744073709551616 == v as nat);
}

proof fn lemma_sum_add(x: Seq<u8>, y: Seq<u8>)
    ensures
        sum(x + y) == sum(x) + sum(y),
    decreases y.len(),
{
    if y.len() > 0 {
        lemma_sum_add(x, y.drop_last());
        assert((x + y).drop_last() =~= x + y.drop_last());
        assert((x + y).last() == y.last());
    }
}

proof fn lemma_sum_bound(s: Seq<u8>)
    ensures
        sum(s) <= 255 * s.len(),
    decreases s.len(),
{
    if s.len() > 0 {
        lemma_sum_bound(s.drop_last());
    }
}

/// Two byte strings of one length that differ in exactly one place have different sums: the check sees
/// every single-byte change.
proof fn lemma_sum_one_diff(s: Seq<u8>, i: int, b: u8)
    requires
        0 <= i < s.len(),
        b != s[i],
    ensures
        sum(s.update(i, b)) != sum(s),
    decreases s.len(),
{
    let t = s.update(i, b);
    if i == s.len() - 1 {
        assert(t.drop_last() =~= s.drop_last());
        assert(t.last() == b);
    } else {
        lemma_sum_one_diff(s.drop_last(), i, b);
        assert(t.drop_last() =~= s.drop_last().update(i, b));
        assert(t.last() == s.last());
    }
}

proof fn lemma_dots_add(x: Seq<u8>, y: Seq<u8>)
    ensures
        dots(x + y) == dots(x) + dots(y),
    decreases y.len(),
{
    if y.len() > 0 {
        lemma_dots_add(x, y.drop_last());
        assert((x + y).drop_last() =~= x + y.drop_last());
        assert((x + y).last() == y.last());
    }
}

proof fn lemma_dots_zero(s: Seq<u8>)
    requires
        forall|i: int| 0 <= i < s.len() ==> #[trigger] s[i] != 46,
    ensures
        dots(s) == 0,
    decreases s.len(),
{
    if s.len() > 0 {
        let p = s.drop_last();
        assert forall|i: int| 0 <= i < p.len() implies #[trigger] p[i] != 46 by {
            assert(p[i] == s[i]);
        }
        lemma_dots_zero(p);
    }
}

proof fn lemma_dots_some(s: Seq<u8>, j: int)
    requires
        0 <= j < s.len(),
        s[j] == 46,
    ensures
        dots(s) >= 1,
    decreases s.len(),
{
    if j < s.len() - 1 {
        lemma_dots_some(s.drop_last(), j);
    }
}

/// The body has four dots: after the version, and one after each of scope, id, mark.
proof fn lemma_body_dots(scope: Seq<u8>, id: Seq<u8>, mark: Seq<u8>, offset: u64)
    requires
        fields_ok(scope, id, mark),
    ensures
        dots(body(scope, id, mark, offset)) == 4,
{
    let h16 = hex_n(offset as nat, 16);
    lemma_hex_n_len(offset as nat, 16);
    lemma_hex_n_bytes(offset as nat, 16);
    assert forall|i: int| 0 <= i < scope.len() implies #[trigger] scope[i] != 46 by {
        assert(scope_byte(scope[i]));
    }
    assert forall|i: int| 0 <= i < id.len() implies #[trigger] id[i] != 46 by {
        assert(id_byte(id[i]));
    }
    assert forall|i: int| 0 <= i < mark.len() implies #[trigger] mark[i] != 46 by {
        assert(mark_byte(mark[i]));
    }
    assert forall|i: int| 0 <= i < h16.len() implies #[trigger] h16[i] != 46 by {
        assert(is_hex(h16[i]));
    }
    lemma_dots_zero(scope);
    lemma_dots_zero(id);
    lemma_dots_zero(mark);
    lemma_dots_zero(h16);
    let head = seq![106u8, 49u8, 46u8];
    reveal_with_fuel(dots, 4);
    assert(head.drop_last() =~= seq![106u8, 49u8]);
    assert(seq![106u8, 49u8].drop_last() =~= seq![106u8]);
    assert(seq![106u8].drop_last() =~= Seq::<u8>::empty());
    assert(dot().drop_last() =~= Seq::<u8>::empty());
    assert(dots(head) == 1);
    assert(dots(dot()) == 1);
    let e1 = head + scope;
    let e2 = e1 + dot();
    let e3 = e2 + id;
    let e4 = e3 + dot();
    let e5 = e4 + mark;
    let e6 = e5 + dot();
    lemma_dots_add(head, scope);
    lemma_dots_add(e1, dot());
    lemma_dots_add(e2, id);
    lemma_dots_add(e3, dot());
    lemma_dots_add(e4, mark);
    lemma_dots_add(e5, dot());
    lemma_dots_add(e6, h16);
}

/// What the encoding of a valid cursor looks like, position by position. `a`, `b`, `m` are the lengths of
/// scope, id, mark; `n` is the length of the string. The offset's 16 digits are `s[n-21..n-5]`, the check's
/// 4 are `s[n-4..n]`, and `.` stands at `n-22` and `n-5`.
pub open spec fn enc_facts(scope: Seq<u8>, id: Seq<u8>, mark: Seq<u8>, offset: u64) -> bool {
    let s = spec_enc(scope, id, mark, offset);
    let a = scope.len() as int;
    let b = id.len() as int;
    let m = mark.len() as int;
    let n = s.len() as int;
    &&& n == a + b + m + 27
    &&& 30 <= n <= 155
    &&& s[0] == 106 && s[1] == 49 && s[2] == 46
    &&& s.subrange(3, 3 + a) == scope
    &&& forall|j: int| 3 <= j < 3 + a ==> scope_byte(#[trigger] s[j])
    &&& s[3 + a] == 46
    &&& s.subrange(4 + a, 4 + a + b) == id
    &&& forall|j: int| 4 + a <= j < 4 + a + b ==> id_byte(#[trigger] s[j])
    &&& forall|j: int| 5 + a + b <= j < n - 22 ==> mark_byte(#[trigger] s[j])
    &&& forall|j: int| n - 21 <= j < n - 5 ==> is_hex(#[trigger] s[j])
    &&& forall|j: int| n - 4 <= j < n ==> is_hex(#[trigger] s[j])
    &&& s[4 + a + b] == 46
    &&& s.subrange(5 + a + b, 5 + a + b + m) == mark
    &&& s[n - 22] == 46
    &&& s.subrange(n - 21, n - 5) == hex_n(offset as nat, 16)
    &&& s[n - 5] == 46
    &&& s.subrange(n - 4, n) == hex_n(sum(s.subrange(0, n - 5)), 4)
    &&& s.subrange(n - 5, n) == tail(sum(s.subrange(0, n - 5)))
    &&& sum(s.subrange(0, n - 5)) < 65536
    &&& dots(s) == 5
    &&& forall|j: int| n - 21 <= j < n - 5 ==> #[trigger] s[j] != 46
}

pub proof fn lemma_enc_facts(scope: Seq<u8>, id: Seq<u8>, mark: Seq<u8>, offset: u64)
    requires
        fields_ok(scope, id, mark),
    ensures
        enc_facts(scope, id, mark, offset),
{
    let s = spec_enc(scope, id, mark, offset);
    let a = scope.len() as int;
    let b = id.len() as int;
    let m = mark.len() as int;
    let h16 = hex_n(offset as nat, 16);
    lemma_hex_n_len(offset as nat, 16);
    lemma_hex_n_bytes(offset as nat, 16);
    let bd = body(scope, id, mark, offset);
    let x = sum(bd);
    let hx = hex_n(x, 4);
    lemma_hex_n_len(x, 4);
    lemma_hex_n_bytes(x, 4);
    let n = s.len() as int;
    assert(bd.len() == a + b + m + 22);
    assert(n == a + b + m + 27);
    assert(s.subrange(0, n - 5) =~= bd);
    assert(s.subrange(n - 5, n) =~= tail(x));
    assert(s.subrange(n - 4, n) =~= hx);
    assert(s.subrange(3, 3 + a) =~= scope);
    assert(s.subrange(4 + a, 4 + a + b) =~= id);
    assert(s.subrange(5 + a + b, 5 + a + b + m) =~= mark);
    assert(s.subrange(n - 21, n - 5) =~= h16);
    // The check fits four digits: the body is at most 150 bytes of at most 255.
    lemma_sum_bound(bd);
    assert(sum(bd) <= 255 * 150) by (nonlinear_arith)
        requires
            sum(bd) <= 255 * bd.len(),
            bd.len() <= 150,
    ;
    lemma_body_dots(scope, id, mark, offset);
    assert forall|i: int| 0 <= i < hx.len() implies #[trigger] hx[i] != 46 by {
        assert(is_hex(hx[i]));
    }
    lemma_dots_zero(hx);
    lemma_dots_add(bd, tail(x));
    lemma_dots_add(dot(), hx);
    assert(dots(dot()) == 1) by {
        reveal_with_fuel(dots, 2);
        assert(dot().drop_last() =~= Seq::<u8>::empty());
    }
    assert forall|j: int| 3 <= j < 3 + a implies scope_byte(#[trigger] s[j]) by {
        assert(s[j] == scope[j - 3]);
    }
    assert forall|j: int| 4 + a <= j < 4 + a + b implies id_byte(#[trigger] s[j]) by {
        assert(s[j] == id[j - (4 + a)]);
    }
    assert forall|j: int| 5 + a + b <= j < n - 22 implies mark_byte(#[trigger] s[j]) by {
        assert(s[j] == mark[j - (5 + a + b)]);
    }
    assert forall|j: int| n - 21 <= j < n - 5 implies is_hex(#[trigger] s[j]) by {
        assert(s[j] == h16[j - (n - 21)]);
    }
    assert forall|j: int| n - 4 <= j < n implies is_hex(#[trigger] s[j]) by {
        assert(s[j] == hx[j - (n - 4)]);
    }
    assert forall|j: int| n - 21 <= j < n - 5 implies #[trigger] s[j] != 46 by {
        assert(s[j] == h16[j - (n - 21)]);
        assert(is_hex(h16[j - (n - 21)]));
    }
}

/// The facts about an accepted string that the refusals below rest on.
proof fn lemma_accepts_facts(s: Seq<u8>)
    requires
        accepts(s),
    ensures
        30 <= s.len() <= 155,
        s[s.len() - 5] == 46,
        s.subrange(s.len() - 5, s.len() as int) == tail(sum(s.subrange(0, s.len() - 5))),
        sum(s.subrange(0, s.len() - 5)) < 65536,
        dots(s) == 5,
        forall|j: int| s.len() - 21 <= j < s.len() - 5 ==> #[trigger] s[j] != 46,
{
    let (scope, id, mark, offset) = choose|scope: Seq<u8>, id: Seq<u8>, mark: Seq<u8>, offset: u64|
        fields_ok(scope, id, mark) && s == spec_enc(scope, id, mark, offset);
    lemma_enc_facts(scope, id, mark, offset);
}

/// The check, read back: two sums below 65536 with one tail are one sum.
proof fn lemma_tail_injective(x: nat, y: nat)
    requires
        x < 65536,
        y < 65536,
        tail(x) == tail(y),
    ensures
        x == y,
{
    lemma_pow16_values();
    lemma_hex_roundtrip(x, 4);
    lemma_hex_roundtrip(y, 4);
    assert(tail(x).drop_first() =~= hex_n(x, 4));
    assert(tail(y).drop_first() =~= hex_n(y, 4));
    assert(hex_n(x, 4) == hex_n(y, 4));
}

/// C3a: changing any one byte of an accepted string leaves a string that is not accepted.
pub proof fn lemma_single_byte_change_refused(s: Seq<u8>, i: int, b: u8)
    requires
        accepts(s),
        0 <= i < s.len(),
        b != s[i],
    ensures
        !accepts(s.update(i, b)),
{
    let t = s.update(i, b);
    if accepts(t) {
        lemma_accepts_facts(s);
        lemma_accepts_facts(t);
        let n = s.len() as int;
        assert(t.len() == n);
        let bs = s.subrange(0, n - 5);
        let bt = t.subrange(0, n - 5);
        if i < n - 5 {
            assert(bt =~= bs.update(i, b));
            lemma_sum_one_diff(bs, i, b);
            assert(s.subrange(n - 5, n) =~= t.subrange(n - 5, n));
            lemma_tail_injective(sum(bs), sum(bt));
        } else {
            assert(bs =~= bt);
            assert(s.subrange(n - 5, n) == t.subrange(n - 5, n));
            assert(t[i] == s[i]) by {
                assert(t.subrange(n - 5, n)[i - (n - 5)] == t[i]);
                assert(s.subrange(n - 5, n)[i - (n - 5)] == s[i]);
            }
        }
    }
}

/// C3b: a strict prefix of an accepted string is not accepted (a cursor cut short).
pub proof fn lemma_prefix_refused(s: Seq<u8>, k: int)
    requires
        accepts(s),
        0 <= k < s.len(),
    ensures
        !accepts(s.subrange(0, k)),
{
    let t = s.subrange(0, k);
    if accepts(t) {
        lemma_accepts_facts(s);
        lemma_accepts_facts(t);
        let n = s.len() as int;
        // Both have five dots, so the rest of `s` has none; its last five bytes start with one.
        let rest = s.subrange(k, n);
        assert(s =~= t + rest);
        lemma_dots_add(t, rest);
        if k <= n - 5 {
            assert(rest[n - 5 - k] == 46);
            lemma_dots_some(rest, n - 5 - k);
        }
        // So `k` is within four of the end, and `t`'s own `.` at `k-5` falls among `s`'s offset digits.
        assert(t[k - 5] == 46);
        assert(t[k - 5] == s[k - 5]);
    }
}

/// C3c: an accepted string with anything after it is not accepted (a cursor with a tail glued on).
pub proof fn lemma_extension_refused(s: Seq<u8>, extra: Seq<u8>)
    requires
        accepts(s),
        extra.len() > 0,
    ensures
        !accepts(s + extra),
{
    let u = s + extra;
    if accepts(u) {
        assert(u.subrange(0, s.len() as int) =~= s);
        lemma_prefix_refused(u, s.len() as int);
    }
}

// ---------------------------------------------------------------------------------------------
// The cursor and its errors.
// ---------------------------------------------------------------------------------------------

/// The field a refusal names.
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// A `.` that is not where the format puts it, or a missing one.
    Separator,
    Scope,
    Id,
    Mark,
    Offset,
    Check,
}

/// Why a string is not a cursor.
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CursorError {
    /// It does not start `j1`: another format, or not a cursor at all.
    BadVersion,
    /// Shorter than the shortest cursor or longer than the longest.
    BadLength,
    /// Well formed, and its check does not match the bytes before it.
    BadChecksum,
    /// A field is empty, too long, outside its alphabet, or a separator is missing.
    BadField(Field),
}

/// One job incarnation at one output offset: `scope` and `id` name the job, `mark` names this incarnation
/// of it (a reused id has another mark), `offset` is the next output byte not yet delivered. The fields are
/// valid by construction ([`Cursor::new`], [`decode`]), which is what the type invariant says.
#[derive(Debug, PartialEq, Eq)]
pub struct Cursor {
    scope: Vec<u8>,
    id: Vec<u8>,
    mark: Vec<u8>,
    offset: u64,
}

impl Cursor {
    #[verifier::type_invariant]
    spec fn inv(self) -> bool {
        fields_ok(self.scope@, self.id@, self.mark@)
    }

    pub closed spec fn scope_v(self) -> Seq<u8> {
        self.scope@
    }

    pub closed spec fn id_v(self) -> Seq<u8> {
        self.id@
    }

    pub closed spec fn mark_v(self) -> Seq<u8> {
        self.mark@
    }

    pub closed spec fn offset_v(self) -> u64 {
        self.offset
    }

    /// C4: every cursor's fields are within the rules, whoever made it.
    pub fn scope(&self) -> (r: &[u8])
        ensures
            r@ == self.scope_v(),
            fields_ok(self.scope_v(), self.id_v(), self.mark_v()),
    {
        proof {
            use_type_invariant(self);
        }
        self.scope.as_slice()
    }

    pub fn id(&self) -> (r: &[u8])
        ensures
            r@ == self.id_v(),
            fields_ok(self.scope_v(), self.id_v(), self.mark_v()),
    {
        proof {
            use_type_invariant(self);
        }
        self.id.as_slice()
    }

    pub fn mark(&self) -> (r: &[u8])
        ensures
            r@ == self.mark_v(),
            fields_ok(self.scope_v(), self.id_v(), self.mark_v()),
    {
        proof {
            use_type_invariant(self);
        }
        self.mark.as_slice()
    }

    /// The cursor of `(scope, id, mark)` at `offset`, if the three obey the rules.
    pub fn new(scope: &[u8], id: &[u8], mark: &[u8], offset: u64) -> (r: Result<Cursor, CursorError>)
        ensures
            (r is Ok) == fields_ok(scope@, id@, mark@),
            r is Ok ==> (r->Ok_0.scope_v() == scope@ && r->Ok_0.id_v() == id@ && r->Ok_0.mark_v() == mark@
                && r->Ok_0.offset_v() == offset),
    {
        if scope.len() < 1 || scope.len() > 64 || !all_ok(scope, 0, scope.len(), 0) {
            return Err(CursorError::BadField(Field::Scope));
        }
        if id.len() < 1 || id.len() > 32 || !all_ok(id, 0, id.len(), 1) {
            return Err(CursorError::BadField(Field::Id));
        }
        if mark.len() < 1 || mark.len() > 32 || !all_ok(mark, 0, mark.len(), 2) {
            return Err(CursorError::BadField(Field::Mark));
        }
        let c = Cursor { scope: copy_range(scope, 0, scope.len()), id: copy_range(id, 0, id.len()),
            mark: copy_range(mark, 0, mark.len()), offset };
        proof {
            assert(c.scope@ =~= scope@);
            assert(c.id@ =~= id@);
            assert(c.mark@ =~= mark@);
        }
        Ok(c)
    }

    /// C2: the offset, whole.
    pub fn offset(&self) -> (r: u64)
        ensures
            r == self.offset_v(),
    {
        self.offset
    }
}

// ---------------------------------------------------------------------------------------------
// Byte loops. Each one says exactly what it computed.
// ---------------------------------------------------------------------------------------------

/// Which alphabet: 0 scope, 1 id, 2 hex (a mark, an offset, a check).
pub open spec fn kind_ok(kind: u8, b: u8) -> bool {
    if kind == 0 { scope_byte(b) } else if kind == 1 { id_byte(b) } else { is_hex(b) }
}

fn byte_ok(kind: u8, b: u8) -> (r: bool)
    ensures
        r == kind_ok(kind, b),
{
    let digit = 48 <= b && b <= 57;
    if kind == 0 {
        digit || (97 <= b && b <= 122) || b == 45
    } else if kind == 1 {
        digit || (97 <= b && b <= 122)
    } else {
        digit || (97 <= b && b <= 102)
    }
}

/// Whether every byte of `s[from..to]` is in the alphabet `kind`.
fn all_ok(s: &[u8], from: usize, to: usize, kind: u8) -> (r: bool)
    requires
        from <= to <= s.len(),
    ensures
        r == (forall|j: int| from <= j < to ==> kind_ok(kind, #[trigger] s[j])),
{
    let mut i = from;
    while i < to
        invariant
            from <= i <= to <= s.len(),
            forall|j: int| from <= j < i ==> kind_ok(kind, #[trigger] s[j]),
        decreases to - i,
    {
        if !byte_ok(kind, s[i]) {
            return false;
        }
        i += 1;
    }
    true
}

/// The first `.` in `s[from..to]`.
fn find_dot(s: &[u8], from: usize, to: usize) -> (r: Option<usize>)
    requires
        from <= to <= s.len(),
    ensures
        match r {
            Some(i) => from <= i < to && s[i as int] == 46
                && forall|j: int| from <= j < i ==> #[trigger] s[j] != 46,
            None => forall|j: int| from <= j < to ==> #[trigger] s[j] != 46,
        },
{
    let mut i = from;
    while i < to
        invariant
            from <= i <= to <= s.len(),
            forall|j: int| from <= j < i ==> #[trigger] s[j] != 46,
        decreases to - i,
    {
        if s[i] == 46 {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// `s[from..to]` as a vector.
fn copy_range(s: &[u8], from: usize, to: usize) -> (v: Vec<u8>)
    requires
        from <= to <= s.len(),
    ensures
        v@ == s@.subrange(from as int, to as int),
{
    let mut v = Vec::new();
    let mut i = from;
    while i < to
        invariant
            from <= i <= to <= s.len(),
            v@ == s@.subrange(from as int, i as int),
        decreases to - i,
    {
        v.push(s[i]);
        proof {
            assert(s@.subrange(from as int, i as int + 1) =~= s@.subrange(from as int, i as int).push(s[i as int]));
        }
        i += 1;
    }
    v
}

/// Whether two byte strings are the same.
fn same(a: &[u8], b: &[u8]) -> (r: bool)
    ensures
        r == (a@ == b@),
{
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len()
        invariant
            a.len() == b.len(),
            0 <= i <= a.len(),
            forall|j: int| 0 <= j < i ==> a[j] == b[j],
        decreases a.len() - i,
    {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    proof {
        assert(a@ =~= b@);
    }
    true
}

/// Append `s` to `out`.
fn append(out: &mut Vec<u8>, s: &[u8])
    ensures
        final(out)@ == old(out)@ + s@,
{
    let mut i = 0;
    while i < s.len()
        invariant
            0 <= i <= s.len(),
            out@ == old(out)@ + s@.subrange(0, i as int),
        decreases s.len() - i,
    {
        out.push(s[i]);
        proof {
            assert(s@.subrange(0, i as int + 1) =~= s@.subrange(0, i as int).push(s[i as int]));
            assert(old(out)@ + s@.subrange(0, i as int + 1) =~= (old(out)@ + s@.subrange(0, i as int)).push(s[i as int]));
        }
        i += 1;
    }
    proof {
        assert(s@.subrange(0, s.len() as int) =~= s@);
    }
}

/// The sum of `s[..to]`.
fn sum_of(s: &[u8], to: usize) -> (r: u32)
    requires
        to <= s.len(),
        to <= 1000,
    ensures
        r as nat == sum(s@.subrange(0, to as int)),
{
    let to = if cfg!(feature = "mut_check_skips_a_byte") && to > 0 { to - 1 } else { to };
    let mut i = 0;
    let mut total: u32 = 0;
    while i < to
        invariant
            0 <= i <= to <= s.len(),
            to <= 1000,
            total as nat == sum(s@.subrange(0, i as int)),
            total <= 255 * i,
        decreases to - i,
    {
        proof {
            assert(s@.subrange(0, i as int + 1).drop_last() =~= s@.subrange(0, i as int));
            assert(s@.subrange(0, i as int + 1).last() == s[i as int]);
        }
        total += s[i] as u32;
        i += 1;
    }
    total
}

/// Write `v` as `n` hex digits.
fn hex_push(v: u64, n: usize, out: &mut Vec<u8>)
    requires
        n <= 16,
    ensures
        final(out)@ == old(out)@ + hex_n(v as nat, n as nat),
    decreases n,
{
    if n > 0 {
        hex_push(v / 16, n - 1, out);
        let d = (v % 16) as u8;
        let c = if d < 10 { 48 + d } else { 87 + d };
        out.push(c);
        proof {
            assert(c == hexc((v as nat) % 16));
            assert(hex_n(v as nat, n as nat) =~= hex_n((v as nat) / 16, (n - 1) as nat).push(c));
            assert(old(out)@ + hex_n(v as nat, n as nat) =~= (old(out)@ + hex_n((v / 16) as nat, (n - 1) as nat)).push(c));
        }
    }
}

/// The number the `n` hex digits `s[from..from+n]` spell.
fn parse_hex(s: &[u8], from: usize, n: usize) -> (r: u64)
    requires
        from + n <= s.len(),
        n <= 16,
        forall|j: int| from <= j < from + n ==> is_hex(#[trigger] s[j]),
    ensures
        r as nat == hex_val(s@.subrange(from as int, (from + n) as int)),
{
    let mut acc: u64 = 0;
    let mut i = 0;
    while i < n
        invariant
            0 <= i <= n <= 16,
            from + n <= s.len(),
            forall|j: int| from <= j < from + n ==> is_hex(#[trigger] s[j]),
            acc as nat == hex_val(s@.subrange(from as int, (from + i) as int)),
        decreases n - i,
    {
        let b = s[from + i];
        let d: u64 = if b <= 57 { (b - 48) as u64 } else { (b - 87) as u64 };
        proof {
            let p = s@.subrange(from as int, (from + i) as int);
            assert forall|j: int| 0 <= j < p.len() implies is_hex(#[trigger] p[j]) by {
                assert(p[j] == s[from + j]);
            }
            lemma_hex_val_bound(p);
            lemma_pow16_values();
            lemma_pow16_mono(i as nat, 15);
            reveal_with_fuel(pow16, 16);
            let q = s@.subrange(from as int, (from + i + 1) as int);
            assert(q.drop_last() =~= p);
            assert(q.last() == b);
            assert(d as nat == nib(b));
        }
        acc = acc * 16 + d;
        i += 1;
    }
    acc
}

// ---------------------------------------------------------------------------------------------
// Encode and decode.
// ---------------------------------------------------------------------------------------------

/// The bytes of the encoding of `(scope, id, mark, offset)`: the string [`encode`] returns.
pub fn encode_fields(scope: &[u8], id: &[u8], mark: &[u8], offset: u64) -> (v: Vec<u8>)
    requires
        fields_ok(scope@, id@, mark@),
    ensures
        v@ == spec_enc(scope@, id@, mark@, offset),
{
    let mut out: Vec<u8> = Vec::new();
    out.push(106);
    out.push(49);
    out.push(46);
    append(&mut out, scope);
    out.push(46);
    append(&mut out, id);
    out.push(46);
    append(&mut out, mark);
    out.push(46);
    let o = if cfg!(feature = "mut_offset_truncated") { offset & 0xFFFF_FFFF } else { offset };
    hex_push(o, 16, &mut out);
    proof {
        assert(out@ =~= body(scope@, id@, mark@, offset)) by {
            assert(out@ =~= seq![106u8, 49u8, 46u8] + scope@ + dot() + id@ + dot() + mark@ + dot()
                + hex_n(o as nat, 16));
        }
    }
    proof {
        lemma_hex_n_len(offset as nat, 16);
        assert(out@.len() == scope@.len() + id@.len() + mark@.len() + 22);
    }
    let check = sum_of(out.as_slice(), out.len());
    proof {
        assert(out@.subrange(0, out.len() as int) =~= out@);
        lemma_enc_facts(scope@, id@, mark@, offset);
    }
    out.push(46);
    hex_push(check as u64, 4, &mut out);
    proof {
        assert(out@ =~= spec_enc(scope@, id@, mark@, offset));
    }
    out
}

/// The bytes of a cursor's encoding.
pub fn encode_bytes(c: &Cursor) -> (v: Vec<u8>)
    ensures
        v@ == spec_enc(c.scope_v(), c.id_v(), c.mark_v(), c.offset_v()),
{
    proof {
        use_type_invariant(c);
    }
    encode_fields(c.scope.as_slice(), c.id.as_slice(), c.mark.as_slice(), c.offset)
}

/// The decision: whether `input` is exactly the encoding of a cursor, and which. The refusals say which part
/// is wrong; the proof says the answer is `Ok` exactly for the strings [`encode_bytes`] writes (`accepts`),
/// and that what it returns is the cursor that was written.
#[verifier::rlimit(100)]
fn decode_core(input: &[u8]) -> (r: Result<Cursor, CursorError>)
    ensures
        (r is Ok) == accepts(input@),
        r matches Ok(c) ==> input@ == spec_enc(c.scope_v(), c.id_v(), c.mark_v(), c.offset_v()),
        forall|sc: Seq<u8>, id: Seq<u8>, mark: Seq<u8>, off: u64|
            fields_ok(sc, id, mark) && input@ == #[trigger] spec_enc(sc, id, mark, off)
                ==> (r is Ok && r->Ok_0.scope_v() == sc && r->Ok_0.id_v() == id && r->Ok_0.mark_v() == mark
                && r->Ok_0.offset_v() == off),
        r is Ok ==> fields_ok(r->Ok_0.scope_v(), r->Ok_0.id_v(), r->Ok_0.mark_v()),
{
    let s: &[u8] = input;
    proof {
        assert forall|sc: Seq<u8>, id: Seq<u8>, mark: Seq<u8>, off: u64|
            fields_ok(sc, id, mark) && s@ == #[trigger] spec_enc(sc, id, mark, off)
            implies enc_facts(sc, id, mark, off) by {
            lemma_enc_facts(sc, id, mark, off);
        }
    }
    let n = s.len();
    if n < 30 || n > 155 {
        return Err(CursorError::BadLength);
    }
    if s[0] != 106 || s[1] != 49 {
        return Err(CursorError::BadVersion);
    }
    if s[2] != 46 || s[n - 5] != 46 {
        return Err(CursorError::BadField(Field::Separator));
    }
    let end = n - 22;
    let Some(p1) = find_dot(s, 3, end) else {
        return Err(CursorError::BadField(Field::Separator));
    };
    let Some(p2) = find_dot(s, p1 + 1, end) else {
        return Err(CursorError::BadField(Field::Separator));
    };
    if p1 < 4 || p1 > 67 || (!cfg!(feature = "mut_scope_unchecked") && !all_ok(s, 3, p1, 0)) {
        return Err(CursorError::BadField(Field::Scope));
    }
    if p2 < p1 + 2 || p2 > p1 + 33 || (!cfg!(feature = "mut_id_unchecked") && !all_ok(s, p1 + 1, p2, 1)) {
        return Err(CursorError::BadField(Field::Id));
    }
    if end < p2 + 2 || end > p2 + 33 || (!cfg!(feature = "mut_mark_unchecked") && !all_ok(s, p2 + 1, end, 2)) {
        return Err(CursorError::BadField(Field::Mark));
    }
    if !all_ok(s, n - 21, n - 5, 2) {
        return Err(CursorError::BadField(Field::Offset));
    }
    if !all_ok(s, n - 4, n, 2) {
        return Err(CursorError::BadField(Field::Check));
    }
    let offset = parse_hex(s, n - 21, 16);
    let check = parse_hex(s, n - 4, 4);
    let computed = sum_of(s, n - 5);
    proof {
        assert forall|sc: Seq<u8>, id: Seq<u8>, mark: Seq<u8>, off: u64|
            fields_ok(sc, id, mark) && s@ == #[trigger] spec_enc(sc, id, mark, off)
            implies check as nat == computed as nat && offset == off by {
            lemma_enc_facts(sc, id, mark, off);
            lemma_offset_hex_round_trip(off);
            lemma_pow16_values();
            lemma_hex_roundtrip(sum(s@.subrange(0, n - 5)), 4);
        }
    }
    if !cfg!(feature = "mut_check_skipped") && check != computed as u64 {
        return Err(CursorError::BadChecksum);
    }
    let c = Cursor {
        scope: copy_range(s, 3, p1),
        id: copy_range(s, p1 + 1, p2),
        mark: copy_range(s, p2 + 1, end),
        offset,
    };
    proof {
        assert forall|sc: Seq<u8>, id: Seq<u8>, mark: Seq<u8>, off: u64|
            fields_ok(sc, id, mark) && s@ == #[trigger] spec_enc(sc, id, mark, off)
            implies c.scope@ == sc && c.id@ == id && c.mark@ == mark && c.offset == off by {
            lemma_enc_facts(sc, id, mark, off);
            assert(c.scope@ =~= sc);
            assert(c.id@ =~= id);
            assert(c.mark@ =~= mark);
        }
    }
    // The string must be the cursor's own encoding, byte for byte: this is what makes a cursor have one
    // string, and what `Ok` means.
    proof {
        assert(fields_ok(c.scope@, c.id@, c.mark@));
    }
    let again = encode_fields(c.scope.as_slice(), c.id.as_slice(), c.mark.as_slice(), c.offset);
    if !cfg!(feature = "mut_check_skipped") && !same(again.as_slice(), s) {
        return Err(CursorError::BadChecksum);
    }
    proof {
        assert(accepts(input@)) by {
            assert(input@ == s@);
        }
    }
    Ok(c)
}

/// Whether `input` is a cursor, and which: [`decode_core`] under the rules of the cursor's refusals. The
/// proof says the answer is `Ok` exactly for the strings [`encode_bytes`] writes (C1), that what it returns
/// is the cursor that was written (C1), that nothing altered by one byte (C3a), cut short (C3b) or with
/// bytes after it (C3c) from a valid cursor is accepted, and that a decoded cursor obeys the field rules (C4).
pub fn decode_bytes(input: &[u8]) -> (r: Result<Cursor, CursorError>)
    ensures
        (r is Ok) == accepts(input@),
        r matches Ok(c) ==> input@ == spec_enc(c.scope_v(), c.id_v(), c.mark_v(), c.offset_v()),
        forall|sc: Seq<u8>, id: Seq<u8>, mark: Seq<u8>, off: u64|
            fields_ok(sc, id, mark) && input@ == #[trigger] spec_enc(sc, id, mark, off)
                ==> (r is Ok && r->Ok_0.scope_v() == sc && r->Ok_0.id_v() == id && r->Ok_0.mark_v() == mark
                && r->Ok_0.offset_v() == off),
        // C3a: one byte changed in a valid cursor
        forall|orig: Seq<u8>, i: int, b: u8|
            accepts(orig) && 0 <= i < orig.len() && b != orig[i] && input@ == #[trigger] orig.update(i, b)
                ==> r is Err,
        // C3b: a valid cursor cut short
        forall|orig: Seq<u8>, k: int|
            accepts(orig) && 0 <= k < orig.len() && input@ == #[trigger] orig.subrange(0, k) ==> r is Err,
        // C3c: a valid cursor with bytes after it
        forall|orig: Seq<u8>, extra: Seq<u8>|
            accepts(orig) && extra.len() > 0 && input@ == #[trigger] (orig + extra) ==> r is Err,
        // C4: the fields of a decoded cursor obey the rules
        r is Ok ==> fields_ok(r->Ok_0.scope_v(), r->Ok_0.id_v(), r->Ok_0.mark_v()),
{
    let r = decode_core(input);
    if cfg!(feature = "mut_rejects_max_offset") {
        if let Ok(c) = &r {
            if c.offset() == u64::MAX {
                return Err(CursorError::BadField(Field::Offset));
            }
        }
    }
    if cfg!(feature = "mut_prefix_accepted") && r.is_err() {
        // A cursor missing its last byte is read as if the byte were `0`.
        let mut padded: Vec<u8> = Vec::new();
        append(&mut padded, input);
        padded.push(48);
        return decode_core(padded.as_slice());
    }
    if cfg!(feature = "mut_trailing_accepted") && r.is_err() && input.len() > 0
        && input[input.len() - 1] == 10
    {
        // A trailing newline is dropped.
        return decode_core(vstd::slice::slice_subrange(input, 0, input.len() - 1));
    }
    proof {
        assert forall|orig: Seq<u8>, i: int, b: u8|
            accepts(orig) && 0 <= i < orig.len() && b != orig[i] && input@ == #[trigger] orig.update(i, b)
            implies r is Err by {
            lemma_single_byte_change_refused(orig, i, b);
        }
        assert forall|orig: Seq<u8>, k: int|
            accepts(orig) && 0 <= k < orig.len() && input@ == #[trigger] orig.subrange(0, k)
            implies r is Err by {
            lemma_prefix_refused(orig, k);
        }
        assert forall|orig: Seq<u8>, extra: Seq<u8>|
            accepts(orig) && extra.len() > 0 && input@ == #[trigger] (orig + extra)
            implies r is Err by {
            lemma_extension_refused(orig, extra);
        }
    }
    r
}

} // verus!

// `Clone` is outside `verus!`: a clone of valid fields is valid, and the type invariant is checked where a
// cursor is made ([`Cursor::new`], [`decode_bytes`]).
impl Clone for Cursor {
    fn clone(&self) -> Self {
        Self {
            scope: self.scope.clone(),
            id: self.id.clone(),
            mark: self.mark.clone(),
            offset: self.offset,
        }
    }
}

impl Cursor {
    /// A cursor from strings: [`Cursor::new`] on their bytes.
    pub fn from_parts(
        scope: &str,
        id: &str,
        mark: &str,
        offset: u64,
    ) -> Result<Cursor, CursorError> {
        Cursor::new(scope.as_bytes(), id.as_bytes(), mark.as_bytes(), offset)
    }

    /// The scope, as the string it was made from (the fields are ASCII, so this is never the empty fallback).
    pub fn scope_str(&self) -> &str {
        std::str::from_utf8(&self.scope).unwrap_or_default()
    }

    pub fn id_str(&self) -> &str {
        std::str::from_utf8(&self.id).unwrap_or_default()
    }

    pub fn mark_str(&self) -> &str {
        std::str::from_utf8(&self.mark).unwrap_or_default()
    }
}

/// A cursor as the string a caller keeps: `j1.<scope>.<id>.<mark>.<offset>.<check>`.
pub fn encode(c: &Cursor) -> String {
    encode_bytes(c).into_iter().map(char::from).collect()
}

/// The cursor a string holds, or why it is not one.
pub fn decode(s: &str) -> Result<Cursor, CursorError> {
    decode_bytes(s.as_bytes())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    fn cur(offset: u64) -> Cursor {
        Cursor::from_parts("sess-1", "a1b2c3", "00ff00ff00ff00ff1a", offset).unwrap()
    }

    #[test]
    fn a_cursor_round_trips_at_every_offset_edge() {
        for off in [
            0,
            1,
            255,
            256,
            u64::from(u32::MAX),
            u64::from(u32::MAX) + 1,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let c = cur(off);
            let s = encode(&c);
            assert_eq!(decode(&s), Ok(c.clone()), "{s}");
            assert_eq!(decode(&s).unwrap().offset(), off);
        }
    }

    #[test]
    fn the_string_is_the_documented_format() {
        let s = encode(&cur(0x1f));
        assert_eq!(&s[..29], "j1.sess-1.a1b2c3.00ff00ff00ff");
        assert!(s.contains(".000000000000001f."));
        assert_eq!(s.len(), 3 + 6 + 1 + 6 + 1 + 18 + 1 + 16 + 1 + 4);
    }

    #[test]
    fn every_prefix_extension_and_single_byte_change_is_refused() {
        let s = encode(&cur(0xdead_beef_0000_0001));
        for k in 0..s.len() {
            assert!(decode(&s[..k]).is_err(), "prefix {k}");
        }
        for extra in ["0", ".", "\n", ".0000", "j1"] {
            assert!(
                decode(&format!("{s}{extra}")).is_err(),
                "extension {extra:?}"
            );
        }
        let bytes = s.as_bytes();
        for i in 0..bytes.len() {
            for b in 0u8..=127 {
                if b == bytes[i] {
                    continue;
                }
                let mut t = bytes.to_vec();
                t[i] = b;
                let t = String::from_utf8(t).unwrap();
                assert!(decode(&t).is_err(), "byte {i} -> {b}: {t}");
            }
        }
    }

    #[test]
    fn errors_are_typed() {
        let s = encode(&cur(7));
        assert_eq!(decode(""), Err(CursorError::BadLength));
        assert_eq!(
            decode(&s.replacen("j1", "j2", 1)),
            Err(CursorError::BadVersion)
        );
        let (head, check) = s.split_at(s.len() - 4);
        let other = if check == "0000" { "0001" } else { "0000" };
        assert_eq!(
            decode(&format!("{head}{other}")),
            Err(CursorError::BadChecksum)
        );
        assert_eq!(
            decode(&s.replacen("sess-1", "sess_1", 1)),
            Err(CursorError::BadField(Field::Scope))
        );
        assert_eq!(
            decode(&s.replace('.', ",")),
            Err(CursorError::BadField(Field::Separator))
        );
    }

    #[test]
    fn the_fields_obey_the_rules_of_the_job_endpoint() {
        let ok = |s: &str, i: &str, m: &str| Cursor::from_parts(s, i, m, 0);
        assert!(ok(&"a".repeat(64), &"b".repeat(32), &"c".repeat(32)).is_ok());
        assert!(ok(&"a".repeat(65), "b", "c").is_err());
        assert!(ok("a", &"b".repeat(33), "c").is_err());
        assert!(ok("a", "b", &"c".repeat(33)).is_err());
        assert!(ok("", "b", "c").is_err());
        assert!(ok("a", "", "c").is_err());
        assert!(ok("a", "b", "").is_err());
        assert!(ok("A", "b", "c").is_err());
        assert!(ok("a.b", "b", "c").is_err());
        assert!(ok("a", "b-c", "c").is_err(), "an id has no '-'");
        assert!(ok("a", "b", "g").is_err(), "a mark is hex");
        assert!(ok("a-b", "b", "c").is_ok(), "a scope has '-'");
    }
}
