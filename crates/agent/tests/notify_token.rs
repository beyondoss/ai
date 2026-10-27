//! The `bjn_v1` notify token as the crate makes and checks it (`beyond_agent::notify_token`),
//! held to `tests/fixtures/exec/notify-token.json` and to the spec's token requirements.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use beyond_agent::exec_endpoint::jobs::{JobId, JobScope};
use beyond_agent::notify_token::{NotifyKey, Verdict};
use serde_json::Value;

const FIXTURE: &str = include_str!("../../sandbox/tests/fixtures/exec/notify-token.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).unwrap()
}

fn secret(fx: &Value, name: &str) -> [u8; 32] {
    STANDARD
        .decode(fx["secrets"][name].as_str().unwrap())
        .unwrap()
        .try_into()
        .unwrap()
}

fn current() -> NotifyKey {
    NotifyKey::from_secret_bytes(&secret(&fixture(), "current"))
}

fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap_or_else(|| panic!("{k} is a string"))
}

fn verdict(name: &str) -> Verdict {
    match name {
        "valid" => Verdict::Valid,
        "malformed" => Verdict::Malformed,
        "unknown_kid" => Verdict::UnknownKid,
        "bad_mac" => Verdict::BadMac,
        other => panic!("unknown verdict {other}"),
    }
}

/// The minted vector's inputs.
struct Minted {
    tenant: String,
    session: String,
    scope: String,
    id: String,
    token: String,
}

fn minted() -> Minted {
    let fx = fixture();
    let m = &fx["minted"];
    Minted {
        tenant: s(m, "tenant").into(),
        session: s(m, "session_id").into(),
        scope: s(m, "scope").into(),
        id: s(m, "id").into(),
        token: s(m, "token").into(),
    }
}

#[test]
fn each_key_has_the_fixtures_kid() {
    let fx = fixture();
    for name in ["current", "other"] {
        let bytes = secret(&fx, name);
        let want = s(&fx["keys"][name], "kid");
        assert_eq!(NotifyKey::from_secret_bytes(&bytes).kid(), want, "{name}");
        // Derived from the secret exactly as given: X25519 clamps it only when it is used.
        let static_secret = x25519_dalek::StaticSecret::from(bytes);
        assert_eq!(
            NotifyKey::from_seal_secret(&static_secret).kid(),
            want,
            "{name}"
        );
    }
}

#[test]
fn minting_reproduces_the_vector() {
    let m = minted();
    assert_eq!(
        JobScope::for_session(Some(&m.tenant), &m.session).name(),
        m.scope
    );
    let id = JobId::parse(&m.id).unwrap();
    assert_eq!(current().mint(&m.tenant, &m.session, &id), m.token);
    // No time, nonce or replica in it: minting again, or on another replica, gives the same token.
    let again = NotifyKey::from_secret_bytes(&secret(&fixture(), "current"));
    assert_eq!(again.mint(&m.tenant, &m.session, &id), m.token);
}

#[test]
fn every_token_vector_gets_its_verdict() {
    let fx = fixture();
    assert_eq!(fx["verifier_keys"], serde_json::json!(["current"]));
    let key = current();
    let vectors = fx["vectors"].as_array().unwrap();
    assert!(!vectors.is_empty());
    for v in vectors {
        let cx = &v["context"];
        let got = key.verify(
            s(v, "token"),
            s(cx, "tenant"),
            s(cx, "session_id"),
            s(cx, "scope"),
            s(cx, "id"),
        );
        assert_eq!(got, verdict(s(v, "expect")), "{}", s(v, "name"));
    }
}

#[test]
fn verdicts_map_to_their_statuses() {
    assert_eq!(Verdict::Valid.refusal_status(), None);
    assert_eq!(Verdict::Malformed.refusal_status(), Some(401));
    assert_eq!(Verdict::BadMac.refusal_status(), Some(401));
    assert_eq!(Verdict::UnknownKid.refusal_status(), Some(503));
}

/// Every single-byte change to the token, or to any field of the request it is checked against,
/// makes it fail; so do a dropped or an added character.
#[test]
fn any_byte_change_fails() {
    let m = minted();
    let key = current();
    let ok = |token: &str, tenant: &str, session: &str, scope: &str, id: &str| {
        key.verify(token, tenant, session, scope, id) == Verdict::Valid
    };
    assert!(ok(&m.token, &m.tenant, &m.session, &m.scope, &m.id));

    // Each field, each position, each other ASCII byte, and one multi-byte character.
    let variants = |orig: &str| -> Vec<String> {
        let mut out = Vec::new();
        for i in 0..orig.len() {
            for b in 0u8..0x80 {
                if b == orig.as_bytes()[i] {
                    continue;
                }
                let mut bytes = orig.as_bytes().to_vec();
                bytes[i] = b;
                out.push(String::from_utf8(bytes).unwrap());
            }
            out.push(format!("{}é{}", &orig[..i], &orig[i + 1..]));
            out.push(format!("{}{}", &orig[..i], &orig[i + 1..]));
            out.push(format!("{}x{}", &orig[..i], &orig[i..]));
        }
        out.push(format!("{orig}x"));
        out
    };
    let mut checked = 0usize;
    for t in variants(&m.token) {
        assert!(!ok(&t, &m.tenant, &m.session, &m.scope, &m.id), "{t:?}");
        checked += 1;
    }
    for t in variants(&m.tenant) {
        assert!(!ok(&m.token, &t, &m.session, &m.scope, &m.id), "{t:?}");
        checked += 1;
    }
    for t in variants(&m.session) {
        assert!(!ok(&m.token, &m.tenant, &t, &m.scope, &m.id), "{t:?}");
        checked += 1;
    }
    for t in variants(&m.scope) {
        assert!(!ok(&m.token, &m.tenant, &m.session, &t, &m.id), "{t:?}");
        checked += 1;
    }
    for t in variants(&m.id) {
        assert!(!ok(&m.token, &m.tenant, &m.session, &m.scope, &t), "{t:?}");
        checked += 1;
    }
    assert!(checked > 10_000, "{checked}");
    // The length prefixes keep fields apart: moving a byte from one field into the next is not
    // the same request.
    let (t, rest) = m.tenant.split_at(m.tenant.len() - 1);
    assert!(!ok(
        &m.token,
        t,
        &format!("{rest}{}", m.session),
        &m.scope,
        &m.id
    ));
}

/// The tag is compared by `hmac::Mac::verify_slice` (constant time), never by `==`, and in full:
/// a tag that is a correct prefix is not a match, and a difference anywhere is caught.
#[test]
fn the_tag_is_compared_in_constant_time() {
    let src = include_str!("../src/notify_token.rs");
    let code: String = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        code.matches(".verify_slice(&tag)").count(),
        1,
        "verify goes through verify_slice exactly once"
    );
    for forbidden in ["== tag", "tag ==", "tag[..", "verify_truncated", "ct_eq"] {
        assert!(
            !code.contains(forbidden),
            "{forbidden:?} in notify_token.rs"
        );
    }

    let m = minted();
    let key = current();
    let (head, mac) = m.token.rsplit_once('.').unwrap();
    // A truncated tag never verifies, whatever its length.
    for n in 0..mac.len() {
        let t = format!("{head}.{}", &mac[..n]);
        assert_ne!(
            key.verify(&t, &m.tenant, &m.session, &m.scope, &m.id),
            Verdict::Valid,
            "{t}"
        );
    }
    // A wrong tag is BadMac whether it differs in its first character or its last.
    let flip = |i: usize| {
        let mut b = mac.as_bytes().to_vec();
        b[i] = if b[i] == b'A' { b'Q' } else { b'A' };
        format!("{head}.{}", String::from_utf8(b).unwrap())
    };
    for i in [0, mac.len() / 2] {
        assert_eq!(
            key.verify(&flip(i), &m.tenant, &m.session, &m.scope, &m.id),
            Verdict::BadMac
        );
    }
}

/// A replica mints only for the session's own scope: the scope is computed from the tenant and
/// session, never taken from the caller, so a minted token fails against any other scope.
#[test]
fn mints_only_its_own_scope() {
    let key = current();
    let id = JobId::parse("0123456789abcdef").unwrap();
    let sessions = [
        ("acme", "s1"),
        ("acme", "s2"),
        ("globex", "s1"),
        ("a", "cme.s1"),
    ];
    for (tenant, session) in sessions {
        let token = key.mint(tenant, session, &id);
        let own = JobScope::for_session(Some(tenant), session);
        assert_eq!(
            key.verify(&token, tenant, session, own.name(), id.as_str()),
            Verdict::Valid
        );
        for (t2, s2) in sessions {
            if (t2, s2) == (tenant, session) {
                continue;
            }
            let other = JobScope::for_session(Some(t2), s2);
            // Its own URL with another session's scope, and another session's URL and scope.
            assert_eq!(
                key.verify(&token, tenant, session, other.name(), id.as_str()),
                Verdict::BadMac
            );
            assert_eq!(
                key.verify(&token, t2, s2, other.name(), id.as_str()),
                Verdict::BadMac
            );
        }
    }
    // Under exactly one key: another seal secret's token is an unknown kid, not a bad tag.
    let other = NotifyKey::from_secret_bytes(&secret(&fixture(), "other"));
    let token = other.mint("acme", "s1", &id);
    let scope = JobScope::for_session(Some("acme"), "s1");
    assert_eq!(
        key.verify(&token, "acme", "s1", scope.name(), id.as_str()),
        Verdict::UnknownKid
    );
    assert_eq!(
        other.verify(&token, "acme", "s1", scope.name(), id.as_str()),
        Verdict::Valid
    );
}

fn stat(b: u8) -> x25519_dalek::StaticSecret {
    x25519_dalek::StaticSecret::from([b; 32])
}

/// A seal-secret rotation: the replica holding the new secret refuses the old kid, unless it also
/// holds the old secret as the previous one; and it never mints under the previous one.
#[test]
fn previous_seal_secret_verifies_but_never_mints() {
    let job = JobId::parse("0123456789abcdef").unwrap();
    let scope = JobScope::for_session(Some("t"), "s");
    let old = NotifyKey::from_seal_secret(&stat(1));
    let token = old.mint("t", "s", &job);

    let rotated = NotifyKey::from_seal_secret(&stat(2));
    assert_eq!(
        rotated.verify(&token, "t", "s", scope.name(), job.as_str()),
        Verdict::UnknownKid
    );

    let rotating = NotifyKey::from_seal_secret(&stat(2)).with_previous(&stat(1));
    assert_eq!(
        rotating.verify(&token, "t", "s", scope.name(), job.as_str()),
        Verdict::Valid
    );
    // A tampered tag under the previous kid is a bad mac, not an unknown kid.
    let mut bad = token.clone();
    bad.pop();
    bad.push(if token.ends_with('A') { 'B' } else { 'A' });
    assert_ne!(
        rotating.verify(&bad, "t", "s", scope.name(), job.as_str()),
        Verdict::Valid
    );
    // Minting is under the current secret only.
    let minted = rotating.mint("t", "s", &job);
    assert_eq!(minted, rotated.mint("t", "s", &job));
    assert_eq!(rotating.kid(), rotated.kid());
    assert_eq!(
        old.verify(&minted, "t", "s", scope.name(), job.as_str()),
        Verdict::UnknownKid
    );
}
