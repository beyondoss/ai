//! Contract vectors for the virtual key: `verify/key_vectors.json` holds exact token strings for a
//! fixed public seed and the outcome a verifier must give each at a stated clock. The control plane
//! mints these tokens and the gateway checks them, so the file is the shared truth between them.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use beyond_gateway::__test::key::{KeyError, Keyring, VirtualKey, mint, mint_v2, mint_v3};
use serde_json::Value;

fn vectors() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../verify/key_vectors.json");
    serde_json::from_slice(&std::fs::read(path).expect("verify/key_vectors.json")).unwrap()
}

fn signing_key(v: &Value) -> ed25519_dalek::SigningKey {
    ed25519_dalek::SigningKey::from_bytes(&[v["seed_byte"].as_u64().unwrap() as u8; 32])
}

/// The minter reproduces every vector's token byte for byte.
/// claim: A5
#[test]
fn minting_reproduces_every_vector_token() {
    let v = vectors();
    let sk = signing_key(&v);
    let kid = v["kid"].as_u64().unwrap() as u32;
    for row in v["mint"].as_array().unwrap() {
        let id = VirtualKey {
            tenant_id: row["tenant_id"].as_u64().unwrap(),
            vpc_id: row["vpc_id"].as_u64().unwrap(),
            key_id: None,
        };
        let key_id = row["key_id"].as_u64();
        let got = match row["version"].as_u64().unwrap() {
            1 => mint(&id, kid, &sk),
            2 => mint_v2(&id, key_id.unwrap(), kid, &sk),
            3 => mint_v3(&id, key_id.unwrap(), row["exp"].as_u64().unwrap(), kid, &sk),
            n => panic!("version {n}"),
        };
        assert_eq!(got, row["token"].as_str().unwrap(), "{}", row["name"]);
    }
}

/// The verifier gives every vector its stated outcome at its stated clock.
/// claim: A5
#[test]
fn verifying_gives_every_vector_its_outcome() {
    let v = vectors();
    let sk = signing_key(&v);
    let kid = v["kid"].as_u64().unwrap() as u32;
    assert_eq!(v["leeway_secs"], 30, "the leeway is part of the contract");
    let ring: Keyring = [(kid, sk.verifying_key())].into_iter().collect();
    for row in v["verify"].as_array().unwrap() {
        let name = row["name"].as_str().unwrap();
        let got = ring.verify_at(row["token"].as_str().unwrap(), row["now"].as_u64().unwrap());
        match row["outcome"].as_str().unwrap() {
            "ok" => {
                let id = got.unwrap_or_else(|e| panic!("{name}: {e}"));
                let want = &row["identity"];
                assert_eq!(id.tenant_id, want["tenant_id"].as_u64().unwrap(), "{name}");
                assert_eq!(id.vpc_id, want["vpc_id"].as_u64().unwrap(), "{name}");
                assert_eq!(id.key_id, want["key_id"].as_u64(), "{name}");
            }
            want => {
                let label = match got.expect_err(name) {
                    KeyError::Expired => "expired",
                    KeyError::BadSignature => "bad_signature",
                    KeyError::Malformed => "malformed",
                    KeyError::UnknownKid(_) => "unknown_kid",
                    KeyError::BadVersion => "bad_version",
                    other => panic!("{name}: {other}"),
                };
                assert_eq!(label, want, "{name}");
            }
        }
    }
}
