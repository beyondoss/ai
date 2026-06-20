//! Mint a deterministic **dev-only** virtual API key + its public signing key.
//!
//! ⚠️  DEV ONLY — NOT FOR PRODUCTION. This uses a *fixed*, public seed (`[7u8; 32]`)
//! so the keypair is reproducible across runs and across machines. Anyone can
//! reproduce it, so anyone can mint valid tokens for it. Never use it anywhere real.
//!
//! It prints three lines you can paste straight into a dev config:
//!   - `SIGNING_PUBKEY_B64=` — the Ed25519 public key, standard-base64 of the 32 raw
//!     bytes, in the exact form `verifying_key_from_value` accepts for the gateway's
//!     `[signing_keys]` table.
//!   - `KID=` — the key id the token is minted under.
//!   - `DEV_BAI_V1=` — a ready-to-use `bai_v1...` token for tenant 1 / vpc 1.
//!
//! Run it via: `cargo run --example mint_dev_key` (or `mise run ai:mint-dev-key`).

// Examples assert their own invariants — allow the panic surface that the crate
// otherwise denies, matching how the test/bench targets opt out at the file head.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use beyond_ai::key::{Keyring, VirtualKey, mint};
use ed25519_dalek::SigningKey;

fn main() {
    // FIXED seed → deterministic, reproducible keypair. DEV ONLY.
    const DEV_SEED: [u8; 32] = [7u8; 32];
    const KID: u32 = 1;

    eprintln!("================================================================");
    eprintln!("  DEV ONLY — NOT FOR PRODUCTION.");
    eprintln!("  Keypair is derived from a fixed public seed ([7u8; 32]).");
    eprintln!("  It is reproducible by anyone; do not use it for anything real.");
    eprintln!("================================================================");

    let signing_key = SigningKey::from_bytes(&DEV_SEED);
    let verifying_key = signing_key.verifying_key();

    // Standard base64 of the 32 raw public-key bytes — `verifying_key_from_value`
    // tries STANDARD first, so this is what the `[signing_keys]` config loads.
    let pubkey_b64 = STANDARD.encode(verifying_key.to_bytes());

    // Mint the default dev token: tenant 1 / vpc 1, under KID.
    let vk = VirtualKey {
        tenant_id: 1,
        vpc_id: 1,
    };
    let token = mint(&vk, KID, &signing_key);

    // Sanity check: the token must round-trip through a keyring holding this pubkey,
    // and recover the same identity. If this fails the printed values are useless.
    let mut ring = Keyring::new();
    ring.insert(KID, verifying_key);
    let recovered = ring.verify(&token).expect("dev token must verify");
    assert_eq!(recovered, vk, "verified identity must match minted identity");

    println!("SIGNING_PUBKEY_B64={pubkey_b64}");
    println!("KID={KID}");
    println!("DEV_BAI_V1={token}");
}
