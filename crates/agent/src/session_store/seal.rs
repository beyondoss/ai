use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use chacha20poly1305::aead::{AeadInPlace, KeyInit};
use chacha20poly1305::{Key, Tag, XChaCha20Poly1305, XNonce};
use hkdf::Hkdf;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use super::*;

// ---- Sealing -----------------------------------------------------------------------------------

/// Per-tenant sealing keys, derived once from the grant's DEK.
///
/// One codec serves a whole tenant, not one session: a listing, a fork and a preview all open *other*
/// ids through the same codec, so the session id is mixed in as **associated data** per call rather
/// than baked into the key. That keeps one key per tenant (cheap to hold, simple to rotate) while
/// still binding every transcript line to the exact session it was written for — moving a sealed line
/// from one session's file into another's makes it fail to open.
///
/// Three independent subkeys, so a compromise of one surface can't read another:
///
/// | surface | HKDF info | AAD |
/// |---|---|---|
/// | transcript lines | `bsg_v1 transcript` | `tenant ‖ 0x00 ‖ session_id` |
/// | `.listings.json` | `bsg_v1 listing` | `tenant` |
/// | memory documents | `bsg_v1 memory` | `tenant` |
///
/// Memory documents are **not** bound to their path: `rename` moves a document without rewriting it,
/// and a path-bound AAD would turn every rename into a re-seal (and a crash mid-rename into data
/// loss).
pub struct TenantCodec {
    pub(super) tenant: String,
    /// `e1.<kfp8>` — the framing every blob this codec writes carries, so a reader can tell which key
    /// sealed it across a rotation.
    pub(super) frame: String,
    pub(super) transcript: Zeroizing<[u8; 32]>,
    pub(super) listing: Zeroizing<[u8; 32]>,
    pub(super) memory: Zeroizing<[u8; 32]>,
}

impl std::fmt::Debug for TenantCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenantCodec")
            .field("tenant", &self.tenant)
            .field("key", &self.frame)
            .finish()
    }
}

impl TenantCodec {
    /// Derive a tenant's sealing keys from its data-encryption key (`grant.dek`).
    pub fn new(tenant: impl Into<String>, dek: &[u8; 32]) -> Self {
        let fp = Self::subkey(dek, b"bsg_v1 kfp");
        Self {
            tenant: tenant.into(),
            frame: format!("{SEAL_FRAME}{}", hex::encode(&fp[..4])),
            transcript: Self::subkey(dek, b"bsg_v1 transcript"),
            listing: Self::subkey(dek, b"bsg_v1 listing"),
            memory: Self::subkey(dek, b"bsg_v1 memory"),
        }
    }

    /// The tenant this codec seals for.
    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    /// HKDF-SHA256 with no salt — the DEK is already a uniformly random 32 bytes, so a salt would add
    /// nothing; `info` is what separates the three surfaces.
    pub(super) fn subkey(dek: &[u8; 32], info: &[u8]) -> Zeroizing<[u8; 32]> {
        let mut out = Zeroizing::new([0u8; 32]);
        // `expand` fails only for an output longer than 255 × 32 bytes, so 32 never does. The branch
        // is unreachable; it exists because leaving `out` at its all-zero initial value would be a
        // silent downgrade to a fixed key, and this crate refuses to `expect` its way past that.
        if Hkdf::<Sha256>::new(None, dek)
            .expand(info, out.as_mut())
            .is_err()
        {
            out.copy_from_slice(&Sha256::digest([&dek[..], info].concat().as_slice()));
        }
        out
    }

    /// `tenant ‖ 0x00 ‖ session_id` — the transcript AAD, built per opened id.
    pub(super) fn line_aad(&self, id: &str) -> Vec<u8> {
        let mut aad = Vec::with_capacity(self.tenant.len() + 1 + id.len());
        aad.extend_from_slice(self.tenant.as_bytes());
        aad.push(0);
        aad.extend_from_slice(id.as_bytes());
        aad
    }

    pub(super) fn seal(
        &self,
        key: &[u8; 32],
        aad: &[u8],
        plaintext: &[u8],
    ) -> std::io::Result<Vec<u8>> {
        let mut nonce = [0u8; SEAL_NONCE_LEN];
        getrandom::fill(&mut nonce).map_err(std::io::Error::other)?;
        let mut body = Vec::with_capacity(SEAL_NONCE_LEN + plaintext.len() + SEAL_TAG_LEN);
        body.extend_from_slice(&nonce);
        body.extend_from_slice(plaintext);
        let tag = XChaCha20Poly1305::new(Key::from_slice(&key[..]))
            .encrypt_in_place_detached(XNonce::from_slice(&nonce), aad, &mut body[SEAL_NONCE_LEN..])
            .map_err(|_| std::io::Error::other("sealing failed"))?;
        body.extend_from_slice(&tag);
        let mut out = Vec::with_capacity(self.frame.len() + 1 + body.len().div_ceil(3) * 4);
        out.extend_from_slice(self.frame.as_bytes());
        out.push(b':');
        out.extend_from_slice(URL_SAFE_NO_PAD.encode(&body).as_bytes());
        Ok(out)
    }

    pub(super) fn open(&self, key: &[u8; 32], aad: &[u8], blob: &[u8]) -> std::io::Result<Vec<u8>> {
        let bad = || seal_error("sealed blob is malformed");
        let rest = blob.strip_prefix(SEAL_FRAME.as_bytes()).ok_or_else(bad)?;
        let sep = rest.iter().position(|&b| b == b':').ok_or_else(bad)?;
        let mut body = URL_SAFE_NO_PAD
            .decode(&rest[sep + 1..])
            .map_err(|_| bad())?;
        if body.len() < SEAL_NONCE_LEN + SEAL_TAG_LEN {
            return Err(bad());
        }
        let (nonce, tail) = body.split_at_mut(SEAL_NONCE_LEN);
        let nonce = XNonce::from_slice(nonce).to_owned();
        let (ct, tag) = tail.split_at_mut(tail.len() - SEAL_TAG_LEN);
        let tag = Tag::from_slice(tag).to_owned();
        XChaCha20Poly1305::new(Key::from_slice(&key[..]))
            .decrypt_in_place_detached(&nonce, aad, ct, &tag)
            .map_err(|_| {
                seal_error("sealed data failed to open (wrong key or wrong associated data)")
            })?;
        let len = body.len() - SEAL_NONCE_LEN - SEAL_TAG_LEN;
        body.drain(..SEAL_NONCE_LEN);
        body.truncate(len);
        Ok(body)
    }

    /// Seal one transcript line for session `id`. The result never starts with `{`, which is how a
    /// reader tells a sealed line from a plaintext one.
    pub(crate) fn seal_line(&self, id: &str, line: &[u8]) -> std::io::Result<Vec<u8>> {
        self.seal(&self.transcript, &self.line_aad(id), line)
    }

    /// Open one sealed transcript line for session `id`.
    pub(crate) fn open_line(&self, id: &str, line: &[u8]) -> std::io::Result<Vec<u8>> {
        self.open(&self.transcript, &self.line_aad(id), line)
    }

    /// `tenant ‖ 0x00 ‖ session_id ‖ 0x00 ‖ "sidecar:" ‖ kind` — a session sidecar's AAD: bound to its
    /// session *and* its kind, so neither a transcript line nor another sidecar opens as this one.
    pub(super) fn sidecar_aad(&self, id: &str, kind: &str) -> Vec<u8> {
        let mut aad = self.line_aad(id);
        aad.push(0);
        aad.extend_from_slice(b"sidecar:");
        aad.extend_from_slice(kind.as_bytes());
        aad
    }

    /// Seal one whole session sidecar file (e.g. MCP App view state) for session `id`.
    pub fn seal_sidecar(&self, id: &str, kind: &str, bytes: &[u8]) -> std::io::Result<Vec<u8>> {
        self.seal(&self.transcript, &self.sidecar_aad(id, kind), bytes)
    }

    /// Open a sidecar sealed by [`Self::seal_sidecar`]. Never accepts plaintext: a sealed store's
    /// sidecar that is not sealed is not ours.
    pub fn open_sidecar(&self, id: &str, kind: &str, bytes: &[u8]) -> std::io::Result<Vec<u8>> {
        self.open(&self.transcript, &self.sidecar_aad(id, kind), bytes)
    }

    /// Seal the whole `.listings.json` cache.
    pub(crate) fn seal_listing(&self, bytes: &[u8]) -> std::io::Result<Vec<u8>> {
        self.seal(&self.listing, self.tenant.as_bytes(), bytes)
    }

    /// Open the whole `.listings.json` cache.
    pub(crate) fn open_listing(&self, bytes: &[u8]) -> std::io::Result<Vec<u8>> {
        self.open(&self.listing, self.tenant.as_bytes(), bytes)
    }

    /// Seal one memory document (durable or `/session`) as a whole file.
    pub fn seal_doc(&self, bytes: &[u8]) -> std::io::Result<Vec<u8>> {
        self.seal(&self.memory, self.tenant.as_bytes(), bytes)
    }

    /// Open one memory document. Content that isn't framed as sealed is returned verbatim, so a store
    /// that predates sealing still reads.
    pub fn open_doc(&self, bytes: &[u8]) -> std::io::Result<Vec<u8>> {
        if !bytes.starts_with(SEAL_FRAME.as_bytes()) {
            return Ok(bytes.to_vec());
        }
        self.open(&self.memory, self.tenant.as_bytes(), bytes)
    }
}
