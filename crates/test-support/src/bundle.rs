//! Agent-folder bundles for tests: a plain-tar builder (including the entries a safe unpacker must
//! refuse), the content hash a grant names it by, and a loopback server that serves one.

use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use sha2::{Digest, Sha256};

/// One tar entry. `kind` is the raw type flag byte: `b'0'` file, `b'5'` directory, `b'2'` symlink,
/// `b'1'` hardlink, `b'3'`/`b'4'` devices.
pub struct Entry<'a> {
    pub path: &'a str,
    pub kind: u8,
    pub body: &'a [u8],
    pub link: &'a str,
}

/// A regular file.
pub fn file<'a>(path: &'a str, body: &'a str) -> Entry<'a> {
    Entry {
        path,
        kind: b'0',
        body: body.as_bytes(),
        link: "",
    }
}

/// A directory.
pub fn dir(path: &str) -> Entry<'_> {
    Entry {
        path,
        kind: b'5',
        body: b"",
        link: "",
    }
}

/// A ustar archive of `entries`.
pub fn tar(entries: &[Entry<'_>]) -> Vec<u8> {
    let mut out = Vec::new();
    for e in entries {
        let mut h = [0u8; 512];
        h[..e.path.len()].copy_from_slice(e.path.as_bytes());
        h[100..107].copy_from_slice(b"0000644");
        h[108..115].copy_from_slice(b"0000000");
        h[116..123].copy_from_slice(b"0000000");
        h[124..135].copy_from_slice(format!("{:011o}", e.body.len()).as_bytes());
        h[136..147].copy_from_slice(b"00000000000");
        h[148..156].copy_from_slice(b"        ");
        h[156] = e.kind;
        h[157..157 + e.link.len()].copy_from_slice(e.link.as_bytes());
        h[257..263].copy_from_slice(b"ustar\0");
        h[263..265].copy_from_slice(b"00");
        let sum: u32 = h.iter().map(|b| u32::from(*b)).sum();
        h[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(e.body);
        out.resize(out.len().div_ceil(512) * 512, 0);
    }
    out.resize(out.len() + 1024, 0);
    out
}

/// Lowercase hex SHA-256: the `bundle.sha256` a grant names the bytes by.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A loopback HTTP server answering every GET with `bytes`.
pub struct BundleServer {
    pub url: String,
    hits: Arc<AtomicUsize>,
}

/// How many responses are being held across all the servers sharing it, and the most at once.
#[derive(Default)]
pub struct Gauge {
    now: AtomicUsize,
    peak: AtomicUsize,
}

impl Gauge {
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }
}

impl BundleServer {
    pub fn start(bytes: Vec<u8>) -> Self {
        Self::serve(bytes, None)
    }

    /// Like [`Self::start`], but each response is held for `hold` after its request arrives, counted
    /// in `gauge`: a way to see how many fetches are in flight at once.
    pub fn start_held(bytes: Vec<u8>, gauge: Arc<Gauge>, hold: std::time::Duration) -> Self {
        Self::serve(bytes, Some((gauge, hold)))
    }

    fn serve(bytes: Vec<u8>, held: Option<(Arc<Gauge>, std::time::Duration)>) -> Self {
        let listener = crate::ports::listener();
        let url = format!("http://{}/agent.tar", listener.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut stream) = conn else { break };
                counter.fetch_add(1, Ordering::SeqCst);
                let bytes = bytes.clone();
                let held = held.clone();
                std::thread::spawn(move || {
                    let mut buf = [0u8; 4096];
                    let mut seen = Vec::new();
                    while !seen.windows(4).any(|w| w == b"\r\n\r\n") {
                        match stream.read(&mut buf) {
                            Ok(0) | Err(_) => return,
                            Ok(n) => seen.extend_from_slice(&buf[..n]),
                        }
                    }
                    if let Some((gauge, hold)) = &held {
                        let now = gauge.now.fetch_add(1, Ordering::SeqCst) + 1;
                        gauge.peak.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(*hold);
                        gauge.now.fetch_sub(1, Ordering::SeqCst);
                    }
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/x-tar\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        bytes.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(&bytes);
                });
            }
        });
        Self { url, hits }
    }

    /// How many requests the server has answered.
    pub fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }
}
