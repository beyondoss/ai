//! `trust`, `untrust`, `clear-trust`, `trust-status`: the persisted project-trust store.

use std::path::PathBuf;

/// `trust [path]`
pub(crate) fn trust(path: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let dir = match path {
        Some(p) => PathBuf::from(p),
        None => std::env::current_dir()?,
    };
    let mut store = beyond_agent::trust_store::TrustStore::open_default();
    store.trust(&dir)?;
    println!("trusted: {}", dir.display());
    Ok(())
}

/// `untrust [path]`
pub(crate) fn untrust(path: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let dir = match path {
        Some(p) => PathBuf::from(p),
        None => std::env::current_dir()?,
    };
    let mut store = beyond_agent::trust_store::TrustStore::open_default();
    store.distrust(&dir)?;
    println!("untrusted: {}", dir.display());
    Ok(())
}

/// `clear-trust [path]`
pub(crate) fn clear_trust(path: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let dir = match path {
        Some(p) => PathBuf::from(p),
        None => std::env::current_dir()?,
    };
    let mut store = beyond_agent::trust_store::TrustStore::open_default();
    store.clear(&dir)?;
    println!("cleared: {}", dir.display());
    Ok(())
}

/// `trust-status [path]`
pub(crate) fn trust_status(path: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let dir = match path {
        Some(p) => PathBuf::from(p),
        None => std::env::current_dir()?,
    };
    let store = beyond_agent::trust_store::TrustStore::open_default();
    let status = match store.lookup(&dir) {
        beyond_agent::trust_store::Trust::Trusted => "trusted",
        beyond_agent::trust_store::Trust::Untrusted => "untrusted",
        beyond_agent::trust_store::Trust::Unknown => "unknown",
    };
    println!("{status}: {}", dir.display());
    Ok(())
}
