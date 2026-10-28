//! Where a bearer token comes from.
//!
//! A token that never changes is a [`TokenSource::fixed`]. Beyond's exec tokens expire within 24 h,
//! so a long-lived client gives the source a way to mint another ([`TokenSource::from_fn`],
//! [`TokenSource::command`]). The client asks it for a token before its first request, and, on a
//! `401`, once for a new one and then repeats that request once. That is the whole auth logic.
//!
//! A [`TokenSource::command`] mint holds the bearer lock so concurrent callers share one command,
//! and that wait is bounded by [`TOKEN_COMMAND_TIMEOUT`]. A [`TokenSource::from_fn`] mint is the
//! caller's to bound.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

type Mint = dyn Fn() -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> + Send + Sync;

/// How long [`TokenSource::command`] waits for the mint command before killing it.
pub const TOKEN_COMMAND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

async fn token_from_command(
    argv: &[String],
    timeout: std::time::Duration,
) -> Result<String, String> {
    let (program, args) = argv.split_first().ok_or("the token command is empty")?;
    let run = tokio::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .output();
    let out = match tokio::time::timeout(timeout, run).await {
        Ok(result) => {
            result.map_err(|e| format!("could not run the token command {program:?}: {e}"))?
        }
        Err(_) => return Err(format!("the token command {program:?} timed out")),
    };
    if !out.status.success() {
        return Err(format!(
            "the token command {program:?} exited with {}",
            out.status
        ));
    }
    let token = String::from_utf8(out.stdout)
        .map_err(|_| format!("the token command {program:?} printed a token that is not UTF-8"))?;
    let token = token.trim().to_string();
    if token.is_empty() {
        return Err(format!("the token command {program:?} printed nothing"));
    }
    Ok(token)
}

/// A bearer token, or a way to get a fresh one. Cheap to clone.
#[derive(Clone)]
pub struct TokenSource(Inner);

#[derive(Clone)]
enum Inner {
    Fixed(Arc<str>),
    Fresh(Arc<Mint>),
}

impl TokenSource {
    /// A token that does not change. A `401` is then final: there is nothing to ask again.
    pub fn fixed(token: impl Into<String>) -> Self {
        Self(Inner::Fixed(token.into().into()))
    }

    /// A token minted by `mint`, called for the first request and again after a `401`. The error
    /// text becomes the request's error.
    pub fn from_fn<F, Fut>(mint: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, String>> + Send + 'static,
    {
        Self(Inner::Fresh(Arc::new(move || Box::pin(mint()))))
    }

    /// A token printed by a command (`argv[0]` with the rest as arguments), run when one is needed:
    /// stdout, trimmed, and valid UTF-8. A non-zero exit, empty output, a token that is not UTF-8,
    /// or no exit within [`TOKEN_COMMAND_TIMEOUT`] is an error. The child is killed when the
    /// deadline fires.
    pub fn command<I, S>(argv: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let argv: Arc<[String]> = argv.into_iter().map(Into::into).collect();
        Self::from_fn(move || {
            let argv = argv.clone();
            async move { token_from_command(&argv, TOKEN_COMMAND_TIMEOUT).await }
        })
    }

    /// Whether asking again can give a different token.
    pub(crate) fn renews(&self) -> bool {
        matches!(self.0, Inner::Fresh(_))
    }

    /// The current token, minting one if this source mints.
    pub(crate) async fn get(&self) -> Result<String, String> {
        match &self.0 {
            Inner::Fixed(t) => Ok(t.to_string()),
            Inner::Fresh(mint) => mint().await,
        }
    }
}

impl From<&str> for TokenSource {
    fn from(token: &str) -> Self {
        Self::fixed(token)
    }
}

impl From<String> for TokenSource {
    fn from(token: String) -> Self {
        Self::fixed(token)
    }
}

impl std::fmt::Debug for TokenSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0 {
            Inner::Fixed(_) => "TokenSource::Fixed(<redacted>)",
            Inner::Fresh(_) => "TokenSource::Fresh(..)",
        })
    }
}

/// A source plus the token last obtained from it, so concurrent requests share one mint.
pub(crate) struct Bearer {
    source: TokenSource,
    current: tokio::sync::Mutex<Option<String>>,
}

impl Bearer {
    pub(crate) fn new(source: TokenSource) -> Self {
        Self {
            source,
            current: tokio::sync::Mutex::new(None),
        }
    }

    /// The token to send: the one already held, or the source's first.
    pub(crate) async fn token(&self) -> std::io::Result<String> {
        let mut cur = self.current.lock().await;
        if let Some(t) = cur.as_ref() {
            return Ok(t.clone());
        }
        let t = self.mint().await?;
        *cur = Some(t.clone());
        Ok(t)
    }

    /// After a `401` on `stale`: a different token to retry with, or `None` when there is none to
    /// try (a fixed source, or the source gave the same one back). A request that finds another
    /// already replaced `stale` takes that one without asking the source again.
    pub(crate) async fn renew(&self, stale: &str) -> std::io::Result<Option<String>> {
        if !self.source.renews() {
            return Ok(None);
        }
        let mut cur = self.current.lock().await;
        if let Some(t) = cur.as_ref().filter(|t| t.as_str() != stale) {
            return Ok(Some(t.clone()));
        }
        let t = self.mint().await?;
        let changed = t != stale;
        *cur = Some(t.clone());
        Ok(changed.then_some(t))
    }

    async fn mint(&self) -> std::io::Result<String> {
        self.source
            .get()
            .await
            .map_err(|e| std::io::Error::other(format!("token source failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_token_command_must_print_utf8_and_must_finish() {
        let err = token_from_command(
            &["sh".into(), "-c".into(), r"printf '\377'".into()],
            std::time::Duration::from_secs(5),
        )
        .await
        .unwrap_err();
        assert!(err.contains("UTF-8"), "{err}");

        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("pid");
        let script = format!("echo $$ > {}; exec sleep 30", pidfile.display());
        let err = token_from_command(
            &["sh".into(), "-c".into(), script],
            std::time::Duration::from_millis(200),
        )
        .await
        .unwrap_err();
        assert!(err.contains("timed out"), "{err}");
        let pid = std::fs::read_to_string(&pidfile).unwrap();
        let proc = format!("/proc/{}", pid.trim());
        for _ in 0..50 {
            if !std::path::Path::new(&proc).exists() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(
            !std::path::Path::new(&proc).exists(),
            "the token command was not killed"
        );
    }
}
