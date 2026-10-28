//! Where a bearer token comes from.
//!
//! A token that never changes is a [`TokenSource::fixed`]. Beyond's exec tokens expire within 24 h,
//! so a long-lived client gives the source a way to mint another ([`TokenSource::from_fn`],
//! [`TokenSource::command`]). The client asks it for a token before its first request, and, on a
//! `401`, once for a new one and then repeats that request once. That is the whole auth logic.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

type Mint = dyn Fn() -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> + Send + Sync;

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
    /// text becomes the request's error, of kind `auth` (`token source failed: ...`).
    pub fn from_fn<F, Fut>(mint: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, String>> + Send + 'static,
    {
        Self(Inner::Fresh(Arc::new(move || Box::pin(mint()))))
    }

    /// A token printed by a command (`argv[0]` with the rest as arguments), run when one is needed:
    /// stdout, trimmed. A non-zero exit (stderr is in the error), or empty output, is an error.
    pub fn command<I, S>(argv: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let argv: Arc<[String]> = argv.into_iter().map(Into::into).collect();
        Self::from_fn(move || {
            let argv = argv.clone();
            async move {
                let (program, args) = argv.split_first().ok_or("the token command is empty")?;
                let out = tokio::process::Command::new(program)
                    .args(args)
                    .stdin(std::process::Stdio::null())
                    .output()
                    .await
                    .map_err(|e| format!("could not run the token command {program:?}: {e}"))?;
                if !out.status.success() {
                    let said = String::from_utf8_lossy(&out.stderr);
                    let said = said.trim();
                    return Err(format!(
                        "the token command {program:?} exited with {}{}{said}",
                        out.status,
                        if said.is_empty() { "" } else { ": " },
                    ));
                }
                let token = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if token.is_empty() {
                    return Err(format!("the token command {program:?} printed nothing"));
                }
                Ok(token)
            }
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

    /// After a `401` on `stale`: the token to repeat the request with, or `None` for a fixed
    /// source, which has nothing to ask again. The repeat is made whatever the source gives, even
    /// `stale` again, so a call is rejected at most twice. A request that finds another already
    /// replaced `stale` takes that one without asking the source again.
    pub(crate) async fn renew(&self, stale: &str) -> std::io::Result<Option<String>> {
        if !self.source.renews() {
            return Ok(None);
        }
        let mut cur = self.current.lock().await;
        if let Some(t) = cur.as_ref().filter(|t| t.as_str() != stale) {
            return Ok(Some(t.clone()));
        }
        let t = self.mint().await?;
        *cur = Some(t.clone());
        Ok(Some(t))
    }

    async fn mint(&self) -> std::io::Result<String> {
        self.source
            .get()
            .await
            .map_err(|e| std::io::Error::other(format!("token source failed: {e}")))
    }
}
