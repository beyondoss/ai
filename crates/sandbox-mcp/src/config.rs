//! The configuration: environment variables, read once at startup. There are no flags (execd's
//! rule): an MCP client launches this server from a config file that already has an `env` block,
//! and a second place to say the same thing is a second place to get it wrong.
//!
//! An empty variable is unset. A value that does not parse is an error naming the variable.

use serde_json::Value;

/// Where the bearer token comes from.
#[derive(Debug, PartialEq)]
pub enum Token {
    /// The endpoint takes no bearer.
    None,
    /// `SANDBOX_TOKEN`: one token for the life of the process.
    Fixed(String),
    /// `SANDBOX_TOKEN_CMD`: a shell command whose stdout is a token; run again after a `401`.
    Command(String),
}

/// Probe results the server is told to believe instead of what it finds (`SANDBOX_CAPS`), so one
/// sandbox can walk every rung of the file helpers' search ladder. The conformance driver sets it.
#[derive(Debug, Default, PartialEq)]
pub struct CapsOverride {
    pub find_printf: Option<bool>,
    /// `rg`, `grep` or `busybox-grep`.
    pub search_engine: Option<String>,
}

/// What the environment asked for.
#[derive(Debug, PartialEq)]
pub struct Config {
    pub url: String,
    pub token: Token,
    pub jobs_scope: String,
    pub caps: CapsOverride,
    /// `SANDBOX_MCP_LISTEN`: serve streamable HTTP here instead of stdio.
    pub listen: Option<String>,
    pub insecure_no_auth: bool,
}

pub const USAGE: &str = "\
sandbox-mcp takes no arguments; it is configured by environment variables:

  SANDBOX_URL                  the sandbox's exec endpoint, e.g. http://127.0.0.1:8080 (required)
  SANDBOX_TOKEN                the bearer token the endpoint wants
  SANDBOX_TOKEN_CMD            a shell command whose stdout is a fresh token (run again after a 401);
                               not together with SANDBOX_TOKEN
  SANDBOX_JOBS_SCOPE           the name jobs are kept under in the sandbox (default mcp)
  SANDBOX_MCP_LISTEN           host:port: serve streamable HTTP there instead of stdio
  SANDBOX_MCP_INSECURE_NO_AUTH 1|true|0|false: allow SANDBOX_MCP_LISTEN on a non-loopback address
                               (the server holds the sandbox token and checks nobody; default 0)
  SANDBOX_CAPS                 JSON {\"find_printf\":bool,\"search_engine\":\"rg|grep|busybox-grep\"}:
                               force a search rung of the file helpers (for testing)
";

fn boolean(name: &str, v: Option<String>) -> Result<bool, String> {
    match v.as_deref() {
        None | Some("0" | "false") => Ok(false),
        Some("1" | "true") => Ok(true),
        Some(other) => Err(format!("{name}={other:?} is not 1, true, 0 or false")),
    }
}

impl Config {
    /// Read the configuration from `get`, which returns a variable's value (the process environment,
    /// or a map in a test).
    pub fn from_env(get: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let get = |name: &str| get(name).filter(|v| !v.is_empty());
        let url = get("SANDBOX_URL").ok_or(
            "SANDBOX_URL is required: the sandbox's exec endpoint, e.g. http://127.0.0.1:8080",
        )?;
        let token = match (get("SANDBOX_TOKEN"), get("SANDBOX_TOKEN_CMD")) {
            (None, None) => Token::None,
            (Some(t), None) => Token::Fixed(t),
            (None, Some(c)) => Token::Command(c),
            (Some(_), Some(_)) => {
                return Err("set SANDBOX_TOKEN or SANDBOX_TOKEN_CMD, not both".to_string());
            }
        };
        let jobs_scope = get("SANDBOX_JOBS_SCOPE").unwrap_or_else(|| "mcp".to_string());
        if jobs_scope.is_empty()
            || jobs_scope.len() > 64
            || !jobs_scope
                .bytes()
                .all(|b| b.is_ascii_digit() || b.is_ascii_lowercase() || b == b'-')
        {
            return Err(format!(
                "SANDBOX_JOBS_SCOPE={jobs_scope:?} must be 1 to 64 characters of [0-9a-z-]"
            ));
        }
        let caps = match get("SANDBOX_CAPS") {
            None => CapsOverride::default(),
            Some(raw) => {
                let v: Value = serde_json::from_str(&raw)
                    .map_err(|e| format!("SANDBOX_CAPS is not JSON: {e}"))?;
                CapsOverride {
                    find_printf: v["find_printf"].as_bool(),
                    search_engine: v["search_engine"].as_str().map(str::to_string),
                }
            }
        };
        Ok(Self {
            url,
            token,
            jobs_scope,
            caps,
            listen: get("SANDBOX_MCP_LISTEN"),
            insecure_no_auth: boolean(
                "SANDBOX_MCP_INSECURE_NO_AUTH",
                get("SANDBOX_MCP_INSECURE_NO_AUTH"),
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: std::collections::HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| m.get(k).cloned()
    }

    #[test]
    fn the_url_is_required_and_empty_means_unset() {
        assert!(Config::from_env(env(&[])).is_err());
        assert!(Config::from_env(env(&[("SANDBOX_URL", "")])).is_err());
        let c = Config::from_env(env(&[("SANDBOX_URL", "http://x")])).unwrap();
        assert_eq!(c.token, Token::None);
        assert_eq!(c.jobs_scope, "mcp");
        assert_eq!(c.listen, None);
    }

    #[test]
    fn a_fixed_token_and_a_token_command_are_exclusive() {
        let both = env(&[
            ("SANDBOX_URL", "http://x"),
            ("SANDBOX_TOKEN", "a"),
            ("SANDBOX_TOKEN_CMD", "echo a"),
        ]);
        assert!(Config::from_env(both).is_err());
        let cmd = env(&[("SANDBOX_URL", "http://x"), ("SANDBOX_TOKEN_CMD", "echo a")]);
        assert_eq!(
            Config::from_env(cmd).unwrap().token,
            Token::Command("echo a".into())
        );
    }

    #[test]
    fn a_bad_scope_or_caps_names_its_variable() {
        let bad = env(&[("SANDBOX_URL", "u"), ("SANDBOX_JOBS_SCOPE", "Bad Scope")]);
        assert!(
            Config::from_env(bad)
                .unwrap_err()
                .contains("SANDBOX_JOBS_SCOPE")
        );
        let bad = env(&[("SANDBOX_URL", "u"), ("SANDBOX_CAPS", "{")]);
        assert!(Config::from_env(bad).unwrap_err().contains("SANDBOX_CAPS"));
        let ok = env(&[
            ("SANDBOX_URL", "u"),
            (
                "SANDBOX_CAPS",
                r#"{"find_printf":false,"search_engine":"grep"}"#,
            ),
        ]);
        let caps = Config::from_env(ok).unwrap().caps;
        assert_eq!(caps.find_printf, Some(false));
        assert_eq!(caps.search_engine.as_deref(), Some("grep"));
    }
}
