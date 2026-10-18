//! The browser guard on the daemon's TCP front door.
//!
//! Outside service mode the daemon authenticates no caller: whoever can reach the socket drives the
//! session, and the session has `bash`. On a loopback TCP listener that is meant to be "processes on
//! this machine" — but a web browser is a process on this machine that runs code from anywhere:
//!
//! - **Simple-request CSRF.** A page can `fetch()` a `text/plain` POST at `127.0.0.1:<port>` with no
//!   CORS preflight. The page never sees the reply, and doesn't need to: the command already ran.
//! - **Cross-site WebSocket.** Browsers do not apply CORS to a WebSocket upgrade at all; the server
//!   is expected to check `Origin` itself.
//! - **DNS rebinding.** A page on `evil.example` re-points that name at `127.0.0.1`, and its requests
//!   are then *same-origin* to the browser — no `Origin` to refuse. What gives it away is `Host`,
//!   which still names `evil.example`.
//!
//! So a request is refused if it carries a foreign `Origin`, or a `Host` that isn't a loopback name or
//! an IP literal, and a POST must be `Content-Type: application/json` — not a type a page can send
//! cross-origin without a preflight, which this listener never answers (`OPTIONS` is a 405). Absent
//! headers are allowed: every browser sends `Host` on every request and `Origin` on every cross-origin
//! one, so a request without them is curl, the CLI, or another non-browser client — which could just
//! as well have connected directly, so there is nothing to defend against.
//!
//! **Not applied to the Unix socket.** A browser cannot open one; who connects is the socket file's
//! permissions. A front door that splices authenticated browser traffic onto it (Beyond's console
//! proxy does) forwards the browser's own `Origin` and public `Host` and has made its own Origin
//! decision before splicing — refusing them here would break exactly the deployment that is secure.
//!
//! **Not applied in service mode.** Every request there must carry a session grant in the
//! `x-beyond-grant` header. That is not an ambient credential: a browser never attaches it on its own,
//! a page cannot add a custom header to a cross-origin request without a preflight (refused), and a
//! WebSocket upgrade from a browser cannot carry custom headers at all — it can carry the grant only
//! in `Sec-WebSocket-Protocol: beyond.grant, <grant>`, which a page must have been handed the grant
//! to set, so it is no more ambient than the header. Without a grant a forged
//! request is a 401, so there is no confused deputy for `Origin` to guard; and `Host` there is
//! whatever name the load balancer routed, so a loopback rule could never hold.

use std::net::IpAddr;

/// Why a request was refused. The message goes on the wire: it names the rule, which a legitimate
/// local client needs to fix its request, and tells an attacker nothing they didn't send.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Refused {
    /// `Origin` is present and not a loopback origin.
    ForeignOrigin,
    /// `Host` is present and is neither a loopback name nor an IP literal.
    ForeignHost,
    /// A POST whose `Content-Type` is not `application/json`.
    NotJson,
}

impl Refused {
    pub(super) fn message(&self) -> &'static str {
        match self {
            Refused::ForeignOrigin => {
                "cross-origin request refused: Origin must be a loopback origin"
            }
            Refused::ForeignHost => {
                "request refused: Host must be localhost or an IP address (DNS rebinding guard)"
            }
            Refused::NotJson => "POST body must be sent as Content-Type: application/json",
        }
    }
}

/// Check one request's headers. `is_post` adds the Content-Type rule. Every occurrence of a repeated
/// header is checked, so a second `Origin` cannot smuggle a foreign one past the first.
pub(super) fn check(headers: &[(String, String)], is_post: bool) -> Result<(), Refused> {
    let mut content_type_ok = false;
    for (name, value) in headers {
        if name.eq_ignore_ascii_case("origin") {
            if !is_loopback_origin(value) {
                return Err(Refused::ForeignOrigin);
            }
        } else if name.eq_ignore_ascii_case("host") {
            if !is_allowed_host(value) {
                return Err(Refused::ForeignHost);
            }
        } else if name.eq_ignore_ascii_case("content-type") {
            content_type_ok = is_json_content_type(value);
        }
    }
    if is_post && !content_type_ok {
        return Err(Refused::NotJson);
    }
    Ok(())
}

/// `http(s)://<loopback host>[:port]`. Anything else — another site, an IP literal that isn't
/// loopback, `null` (sandboxed iframes, `file://`), an unparseable value — is foreign.
fn is_loopback_origin(origin: &str) -> bool {
    let Some(rest) = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    else {
        return false;
    };
    // An origin is scheme://host[:port] with nothing after; a path means it isn't one.
    if rest.contains('/') {
        return false;
    }
    host_part(rest).is_some_and(is_loopback_host)
}

/// A `Host` that DNS rebinding cannot produce: a loopback name, or any IP literal. Rebinding needs a
/// *name* the attacker controls; a request whose `Host` is an address was sent to that address, and
/// allowing non-loopback addresses keeps a daemon bound on an internal interface (reached by IP)
/// working.
fn is_allowed_host(host: &str) -> bool {
    host_part(host).is_some_and(|h| is_loopback_host(h) || parse_ip(h).is_some())
}

/// `localhost`, `*.localhost` (which browsers resolve to loopback themselves, never via DNS), or a
/// loopback address.
fn is_loopback_host(host: &str) -> bool {
    let host = host.strip_suffix('.').unwrap_or(host);
    if host.eq_ignore_ascii_case("localhost")
        || host.len() > ".localhost".len()
            && host[host.len() - ".localhost".len()..].eq_ignore_ascii_case(".localhost")
    {
        return true;
    }
    parse_ip(host).is_some_and(|ip| match ip {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => {
            v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
    })
}

/// Split `host[:port]` / `[v6][:port]` and return the host. `None` if empty or malformed.
fn host_part(authority: &str) -> Option<&str> {
    let authority = authority.trim();
    if authority.is_empty() || authority.contains('@') {
        return None;
    }
    if let Some(rest) = authority.strip_prefix('[') {
        let (v6, after) = rest.split_once(']')?;
        if !(after.is_empty() || after.strip_prefix(':').is_some_and(is_port)) {
            return None;
        }
        return Some(v6);
    }
    match authority.split_once(':') {
        Some((host, port)) if is_port(port) && !host.is_empty() => Some(host),
        Some(_) => None,
        None => Some(authority),
    }
}

fn is_port(s: &str) -> bool {
    !s.is_empty() && s.len() <= 5 && s.bytes().all(|b| b.is_ascii_digit())
}

/// Bracket-stripped host to address. `host_part` already removed the brackets from a v6 literal.
fn parse_ip(host: &str) -> Option<IpAddr> {
    host.parse().ok()
}

/// `application/json`, case-insensitive, with any parameters (`; charset=utf-8`).
fn is_json_content_type(value: &str) -> bool {
    value
        .split(';')
        .next()
        .is_some_and(|t| t.trim().eq_ignore_ascii_case("application/json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn non_browser_requests_pass() {
        assert_eq!(check(&h(&[]), false), Ok(()));
        assert_eq!(check(&h(&[("Host", "127.0.0.1:8787")]), false), Ok(()));
        assert_eq!(
            check(&h(&[("Content-Type", "application/json")]), true),
            Ok(())
        );
    }

    #[test]
    fn loopback_origins_pass_and_foreign_ones_do_not() {
        for ok in [
            "http://localhost",
            "http://localhost:3000",
            "https://LOCALHOST:1",
            "http://app.localhost:5173",
            "http://127.0.0.1:8787",
            "http://127.9.9.9",
            "http://[::1]:8787",
            "http://[::ffff:127.0.0.1]",
        ] {
            assert_eq!(check(&h(&[("Origin", ok)]), false), Ok(()), "{ok}");
        }
        for bad in [
            "https://evil.example",
            "null",
            "",
            "file://",
            "http://10.0.0.5",
            "http://localhost.evil.example",
            "http://evillocalhost",
            "http://127.0.0.1.evil.example",
            "http://localhost/path",
            "http://user@localhost",
            "ws://localhost",
            "http://localhost:notaport",
        ] {
            assert_eq!(
                check(&h(&[("Origin", bad)]), false),
                Err(Refused::ForeignOrigin),
                "{bad}"
            );
        }
    }

    #[test]
    fn a_second_origin_header_is_checked_too() {
        let headers = h(&[
            ("Origin", "http://localhost"),
            ("origin", "https://evil.example"),
        ]);
        assert_eq!(check(&headers, false), Err(Refused::ForeignOrigin));
    }

    #[test]
    fn rebinding_hosts_are_refused_and_addresses_pass() {
        for ok in [
            "localhost",
            "localhost:8787",
            "127.0.0.1",
            "127.0.0.1:8787",
            "[::1]:8787",
            "10.1.2.3:8787",
            "foo.localhost",
            "localhost.",
        ] {
            assert_eq!(check(&h(&[("Host", ok)]), false), Ok(()), "{ok}");
        }
        for bad in [
            "rebound.evil.example",
            "rebound.evil.example:8787",
            "x",
            "",
            "[::1",
            "127.0.0.1:",
        ] {
            assert_eq!(
                check(&h(&[("Host", bad)]), false),
                Err(Refused::ForeignHost),
                "{bad}"
            );
        }
    }

    #[test]
    fn a_post_must_be_json() {
        for ok in [
            "application/json",
            "Application/JSON",
            "application/json; charset=utf-8",
        ] {
            assert_eq!(check(&h(&[("Content-Type", ok)]), true), Ok(()), "{ok}");
        }
        for bad in [
            "text/plain",
            "application/x-www-form-urlencoded",
            "multipart/form-data; boundary=x",
            "application/jsonx",
        ] {
            assert_eq!(
                check(&h(&[("Content-Type", bad)]), true),
                Err(Refused::NotJson),
                "{bad}"
            );
        }
        assert_eq!(check(&h(&[]), true), Err(Refused::NotJson));
        // A GET (WebSocket upgrade) has no body, so no Content-Type rule.
        assert_eq!(check(&h(&[]), false), Ok(()));
    }
}
