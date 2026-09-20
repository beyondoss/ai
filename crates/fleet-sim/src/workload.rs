//! The client side: what a tenant's session actually does to a replica.
//!
//! Everything here speaks the real wire protocol against a real replica. The point of a simulator is
//! that nothing in the path is stubbed except the two things outside this repo's boundary — the model
//! (a mock server replaying fixed turns) and the sandbox (an exec endpoint the scenarios point
//! somewhere harmless).

use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

/// How long any single frame may take before a session is called stuck. Generous: a loaded
/// simulator is still a correct one, and a deadline that fails a slow run teaches nothing.
const FRAME_WAIT: Duration = Duration::from_secs(60);

pub type Ws =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Open a session on `port`, presenting `grant`.
pub async fn connect(port: u16, session_id: &str, grant: &str) -> Result<Ws, String> {
    let url = format!("ws://127.0.0.1:{port}/_beyond/agent?session_id={session_id}");
    let mut req = url
        .into_client_request()
        .map_err(|e| format!("request: {e}"))?;
    req.headers_mut().insert(
        "x-beyond-grant",
        grant.parse().map_err(|e| format!("grant header: {e}"))?,
    );
    let (ws, _) = tokio_tungstenite::connect_async(req)
        .await
        .map_err(|e| format!("connect: {e}"))?;
    Ok(ws)
}

/// Ask a replica whether it will serve this session, without holding a session open.
///
/// Returns the HTTP status, which is the whole point: `503` is "not mine, retry", `421` is "wrong
/// shard", `401`/`403` are the grant's problem. A successful upgrade is reported as 101 and the
/// socket is dropped immediately — the caller reconnects if it wants to drive the session.
pub async fn probe_session(port: u16, session_id: &str, grant: &str) -> Result<u16, String> {
    match connect(port, session_id, grant).await {
        Ok(ws) => {
            drop(ws);
            Ok(101)
        }
        Err(e) => {
            // tungstenite reports a refused upgrade as an HTTP error carrying the status.
            for code in [503u16, 421, 403, 401, 400, 404, 429, 500] {
                if e.contains(&code.to_string()) {
                    return Ok(code);
                }
            }
            Err(e)
        }
    }
}

pub async fn send(ws: &mut Ws, cmd: Value) -> Result<(), String> {
    ws.send(Message::Text(cmd.to_string().into()))
        .await
        .map_err(|e| format!("send: {e}"))
}

/// Read frames until one satisfies `matches`, returning it along with everything seen on the way.
pub async fn read_until(
    ws: &mut Ws,
    mut matches: impl FnMut(&Value) -> bool,
) -> Result<(Value, Vec<Value>), String> {
    let mut seen = Vec::new();
    loop {
        let frame = tokio::time::timeout(FRAME_WAIT, ws.next())
            .await
            .map_err(|_| format!("timed out after {FRAME_WAIT:?}; saw {} frames", seen.len()))?;
        let Some(msg) = frame else {
            return Err(format!("socket closed after {} frames", seen.len()));
        };
        let msg = msg.map_err(|e| format!("read: {e}"))?;
        let Message::Text(text) = msg else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if matches(&v) {
            return Ok((v, seen));
        }
        seen.push(v);
    }
}

/// Send a command and wait for its `response`.
pub async fn command(ws: &mut Ws, cmd: Value, name: &str) -> Result<Value, String> {
    send(ws, cmd).await?;
    let (resp, _) = read_until(ws, |f| f["type"] == "response" && f["command"] == name).await?;
    Ok(resp)
}

/// Run one prompt to completion, returning the terminal `response` frame.
pub async fn prompt(ws: &mut Ws, message: &str) -> Result<Value, String> {
    command(
        ws,
        json!({ "type": "prompt", "message": message }),
        "prompt",
    )
    .await
}

/// The session's committed transcript, as the *client* sees it.
///
/// The checker compares this against what the client was told earlier: the invariant is that no
/// acknowledged message is missing after an ownership change. Reading it back through the protocol
/// rather than off disk is deliberate — it exercises the real replay path, and it needs no key.
pub async fn transcript(ws: &mut Ws) -> Result<Vec<Value>, String> {
    let resp = command(ws, json!({ "type": "get_messages" }), "get_messages").await?;
    Ok(resp["data"]["messages"]
        .as_array()
        .cloned()
        .unwrap_or_default())
}

/// One bare HTTP GET against a replica: `/livez`, `/readyz`, or a metrics listener's `/metrics`.
///
/// Hand-rolled for the same reason the mock model server is: a client with no framework of its own
/// has nothing to agree with the thing under test about, and these are three-line requests.
pub async fn http_get(port: u16, path: &str) -> Result<(u16, String), String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .map_err(|e| format!("connect {port}: {e}"))?;
    s.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes())
        .await
        .map_err(|e| format!("write: {e}"))?;
    let mut out = Vec::new();
    // A peer that closes while we are still reading surfaces as a reset rather than EOF; what it
    // already sent is still a valid response.
    if let Err(e) = s.read_to_end(&mut out).await
        && out.is_empty()
    {
        return Err(format!("read {path}: {e}"));
    }
    let text = String::from_utf8_lossy(&out).into_owned();
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    let body = text.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");
    Ok((status, body.to_owned()))
}
