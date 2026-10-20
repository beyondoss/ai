//! Keep a slow MCP server through rmcp's discover → initialize fallback, on every transport.
//!
//! rmcp's `Auto` lifecycle sends `server/discover` and, after 10 s without an answer, falls back to a
//! legacy `initialize` on the same connection. A server that was merely slow — an `npx` cold start
//! on a loaded host, a cold-start serverless MCP host — then answers the discover *after* the
//! initialize went out, and rmcp, waiting for the initialize's answer, failed the whole handshake on
//! it ("uncorrelated error response: expected id 1, error response carried 0"): the server and every
//! tool it had were gone for the run. In one A/B that was 6 of 132 MCP runs; it read as a broken
//! search (`tools.search("browser")` → `[]`) because the browser server was simply absent.
//!
//! That answer is stale by construction: sending `initialize` abandoned the discover. [`Guard`]
//! wraps a connection's transport, sees the client's messages as rmcp hands them over — before any
//! transport queues them, which matters over streamable HTTP, where the worker posts `initialize`
//! only once the slow discover POST has returned — and drops that one answer. It tracks nothing
//! once the handshake is over.
//!
//! The other half of the same fallback: rmcp only drops back to `initialize` when the discover is
//! answered by a *correlated* JSON-RPC error that is not a modern-era rejection. Public servers that
//! predate `2026-07-28` reject the probe in their own way — DeepWiki answers HTTP 400 with code
//! -32600 and the made-up id `"server-error"` — and rmcp treats that as a fatal "uncorrelated error
//! response", so the server and its tools were gone. Any error that arrives while the discover is
//! the only request in flight can only be its answer, so [`Guard::probe_rejected`] makes it one:
//! the discover's id, and "method not found" in place of a modern-era code. Transport failures
//! (refused, timed out, a 5xx with no JSON-RPC body) produce no error message and stay errors.

use rmcp::RoleClient;
use rmcp::model::{
    ClientJsonRpcMessage, ClientRequest, ErrorCode, RequestId, ServerJsonRpcMessage,
};
use rmcp::transport::Transport;

/// A client transport with the handshake's stale discover answer filtered out — see the module doc.
pub(crate) struct Guard<T> {
    inner: T,
    /// `server/discover`, until it is answered or `initialize` follows it.
    discover: Option<RequestId>,
    /// A `server/discover` abandoned by `initialize`, until its late answer has been dropped.
    abandoned: Option<RequestId>,
    /// That `initialize`, until it is answered — the end of the handshake.
    initialize: Option<RequestId>,
}

impl<T> Guard<T> {
    pub(crate) fn new(inner: T) -> Self {
        Self {
            inner,
            discover: None,
            abandoned: None,
            initialize: None,
        }
    }

    fn sent(&mut self, message: &ClientJsonRpcMessage) {
        let ClientJsonRpcMessage::Request(request) = message else {
            return;
        };
        match &request.request {
            ClientRequest::DiscoverRequest(_) => self.discover = Some(request.id.clone()),
            ClientRequest::InitializeRequest(_) => {
                self.abandoned = self.discover.take();
                self.initialize = Some(request.id.clone());
            }
            _ => {}
        }
    }

    /// Turn any JSON-RPC error that arrives while only the discover is outstanding into the
    /// correlated "method not found" rmcp falls back to `initialize` on — see the module doc. A
    /// version-negotiation error (`UNSUPPORTED_PROTOCOL_VERSION`) keeps its code: rmcp retries
    /// those itself.
    fn probe_rejected(&self, message: &mut ServerJsonRpcMessage) {
        let (Some(discover), ServerJsonRpcMessage::Error(error)) = (&self.discover, message) else {
            return;
        };
        if error.id.as_ref() != Some(discover) {
            tracing::debug!(id = ?error.id, "MCP `server/discover` rejected with an uncorrelated error");
            error.id = Some(discover.clone());
        }
        if error.error.code != ErrorCode::UNSUPPORTED_PROTOCOL_VERSION {
            error.error.code = ErrorCode::METHOD_NOT_FOUND;
        }
    }

    /// Whether `message` is the late answer to the abandoned discover; ends the tracking it settles.
    fn stale(&mut self, message: &ServerJsonRpcMessage) -> bool {
        let id = match message {
            ServerJsonRpcMessage::Response(response) => Some(&response.id),
            ServerJsonRpcMessage::Error(error) => error.id.as_ref(),
            _ => None,
        };
        let Some(id) = id else {
            return false;
        };
        if self.abandoned.as_ref() == Some(id) {
            self.abandoned = None;
            tracing::debug!("dropped a late answer to an abandoned MCP `server/discover`");
            return true;
        }
        if self.discover.as_ref() == Some(id) {
            self.discover = None;
        } else if self.initialize.as_ref() == Some(id) {
            // Answered in order: an abandoned discover still unanswered now never will be.
            self.abandoned = None;
            self.initialize = None;
        }
        false
    }
}

impl<T: Transport<RoleClient>> Transport<RoleClient> for Guard<T> {
    type Error = T::Error;

    fn send(
        &mut self,
        item: ClientJsonRpcMessage,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        self.sent(&item);
        self.inner.send(item)
    }

    async fn receive(&mut self) -> Option<ServerJsonRpcMessage> {
        loop {
            let mut message = self.inner.receive().await?;
            self.probe_rejected(&mut message);
            if !self.stale(&message) {
                return Some(message);
            }
        }
    }

    fn close(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.inner.close()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn client(v: serde_json::Value) -> ClientJsonRpcMessage {
        serde_json::from_value(v).unwrap()
    }

    fn server(v: serde_json::Value) -> ServerJsonRpcMessage {
        serde_json::from_value(v).unwrap()
    }

    fn discover() -> ClientJsonRpcMessage {
        client(json!({ "jsonrpc": "2.0", "id": 0, "method": "server/discover", "params": {} }))
    }

    fn initialize() -> ClientJsonRpcMessage {
        client(
            json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "t", "version": "0" },
        } }),
        )
    }

    fn late() -> ServerJsonRpcMessage {
        server(json!({ "jsonrpc": "2.0", "id": 0, "error": { "code": -32601, "message": "nope" } }))
    }

    fn init_ok() -> ServerJsonRpcMessage {
        server(json!({ "jsonrpc": "2.0", "id": 1, "result": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "serverInfo": { "name": "s", "version": "0" },
        } }))
    }

    fn quiet(g: &Guard<()>) -> bool {
        g.discover.is_none() && g.abandoned.is_none() && g.initialize.is_none()
    }

    #[test]
    fn any_error_answering_the_discover_becomes_method_not_found() {
        for (code, id) in [
            (-32600, json!("server-error")),
            (-32602, json!(0)),
            (-32000, json!(7)),
            (-32001, json!(0)),
        ] {
            let mut g = Guard::new(());
            g.sent(&discover());
            let mut m = server(json!({ "jsonrpc": "2.0", "id": id,
                "error": { "code": code, "message": "no" } }));
            g.probe_rejected(&mut m);
            let ServerJsonRpcMessage::Error(e) = &m else {
                panic!("not an error")
            };
            assert_eq!(e.id, Some(RequestId::Number(0)));
            assert_eq!(e.error.code, ErrorCode::METHOD_NOT_FOUND);
        }

        // Negotiation keeps its code; an id-less error is correlated.
        let mut g = Guard::new(());
        g.sent(&discover());
        let mut m = server(json!({ "jsonrpc": "2.0",
            "error": { "code": ErrorCode::UNSUPPORTED_PROTOCOL_VERSION.0, "message": "v" } }));
        g.probe_rejected(&mut m);
        let ServerJsonRpcMessage::Error(e) = &m else {
            panic!("not an error")
        };
        assert_eq!(e.id, Some(RequestId::Number(0)));
        assert_eq!(e.error.code, ErrorCode::UNSUPPORTED_PROTOCOL_VERSION);

        // Not while the discover is outstanding: untouched (an `initialize` error stays one).
        let mut g = Guard::new(());
        g.sent(&discover());
        g.sent(&initialize());
        g.discover = None;
        let mut m = server(json!({ "jsonrpc": "2.0", "id": "x",
            "error": { "code": -32600, "message": "no" } }));
        g.probe_rejected(&mut m);
        let ServerJsonRpcMessage::Error(e) = &m else {
            panic!("not an error")
        };
        assert_eq!(e.id, Some(RequestId::String("x".into())));
        assert_eq!(e.error.code, ErrorCode(-32600));
    }

    #[test]
    fn a_late_answer_to_an_abandoned_discover_is_dropped_once() {
        // Answered in time: delivered.
        let mut g = Guard::new(());
        g.sent(&discover());
        assert!(!g.stale(&late()));
        g.sent(&initialize());
        assert!(!g.stale(&init_ok()));
        assert!(quiet(&g));

        // Answered after `initialize` followed it: dropped, once; the initialize answer goes on.
        let mut g = Guard::new(());
        g.sent(&discover());
        g.sent(&initialize());
        assert!(g.stale(&late()));
        assert!(!g.stale(&init_ok()));
        assert!(!g.stale(&late()));
        assert!(quiet(&g));

        // Never answered: the initialize answer ends the tracking.
        let mut g = Guard::new(());
        g.sent(&discover());
        g.sent(&initialize());
        assert!(!g.stale(&init_ok()));
        assert!(quiet(&g));
    }
}
