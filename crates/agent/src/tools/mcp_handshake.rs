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

use rmcp::RoleClient;
use rmcp::model::{ClientJsonRpcMessage, ClientRequest, RequestId, ServerJsonRpcMessage};
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
            let message = self.inner.receive().await?;
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
