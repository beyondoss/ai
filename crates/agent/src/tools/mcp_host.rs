//! Host-facing callbacks for MCP server→client requests (elicitation, sampling).
//!
//! Same shape as [`crate::approval::ApprovalGate`]: an abstract trait with one implementation per
//! host. `serve` broadcasts a frame and parks a oneshot; the default declines so `run` (no UI) never
//! hangs. Installed into process-scoped hubs that every MCP [`ClientHandler`](rmcp::ClientHandler)
//! consults — connect happens before `serve`'s gate exists, so the hub is the late-binding seam.

#![expect(deprecated)] // Sampling: SEP-2577-deprecated but still on the wire.

use std::sync::Arc;

use async_trait::async_trait;
use rmcp::ErrorData as McpError;
use rmcp::model::{
    CreateMessageRequestParams, CreateMessageResult, ElicitRequestParams, ElicitResult,
    ElicitationAction, SamplingMessage,
};
use serde_json::{Value, json};

/// One pending elicitation for a human (or host UI) to answer.
#[derive(Clone, Debug)]
pub struct ElicitationAsk {
    /// Which MCP server asked.
    pub server: String,
    /// Wire params (form schema / URL).
    pub params: ElicitRequestParams,
}

/// Fail-closed outcomes for an unanswered elicitation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElicitationError {
    /// No host UI is attached (e.g. headless `run`).
    NoClient,
    /// The run was cancelled while waiting.
    Cancelled,
    /// The host did not answer in time.
    TimedOut,
}

impl ElicitationError {
    fn to_result(&self) -> ElicitResult {
        // Decline rather than cancel: cancel is "user dismissed"; decline is "client will not provide".
        ElicitResult::new(ElicitationAction::Decline)
    }
}

/// Ask a human (or host) to fulfill an MCP elicitation.
#[async_trait]
pub trait ElicitationGate: Send + Sync {
    async fn elicit(&self, ask: ElicitationAsk) -> Result<ElicitResult, ElicitationError>;
}

/// Always declines — safe default when no UI is wired.
#[derive(Debug, Default, Clone, Copy)]
pub struct DeclineElicitation;

#[async_trait]
impl ElicitationGate for DeclineElicitation {
    async fn elicit(&self, _ask: ElicitationAsk) -> Result<ElicitResult, ElicitationError> {
        Err(ElicitationError::NoClient)
    }
}

/// Late-bound elicitation target. `serve` installs a real gate after connect.
#[derive(Clone)]
pub struct ElicitationHub {
    inner: Arc<std::sync::RwLock<Arc<dyn ElicitationGate>>>,
}

impl Default for ElicitationHub {
    fn default() -> Self {
        Self::new()
    }
}

impl ElicitationHub {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(std::sync::RwLock::new(
                Arc::new(DeclineElicitation) as Arc<dyn ElicitationGate>
            )),
        }
    }

    pub fn install(&self, gate: Arc<dyn ElicitationGate>) {
        if let Ok(mut guard) = self.inner.write() {
            *guard = gate;
        }
    }

    pub async fn elicit(&self, ask: ElicitationAsk) -> ElicitResult {
        let gate = self
            .inner
            .read()
            .ok()
            .map(|g| g.clone())
            .unwrap_or_else(|| Arc::new(DeclineElicitation) as Arc<dyn ElicitationGate>);
        match gate.elicit(ask).await {
            Ok(result) => result,
            Err(err) => {
                tracing::debug!(?err, "MCP elicitation unanswered; declining");
                err.to_result()
            }
        }
    }
}

/// Host-side LLM sampling for MCP `sampling/createMessage` (SEP-2577-deprecated, still on the wire).
#[async_trait]
pub trait SamplingGate: Send + Sync {
    async fn create_message(
        &self,
        params: CreateMessageRequestParams,
    ) -> Result<CreateMessageResult, McpError>;
}

/// Rejects sampling — default until a host opts in.
#[derive(Debug, Default, Clone, Copy)]
pub struct RejectSampling;

#[async_trait]
impl SamplingGate for RejectSampling {
    async fn create_message(
        &self,
        _params: CreateMessageRequestParams,
    ) -> Result<CreateMessageResult, McpError> {
        Err(McpError::method_not_found::<
            rmcp::model::CreateMessageRequestMethod,
        >())
    }
}

/// Late-bound sampling target.
#[derive(Clone)]
pub struct SamplingHub {
    inner: Arc<std::sync::RwLock<Arc<dyn SamplingGate>>>,
}

impl Default for SamplingHub {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplingHub {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(std::sync::RwLock::new(
                Arc::new(RejectSampling) as Arc<dyn SamplingGate>
            )),
        }
    }

    pub fn install(&self, gate: Arc<dyn SamplingGate>) {
        if let Ok(mut guard) = self.inner.write() {
            *guard = gate;
        }
    }

    pub async fn create_message(
        &self,
        params: CreateMessageRequestParams,
    ) -> Result<CreateMessageResult, McpError> {
        let gate = self
            .inner
            .read()
            .ok()
            .map(|g| g.clone())
            .unwrap_or_else(|| Arc::new(RejectSampling) as Arc<dyn SamplingGate>);
        gate.create_message(params).await
    }
}

/// Process-scoped hubs every MCP client handler shares. Connect runs before `serve` can install
/// gates; hubs are the seam.
#[derive(Clone, Default)]
pub struct McpHost {
    pub elicitation: ElicitationHub,
    pub sampling: SamplingHub,
}

impl McpHost {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Serialize elicitation params for a serve frame / test assertion.
pub fn elicitation_params_json(params: &ElicitRequestParams) -> Value {
    serde_json::to_value(params).unwrap_or_else(|_| json!({}))
}

/// Build a trivial assistant sampling result (test / stub hosts).
pub fn sampling_text_result(
    model: impl Into<String>,
    text: impl Into<String>,
) -> CreateMessageResult {
    CreateMessageResult::new(SamplingMessage::assistant_text(text), model.into())
        .with_stop_reason(CreateMessageResult::STOP_REASON_END_TURN)
}
