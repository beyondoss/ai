//! The exec-endpoint double now lives in `beyond-ai-test-support`: the fleet simulator has to stand
//! one up per session for the same reason the service suites do, and a second implementation of the
//! v1.1 protocol would be a second thing to get subtly wrong about `stdin_base64` and the response
//! cap.
pub use beyond_ai_test_support::exec_mock::*;
