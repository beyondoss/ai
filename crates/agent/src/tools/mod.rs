//! The agent's coding tools — pi's tool set, ported.
//!
//! Each tool implements [`agent_core::Tool`]; [`default_registry`] assembles the set the agent
//! advertises to the model. The Beyond platform tools (fork/sync/logs) register here too once added.

use std::sync::Arc;

use agent_core::ToolRegistry;

pub mod bash;
pub mod edit;
pub mod exec;
pub mod find;
pub mod grep;
pub mod ls;
pub mod read;
pub mod write;

/// The default coding tool set: read, write, edit, bash, ls, grep, find.
pub fn default_registry() -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(read::Read));
    reg.register(Arc::new(write::Write));
    reg.register(Arc::new(edit::Edit));
    reg.register(Arc::new(ls::Ls));
    reg.register(Arc::new(grep::Grep));
    reg.register(Arc::new(find::Find));
    reg.register(Arc::new(bash::Bash::real()));
    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_has_the_pi_tool_set() {
        let reg = default_registry();
        for name in ["read", "write", "edit", "bash", "ls", "grep", "find"] {
            assert!(reg.get(name).is_some(), "missing tool: {name}");
        }
        assert_eq!(reg.len(), 7);
    }
}
